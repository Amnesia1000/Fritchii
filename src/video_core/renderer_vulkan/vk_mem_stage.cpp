// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// The memory stage's seam (fable_memstage_step2_spec.md, sub-step 2a).
//
// Every Vulkan command the buffer cache runtime used to record itself is now an entry: the
// runtime function keeps its own work (null replacement, the route of an upload, staging) and
// pushes an entry; MemPlayer::Exec turns the entry into the scheduler or descriptor queue calls
// the function made before. Exec is the only copy of that code.
//
// EDEN_NX2_EXP123 (SEAM, live, read once a frame in TickFrame):
//   0: every entry is executed the moment it is pushed: today's calls at today's moments.
//   1: inside a fill bracket (MemStage::Fill, FillPlay, Ref) the entries wait in the arena and
//      are played where the call site plays them: right after the call, as today. Outside a
//      bracket (a site that has none, another thread) they still run at once.
//
// The order rule: pending entries are played before anything else is recorded. The scheduler's
// record functions, its flush, finish, dispatch and state reset, and the descriptor queues' adds
// and Acquire, call SeamHook::Watch; while a fill is open or entries are pending, OnUse plays them
// first and counts:
//   around (5.1): something recorded or added a descriptor inside a fill without going through
//                 the player or a sync: a runtime path that was not converted. Must be 0.
//   late (5.2):   entries were still pending outside a fill when something else recorded: a call
//                 site misses its play. Must be 0. The picture is right either way.
// A sync (SeamClock::Finish, Wait, SubmitForHostRead, the multi-range bind, the kept words wait)
// plays the pending entries first and is not counted.
//
// Only the thread that owns the arena defers (the first to open a fill while none is pending);
// a fill on another thread at the same moment runs its entries at once. Pending entries are only
// ever played by the thread that pushed them.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include <boost/container/static_vector.hpp>
#include <fmt/format.h>

#include "common/assert.h"
#include "common/logging.h"
#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_perf.h"
#include "video_core/renderer_vulkan/vk_compute_pass.h"
#include "video_core/renderer_vulkan/vk_mem_stage.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
namespace {

thread_local u32 t_fill_depth = 0;
thread_local u32 t_play_depth = 0;
thread_local MemStage::Site t_site = MemStage::Site::Count;
thread_local char t_thread_marker = 0;

std::atomic<MemStage*> g_stage{nullptr};

uintptr_t ThisThread() noexcept {
    return reinterpret_cast<uintptr_t>(&t_thread_marker);
}

struct PlayDepth {
    PlayDepth() noexcept {
        ++t_play_depth;
    }
    ~PlayDepth() {
        --t_play_depth;
    }
};

/// Stats counters: a plain load and store, no locked instruction. A count racing with another
/// thread's may be lost; the stats line does not need more.
void Bump(std::atomic<u64>& counter, u64 amount = 1) noexcept {
    counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}

constexpr u32 Align8(u32 value) noexcept {
    return (value + 7U) & ~7U;
}

template <typename T>
const T* PayloadAt(const MemStage::Entry& entry, u32 offset) noexcept {
    return reinterpret_cast<const T*>(reinterpret_cast<const u8*>(&entry) + offset);
}

/// Where an entry executed at once is written: per thread, reused for every push.
// One scratch arena per nesting level of immediate pushes: an Exec that pushed again on the same
// thread would otherwise reset (or grow and free) the arena holding the entry it is still
// reading. No path does that today (review of 2a); this keeps it safe if one appears.
thread_local u32 t_scratch_depth = 0;

MemStage::Arena& Scratch() {
    constexpr u32 Levels = 4;
    thread_local std::array<std::unique_ptr<MemStage::Arena>, Levels> arenas;
    ASSERT_MSG(t_scratch_depth < Levels, "memory seam: immediate pushes nested too deep");
    auto& slot = arenas[t_scratch_depth];
    if (!slot) {
        slot = std::make_unique<MemStage::Arena>(4096);
    }
    return *slot;
}

struct ScratchDepth {
    ScratchDepth() noexcept {
        ++t_scratch_depth;
    }
    ~ScratchDepth() {
        --t_scratch_depth;
    }
};

void HostReadBarrier(vk::CommandBuffer cmdbuf) {
    const VkMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                           barrier);
}

constexpr VkMemoryBarrier COPY_READ_BARRIER{
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .pNext = nullptr,
    .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
};
constexpr VkMemoryBarrier COPY_WRITE_BARRIER{
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .pNext = nullptr,
    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
};

constexpr std::array<const char*, static_cast<size_t>(MemStage::Cmd::Count)> CMD_NAMES{
    "upload ahead", "upload batched", "copy",  "pre-copy barrier", "post-copy barrier",
    "clear",        "quad table",     "null init", "index",        "index always",
    "vertex",       "vertices",       "xfb",   "xfbs",             "convert",
    "descriptor",
};
constexpr std::array<const char*, static_cast<size_t>(MemStage::Site::Count)> SITE_NAMES{
    "geometry",      "stage",      "dispatch before", "dispatch indirect", "dispatch after",
    "draw count",    "draw indirect", "list words",   "list args",         "list pages",
    "invalidation",  "download",   "download for read", "frame",           "dma",
    "inline upload", "unmap",      "image copy",      "query",             "fence",
};
constexpr std::array<const char*, static_cast<size_t>(MemStage::Sync::Count)> SYNC_NAMES{
    "finish", "wait", "submit for host read", "multi-range", "kept words",
};

const char* UseName(SeamHook::Use use) {
    switch (use) {
    case SeamHook::Use::Record:
        return "record";
    case SeamHook::Use::RecordUpload:
        return "record upload";
    case SeamHook::Use::OutsideRenderPass:
        return "outside render pass";
    case SeamHook::Use::RenderPass:
        return "render pass";
    case SeamHook::Use::Flush:
        return "flush";
    case SeamHook::Use::Finish:
        return "finish";
    case SeamHook::Use::DispatchWork:
        return "dispatch work";
    case SeamHook::Use::InvalidateState:
        return "invalidate state";
    case SeamHook::Use::DescriptorAdd:
        return "descriptor add";
    case SeamHook::Use::DescriptorAcquire:
        return "descriptor acquire";
    }
    return "?";
}

/// A fill at one of these sites is a packet of its own (the others belong to a draw or dispatch).
bool IsOpSite(MemStage::Site site) {
    switch (site) {
    case MemStage::Site::Invalidation:
    case MemStage::Site::Download:
    case MemStage::Site::DownloadForRead:
    case MemStage::Site::Frame:
    case MemStage::Site::Dma:
    case MemStage::Site::InlineUpload:
    case MemStage::Site::Unmap:
    case MemStage::Site::ImageCopy:
    case MemStage::Site::Query:
    case MemStage::Site::Fence:
        return true;
    default:
        return false;
    }
}

/// FRITCHII_MEMSTAGE_TICKS (boot): 0 the scheduler's ticks, 1 packet ticks, 2 packet ticks
/// checked against the scheduler's answer.
u32 TicksMode() {
    const char* const value = std::getenv("FRITCHII_MEMSTAGE_TICKS");
    if (value == nullptr || value[0] < '0' || value[0] > '9') {
        return 0;
    }
    return (std::min)(static_cast<u32>(std::atoi(value)), 2U);
}

bool IsRecord(SeamHook::Use use) {
    switch (use) {
    case SeamHook::Use::Record:
    case SeamHook::Use::RecordUpload:
    case SeamHook::Use::OutsideRenderPass:
    case SeamHook::Use::RenderPass:
    case SeamHook::Use::DescriptorAdd:
        return true;
    default:
        return false;
    }
}

} // Anonymous namespace

// ---------------------------------------------------------------------------------------------
// The hook

void SeamHook::OnUse(Use use, const std::source_location& location) {
    if (t_play_depth != 0) {
        return;
    }
    MemStage* const stage = g_stage.load(std::memory_order_acquire);
    if (stage == nullptr) {
        return;
    }
    const bool in_fill = t_fill_depth != 0;
    if (in_fill) {
        // A use inside a fill that is not the player's: a record is 5.1; a flush, finish or
        // state reset that did not go through the clock is counted beside it.
        stage->Note(use, location, true);
    }
    if (stage->OwnedHere() && stage->pending) {
        if (!in_fill) {
            stage->Note(use, location, false);
        }
        stage->PlayPending(true);
    }
}

void SeamHook::OnTickEnd(u64 tick) {
    MemStage* const stage = g_stage.load(std::memory_order_acquire);
    if (stage != nullptr && stage->ticks) {
        stage->ticks->Close(tick);
    }
}

// ---------------------------------------------------------------------------------------------
// Arena

MemStage::Arena::Arena(u32 initial_capacity)
    : data{std::make_unique_for_overwrite<u8[]>(initial_capacity)}, capacity{initial_capacity} {}

MemStage::Arena::~Arena() = default;

u32 MemStage::Arena::Reserve(u32 bytes) {
    if (bytes > capacity - cursor) {
        u64 grown = capacity;
        while (grown < u64{cursor} + bytes) {
            grown *= 2;
        }
        ASSERT(grown <= 0x80000000ULL);
        auto larger = std::make_unique_for_overwrite<u8[]>(static_cast<size_t>(grown));
        std::memcpy(larger.get(), data.get(), cursor);
        data = std::move(larger);
        capacity = static_cast<u32>(grown);
    }
    const u32 offset = cursor;
    cursor += bytes;
    return offset;
}

// ---------------------------------------------------------------------------------------------
// Brackets

MemStage::Fill::Fill(MemStage& stage_, Site site_, u32 index_)
    : stage{stage_}, site{site_}, index{index_}, previous_site{t_site} {
    if (t_fill_depth++ == 0) {
        const uintptr_t me = ThisThread();
        uintptr_t expected = 0;
        if (stage.owner.load(std::memory_order_relaxed) != me &&
            stage.owner.compare_exchange_strong(expected, me, std::memory_order_acquire)) {
            // The arena is reset when the next packet begins: nothing in it is pending.
            stage.claim_deferring = stage.seam_on.load(std::memory_order_relaxed);
            stage.arena.Reset();
            stage.play_cursor = 0;
            stage.last_entry = ~0U;
        }
        if (stage.ticks && IsOpSite(site)) {
            stage.op_number = stage.ticks->BeginPacket();
        }
    }
    begin = stage.OwnedHere() ? stage.arena.Size() : 0;
    t_site = site;
    SeamHook::g_watch.fetch_add(1, std::memory_order_relaxed);
    Bump(stage.counters.site_fills[static_cast<size_t>(site)]);
}

MemStage::Fill::~Fill() {
    SeamHook::g_watch.fetch_sub(1, std::memory_order_relaxed);
    if (stage.OwnedHere()) {
        stage.CloseSection(site, index, begin);
    }
    t_site = previous_site;
    if (--t_fill_depth == 0 && stage.OwnedHere() && !stage.pending) {
        stage.owner.store(0, std::memory_order_release);
    }
}

MemStage::FillPlay::~FillPlay() {
    fill.reset();
    stage.Play();
}

MemStage::SyncScope::SyncScope(MemStage* stage, Sync kind) {
    if (stage != nullptr) {
        stage->EnterSync(kind);
    }
    ++t_play_depth;
}

MemStage::SyncScope::~SyncScope() {
    --t_play_depth;
}

// ---------------------------------------------------------------------------------------------
// MemStage

MemStage::MemStage(MemPlayer& player_) : player{player_}, arena{256 * 1024} {
    player.stage = this;
    if (const u32 mode = TicksMode(); mode != 0) {
        ticks = std::make_unique<SeamTicks>(player.scheduler.GetMasterSemaphore(), mode == 2);
        LOG_INFO(Render_Vulkan, "Memory seam: packet ticks (FRITCHII_MEMSTAGE_TICKS={}){}", mode,
                 mode == 2 ? ", checked against the scheduler's ticks" : "");
    }
    g_stage.store(this, std::memory_order_release);
    if (ticks) {
        SeamHook::g_tick_watch.store(true, std::memory_order_release);
    }
}

MemStage::~MemStage() {
    MemStage* expected = this;
    if (g_stage.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel) && ticks) {
        SeamHook::g_tick_watch.store(false, std::memory_order_release);
    }
    if (pending) {
        SeamHook::g_watch.fetch_sub(1, std::memory_order_relaxed);
    }
}

bool MemStage::OwnedHere() const noexcept {
    return owner.load(std::memory_order_relaxed) == ThisThread();
}

bool MemStage::Deferring() const noexcept {
    return t_fill_depth != 0 && t_play_depth == 0 && OwnedHere() && claim_deferring;
}

void MemStage::SetPending() {
    if (!pending) {
        pending = true;
        if (ticks) {
            ticks->SetUnplayed(ticks->Current());
        }
        SeamHook::g_watch.fetch_add(1, std::memory_order_relaxed);
    }
}

void MemStage::PlayPending([[maybe_unused]] bool late) {
    if (!pending) {
        return;
    }
    PlayDepth depth;
    Bump(counters.plays);
    while (play_cursor < arena.Size()) {
        const Entry& entry = *reinterpret_cast<const Entry*>(arena.At(play_cursor));
        play_cursor += entry.bytes;
        player.Exec(entry);
    }
    pending = false;
    if (ticks) {
        ticks->ClearUnplayed();
    }
    last_entry = ~0U;
    SeamHook::g_watch.fetch_sub(1, std::memory_order_relaxed);
}

void MemStage::Play() {
    if (!OwnedHere()) {
        return;
    }
    PlayPending(false);
    if (t_fill_depth == 0) {
        owner.store(0, std::memory_order_release);
    }
}

void MemStage::CloseSection(Site site, u32 index, u32 begin) {
    if (!claim_deferring) {
        return;
    }
    const Span span{begin, arena.Size()};
    switch (site) {
    case Site::Geometry:
        draw.geometry = span;
        break;
    case Site::Stage:
        if (index < draw.stage.size()) {
            draw.stage[index] = span;
        }
        break;
    case Site::DispatchBefore:
        dispatch.before = span;
        break;
    case Site::DispatchAfter:
        dispatch.after = span;
        break;
    default:
        if (t_fill_depth == 1) {
            op.number = ticks ? op_number : ++packet_number;
            op.entries = span;
        }
        break;
    }
}

void MemStage::EnterSync(Sync kind) {
    Bump(counters.syncs[static_cast<size_t>(kind)]);
    if (OwnedHere()) {
        PlayPending(false);
    }
}

void MemStage::BeginDraw(u32 repack_mask, u32 repack_dirty) {
    Bump(counters.draws);
    draw = DrawPacket{};
    draw.number = ticks ? ticks->BeginPacket() : ++packet_number;
    draw.fill_tick = player.scheduler.CurrentTick();
    draw.repack_mask = repack_mask;
    draw.repack_dirty = repack_dirty;
}

void MemStage::BeginDispatch() {
    Bump(counters.dispatches);
    dispatch = DispatchPacket{};
    dispatch.number = ticks ? ticks->BeginPacket() : ++packet_number;
    dispatch.fill_tick = player.scheduler.CurrentTick();
}

void MemStage::CountPush(Cmd kind) noexcept {
    Bump(counters.entries[static_cast<size_t>(kind)]);
    if (t_fill_depth != 0 && t_site != Site::Count) {
        Bump(counters.site_entries[static_cast<size_t>(t_site)]);
    } else {
        Bump(counters.outside[static_cast<size_t>(kind)]);
    }
}

void MemStage::PushParts(Cmd kind, u8 flags, u32 count, std::initializer_list<Part> parts) {
    u32 bytes = static_cast<u32>(sizeof(Entry));
    for (const Part& part : parts) {
        bytes += Align8(part.bytes);
    }
    CountPush(kind);
    const bool defer = Deferring();
    Arena& target = defer ? arena : Scratch();
    if (!defer) {
        target.Reset();
    }
    const u32 offset = target.Reserve(bytes);
    u8* const out = target.At(offset);
    const Entry entry{
        .kind = kind,
        .flags = flags,
        .count = static_cast<u16>((std::min)(count, 0xffffU)),
        .bytes = bytes,
    };
    std::memcpy(out, &entry, sizeof(Entry));
    u32 at = static_cast<u32>(sizeof(Entry));
    for (const Part& part : parts) {
        if (part.bytes != 0) {
            std::memcpy(out + at, part.data, part.bytes);
        }
        at += Align8(part.bytes);
    }
    if (defer) {
        last_entry = offset;
        Bump(counters.deferred);
        SetPending();
        return;
    }
    PlayDepth depth;
    const ScratchDepth scratch_depth;
    player.Exec(*reinterpret_cast<const Entry*>(out));
}

void MemStage::Push(Cmd kind, u8 flags) {
    PushParts(kind, flags, 0, {});
}

void MemStage::PushCopy(Cmd kind, u8 flags, VkBuffer src, VkBuffer dst,
                        std::span<const VkBufferCopy> copies) {
    const CopyPayload payload{
        .src = src,
        .dst = dst,
        .count = static_cast<u32>(copies.size()),
        .reserved = 0,
    };
    PushParts(kind, flags, payload.count, {PartOf(payload), PartOf(copies)});
}

void MemStage::PushClear(VkBuffer dst, u64 offset, u64 size, u32 value) {
    const ClearPayload payload{.dst = dst, .offset = offset, .size = size, .value = value};
    PushParts(Cmd::Clear, OutsideRenderPass, 1, {PartOf(payload)});
}

void MemStage::PushQuadLut(VkBuffer src, VkDeviceSize src_offset, VkBuffer dst,
                           VkDeviceSize size) {
    const QuadLutPayload payload{.src = src, .src_offset = src_offset, .dst = dst, .size = size};
    PushParts(Cmd::QuadLutUpload, OutsideRenderPass, 1, {PartOf(payload)});
}

void MemStage::PushNullInit(VkBuffer buffer) {
    const NullInitPayload payload{.buffer = buffer};
    PushParts(Cmd::NullInit, OutsideRenderPass, 1, {PartOf(payload)});
}

void MemStage::PushIndex(Cmd kind, u8 flags, VkBuffer buffer, VkDeviceSize offset,
                         VkIndexType type) {
    const IndexPayload payload{.buffer = buffer, .offset = offset, .type = type};
    PushParts(kind, flags, 1, {PartOf(payload)});
}

void MemStage::PushVertex(u8 flags, u32 index, VkBuffer buffer, u32 offset, u32 size,
                          u32 stride) {
    const VertexOne payload{
        .buffer = buffer, .index = index, .offset = offset, .size = size, .stride = stride};
    PushParts(Cmd::BindVertex, flags, 1, {PartOf(payload)});
}

void MemStage::PushVertices(u8 flags, u32 first, std::span<const VkBuffer> buffers,
                            std::span<const VkDeviceSize> offsets,
                            std::span<const VkDeviceSize> sizes,
                            std::span<const VkDeviceSize> strides) {
    const u32 count = static_cast<u32>(buffers.size());
    const VerticesPayload payload{.first = first, .count = count};
    PushParts(Cmd::BindVertices, flags, count,
              {PartOf(payload), PartOf(buffers), PartOf(offsets.first(count)),
               PartOf(sizes.first(count)), PartOf(strides.first(count))});
}

void MemStage::PushXfb(u32 index, VkBuffer buffer, u32 offset, u32 size) {
    const XfbOne payload{.buffer = buffer, .index = index, .offset = offset, .size = size};
    PushParts(Cmd::BindXfb, 0, 1, {PartOf(payload)});
}

void MemStage::PushXfbs(std::span<const VkBuffer> buffers, std::span<const VkDeviceSize> offsets,
                        std::span<const VkDeviceSize> sizes) {
    const u32 count = static_cast<u32>(buffers.size());
    const XfbsPayload payload{.count = count, .reserved = 0};
    PushParts(Cmd::BindXfbs, 0, count,
              {PartOf(payload), PartOf(buffers), PartOf(offsets.first(count)),
               PartOf(sizes.first(count))});
}

void MemStage::PushConvert(Pass pass, const ComputePassRun& run) {
    // No OutsideRenderPass flag: the pass's Record asks for it itself, after its descriptor work.
    const ConvertPayload payload{.pass = pass, .run = run};
    PushParts(Cmd::Convert, 0, 1, {PartOf(payload)});
}

void MemStage::PushDescriptor(const DescriptorUpdateEntry& descriptor, bool texel) {
    static_assert(sizeof(DescriptorUpdateEntry) % 8 == 0);
    if (texel) {
        Bump(counters.texel);
    }
    if (!Deferring()) {
        CountPush(Cmd::Descriptors);
        PlayDepth depth;
        player.AddDescriptors(&descriptor, 1);
        return;
    }
    // Consecutive descriptors make one entry: one AddEntries call when played.
    if (last_entry != ~0U) {
        const Entry& last = *reinterpret_cast<const Entry*>(arena.At(last_entry));
        if (last.kind == Cmd::Descriptors && last_entry + last.bytes == arena.Size()) {
            CountPush(Cmd::Descriptors);
            const u32 at = arena.Reserve(static_cast<u32>(sizeof(DescriptorUpdateEntry)));
            std::memcpy(arena.At(at), &descriptor, sizeof(DescriptorUpdateEntry));
            Entry& grown = *reinterpret_cast<Entry*>(arena.At(last_entry));
            grown.bytes += static_cast<u32>(sizeof(DescriptorUpdateEntry));
            auto& payload =
                *reinterpret_cast<DescriptorsPayload*>(arena.At(last_entry + sizeof(Entry)));
            ++payload.count;
            grown.count = static_cast<u16>((std::min)(payload.count, 0xffffU));
            Bump(counters.deferred);
            return;
        }
    }
    const DescriptorsPayload payload{.count = 1, .reserved = 0};
    PushParts(Cmd::Descriptors, 0, 1, {PartOf(payload), PartOf(descriptor)});
}

void MemStage::Note(SeamHook::Use use, const std::source_location& location, bool around) {
    if (around && !IsRecord(use)) {
        Bump(counters.unclocked);
    } else {
        Bump(around ? counters.around : counters.late);
    }
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_WARNING(Render_Vulkan,
                    "Memory seam: {} ({}) {}, site {}: {}:{} {}",
                    around ? (IsRecord(use) ? "a record around the packet"
                                            : "a scheduler use not through the clock")
                           : "a late play",
                    around ? "5.1" : "5.2", UseName(use),
                    t_site == Site::Count ? "none" : SITE_NAMES[static_cast<size_t>(t_site)],
                    location.file_name(), location.line(), location.function_name());
    }
}

void MemStage::TickFrame() {
    const bool on = VideoCore::Nx2Debug::Experiment(123) != 0;
    if (OwnedHere() && pending) {
        // Nothing should be pending between frames: a site that misses its play.
        Note(SeamHook::Use::Flush, std::source_location::current(), false);
        PlayPending(true);
    }
    if (OwnedHere() && t_fill_depth == 0) {
        owner.store(0, std::memory_order_release);
    }
    if (on != seam_on.load(std::memory_order_relaxed)) {
        seam_on.store(on, std::memory_order_relaxed);
        player.ForgetBindings();
        LOG_INFO(Render_Vulkan, "Memory seam (EXP123): {}",
                 on ? "1, recorder and player" : "0, entries run as pushed");
    }
    static const bool stats_env = std::getenv("FRITCHII_MEMSTAGE_SEAM_STATS") != nullptr;
    if (on || stats_env) {
        LogStats();
    }
    if (ticks) {
        ticks->LogStats();
    }
}

void MemStage::LogStats() {
    const u64 now = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count());
    if (last_log_ns == 0) {
        last_log_ns = now;
        Snapshot(reported);
        return;
    }
    if (now - last_log_ns < 10'000'000'000ULL) {
        return;
    }
    const double seconds = static_cast<double>(now - last_log_ns) / 1e9;
    last_log_ns = now;
    Counts current;
    Snapshot(current);
    const auto delta = [&](u64 Counts::*member) { return current.*member - reported.*member; };
    const auto rate = [seconds](u64 value) { return static_cast<double>(value) / seconds; };

    std::string kinds;
    std::string outside;
    for (size_t i = 0; i < CMD_NAMES.size(); ++i) {
        if (const u64 n = current.entries[i] - reported.entries[i]; n != 0) {
            kinds += fmt::format("{}{} {:.0f}", kinds.empty() ? "" : ", ", CMD_NAMES[i], rate(n));
        }
        if (const u64 n = current.outside[i] - reported.outside[i]; n != 0) {
            outside +=
                fmt::format("{}{} {:.0f}", outside.empty() ? "" : ", ", CMD_NAMES[i], rate(n));
        }
    }
    std::string sites;
    u64 draw_entries = 0;
    u64 dispatch_entries = 0;
    for (size_t i = 0; i < SITE_NAMES.size(); ++i) {
        const u64 fills = current.site_fills[i] - reported.site_fills[i];
        const u64 entries = current.site_entries[i] - reported.site_entries[i];
        switch (static_cast<Site>(i)) {
        case Site::Geometry:
        case Site::Stage:
        case Site::DrawCount:
        case Site::DrawIndirect:
        case Site::ListArgs:
            draw_entries += entries;
            break;
        case Site::DispatchBefore:
        case Site::DispatchIndirect:
        case Site::DispatchAfter:
            dispatch_entries += entries;
            break;
        default:
            break;
        }
        if (fills != 0) {
            sites += fmt::format("{}{} {:.0f} ({:.2f} entries each)", sites.empty() ? "" : ", ",
                                 SITE_NAMES[i], rate(fills),
                                 static_cast<double>(entries) / static_cast<double>(fills));
        }
    }
    std::string syncs;
    for (size_t i = 0; i < SYNC_NAMES.size(); ++i) {
        if (const u64 n = current.syncs[i] - reported.syncs[i]; n != 0) {
            syncs += fmt::format("{}{} {:.0f}", syncs.empty() ? "" : ", ", SYNC_NAMES[i], rate(n));
        }
    }
    const u64 draws = delta(&Counts::draws);
    const u64 dispatches = delta(&Counts::dispatches);
    const size_t fence = static_cast<size_t>(Site::Fence);
    const u64 fence_fills = current.site_fills[fence] - reported.site_fills[fence];
    const u64 fence_entries = current.site_entries[fence] - reported.site_entries[fence];
    LOG_INFO(Render_Vulkan,
             "Memory seam (EXP123={}) {:.0f} s: draws {:.0f}/s ({:.2f} entries each), "
             "dispatches {:.0f}/s ({:.2f} each), fence commits {:.0f}/s ({:.2f} each); "
             "deferred {:.0f}/s, plays {:.0f}/s; around the packet (5.1) {}, late plays (5.2) {}, "
             "scheduler uses not through the clock {}",
             seam_on.load(std::memory_order_relaxed) ? 1 : 0, seconds, rate(draws),
             draws != 0 ? static_cast<double>(draw_entries) / static_cast<double>(draws) : 0.0,
             rate(dispatches),
             dispatches != 0 ? static_cast<double>(dispatch_entries) / static_cast<double>(dispatches)
                             : 0.0,
             rate(fence_fills),
             fence_fills != 0 ? static_cast<double>(fence_entries) / static_cast<double>(fence_fills)
                              : 0.0,
             rate(delta(&Counts::deferred)), rate(delta(&Counts::plays)),
             delta(&Counts::around), delta(&Counts::late), delta(&Counts::unclocked));
    LOG_INFO(Render_Vulkan, "  memory seam entries/s by kind: {}; texel buffer descriptors {:.0f}/s",
             kinds.empty() ? "none" : kinds, rate(delta(&Counts::texel)));
    LOG_INFO(Render_Vulkan, "  memory seam fills/s by site: {}", sites.empty() ? "none" : sites);
    LOG_INFO(Render_Vulkan, "  memory seam syncs/s: {}; entries/s run outside any fill: {}",
             syncs.empty() ? "none" : syncs, outside.empty() ? "none" : outside);
    reported = current;
}

void MemStage::Snapshot(Counts& out) const {
    for (size_t i = 0; i < out.entries.size(); ++i) {
        out.entries[i] = counters.entries[i].load(std::memory_order_relaxed);
        out.outside[i] = counters.outside[i].load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < out.site_fills.size(); ++i) {
        out.site_fills[i] = counters.site_fills[i].load(std::memory_order_relaxed);
        out.site_entries[i] = counters.site_entries[i].load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < out.syncs.size(); ++i) {
        out.syncs[i] = counters.syncs[i].load(std::memory_order_relaxed);
    }
    out.draws = counters.draws.load(std::memory_order_relaxed);
    out.dispatches = counters.dispatches.load(std::memory_order_relaxed);
    out.deferred = counters.deferred.load(std::memory_order_relaxed);
    out.plays = counters.plays.load(std::memory_order_relaxed);
    out.around = counters.around.load(std::memory_order_relaxed);
    out.late = counters.late.load(std::memory_order_relaxed);
    out.unclocked = counters.unclocked.load(std::memory_order_relaxed);
    out.texel = counters.texel.load(std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------------------------
// SeamClock. With FRITCHII_MEMSTAGE_TICKS = 0 (or no stage), Now, IsFree and KnownGpu read the
// scheduler's master semaphore, exactly what Scheduler::CurrentTick and IsFree read; with 1 or 2
// they ask SeamTicks (below). The syncs play the pending entries first.

SeamClock::SeamClock(Scheduler& scheduler_, MemStage* stage_)
    : scheduler{&scheduler_}, master{&scheduler_.GetMasterSemaphore()}, stage{stage_},
      ticks{stage_ != nullptr ? stage_->Ticks() : nullptr} {}

void SeamClock::Finish(std::source_location location) {
    MemStage::SyncScope sync{stage, MemStage::Sync::Finish};
    scheduler->Finish({}, {}, location);
}

void SeamClock::Wait(u64 stamp) {
    MemStage::SyncScope sync{stage, MemStage::Sync::Wait};
    // Translated after the scope played the pending entries: a number still unpublished then is
    // recorded in the current tick, which Scheduler::Wait flushes (and so publishes).
    scheduler->Wait(ticks != nullptr ? ticks->WaitTickOf(stamp) : stamp);
}

void SeamClock::WaitTick(u64 tick) {
    MemStage::SyncScope sync{stage, MemStage::Sync::Wait};
    scheduler->Wait(tick);
}

u64 SeamClock::SubmitForHostRead(std::source_location location) {
    MemStage::SyncScope sync{stage, MemStage::Sync::SubmitForHostRead};
    scheduler->RequestOutsideRenderPassOperationContext(location);
    scheduler->Record(HostReadBarrier);
    if (ticks == nullptr) {
        return scheduler->Flush({}, {}, location);
    }
    // Nothing is pending after the scope: the number current now is the one this flush closes,
    // and the stamp of the tick it submits.
    const u64 stamp = ticks->Now();
    const u64 tick = scheduler->Flush({}, {}, location);
    ticks->CheckHostRead(stamp, tick);
    return stamp;
}

// ---------------------------------------------------------------------------------------------
// Packet ticks: FRITCHII_MEMSTAGE_TICKS >= 1 (boot), fable_memstage_step2_spec.md 7.2, build
// step 4.
//
// A stamp is a packet number, not a scheduler tick. The stage counts its packets up (a draw, a
// dispatch and an operation packet each take the next number: BeginPacket), and every stamp the
// buffer cache side takes (SeamClock::Now: buffer usage, staging and stream regions, recycled
// buffers, write ticks, probes, list page copies, deleted buffers) is the number current then.
// The thread that records publishes, at the end of every scheduler tick, "every number up to N
// was recorded in tick K" (Close, called by Scheduler::SubmitExecution right before the tick
// advances). A stamp is free when the tick its number was published with is complete.
//
// The published table is a short ring of (N, K), in order, for the ticks not yet seen complete;
// `free_number` is the highest N whose K is known complete. IsFree is a load or two in the
// common cases (at or below free_number: free; above the last published N: not free) and
// otherwise advances free_number to the master semaphore's known GPU tick under the lock. So it
// never asks the GPU more than Scheduler::IsFree did (it does not refresh).
//
// What N a close publishes. Entries the stage filled and the player has not played are not
// recorded yet: their number (`unplayed_from`, set when the first entry starts waiting) and every
// later one stay open, and are published with a later tick, the one their play lands in. That is
// what makes sub-step 2b's tick guard (StagingBufferPool::Retick) unnecessary with packet ticks:
// a staging buffer stamped at an early fill is not free before the tick its copy is recorded in.
// Otherwise the close takes the current number and moves the stage on to the next, so a stamp
// taken after the tick ended is never published with it. On one thread each number then lives
// in exactly one tick, and every answer is today's: IsFree(N) == IsFree(the tick N was handed
// out in).
//
// Step 6 (the stage on its own thread, ahead of the GPU thread) keeps the stamps, the ring and
// the reads as they are. Only Close changes: the GPU thread publishes the last packet it has
// played and never moves `open` (the stage alone does). Waits on a stamp not yet published
// (TickOf falls back to the current tick, right only when the two are level) become a join:
// the stage asks the GPU thread to play up to that packet and flush.
//
// Users that compare stamps for equality (the stream uniform reuse and its keeps: "same tick, so
// the stream copy is still there") take Epoch(): the number of closes. A copy made at epoch E
// carries a number not yet published; its stream region is only recycled once that number is
// published and complete, which is a later close, so "still epoch E" keeps meaning "not
// recycled" in step 6. On one thread there is one close per tick and the epoch is the tick.
//
// FRITCHII_MEMSTAGE_TICKS = 2 checks every answer against the scheduler's: each stamp handed out
// notes the tick current then (a ring keyed by number, 2^18 numbers back), and IsFree, the
// stream regions' known-GPU test, the waits, the host read flushes and the epoch compare with
// what that tick says. The "Memory seam ticks" line every 10 s:
//   early:     the packet answer frees (or waits for) less than the tick answer. Unsafe; must be 0.
//   late:      the packet answer holds longer. Must be 0 on one thread.
//   carried:   numbers a tick ended with unplayed (an early fill across a flush): skipped, the
//              packet answer is the right one there and the tick answer is the one 2b guards.
//   split:     a number handed out in two ticks without a carry (a tick ended without a close).
//              Must be 0.
//   unchecked: stamps older than the ring.
//   epoch off: Epoch() differed from the scheduler's tick. Must be 0.
// Deleted buffers held past their frames (buffer_cache.h's ring waits for their stamp as well)
// are counted with 1 and 2; on one thread they must be 0.

namespace {
constexpr u64 StampSlots = u64{1} << 18;
constexpr std::array<const char*, 4> CHECK_NAMES{"is free", "stream known", "wait", "host read"};

thread_local u64 t_known_tick = 0;
std::atomic<u32> g_ticks_logged{0};

u64 SteadyNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}
} // Anonymous namespace

SeamTicks::SeamTicks(MasterSemaphore& master_, bool verify_) : master{&master_}, verify{verify_} {
    epoch.store(master->CurrentTick(), std::memory_order_release);
    if (verify) {
        slots = std::make_unique<StampSlot[]>(StampSlots);
    }
}

SeamTicks::~SeamTicks() = default;

u64 SeamTicks::Advance(u64 known) {
    std::scoped_lock lock{mutex};
    return AdvanceLocked(known);
}

u64 SeamTicks::AdvanceLocked(u64 known) {
    u64 free = free_number.load(std::memory_order_relaxed);
    while (!ring.empty() && ring.front().tick <= known) {
        free = ring.front().last;
        ring.pop_front();
    }
    // free_number before advanced_gpu: a reader that sees the new known GPU tick there sees the
    // numbers it frees.
    free_number.store(free, std::memory_order_release);
    if (known > advanced_gpu.load(std::memory_order_relaxed)) {
        advanced_gpu.store(known, std::memory_order_release);
    }
    return free;
}

void SeamTicks::Close(u64 tick) {
    std::scoped_lock lock{mutex};
    const u64 unplayed = unplayed_from.load(std::memory_order_acquire);
    u64 last;
    if (unplayed != 0) {
        last = unplayed - 1;
        Bump(counters.carried);
        if (verify) {
            const u64 to = open.load(std::memory_order_acquire);
            carried_ranges.push_back(Range{.from = unplayed, .to = to});
            if (carried_ranges.size() > 1024) {
                carried_ranges.pop_front();
            }
            carried_to.store(to, std::memory_order_release);
        }
    } else {
        last = open.fetch_add(1, std::memory_order_acq_rel);
    }
    if (last > published.load(std::memory_order_relaxed)) {
        ring.push_back(Published{.last = last, .tick = tick});
        published.store(last, std::memory_order_release);
    }
    epoch.fetch_add(1, std::memory_order_acq_rel);
    Bump(counters.closes);
    AdvanceLocked(master->KnownGpuTick());
    if (const u64 depth = ring.size(); depth > counters.max_depth.load(std::memory_order_relaxed)) {
        counters.max_depth.store(depth, std::memory_order_relaxed);
    }
}

u64 SeamTicks::TickOf(u64 stamp) {
    std::scoped_lock lock{mutex};
    if (stamp <= free_number.load(std::memory_order_relaxed)) {
        // Complete: any complete tick does for a wait or a submitted test.
        return advanced_gpu.load(std::memory_order_relaxed);
    }
    const auto it = std::lower_bound(
        ring.begin(), ring.end(), stamp,
        [](const Published& entry, u64 value) { return entry.last < value; });
    // Not published: still open, so its uses are in the tick being recorded.
    return it != ring.end() ? it->tick : master->CurrentTick();
}

u64 SeamTicks::WaitTickOf(u64 stamp) {
    const u64 tick = TickOf(stamp);
    if (verify) [[unlikely]] {
        u64 old_tick{};
        if (stamp > free_number.load(std::memory_order_acquire) && OldTick(stamp, old_tick)) {
            Bump(counters.checks[static_cast<size_t>(Check::Wait)]);
            if (tick != old_tick) {
                Disagree(Check::Wait, tick < old_tick, stamp, old_tick, tick);
            }
        }
    }
    return tick;
}

void SeamTicks::CheckHostRead(u64 stamp, u64 tick) {
    if (!verify) {
        return;
    }
    const u64 published_tick = TickOf(stamp);
    if (stamp <= free_number.load(std::memory_order_acquire)) {
        return;
    }
    Bump(counters.checks[static_cast<size_t>(Check::HostRead)]);
    if (published_tick != tick) {
        Disagree(Check::HostRead, published_tick < tick, stamp, tick, published_tick);
    }
}

void SeamTicks::NoteHeld(u64 newly_held) noexcept {
    if (newly_held != 0) {
        Bump(counters.held, newly_held);
    }
}

void SeamTicks::NoteStamp(u64 number) noexcept {
    StampSlot& slot = slots[number & (StampSlots - 1)];
    const u64 tick = master->CurrentTick();
    if (slot.number.load(std::memory_order_acquire) != number) {
        slot.first.store(tick, std::memory_order_relaxed);
        slot.last.store(tick, std::memory_order_relaxed);
        slot.number.store(number, std::memory_order_release);
    } else if (slot.last.load(std::memory_order_relaxed) != tick) {
        slot.last.store(tick, std::memory_order_relaxed);
    }
}

bool SeamTicks::OldTick(u64 stamp, u64& tick) {
    if (stamp == 0 || stamp == Never) {
        // Never stamped (free) and the deferred staging buffers' "never free".
        tick = stamp;
        return true;
    }
    const StampSlot& slot = slots[stamp & (StampSlots - 1)];
    if (slot.number.load(std::memory_order_acquire) != stamp) {
        Bump(counters.unchecked);
        return false;
    }
    const u64 first = slot.first.load(std::memory_order_relaxed);
    const u64 last = slot.last.load(std::memory_order_relaxed);
    bool carried = false;
    if (stamp <= carried_to.load(std::memory_order_acquire)) {
        std::scoped_lock lock{mutex};
        carried = std::ranges::any_of(carried_ranges, [stamp](const Range& range) {
            return stamp >= range.from && stamp <= range.to;
        });
    }
    if (carried) {
        Bump(counters.carried_skipped);
        return false;
    }
    if (first != last) {
        Bump(counters.split);
        if (g_ticks_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_WARNING(Render_Vulkan,
                        "Memory seam ticks: packet {} was handed out in ticks {} and {} without "
                        "a carry (a tick ended without a close)",
                        stamp, first, last);
        }
        return false;
    }
    tick = first;
    return true;
}

bool SeamTicks::IsFreeChecked(u64 stamp) {
    const u64 known_before = master->KnownGpuTick();
    const bool free = IsFreeFast(stamp);
    const u64 known_after = master->KnownGpuTick();
    u64 old_tick{};
    if (!OldTick(stamp, old_tick)) {
        return free;
    }
    Bump(counters.checks[static_cast<size_t>(Check::IsFree)]);
    // The packet answer came from a known GPU tick between the two reads.
    if (free && old_tick > known_after) {
        Disagree(Check::IsFree, true, stamp, old_tick, known_after);
    } else if (!free && old_tick <= known_before) {
        Disagree(Check::IsFree, false, stamp, old_tick, known_before);
    }
    return free;
}

u64 SeamTicks::KnownFreeChecked(u64 known) {
    std::scoped_lock lock{mutex};
    const u64 free = AdvanceLocked(known);
    // The tick that answer stands for (another thread may have advanced past `known`).
    t_known_tick = advanced_gpu.load(std::memory_order_relaxed);
    return free;
}

void SeamTicks::CheckKnown(u64 stamp, bool free) {
    u64 old_tick{};
    if (!OldTick(stamp, old_tick)) {
        return;
    }
    Bump(counters.checks[static_cast<size_t>(Check::Known)]);
    const bool old_free = old_tick <= t_known_tick;
    if (free != old_free) {
        Disagree(Check::Known, free, stamp, old_tick, t_known_tick);
    }
}

void SeamTicks::CheckEpoch(u64 value) {
    const u64 tick = master->CurrentTick();
    if (value != tick) {
        Bump(counters.epoch_off);
        if (g_ticks_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_WARNING(Render_Vulkan, "Memory seam ticks: epoch {} while the tick is {}", value,
                        tick);
        }
    }
}

void SeamTicks::Disagree(Check kind, bool early, u64 stamp, u64 old_tick, u64 detail) {
    const size_t index = static_cast<size_t>(kind);
    Bump(early ? counters.early[index] : counters.late[index]);
    if (g_ticks_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_WARNING(Render_Vulkan,
                    "Memory seam ticks: {} answer {} for packet {}: tick {} by the scheduler's "
                    "ticks, {} by the packet ticks (open {}, free up to {}, published up to {})",
                    CHECK_NAMES[index], early ? "early" : "late", stamp, old_tick, detail,
                    open.load(std::memory_order_relaxed),
                    free_number.load(std::memory_order_relaxed),
                    published.load(std::memory_order_relaxed));
    }
}

void SeamTicks::Take(Snap& out) const noexcept {
    out.numbers = open.load(std::memory_order_relaxed);
    out.closes = counters.closes.load(std::memory_order_relaxed);
    out.carried = counters.carried.load(std::memory_order_relaxed);
    out.held = counters.held.load(std::memory_order_relaxed);
    for (size_t i = 0; i < out.checks.size(); ++i) {
        out.checks[i] = counters.checks[i].load(std::memory_order_relaxed);
        out.early[i] = counters.early[i].load(std::memory_order_relaxed);
        out.late[i] = counters.late[i].load(std::memory_order_relaxed);
    }
    out.carried_skipped = counters.carried_skipped.load(std::memory_order_relaxed);
    out.unchecked = counters.unchecked.load(std::memory_order_relaxed);
    out.split = counters.split.load(std::memory_order_relaxed);
    out.epoch_off = counters.epoch_off.load(std::memory_order_relaxed);
}

void SeamTicks::LogStats() {
    const u64 now = SteadyNs();
    if (last_log_ns == 0) {
        last_log_ns = now;
        Take(reported);
        return;
    }
    if (now - last_log_ns < 10'000'000'000ULL) {
        return;
    }
    const double seconds = static_cast<double>(now - last_log_ns) / 1e9;
    last_log_ns = now;
    Snap current;
    Take(current);
    const auto rate = [seconds](u64 value) { return static_cast<double>(value) / seconds; };
    const u64 depth = counters.max_depth.exchange(0, std::memory_order_relaxed);
    LOG_INFO(Render_Vulkan,
             "Memory seam ticks (FRITCHII_MEMSTAGE_TICKS={}) {:.0f} s: packets {:.0f}/s, tick "
             "closes {:.0f}/s, published ticks waiting at most {}, carried closes {}, deleted "
             "buffers held past their frames {}",
             verify ? 2 : 1, seconds, rate(current.numbers - reported.numbers),
             rate(current.closes - reported.closes), depth, current.carried - reported.carried,
             current.held - reported.held);
    if (verify) {
        std::string checks;
        u64 early = 0;
        u64 late = 0;
        for (size_t i = 0; i < CHECK_NAMES.size(); ++i) {
            const u64 e = current.early[i] - reported.early[i];
            const u64 l = current.late[i] - reported.late[i];
            early += e;
            late += l;
            checks += fmt::format("{}{} {} (early {}, late {})", checks.empty() ? "" : ", ",
                                  CHECK_NAMES[i], current.checks[i] - reported.checks[i], e, l);
        }
        LOG_INFO(Render_Vulkan,
                 "  memory seam ticks check: early {}, late {}; {}; carried skipped {}, split {}, "
                 "unchecked {}, epoch off {}",
                 early, late, checks, current.carried_skipped - reported.carried_skipped,
                 current.split - reported.split, current.unchecked - reported.unchecked,
                 current.epoch_off - reported.epoch_off);
    }
    reported = current;
}

// ---------------------------------------------------------------------------------------------
// MemPlayer

MemPlayer::MemPlayer(const Device& device_, MemoryAllocator& memory_allocator_,
                     Scheduler& scheduler_, GuestDescriptorQueue& guest_descriptor_queue_,
                     ComputePassDescriptorQueue& compute_pass_descriptor_queue_)
    : device{device_}, memory_allocator{memory_allocator_}, scheduler{scheduler_},
      guest_descriptor_queue{guest_descriptor_queue_},
      compute_pass_descriptor_queue{compute_pass_descriptor_queue_} {
    static_assert(NUM_BOUND_VERTEX_BUFFERS >= VideoCommon::NUM_VERTEX_BUFFERS);
}

MemPlayer::~MemPlayer() = default;

void MemPlayer::SetPasses(Uint8Pass* uint8, QuadIndexedPass* quad_indexed,
                          VertexRepackPass* vertex_repack) noexcept {
    uint8_pass = uint8;
    quad_indexed_pass = quad_indexed;
    vertex_repack_pass = vertex_repack;
}

void MemPlayer::ForgetBindings() noexcept {
    bound_index_valid = false;
    for (BoundVertexBuffer& bound : bound_vertex_buffers) {
        bound.epoch = ~u64{0};
    }
}

void MemPlayer::AddDescriptors(const DescriptorUpdateEntry* entries, size_t count) {
    guest_descriptor_queue.AddEntries(entries, count);
}

void MemPlayer::Exec(const MemStage::Entry& entry) {
    using Cmd = MemStage::Cmd;
    constexpr u32 HEAD = static_cast<u32>(sizeof(MemStage::Entry));
    if ((entry.flags & MemStage::OutsideRenderPass) != 0) {
        scheduler.RequestOutsideRenderPassOperationContext();
    }
    switch (entry.kind) {
    case Cmd::UploadAhead:
    case Cmd::UploadBatched:
    case Cmd::Copy: {
        const auto& payload = *PayloadAt<MemStage::CopyPayload>(entry, HEAD);
        const VkBufferCopy* const copies = PayloadAt<VkBufferCopy>(
            entry, HEAD + Align8(static_cast<u32>(sizeof(MemStage::CopyPayload))));
        const VkBuffer src_buffer = payload.src;
        const VkBuffer dst_buffer = payload.dst;
        if (entry.kind == Cmd::UploadBatched) {
            scheduler.RecordUpload(src_buffer, dst_buffer,
                                   std::span<const VkBufferCopy>{copies, payload.count});
            break;
        }
        const boost::container::small_vector<VkBufferCopy, 8> vk_copies(copies,
                                                                         copies + payload.count);
        if (entry.kind == Cmd::UploadAhead) {
            scheduler.RecordWithUploadBuffer([src_buffer, dst_buffer, vk_copies](
                                                 vk::CommandBuffer, vk::CommandBuffer upload_cmdbuf) {
                upload_cmdbuf.CopyBuffer(src_buffer, dst_buffer, vk_copies);
            });
            break;
        }
        const bool barrier = (entry.flags & MemStage::Barrier) != 0;
        scheduler.Record([src_buffer, dst_buffer, vk_copies, barrier](vk::CommandBuffer cmdbuf) {
            if (barrier) {
                cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, COPY_READ_BARRIER);
            }
            cmdbuf.CopyBuffer(src_buffer, dst_buffer, vk_copies);
            if (barrier) {
                // Later transfers too: a reordered upload of the same bytes in a later submission
                // (RecordWithUploadBuffer, ahead of that submission's work) has no barrier of its
                // own before it, and in-order copies don't count as uses for CanReorderUpload.
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, 0,
                                       COPY_WRITE_BARRIER);
            }
        });
        break;
    }
    case Cmd::PreCopyBarrier:
        scheduler.Record([](vk::CommandBuffer cmdbuf) {
            cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0, COPY_READ_BARRIER);
        });
        break;
    case Cmd::PostCopyBarrier:
        scheduler.Record([](vk::CommandBuffer cmdbuf) {
            // Later transfers too, as after CopyBuffer's own barrier: the downloads, gathers and
            // compute uploads this closes are in-order copies a later reordered upload of the
            // same bytes would not wait for.
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, 0,
                                   COPY_WRITE_BARRIER);
        });
        break;
    case Cmd::Clear: {
        const auto& payload = *PayloadAt<MemStage::ClearPayload>(entry, HEAD);
        scheduler.Record([dest_buffer = payload.dst, offset = payload.offset, size = payload.size,
                          value = payload.value](vk::CommandBuffer cmdbuf) {
            cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0, COPY_READ_BARRIER);
            cmdbuf.FillBuffer(dest_buffer, offset, size, value);
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   vk::PIPELINE_STAGE_GRAPHICS_COMPUTE, 0, COPY_WRITE_BARRIER);
        });
        break;
    }
    case Cmd::QuadLutUpload: {
        const auto& payload = *PayloadAt<MemStage::QuadLutPayload>(entry, HEAD);
        scheduler.Record([src_buffer = payload.src, src_offset = payload.src_offset,
                          dst_buffer = payload.dst,
                          size_bytes = payload.size](vk::CommandBuffer cmdbuf) {
            const VkBufferCopy copy{
                .srcOffset = src_offset,
                .dstOffset = 0,
                .size = size_bytes,
            };
            const VkBufferMemoryBarrier write_barrier{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_INDEX_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = dst_buffer,
                .offset = 0,
                .size = size_bytes,
            };
            cmdbuf.CopyBuffer(src_buffer, dst_buffer, copy);
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, write_barrier);
        });
        break;
    }
    case Cmd::NullInit: {
        const auto& payload = *PayloadAt<MemStage::NullInitPayload>(entry, HEAD);
        scheduler.Record([buffer = payload.buffer](vk::CommandBuffer cmdbuf) {
            cmdbuf.FillBuffer(buffer, 0, VK_WHOLE_SIZE, 0);
        });
        break;
    }
    case Cmd::BindIndex: {
        const auto& payload = *PayloadAt<MemStage::IndexPayload>(entry, HEAD);
        const VkBuffer vk_buffer = payload.buffer;
        const VkDeviceSize vk_offset = payload.offset;
        const VkIndexType vk_index_type = payload.type;
        // The same index buffer is still bound in this command buffer: skip the rebind.
        if (VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::IndexRebind) &&
            !VideoCommon::Nx2Perf::Verify()) {
            const u64 epoch = scheduler.RecordingEpoch();
            if (bound_index_valid && vk_buffer == bound_index_buffer &&
                vk_offset == bound_index_offset && vk_index_type == bound_index_type &&
                epoch == bound_index_epoch) {
                break;
            }
            bound_index_valid = true;
            bound_index_buffer = vk_buffer;
            bound_index_offset = vk_offset;
            bound_index_type = vk_index_type;
            bound_index_epoch = epoch;
        }
        scheduler.Record([vk_buffer, vk_offset, vk_index_type](vk::CommandBuffer cmdbuf) {
            cmdbuf.BindIndexBuffer(vk_buffer, vk_offset, vk_index_type);
        });
        break;
    }
    case Cmd::BindIndexAlways: {
        bound_index_valid = false;
        if ((entry.flags & MemStage::ForgetOnly) != 0) {
            break;
        }
        const auto& payload = *PayloadAt<MemStage::IndexPayload>(entry, HEAD);
        scheduler.Record([buffer = payload.buffer, offset = payload.offset,
                          type = payload.type](vk::CommandBuffer cmdbuf) {
            cmdbuf.BindIndexBuffer(buffer, offset, type);
        });
        break;
    }
    case Cmd::BindVertex: {
        const auto& payload = *PayloadAt<MemStage::VertexOne>(entry, HEAD);
        const u32 index = payload.index;
        if (index < VideoCommon::NUM_VERTEX_BUFFERS) {
            bound_vertex_buffers[index].epoch = ~u64{0};
        }
        if ((entry.flags & MemStage::ForgetOnly) != 0) {
            break;
        }
        const VkBuffer buffer = payload.buffer;
        const u32 offset = payload.offset;
        if ((entry.flags & MemStage::DynamicState) != 0) {
            scheduler.Record([index, buffer, offset, size = payload.size,
                              stride = payload.stride](vk::CommandBuffer cmdbuf) {
                const VkDeviceSize vk_offset = buffer != VK_NULL_HANDLE ? offset : 0;
                const VkDeviceSize vk_size = buffer != VK_NULL_HANDLE ? size : VK_WHOLE_SIZE;
                const VkDeviceSize vk_stride = stride;
                cmdbuf.BindVertexBuffers2EXT(index, 1, &buffer, &vk_offset, &vk_size, &vk_stride);
            });
        } else {
            scheduler.Record([index, buffer, offset](vk::CommandBuffer cmdbuf) {
                cmdbuf.BindVertexBuffer(index, buffer, offset);
            });
        }
        break;
    }
    case Cmd::BindVertices: {
        const auto& payload = *PayloadAt<MemStage::VerticesPayload>(entry, HEAD);
        const u32 first = payload.first;
        const u32 binding_count = payload.count;
        const u32 handles_at = HEAD + Align8(static_cast<u32>(sizeof(MemStage::VerticesPayload)));
        const u32 array_bytes = binding_count * 8;
        const VkBuffer* const handles = PayloadAt<VkBuffer>(entry, handles_at);
        const VkDeviceSize* const offsets = PayloadAt<VkDeviceSize>(entry, handles_at + array_bytes);
        const VkDeviceSize* const sizes =
            PayloadAt<VkDeviceSize>(entry, handles_at + 2 * array_bytes);
        const VkDeviceSize* const strides =
            PayloadAt<VkDeviceSize>(entry, handles_at + 3 * array_bytes);
        // The same buffers are still bound in this command buffer (as for index buffers): skip
        // the rebind. Draws re-resolve their vertex streams when the guest rewrites unchanged
        // registers.
        if (!VideoCommon::Nx2Perf::Verify()) {
            const u64 epoch = scheduler.RecordingEpoch();
            bool unchanged = true;
            for (u32 i = 0; unchanged && i < binding_count; ++i) {
                const BoundVertexBuffer& bound = bound_vertex_buffers[first + i];
                unchanged = bound.epoch == epoch && bound.buffer == handles[i] &&
                            bound.offset == offsets[i] && bound.size == sizes[i] &&
                            bound.stride == strides[i];
            }
            if (unchanged) {
                break;
            }
            for (u32 i = 0; i < binding_count; ++i) {
                bound_vertex_buffers[first + i] = {handles[i], offsets[i], sizes[i], strides[i],
                                                   epoch};
            }
        }
        using Handles = boost::container::static_vector<VkBuffer, NUM_BOUND_VERTEX_BUFFERS>;
        using Sizes = boost::container::static_vector<VkDeviceSize, NUM_BOUND_VERTEX_BUFFERS>;
        Handles buffer_handles(handles, handles + binding_count);
        Sizes buffer_offsets(offsets, offsets + binding_count);
        if ((entry.flags & MemStage::DynamicState) != 0) {
            Sizes buffer_sizes(sizes, sizes + binding_count);
            Sizes buffer_strides(strides, strides + binding_count);
            scheduler.Record([first, binding_count, buffer_handles_ = std::move(buffer_handles),
                              offsets_ = std::move(buffer_offsets),
                              sizes_ = std::move(buffer_sizes),
                              strides_ = std::move(buffer_strides)](vk::CommandBuffer cmdbuf) {
                cmdbuf.BindVertexBuffers2EXT(first, binding_count, buffer_handles_.data(),
                                             offsets_.data(), sizes_.data(), strides_.data());
            });
        } else {
            scheduler.Record([first, binding_count, buffer_handles_ = std::move(buffer_handles),
                              offsets_ = std::move(buffer_offsets)](vk::CommandBuffer cmdbuf) {
                cmdbuf.BindVertexBuffers(first, binding_count, buffer_handles_.data(),
                                         offsets_.data());
            });
        }
        break;
    }
    case Cmd::BindXfb: {
        const auto& payload = *PayloadAt<MemStage::XfbOne>(entry, HEAD);
        scheduler.Record([index = payload.index, buffer = payload.buffer, offset = payload.offset,
                          size = payload.size](vk::CommandBuffer cmdbuf) {
            const VkDeviceSize vk_offset = offset;
            const VkDeviceSize vk_size = size;
            cmdbuf.BindTransformFeedbackBuffersEXT(index, 1, &buffer, &vk_offset, &vk_size);
        });
        break;
    }
    case Cmd::BindXfbs: {
        const auto& payload = *PayloadAt<MemStage::XfbsPayload>(entry, HEAD);
        const u32 count = payload.count;
        const u32 handles_at = HEAD + Align8(static_cast<u32>(sizeof(MemStage::XfbsPayload)));
        const VkBuffer* const handles = PayloadAt<VkBuffer>(entry, handles_at);
        const VkDeviceSize* const offsets = PayloadAt<VkDeviceSize>(entry, handles_at + count * 8);
        const VkDeviceSize* const sizes =
            PayloadAt<VkDeviceSize>(entry, handles_at + 2 * count * 8);
        using Handles = boost::container::static_vector<VkBuffer, NUM_BOUND_VERTEX_BUFFERS>;
        using Sizes = boost::container::static_vector<VkDeviceSize, NUM_BOUND_VERTEX_BUFFERS>;
        scheduler.Record([buffer_handles_ = Handles(handles, handles + count),
                          offsets_ = Sizes(offsets, offsets + count),
                          sizes_ = Sizes(sizes, sizes + count)](vk::CommandBuffer cmdbuf) {
            cmdbuf.BindTransformFeedbackBuffersEXT(0, static_cast<u32>(buffer_handles_.size()),
                                                   buffer_handles_.data(), offsets_.data(),
                                                   sizes_.data());
        });
        break;
    }
    case Cmd::Convert: {
        const auto& payload = *PayloadAt<MemStage::ConvertPayload>(entry, HEAD);
        switch (payload.pass) {
        case MemStage::Pass::Uint8:
            uint8_pass->Record(payload.run);
            break;
        case MemStage::Pass::QuadIndexed:
            quad_indexed_pass->Record(payload.run);
            break;
        case MemStage::Pass::VertexRepack:
            vertex_repack_pass->Record(payload.run);
            break;
        }
        break;
    }
    case Cmd::Descriptors: {
        const auto& payload = *PayloadAt<MemStage::DescriptorsPayload>(entry, HEAD);
        AddDescriptors(PayloadAt<DescriptorUpdateEntry>(
                           entry, HEAD + Align8(static_cast<u32>(sizeof(MemStage::DescriptorsPayload)))),
                       payload.count);
        break;
    }
    case Cmd::Count:
        ASSERT(false);
        break;
    }
}

// ---------------------------------------------------------------------------------------------
// The list draw facility. Its stamps are the tick a copy was recorded in, its wait is on that
// tick and the draw takes its result here: facts about what the GPU thread recorded. Its two
// buffers are created with the main allocator.

void MemPlayer::RecordListDrawArgs(VkBuffer source, VkDeviceSize offset, bool mesh) {
    if (source == VK_NULL_HANDLE) {
        return;
    }
    stage->Play();
    PlayDepth depth;
    if (!list_draw_buffer) {
        const VkBufferCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = VkDeviceSize{LIST_DRAW_SLOTS} * LIST_DRAW_SLOT_BYTES,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        };
        list_draw_buffer = memory_allocator.CreateBuffer(create_info, MemoryUsage::Upload);
        if (device.HasDebuggingToolAttached()) {
            list_draw_buffer.SetObjectNameEXT("List draw arguments");
        }
        // Written once: the host GPU only ever copies over words 0 and 2 of the vertex command
        // and word 0 of the task/mesh command.
        const std::span<u8> mapped = list_draw_buffer.Mapped();
        constexpr std::array<u32, LIST_DRAW_SLOT_BYTES / sizeof(u32)> slot{0, 1, 0, 0, 0, 1, 1, 0};
        for (u32 i = 0; i < LIST_DRAW_SLOTS; ++i) {
            std::memcpy(mapped.data() + size_t{i} * LIST_DRAW_SLOT_BYTES, slot.data(),
                        LIST_DRAW_SLOT_BYTES);
        }
        list_draw_buffer.Flush();
    }
    // The copies and the draws that read them run in the order they are recorded: a slot is
    // free again as soon as the next copy into it is recorded after its draw.
    const VkDeviceSize slot = VkDeviceSize{list_draw_cursor} * LIST_DRAW_SLOT_BYTES;
    list_draw_cursor = (list_draw_cursor + 1) % LIST_DRAW_SLOTS;
    const VkDeviceSize args = slot + (mesh ? 16 : 0);
    scheduler.RequestOutsideRenderPassOperationContext();
    // The front's copy of the words, with the same barriers.
    VkBuffer keep_buffer = VK_NULL_HANDLE;
    VkDeviceSize keep_offset = 0;
    if (const u64 keep_id = list_draw_keep_id; keep_id != 0) {
        EnsureListKeepBuffer();
        const u32 keep_slot = static_cast<u32>(keep_id % LIST_KEEP_SLOTS);
        keep_buffer = *list_keep_buffer;
        keep_offset = VkDeviceSize{keep_slot} * LIST_KEEP_SLOT_BYTES;
        list_keeps[keep_slot] = {keep_id, scheduler.CurrentTick(), true, 2, {}};
    }
    scheduler.Record([source, offset, destination = *list_draw_buffer, args, mesh, keep_buffer,
                      keep_offset](vk::CommandBuffer cmdbuf) {
        static constexpr VkMemoryBarrier READ_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        };
        static constexpr VkMemoryBarrier WRITE_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_HOST_READ_BIT,
        };
        // (count, first) -> word 0 and, for vertex draws, word 2.
        const std::array<VkBufferCopy, 2> copies{{
            {.srcOffset = offset, .dstOffset = args, .size = sizeof(u32)},
            {.srcOffset = offset + sizeof(u32), .dstOffset = args + 2 * sizeof(u32),
             .size = sizeof(u32)},
        }};
        cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, READ_BARRIER);
        cmdbuf.CopyBuffer(source, destination,
                          std::span<const VkBufferCopy>{copies.data(), mesh ? 1U : 2U});
        if (keep_buffer != VK_NULL_HANDLE) {
            const VkBufferCopy keep{
                .srcOffset = offset, .dstOffset = keep_offset, .size = 2 * sizeof(u32)};
            cmdbuf.CopyBuffer(source, keep_buffer, keep);
        }
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               keep_buffer != VK_NULL_HANDLE
                                   ? VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_HOST_BIT
                                   : VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                               0, WRITE_BARRIER);
    });
    list_draw_args = {*list_draw_buffer, args, mesh};
}

void MemPlayer::EnsureListKeepBuffer() {
    if (list_keep_buffer) {
        return;
    }
    const VkBufferCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = VkDeviceSize{LIST_KEEP_SLOTS} * LIST_KEEP_SLOT_BYTES,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    list_keep_buffer = memory_allocator.CreateBuffer(create_info, MemoryUsage::Download);
    if (device.HasDebuggingToolAttached()) {
        list_keep_buffer.SetObjectNameEXT("Kept list words");
    }
}

void MemPlayer::KeepListWords(u64 keep_id, VkBuffer source, VkDeviceSize offset, u32 count) {
    if (keep_id == 0 || source == VK_NULL_HANDLE || count == 0 || count > 3) {
        return;
    }
    stage->Play();
    PlayDepth depth;
    EnsureListKeepBuffer();
    const u32 slot = static_cast<u32>(keep_id % LIST_KEEP_SLOTS);
    const VkDeviceSize keep_offset = VkDeviceSize{slot} * LIST_KEEP_SLOT_BYTES;
    scheduler.RequestOutsideRenderPassOperationContext();
    list_keeps[slot] = {keep_id, scheduler.CurrentTick(), true, count, {}};
    scheduler.Record([source, offset, destination = *list_keep_buffer, keep_offset,
                      count](vk::CommandBuffer cmdbuf) {
        static constexpr VkMemoryBarrier READ_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        };
        static constexpr VkMemoryBarrier HOST_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        };
        cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, READ_BARRIER);
        const VkBufferCopy copy{
            .srcOffset = offset, .dstOffset = keep_offset, .size = VkDeviceSize{count} * 4};
        cmdbuf.CopyBuffer(source, destination, copy);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                               HOST_BARRIER);
    });
}

void MemPlayer::KeepListWordsRead(u64 keep_id, std::span<const u32> words) {
    if (keep_id == 0 || words.empty() || words.size() > 3) {
        return;
    }
    ListKeep& keep = list_keeps[keep_id % LIST_KEEP_SLOTS];
    keep = {keep_id, 0, false, static_cast<u32>(words.size()), {}};
    std::ranges::copy(words, keep.words.begin());
}

bool MemPlayer::ReadKeptListWords(u64 keep_id, std::span<u32> words) {
    const ListKeep& keep = list_keeps[keep_id % LIST_KEEP_SLOTS];
    if (keep_id == 0 || keep.keep_id != keep_id || words.size() > keep.count) {
        return false;
    }
    if (!keep.on_gpu) {
        std::copy_n(keep.words.begin(), words.size(), words.begin());
        return true;
    }
    {
        MemStage::SyncScope sync{stage, MemStage::Sync::KeptWords};
        scheduler.Wait(keep.tick);
    }
    list_keep_buffer.Invalidate();
    const VkDeviceSize keep_offset = VkDeviceSize{keep_id % LIST_KEEP_SLOTS} * LIST_KEEP_SLOT_BYTES;
    std::memcpy(words.data(), list_keep_buffer.Mapped().data() + keep_offset, words.size_bytes());
    return true;
}

} // namespace Vulkan
