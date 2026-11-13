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
//   2: as 1, and each draw and dispatch is filled once, before the texture work (sub-step 2b):
//      FillDraw at the rasterizer's draw site (SetUniformBuffersState, the storage binds, the
//      texture buffer unbinds, UpdateGraphicsBuffers, BindHostGeometryBuffers, every stage's
//      BindHostStageBuffers), FillDispatch before ComputePipeline::Configure (the uniform state,
//      the storage binds, the shader hash, the texture buffer unbinds, UpdateComputeBuffers,
//      BindHostComputeBuffers). The packet is held apart from the arena (held_arena), so the
//      texture side's records do not play it; Configure plays its sections at today's places
//      (PlayHeld: the geometry after the texture buffer binds, each stage after Acquire, the
//      dispatch's `before` before the image descriptors) and PlayRestAfter plays whatever is
//      left when Configure returns, true or false: the cache believes its uploads are on their
//      way. Descriptors left over then are dropped (the draw is dropped too). Pipelines with
//      texel buffers keep the fill where the calls are (LateFill: the texture buffer binds need
//      the texture cache's views). The descriptor entries take the pipeline's form (the fill runs
//      before Acquire; DescriptorForm).
//
// SEAM = 2's checks:
//   5.5, the tick guard: the staging taken at the fill is stamped with the fill's tick. If the
//        scheduler flushed between the fill and a play, the play counts it, gives every cached
//        staging buffer stamped with the fill's tick the current one (StagingBufferPool::Retick)
//        and restamps the buffers the fill marked used. Expected near 0; the first 16 flushes
//        inside a held packet are logged with their caller.
//   5.3, the read log (FRITCHII_MEMSTAGE_VERIFY=N, boot): every Nth draw, dispatch and operation
//        packet logs what its fill read from game memory (the buffer cache's MemReadLog lines:
//        uniforms copied to the stream, buffer uploads with their staged bytes, bound cached
//        ranges, storage and table addresses, compute pointers, the inline page write). At the
//        old read position (the packet's first play; an operation's end) the bytes and values are
//        read again and compared, and each bound range is asked whether the CPU wrote it since.
//        Differences are counted by kind, the first 16 logged. Must be 0.
//   The held descriptors' form is compared with the queue's after Acquire. Must be 0.
// The zones "memory fill" (fills at SEAM = 2, with what the cache does inside them) and "memory
// play" (the plays at SEAM = 2) go to EDEN_NX2_GPU_PROFILE: "memory fill" ms/s is what leaves the
// back in step 6.
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
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pass.h"
#include "video_core/renderer_vulkan/vk_mem_stage.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
namespace {

thread_local u32 t_fill_depth = 0;
thread_local u32 t_play_depth = 0;
thread_local MemStage::Site t_site = MemStage::Site::Count;
thread_local char t_thread_marker = 0;
/// This thread holds a filled draw or dispatch packet (SEAM = 2) until its plays.
thread_local bool t_holding = false;

/// Per packet, the bytes the read log copies or reads again at the compare.
constexpr u64 READ_LOG_BUDGET = 1ULL << 20;

namespace Profile = VideoCommon::Nx2Profile;

u64 NowNs() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

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
    "draw fill",     "dispatch fill",
};
constexpr std::array<const char*, static_cast<size_t>(VideoCommon::MemReadLog::Kind::Count)>
    READ_KIND_NAMES{
        "uniform",          "upload",         "bound range",         "storage address",
        "storage size",     "table address",  "compute pointer",     "inline page write",
        "handle program word", "handle program storage base", "handle program storage word",
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
    if (t_holding && (use == Use::Flush || use == Use::Finish)) [[unlikely]] {
        // SEAM = 2: a submit between a held packet's fill and its plays (the tick guard's case).
        stage->NoteHeldFlush(use, location);
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
        if (stage.seam_value.load(std::memory_order_relaxed) >= 2) {
            // SEAM = 2: the outermost bracket times the fill; an operation packet (not a held
            // draw or dispatch) may log its reads.
            zone.emplace(Profile::Zone::MemoryFill);
            if (stage.verify > 0 && !stage.filling_held && stage.OwnedHere()) {
                reads = stage.StartReadLog(stage.op_log, ++stage.op_fills);
            }
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
    if (reads) {
        // An operation packet is played where it is filled: its old read position is here.
        stage.StopReadLog(stage.op_log, false);
        stage.CompareReads(stage.op_log, SITE_NAMES[static_cast<size_t>(site)], stage.op_fills, 0,
                           0);
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

MemStage::MemStage(MemPlayer& player_)
    : player{player_}, arena{256 * 1024}, held_arena{256 * 1024} {
    player.stage = this;
    if (const char* const value = std::getenv("FRITCHII_MEMSTAGE_VERIFY")) {
        verify = (std::max)(0, std::atoi(value));
    }
    g_stage.store(this, std::memory_order_release);
}

MemStage::~MemStage() {
    MemStage* expected = this;
    g_stage.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel);
    if (pending) {
        SeamHook::g_watch.fetch_sub(1, std::memory_order_relaxed);
    }
    if (held != HeldKind::None) {
        SeamHook::g_watch.fetch_sub(1, std::memory_order_relaxed);
    }
    VideoCommon::MemReadLog::g_on.store(false, std::memory_order_relaxed);
}

void MemStage::Attach(Cache& cache_, StagingBufferPool& staging_pool_) noexcept {
    cache = &cache_;
    staging_pool = &staging_pool_;
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
        SeamHook::g_watch.fetch_add(1, std::memory_order_relaxed);
    }
}

void MemStage::PlayPending([[maybe_unused]] bool late) {
    if (!pending) {
        return;
    }
    std::optional<Profile::Scope> zone;
    if (seam_value.load(std::memory_order_relaxed) >= 2) {
        zone.emplace(Profile::Zone::MemoryPlay);
    }
    PlayDepth depth;
    Bump(counters.plays);
    if (filling_held) [[unlikely]] {
        // A sync inside a held fill (SEAM = 2): the scheduler work it needs runs now, but the
        // descriptor entries wait for the packet's plays (no Acquire yet). Played entries are
        // marked, and the cursor stays at the first descriptor entry left.
        u32 first_left = ~0U;
        for (u32 at = play_cursor; at < arena.Size();) {
            Entry& entry = *reinterpret_cast<Entry*>(arena.At(at));
            const u32 here = at;
            at += entry.bytes;
            if ((entry.flags & Played) != 0) {
                continue;
            }
            if (entry.kind == Cmd::Descriptors) {
                first_left = (std::min)(first_left, here);
                continue;
            }
            entry.flags |= Played;
            player.Exec(entry);
        }
        play_cursor = first_left != ~0U ? first_left : arena.Size();
    } else {
        while (play_cursor < arena.Size()) {
            const Entry& entry = *reinterpret_cast<const Entry*>(arena.At(play_cursor));
            play_cursor += entry.bytes;
            player.Exec(entry);
        }
    }
    pending = false;
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
    case Site::Draw:
    case Site::Dispatch:
        // A held packet's whole fill: its sections are the brackets inside.
        break;
    default:
        if (t_fill_depth == 1) {
            op.number = ++packet_number;
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
    // A held packet ends at its Configure's PlayRestAfter; this is only a safety net.
    PlayRest();
    Bump(counters.draws);
    draw = DrawPacket{};
    draw.number = ++packet_number;
    draw.fill_tick = player.scheduler.CurrentTick();
    draw.repack_mask = repack_mask;
    draw.repack_dirty = repack_dirty;
}

void MemStage::BeginDispatch() {
    PlayRest();
    Bump(counters.dispatches);
    dispatch = DispatchPacket{};
    dispatch.number = ++packet_number;
    dispatch.fill_tick = player.scheduler.CurrentTick();
}

// ---------------------------------------------------------------------------------------------
// SEAM = 2: the held draw and dispatch packets

bool MemStage::ClaimHeld() {
    if (t_fill_depth != 0 || t_play_depth != 0) {
        // Inside another fill or a play: not a draw site. Fill where the calls are.
        return false;
    }
    PlayRest();
    const uintptr_t me = ThisThread();
    uintptr_t expected = 0;
    if (owner.load(std::memory_order_relaxed) != me &&
        !owner.compare_exchange_strong(expected, me, std::memory_order_acquire)) {
        // Another thread's fill holds the arena this instant.
        return false;
    }
    if (pending) {
        // Left pending by a site that misses its play (5.2): played before the packet begins.
        Note(SeamHook::Use::Flush, std::source_location::current(), false);
        PlayPending(true);
    }
    arena.Reset();
    play_cursor = 0;
    last_entry = ~0U;
    claim_deferring = true;
    return true;
}

void MemStage::TakeHeld(HeldKind kind) {
    // The arena holding the packet becomes the held one; the next fill starts on an empty arena.
    // Entries a sync played during the fill are behind play_cursor.
    held_arena.Swap(arena);
    held_cursor = play_cursor;
    arena.Reset();
    play_cursor = 0;
    last_entry = ~0U;
    if (pending) {
        pending = false;
        SeamHook::g_watch.fetch_sub(1, std::memory_order_relaxed);
    }
    held_marked.swap(marked);
    marked.clear();
    held = kind;
    t_holding = true;
    // Watched while held: a flush before the plays is logged with its caller.
    SeamHook::g_watch.fetch_add(1, std::memory_order_relaxed);
}

const MemStage::DrawPacket* MemStage::FillDraw(Cache& cache_, const DrawFacts& facts,
                                               bool is_indexed) {
    if (seam_value.load(std::memory_order_relaxed) < 2) {
        return nullptr;
    }
    if (facts.has_texel_buffers) {
        // The texture buffer binds take their address, size and format from the texture
        // cache's views (FillImageViews): this draw fills where the calls are.
        draw.flags |= LateFill;
        Bump(counters.late_draws);
        return nullptr;
    }
    if (!ClaimHeld()) {
        Bump(counters.unheld);
        return nullptr;
    }
    const bool timed = Profile::Enabled();
    const u64 start = timed ? NowNs() : 0;
    Bump(counters.held_draws);
    draw.fill_tick = player.scheduler.CurrentTick();
    draw.flags |= Held;
    held_hashes = facts.hashes;
    held_form = player.guest_descriptor_queue.FormFor(facts.uses_descriptor_buffer);
    fill_form = held_form ? 1 : 0;
    marked.clear();
    t_marked = &marked;
    filling_held = true;
    const bool logging = verify > 0 && StartReadLog(held_log, draw.number);
    {
        // Today's calls in today's relative order (ConfigureImpl): the uniform state, per stage
        // the storage binds, the texture buffer unbinds, then the geometry and each stage's
        // buffers, each in its own section.
        Fill whole{*this, Site::Draw};
        cache_.SetUniformBuffersState(*facts.uniform_masks, facts.uniform_sizes);
        for (u32 stage = 0; stage < 5; ++stage) {
            if ((facts.stages & (1U << stage)) == 0) {
                continue;
            }
            cache_.UnbindGraphicsStorageBuffers(stage);
            size_t ssbo_index = 0;
            for (const auto& desc : facts.stage_infos[stage].storage_buffers_descriptors) {
                ASSERT(desc.count == 1);
                cache_.BindGraphicsStorageBuffer(stage, ssbo_index, desc.cbuf_index,
                                                 desc.cbuf_offset, desc.is_written,
                                                 desc.indirect_offset);
                ++ssbo_index;
            }
        }
        for (u32 stage = 0; stage < 5; ++stage) {
            if ((facts.stages & (1U << stage)) != 0) {
                cache_.UnbindGraphicsTextureBuffers(stage);
            }
        }
        last_entry = ~0U;
        {
            Fill geometry{*this, Site::Geometry};
            cache_.UpdateGraphicsBuffers(is_indexed);
            cache_.BindHostGeometryBuffers(is_indexed);
        }
        for (u32 stage = 0; stage < 5; ++stage) {
            if ((facts.stages & (1U << stage)) == 0) {
                continue;
            }
            // A stage's descriptors never join the previous stage's entry: each is played at
            // its own stage, before that stage's image descriptors.
            last_entry = ~0U;
            Fill section{*this, Site::Stage, stage};
            cache_.BindHostStageBuffers(stage);
        }
        last_entry = ~0U;
        if (logging) {
            StopReadLog(held_log, true);
        }
        TakeHeld(HeldKind::Draw);
    }
    filling_held = false;
    t_marked = nullptr;
    fill_form = -1;
    if (timed) {
        Bump(counters.fill_ns, NowNs() - start);
    }
    return &draw;
}

const MemStage::DispatchPacket* MemStage::FillDispatch(Cache& cache_,
                                                       const DispatchFacts& facts) {
    if (seam_value.load(std::memory_order_relaxed) < 2) {
        return nullptr;
    }
    if (facts.has_texel_buffers) {
        dispatch.flags |= LateFill;
        Bump(counters.late_dispatches);
        return nullptr;
    }
    if (!ClaimHeld()) {
        Bump(counters.unheld);
        return nullptr;
    }
    const bool timed = Profile::Enabled();
    const u64 start = timed ? NowNs() : 0;
    Bump(counters.held_dispatches);
    dispatch.fill_tick = player.scheduler.CurrentTick();
    dispatch.flags |= Held;
    held_hashes = {facts.shader_hash, 0};
    held_form = player.guest_descriptor_queue.FormFor(facts.uses_descriptor_buffer);
    fill_form = held_form ? 1 : 0;
    marked.clear();
    t_marked = &marked;
    filling_held = true;
    const bool logging = verify > 0 && StartReadLog(held_log, dispatch.number);
    {
        // ComputePipeline::Configure's calls in its order: the uniform state, the storage binds
        // and the shader hash, the texture buffer unbinds, then the `before` section.
        Fill whole{*this, Site::Dispatch};
        cache_.SetComputeUniformBufferState(facts.uniform_mask, facts.uniform_sizes);
        cache_.UnbindComputeStorageBuffers();
        cache_.SetComputeShaderHash(facts.shader_hash);
        size_t ssbo_index = 0;
        for (const auto& desc : facts.info->storage_buffers_descriptors) {
            ASSERT(desc.count == 1);
            cache_.BindComputeStorageBuffer(ssbo_index, desc.cbuf_index, desc.cbuf_offset,
                                            desc.is_written, desc.indirect_offset,
                                            desc.write_extent);
            ++ssbo_index;
        }
        cache_.UnbindComputeTextureBuffers();
        last_entry = ~0U;
        {
            Fill before{*this, Site::DispatchBefore};
            cache_.UpdateComputeBuffers();
            cache_.BindHostComputeBuffers();
        }
        last_entry = ~0U;
        if (logging) {
            StopReadLog(held_log, true);
        }
        TakeHeld(HeldKind::Dispatch);
    }
    filling_held = false;
    t_marked = nullptr;
    fill_form = -1;
    if (timed) {
        Bump(counters.fill_ns, NowNs() - start);
    }
    return &dispatch;
}

void MemStage::CheckHeld() {
    u64& fill_tick = held == HeldKind::Draw ? draw.fill_tick : dispatch.fill_tick;
    const u64 now = player.scheduler.CurrentTick();
    if (now != fill_tick) [[unlikely]] {
        // 5.5: the scheduler flushed between the fill and this play. What the fill stamped with
        // its tick is used by this later submission: stamp it with the current one.
        Bump(counters.tick_moved);
        if (ticks_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_WARNING(Render_Vulkan,
                        "Memory seam (5.5): the tick moved from {} to {} between the fill and the "
                        "play of {} {} (programs {:016x} {:016x}); restamped",
                        fill_tick, now, held == HeldKind::Draw ? "draw" : "dispatch",
                        held == HeldKind::Draw ? draw.number : dispatch.number, held_hashes[0],
                        held_hashes[1]);
        }
        if (staging_pool != nullptr) {
            staging_pool->Retick(fill_tick);
        }
        for (Buffer* const buffer : held_marked) {
            buffer->Restamp(fill_tick);
        }
        fill_tick = now;
    }
    if (held_log_pending) {
        // 5.3: the packet's first play is the old read position.
        held_log_pending = false;
        const bool is_draw = held == HeldKind::Draw;
        CompareReads(held_log, is_draw ? "draw" : "dispatch",
                     is_draw ? draw.number : dispatch.number, held_hashes[0], held_hashes[1]);
    }
}

void MemStage::PlayHeld(Span span) {
    if (held == HeldKind::None) {
        return;
    }
    PlayHeldTo(span.end, false);
}

void MemStage::PlayHeldTo(u32 end, bool rest) {
    const bool timed = Profile::Enabled();
    const u64 start = timed ? NowNs() : 0;
    const Profile::Scope zone{Profile::Zone::MemoryPlay};
    CheckHeld();
    const u32 stop = (std::min)(end, held_arena.Size());
    if (held_cursor >= stop) {
        return;
    }
    if (rest) {
        // Configure returned before playing every section (false, or a path without the play).
        Bump(counters.rest_plays);
        (held == HeldKind::Draw ? draw.flags : dispatch.flags) |= RestPlayed;
    }
    PlayDepth depth;
    while (held_cursor < stop) {
        const Entry& entry = *reinterpret_cast<const Entry*>(held_arena.At(held_cursor));
        held_cursor += entry.bytes;
        if ((entry.flags & Played) != 0) {
            continue;
        }
        if (entry.kind == Cmd::Descriptors) {
            if (rest) {
                // No Acquire for them: the draw or dispatch is dropped.
                Bump(counters.rest_descriptors, entry.count);
                continue;
            }
            if (player.guest_descriptor_queue.UsesDescriptorBuffer() != held_form) [[unlikely]] {
                Bump(counters.form_mismatch);
            }
        }
        player.Exec(entry);
    }
    if (timed) {
        Bump(counters.play_ns, NowNs() - start);
    }
}

void MemStage::PlayRest() {
    if (held == HeldKind::None) {
        return;
    }
    PlayHeldTo(held_arena.Size(), true);
    held = HeldKind::None;
    t_holding = false;
    SeamHook::g_watch.fetch_sub(1, std::memory_order_relaxed);
    held_arena.Reset();
    held_cursor = 0;
    held_marked.clear();
    held_log.Clear();
    held_log_pending = false;
}

void MemStage::NoteHeldFlush(SeamHook::Use use, const std::source_location& location) {
    if (t_play_depth != 0) {
        return;
    }
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_WARNING(Render_Vulkan,
                    "Memory seam (5.5): a {} between the fill and the plays of {} {}: {}:{} {}",
                    UseName(use), held == HeldKind::Draw ? "draw" : "dispatch",
                    held == HeldKind::Draw ? draw.number : dispatch.number, location.file_name(),
                    location.line(), location.function_name());
    }
}

// ---------------------------------------------------------------------------------------------
// The read log (5.3)

bool MemStage::StartReadLog(ReadLog& log, u64 number) {
    if (verify <= 0 || number % static_cast<u64>(verify) != 0 || t_log != nullptr) {
        return false;
    }
    log.Clear();
    t_log = &log;
    return true;
}

void MemStage::StopReadLog(ReadLog& log, bool held_packet) {
    t_log = nullptr;
    for (ReadRecord& record : log.records) {
        if (record.how != ReadRecord::How::Range) {
            continue;
        }
        // The tracker's answer when the fill ends; at the compare a range the CPU wrote since
        // (clean now, modified then) is a difference. An operation is compared at once.
        record.value = !held_packet || cache == nullptr ||
                               cache->IsRegionCpuModified(record.address, record.size)
                           ? 1
                           : 0;
    }
    if (held_packet) {
        held_log_pending = true;
    }
}

void MemStage::CompareReads(ReadLog& log, const char* what, u64 number, u64 hash_a, u64 hash_b) {
    using How = ReadRecord::How;
    for (const ReadRecord& record : log.records) {
        bool differs = false;
        switch (record.how) {
        case How::Bytes:
        case How::Staged: {
            compare_scratch.resize(record.size);
            static_cast<Tegra::MaxwellDeviceMemoryManager*>(record.memory)
                ->ReadBlockUnsafe(record.address, compare_scratch.data(), record.size);
            const u8* const expected = record.how == How::Bytes
                                           ? log.bytes.data() + record.value
                                           : static_cast<const u8*>(record.data);
            differs = std::memcmp(compare_scratch.data(), expected, record.size) != 0;
            break;
        }
        case How::Value: {
            const auto* const memory = static_cast<const Tegra::MemoryManager*>(record.memory);
            const u64 now = record.size == 4 ? u64{memory->Read<u32>(record.address)}
                                             : memory->Read<u64>(record.address);
            differs = now != record.value;
            break;
        }
        case How::Range:
            differs = record.value == 0 && cache != nullptr &&
                      cache->IsRegionCpuModified(record.address, record.size);
            break;
        }
        if (!differs) {
            continue;
        }
        Bump(counters.read_diffs[static_cast<size_t>(record.kind)]);
        if (diffs_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_WARNING(Render_Vulkan,
                        "Memory seam (5.3): {} {} read {} at {:#x}+{:#x}: game memory changed "
                        "between the fill and the old read position (programs {:016x} {:016x})",
                        what, number, READ_KIND_NAMES[static_cast<size_t>(record.kind)],
                        record.address, record.size, hash_a, hash_b);
        }
    }
    Bump(counters.read_packets);
    Bump(counters.read_records, log.records.size());
    Bump(counters.read_skipped, log.skipped);
    log.Clear();
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
    const int value = VideoCore::Nx2Debug::Experiment(123);
    const bool on = value != 0;
    // A held packet never outlives its draw or dispatch; a safety net, as in BeginDraw.
    PlayRest();
    if (OwnedHere() && pending) {
        // Nothing should be pending between frames: a site that misses its play.
        Note(SeamHook::Use::Flush, std::source_location::current(), false);
        PlayPending(true);
    }
    if (OwnedHere() && t_fill_depth == 0) {
        owner.store(0, std::memory_order_release);
    }
    if (value != seam_value.load(std::memory_order_relaxed)) {
        seam_value.store(value, std::memory_order_relaxed);
        seam_on.store(on, std::memory_order_relaxed);
        player.ForgetBindings();
        VideoCommon::MemReadLog::g_on.store(value >= 2 && verify > 0, std::memory_order_relaxed);
        LOG_INFO(Render_Vulkan, "Memory seam (EXP123): {}{}",
                 value >= 2  ? "2, one fill per draw and dispatch before the texture work"
                 : on        ? "1, recorder and player"
                             : "0, entries run as pushed",
                 value >= 2 && verify > 0
                     ? fmt::format(" (FRITCHII_MEMSTAGE_VERIFY: every {}th packet's reads "
                                   "compared)",
                                   verify)
                     : std::string{});
    }
    static const bool stats_env = std::getenv("FRITCHII_MEMSTAGE_SEAM_STATS") != nullptr;
    if (on || stats_env) {
        LogStats();
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
        case Site::Draw:
            draw_entries += entries;
            break;
        case Site::Dispatch:
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
             seam_value.load(std::memory_order_relaxed), seconds, rate(draws),
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
    if (seam_value.load(std::memory_order_relaxed) >= 2) {
        std::string diffs;
        u64 diff_total = 0;
        for (size_t i = 0; i < READ_KIND_NAMES.size(); ++i) {
            if (const u64 n = current.read_diffs[i] - reported.read_diffs[i]; n != 0) {
                diffs += fmt::format("{}{} {}", diffs.empty() ? "" : ", ", READ_KIND_NAMES[i], n);
                diff_total += n;
            }
        }
        const u64 held_draws = delta(&Counts::held_draws);
        const u64 held_dispatches = delta(&Counts::held_dispatches);
        const u64 held_packets = held_draws + held_dispatches;
        LOG_INFO(Render_Vulkan,
                 "  memory seam 2b: held draws {:.0f}/s, late fill draws {:.0f}/s (texel "
                 "buffers), held dispatches {:.0f}/s, late fill dispatches {:.0f}/s, not held "
                 "(arena busy) {}; tick moved (5.5) {}; rest plays {} ({} descriptors dropped); "
                 "descriptor form mismatches {}; fill {:.3f} ms/s ({:.2f} us a packet), play "
                 "{:.3f} ms/s (inclusive, with EDEN_NX2_GPU_PROFILE)",
                 rate(held_draws), rate(delta(&Counts::late_draws)), rate(held_dispatches),
                 rate(delta(&Counts::late_dispatches)), delta(&Counts::unheld),
                 delta(&Counts::tick_moved), delta(&Counts::rest_plays),
                 delta(&Counts::rest_descriptors), delta(&Counts::form_mismatch),
                 static_cast<double>(delta(&Counts::fill_ns)) / 1e6 / seconds,
                 held_packets != 0 ? static_cast<double>(delta(&Counts::fill_ns)) / 1e3 /
                                         static_cast<double>(held_packets)
                                   : 0.0,
                 static_cast<double>(delta(&Counts::play_ns)) / 1e6 / seconds);
        if (verify > 0) {
            LOG_INFO(Render_Vulkan,
                     "  memory seam read log (FRITCHII_MEMSTAGE_VERIFY={}): {} packets, {} reads "
                     "compared, {} not logged (budget); differences (5.3) {}{}{}",
                     verify, delta(&Counts::read_packets), delta(&Counts::read_records),
                     delta(&Counts::read_skipped), diff_total, diffs.empty() ? "" : ": ", diffs);
        }
    }
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
    out.held_draws = counters.held_draws.load(std::memory_order_relaxed);
    out.late_draws = counters.late_draws.load(std::memory_order_relaxed);
    out.held_dispatches = counters.held_dispatches.load(std::memory_order_relaxed);
    out.late_dispatches = counters.late_dispatches.load(std::memory_order_relaxed);
    out.unheld = counters.unheld.load(std::memory_order_relaxed);
    out.tick_moved = counters.tick_moved.load(std::memory_order_relaxed);
    out.rest_plays = counters.rest_plays.load(std::memory_order_relaxed);
    out.rest_descriptors = counters.rest_descriptors.load(std::memory_order_relaxed);
    out.form_mismatch = counters.form_mismatch.load(std::memory_order_relaxed);
    out.read_packets = counters.read_packets.load(std::memory_order_relaxed);
    out.read_records = counters.read_records.load(std::memory_order_relaxed);
    out.read_skipped = counters.read_skipped.load(std::memory_order_relaxed);
    out.fill_ns = counters.fill_ns.load(std::memory_order_relaxed);
    out.play_ns = counters.play_ns.load(std::memory_order_relaxed);
    for (size_t i = 0; i < out.read_diffs.size(); ++i) {
        out.read_diffs[i] = counters.read_diffs[i].load(std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------------------------
// SeamClock. Now, IsFree and KnownGpu read the scheduler's master semaphore, exactly what
// Scheduler::CurrentTick and IsFree read. The syncs play the pending entries first.

SeamClock::SeamClock(Scheduler& scheduler_, MemStage* stage_)
    : scheduler{&scheduler_}, master{&scheduler_.GetMasterSemaphore()}, stage{stage_} {}

void SeamClock::Finish(std::source_location location) {
    MemStage::SyncScope sync{stage, MemStage::Sync::Finish};
    scheduler->Finish({}, {}, location);
}

void SeamClock::Wait(u64 tick) {
    MemStage::SyncScope sync{stage, MemStage::Sync::Wait};
    scheduler->Wait(tick);
}

u64 SeamClock::SubmitForHostRead(std::source_location location) {
    MemStage::SyncScope sync{stage, MemStage::Sync::SubmitForHostRead};
    scheduler->RequestOutsideRenderPassOperationContext(location);
    scheduler->Record(HostReadBarrier);
    return scheduler->Flush({}, {}, location);
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

// The buffer cache's read log lines (buffer_cache.h). g_on is set while SEAM = 2 runs with
// FRITCHII_MEMSTAGE_VERIFY; only a fill that logs (every Nth packet, on its own thread) has a log.
namespace VideoCommon::MemReadLog {

using Vulkan::MemStage;

void NoteBytes(Kind kind, Tegra::MaxwellDeviceMemoryManager& memory, u64 address, u64 size) {
    MemStage::ReadLog* const log = MemStage::t_log;
    if (log == nullptr || size == 0) {
        return;
    }
    if (log->budget + size > Vulkan::READ_LOG_BUDGET) {
        ++log->skipped;
        return;
    }
    log->budget += size;
    const size_t at = log->bytes.size();
    log->bytes.resize(at + static_cast<size_t>(size));
    memory.ReadBlockUnsafe(address, log->bytes.data() + at, static_cast<size_t>(size));
    log->records.push_back({
        .kind = kind,
        .how = MemStage::ReadRecord::How::Bytes,
        .size = static_cast<u32>(size),
        .address = address,
        .value = at,
        .data = nullptr,
        .memory = &memory,
    });
}

void NoteStaged(Kind kind, Tegra::MaxwellDeviceMemoryManager& memory, u64 address,
                const u8* staged, u64 size) {
    MemStage::ReadLog* const log = MemStage::t_log;
    if (log == nullptr || size == 0) {
        return;
    }
    if (log->budget + size > Vulkan::READ_LOG_BUDGET) {
        ++log->skipped;
        return;
    }
    log->budget += size;
    log->records.push_back({
        .kind = kind,
        .how = MemStage::ReadRecord::How::Staged,
        .size = static_cast<u32>(size),
        .address = address,
        .value = 0,
        .data = staged,
        .memory = &memory,
    });
}

void NoteValue(Kind kind, Tegra::MemoryManager* memory, u64 address, u64 value, u32 size) {
    MemStage::ReadLog* const log = MemStage::t_log;
    if (log == nullptr || memory == nullptr) {
        return;
    }
    log->records.push_back({
        .kind = kind,
        .how = MemStage::ReadRecord::How::Value,
        .size = size,
        .address = address,
        .value = value,
        .data = nullptr,
        .memory = memory,
    });
}

void NoteRange(Kind kind, u64 address, u64 size) {
    MemStage::ReadLog* const log = MemStage::t_log;
    if (log == nullptr || size == 0) {
        return;
    }
    log->records.push_back({
        .kind = kind,
        .how = MemStage::ReadRecord::How::Range,
        .size = static_cast<u32>(size),
        .address = address,
        .value = 0,
        .data = nullptr,
        .memory = nullptr,
    });
}

} // namespace VideoCommon::MemReadLog
