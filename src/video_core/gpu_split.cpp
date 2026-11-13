// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <thread>

#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <cstring>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "common/dkb_cpu_trace.h"
#include "common/logging.h"
#include "common/nx2_hitch.h"
#include "common/nx2_timeline.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core.h"
#include "video_core/engines/engine_interface.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/nx2_fastdraw_count.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_split_stats.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/nx2_runahead.h"

namespace VideoCommon {

namespace {

// Every front wait for the back, by site, always on (fable52_answer_13 section 3: the macro DMA
// read drains were half the front's wait in the Stalchild fight and had no line). Per site: how
// many, their time, and the ring words the back had left when each began ("behind": more words
// is more queued work, the same words taking longer is a slower back). Front thread only. Logged
// every 5 s with the cache lock waits per thread role (Nx2Hitch::g_lock_waits).
struct SiteWaitStats {
    std::array<u64, static_cast<size_t>(DrainSite::Count)> count{};
    std::array<u64, static_cast<size_t>(DrainSite::Count)> ns{};
    std::array<u64, static_cast<size_t>(DrainSite::Count)> behind{};
    std::array<u64, static_cast<size_t>(Common::Nx2Hitch::Role::Count)> lock_ns{};
    std::array<u64, static_cast<size_t>(Common::Nx2Hitch::Role::Count)> lock_count{};
    u64 since{};
};
SiteWaitStats g_site_waits;

void NoteSiteWait(DrainSite site, u64 behind, u64 ns) {
    auto& stats = g_site_waits;
    const auto i = static_cast<size_t>(site);
    ++stats.count[i];
    stats.ns[i] += ns;
    stats.behind[i] += behind;
    const u64 now = Common::Nx2Hitch::NowNs();
    if (stats.since == 0) {
        stats.since = now;
        return;
    }
    const double window = static_cast<double>(now - stats.since) / 1e9;
    if (window < 5.0) {
        return;
    }
    std::vector<size_t> order;
    for (size_t s = 0; s < stats.count.size(); ++s) {
        if (stats.count[s] != 0) {
            order.push_back(s);
        }
    }
    std::ranges::sort(order, [&](size_t a, size_t b) { return stats.ns[a] > stats.ns[b]; });
    std::string sites;
    for (const size_t s : order) {
        sites += fmt::format("{}{} {:.0f}/s {:.1f} ms/s behind {:.0f}", sites.empty() ? "" : "; ",
                             drain_site_names[s], static_cast<double>(stats.count[s]) / window,
                             static_cast<double>(stats.ns[s]) / 1e6 / window,
                             static_cast<double>(stats.behind[s]) /
                                 static_cast<double>(stats.count[s]));
    }
    std::array<double, static_cast<size_t>(Common::Nx2Hitch::Role::Count)> lock_ms{};
    std::array<double, static_cast<size_t>(Common::Nx2Hitch::Role::Count)> lock_rate{};
    for (size_t r = 0; r < lock_ms.size(); ++r) {
        const u64 total_ns = Common::Nx2Hitch::g_lock_waits.ns[r].load(std::memory_order_relaxed);
        const u64 total = Common::Nx2Hitch::g_lock_waits.count[r].load(std::memory_order_relaxed);
        lock_ms[r] = static_cast<double>(total_ns - stats.lock_ns[r]) / 1e6 / window;
        lock_rate[r] = static_cast<double>(total - stats.lock_count[r]) / window;
        stats.lock_ns[r] = total_ns;
        stats.lock_count[r] = total;
    }
    LOG_INFO(HW_GPU,
             "Front waits for the back by site (ring words behind at the start): {}; cache lock "
             "waits: guest cores {:.1f} ms/s ({:.0f}/s), GPU back {:.1f} ms/s ({:.0f}/s), GPU "
             "front {:.1f} ms/s, other threads {:.1f} ms/s",
             sites, lock_ms[1], lock_rate[1], lock_ms[3], lock_rate[3], lock_ms[2], lock_ms[0]);
    stats.count = {};
    stats.ns = {};
    stats.behind = {};
    stats.since = now;
}

void Pause() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#else
    std::this_thread::yield();
#endif
}

[[maybe_unused]] std::atomic<u32> g_back_thread_id{};

/// EXP27 & 4: how far ahead of the entry it runs the back prefetches ring words (16 lines).
constexpr u64 RingPrefetchWords = 128;

void PrefetchRingLine(const u64* address) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_prefetch(reinterpret_cast<const char*>(address), _MM_HINT_T0);
#elif defined(__clang__) || defined(__GNUC__)
    __builtin_prefetch(address, 0, 3);
#else
    static_cast<void>(address);
#endif
}

/// EDEN_NX2_FASTDRAW_COUNT: a deferred call that ends delta draw reuse, by its thunk (Fable plan
/// D1: the per call site histogram in the counter's log).
void NoteFastDrawCallBreak(u64 thunk) {
    Nx2FastDrawCount::Break(Nx2FastDrawCount::Reason::Call);
#ifdef _WIN32
    static const bool based = [] {
        Nx2FastDrawCount::g_module_base.store(reinterpret_cast<u64>(GetModuleHandleW(nullptr)),
                                              std::memory_order_relaxed);
        return true;
    }();
    static_cast<void>(based);
#endif
    Nx2FastDrawCount::NoteCallSite(thunk);
}

} // Anonymous namespace

void GpuSplit::CheckOwner(const char* where) {
#ifdef _WIN32
    // Also while the back is idle: its idle hook submits (Scheduler::FlushDeferred).
    if (!instance.back_running.load(std::memory_order_acquire)) {
        return;
    }
    const DWORD thread = GetCurrentThreadId();
    if (thread == g_back_thread_id.load(std::memory_order_relaxed)) {
        return;
    }
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1) >= 12) {
        return;
    }
    std::string stack;
    void* frames[40];
    const USHORT count = RtlCaptureStackBackTrace(1, 40, frames, nullptr);
    for (USHORT i = 0; i < count; ++i) {
        HMODULE module{};
        char name[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               static_cast<LPCSTR>(frames[i]), &module)) {
            GetModuleFileNameA(module, name, sizeof(name));
        }
        const char* base_name = std::strrchr(name, '\\');
        stack += fmt::format(" {}+{:#x}", base_name ? base_name + 1 : name,
                             reinterpret_cast<uintptr_t>(frames[i]) -
                                 reinterpret_cast<uintptr_t>(module));
    }
    LOG_ERROR(HW_GPU, "GPU split owner check: {} on thread {} while the back runs; stack:{}",
              where, thread, stack);
#else
    (void)where;
#endif
}

GpuSplit GpuSplit::instance;

void GpuSplit::MaybeStart(Core::System& system_) {
    static const char mode = [] {
        const char* value = std::getenv("EDEN_NX2_SPLIT");
        return value != nullptr ? value[0] : '2';
    }();
    const bool wanted = mode == '1' || mode == '2';
    if (!wanted || !Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
        return;
    }
    // The front reads command lists straight from guest memory; a read that flushes the caches
    // first (safe DMA, accurate GPU) would touch back-owned state. The settings can change
    // mid-game: hand the engines back to the front (and start again when they allow it).
    const bool safe_reads = Settings::IsDMALevelDefault() ? Settings::IsGPULevelHigh()
                                                          : Settings::IsDMALevelSafe();
    const bool allowed = !safe_reads && !Settings::IsGPULevelHigh() &&
                         Settings::values.renderer_backend.GetValue() == Settings::RendererBackend::Vulkan;
    if (active && !allowed) {
        Drain();
        Stop();
        LOG_INFO(HW_GPU, "GPU thread split: stopped, the accuracy settings need the caches on the front");
        return;
    }
    if (active || !allowed) {
        return;
    }
    system = &system_;
    check_owner = std::getenv("EDEN_NX2_SPLIT_CHECK") != nullptr;
    static const bool switches_read = [] {
        const auto on = [](const char* name) {
            const char* value = std::getenv(name);
            return value != nullptr && value[0] != '0';
        };
        macro_sink = on("EDEN_NX2_MACRO_SINK");
        compute_sink = on("EDEN_NX2_COMPUTE_SINK");
        ring_pairs = on("EDEN_NX2_RING_PAIRS");
        return true;
    }();
    (void)switches_read;
    ring = std::make_unique<u64[]>(RingWords);
    ++session;
    // EDEN_NX2_RUNAHEAD: Ocarina of Time only.
    Runahead::Reset();
    Runahead::SetProgramId(system_.GetApplicationProcessProgramID());
    back_thread = std::jthread([this](std::stop_token stop_token) { BackLoop(stop_token); });
    active = true;
    const char* jit = std::getenv("EDEN_NX2_MME_JIT");
    front_macros = mode == '2' && (jit == nullptr || jit[0] != '0');
    LOG_INFO(HW_GPU, "GPU thread split: engine methods run on a second thread{}",
             front_macros ? ", macros on the first" : "");
}

void GpuSplit::Stop() {
    if (!back_thread.joinable()) {
        return;
    }
    back_thread.request_stop();
    wake.fetch_add(1, std::memory_order_seq_cst);
    wake.notify_one();
    back_thread.join();
    active = false;
    // The split is static and outlives the emulation session. The next session's back thread
    // starts reading at 0: with the old positions kept it skipped through the new, empty ring a
    // wrap at a time and never reached the old write position (the next game's first GPU work
    // never ran, it waited on its syncpoint forever), and the engines it held were freed.
    ring.reset();
    write_position = 0;
    ForgetFencesQuiet();
    last_writer_position = 0;
    last_memory_writer_position = 0;
    last_unranged_position = 0;
    unranged_without_exp120 = 0;
    pending_write_count = 0;
    writer_range_count = 0;
    last_unbounded_writer_position = 0;
    writer_range_memory = nullptr;
    writer_range_sealed = 0;
    ClearPageWatch();
    last_dispatch_position = 0;
    last_unrecorded_dispatch_position = 0;
    dispatch_records = false;
    back_dispatch_records = false;
    dispatch_write_count.store(0, std::memory_order_relaxed);
    back_entry_position = 0;
    async_writes.clear();
    last_sync_position = 0;
    last_memory_write_position = 0;
    previous_memory_write_position = 0;
    last_method_write_position = 0;
    previous_method_write_position = 0;
    if (write_stamps) {
        // The next session's positions start at 0 again.
        std::fill_n(block_stamps.get(), BlockStamps, StampSlot{});
        std::fill_n(page_stamps.get(), PageStamps, StampSlot{});
    }
    write_stamp_floor = 0;
    unknown_write_position = 0;
    write_stamp_tag = 0;
    ranged_writes.clear();
    last_function_position = 0;
    writes.Reset();
    note_memory = nullptr;
    note_page = ~GPUVAddr{0};
    published_local = 0;
    consumed_cache = 0;
    publish_every = PublishEvery;
    publish_waiting = false;
    front_engine = nullptr;
    pairs_header = nullptr;
    pairs_count = 0;
    delta_header = nullptr;
    delta_count = 0;
    pending_flush = nullptr;
    pending_flush_argument = nullptr;
    fold_count = 0;
    fold_flush = nullptr;
    fold_context = nullptr;
    back_engine = nullptr;
    front_macros = false;
    published.store(0, std::memory_order_relaxed);
    consumed.store(0, std::memory_order_relaxed);
    back_idle_at.store(~u64{0}, std::memory_order_relaxed);
    back_waiting.store(false, std::memory_order_relaxed);
    front_waiting.store(false, std::memory_order_relaxed);
    // Run-ahead's positions are ring positions too; queued chunks' bytes are in guest memory.
    Runahead::Reset();
}

u64* GpuSplit::ReserveSlow(size_t words) {
    size_t index = write_position & RingMask;
    if (index + words > RingWords) {
        // Entries never wrap: skip to the start of the ring.
        const size_t rest = RingWords - index;
        WaitForSpace(rest);
        ring[index] = static_cast<u64>(Kind::Wrap);
        write_position += rest;
        index = 0;
    }
    WaitForSpace(words);
    return &ring[index];
}

void GpuSplit::WaitForSpace(size_t words) {
    if (write_position + words - consumed_cache <= RingWords) {
        return;
    }
    Publish();
    Nx2Profile::Scope scope{Nx2Profile::Zone::SplitDrain};
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (write_position + words - consumed_cache <= RingWords) {
            return;
        }
        if (spins < 4096) {
            Pause();
        } else {
            std::this_thread::yield();
        }
    }
}

void GpuSplit::Publish() {
    RunPendingFlush();
    // The back may read what is published: the open blocks are final from here on.
    pairs_header = nullptr;
    delta_header = nullptr;
    if (published_local == write_position) {
        return;
    }
    if (Nx2SplitStats::Enabled()) [[unlikely]] {
        Nx2SplitStats::NotePublish(write_position - published_local);
    }
    published_local = write_position;
    publish_every = std::min<size_t>(publish_every * 2, PublishEvery);
    publish_waiting = (VideoCore::Nx2Debug::Experiment(39) & 1) != 0;
    published.store(write_position, std::memory_order_seq_cst);
    if (back_waiting.load(std::memory_order_seq_cst)) {
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }
    if ((++publishes & 1023) == 0) [[unlikely]] {
        ReportBatched();
    }
}

void GpuSplit::PublishBeforeWait() {
    if ((VideoCore::Nx2Debug::Experiment(27) & 1) != 0) {
        Publish(); // Sends the held-back work first (RunPendingFlush)
    }
}

void GpuSplit::ReportBatched() {
    const auto now = std::chrono::steady_clock::now();
    if (batched_since == std::chrono::steady_clock::time_point{}) {
        batched_since = now;
        batched.fill(0);
        return;
    }
    const std::chrono::duration<double> elapsed = now - batched_since;
    if (elapsed.count() < 10.0) {
        return;
    }
    const auto rate = [&](Batched counter) {
        return static_cast<double>(batched[static_cast<size_t>(counter)]) / elapsed.count();
    };
    if (std::ranges::any_of(batched, [](u64 count) { return count != 0; })) {
        LOG_INFO(HW_GPU,
                 "GPU split batched writes per second: split_batch.macro_sunk {:.0f}, "
                 "split_batch.compute_sunk {:.0f}, split_batch.compute_same {:.0f}, "
                 "split_batch.pairs {:.0f} in split_batch.pair_blocks {:.0f}, "
                 "split_batch.delta {:.0f} in split_batch.delta_blocks {:.0f}, "
                 "split_batch.draw_batches {:.0f} for split_batch.draw_repeats {:.0f} more groups, "
                 "split_batch.same_index_first {:.0f}",
                 rate(Batched::MacroSunk), rate(Batched::ComputeSunk), rate(Batched::ComputeSame),
                 rate(Batched::Pairs), rate(Batched::PairBlocks), rate(Batched::DeltaWrites),
                 rate(Batched::DeltaBlocks), rate(Batched::DrawBatches),
                 rate(Batched::DrawRepeats), rate(Batched::SameIndexFirst));
    }
    batched.fill(0);
    batched_since = now;
}

void GpuSplit::Drain(DrainSite site) {
    RunPendingFlush(); // Everything sent so far includes the held-back work
    consumed_cache = consumed.load(std::memory_order_acquire);
    if (consumed_cache == write_position) {
        return;
    }
    Publish();
    Nx2Profile::Scope scope{Nx2Profile::Zone::SplitDrain};
    // "behind": the ring words the back had left to run, which tells a slow back from a front
    // far ahead (tools/nx2/critical_path.py).
    Common::Nx2Timeline::ArgSpan timeline_span{drain_site_names[static_cast<size_t>(site)],
                                               {"behind", write_position - consumed_cache}};
    const u64 start = Nx2Profile::Enabled() ? Nx2Profile::Ticks() : 0;
    const u64 begin_ns = Common::Nx2Hitch::NowNs();
    const u64 behind = write_position - consumed_cache;
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (consumed_cache == write_position) {
            publish_every = 8;
            Nx2Profile::AddDrain(site, start);
            NoteSiteWait(site, behind, Common::Nx2Hitch::NowNs() - begin_ns);
            return;
        }
        Backoff(spins, write_position);
    }
}

bool GpuSplit::WaitForWriters() {
    return WaitFor(std::min(last_writer_position, write_position), DrainSite::ListWriters);
}

bool GpuSplit::WaitForMemoryWriters() {
    const u64 target = (std::max)(last_writer_position, last_memory_writer_position);
    return WaitFor((std::min)(target, write_position), DrainSite::MacroDmaRead);
}

bool GpuSplit::WaitForWritesTo(u64 begin, u64 size) {
    consumed_cache = consumed.load(std::memory_order_acquire);
    const u64 end = begin + size;
    const u64 remembered = (std::min<u64>)(pending_write_count, pending_writes.size());
    u64 target = 0;
    u64 scanned = 0;
    for (; scanned < remembered; ++scanned) {
        const PendingWrite& write =
            pending_writes[(pending_write_count - 1 - scanned) % pending_writes.size()];
        if (write.position <= consumed_cache) {
            break;
        }
        if (write.begin < end && begin < write.end) {
            target = (std::max)(target, write.position);
        }
    }
    if (scanned == pending_writes.size()) {
        return false;
    }
    if (target != 0) {
        WaitFor(target, DrainSite::MacroDmaRead);
    }
    return true;
}

bool GpuSplit::HasQueuedWritesTo(u64 begin, u64 size) {
    consumed_cache = consumed.load(std::memory_order_acquire);
    const u64 end = begin + size;
    const u64 remembered = (std::min<u64>)(pending_write_count, pending_writes.size());
    for (u64 scanned = 0; scanned < remembered; ++scanned) {
        const PendingWrite& write =
            pending_writes[(pending_write_count - 1 - scanned) % pending_writes.size()];
        if (write.position <= consumed_cache) {
            return false;
        }
        if (write.begin < end && begin < write.end) {
            return true;
        }
    }
    return remembered == pending_writes.size();
}

bool GpuSplit::OverlayQueuedWrites(u64 since, u64 begin, u32* out, u32 words) {
    if ((begin & 3) != 0) {
        return false;
    }
    const u64 end = begin + u64{words} * sizeof(u32);
    const u64 remembered = (std::min<u64>)(pending_write_count, pending_writes.size());
    // Newest first, back to what the back had consumed before guest memory was read.
    std::array<u32, 64> overlapping;
    u32 found = 0;
    u64 scanned = 0;
    for (; scanned < remembered; ++scanned) {
        const u64 index = (pending_write_count - 1 - scanned) % pending_writes.size();
        const PendingWrite& write = pending_writes[index];
        if (write.position <= since) {
            break;
        }
        if (write.begin < end && begin < write.end) {
            if (write.words == 0 || (write.begin & 3) != 0 || found == overlapping.size()) {
                // EDEN_NX2_OVERLAY_TRACE: which queued writes keep reads waiting.
                static const bool trace = std::getenv("EDEN_NX2_OVERLAY_TRACE") != nullptr;
                static std::atomic<u32> traced{};
                if (trace && traced.fetch_add(1, std::memory_order_relaxed) < 48) {
                    LOG_INFO(HW_GPU, "overlay: read {:#x}+{} blocked by {} write {:#x}-{:#x}",
                             begin, words * 4, write.what, write.begin, write.end);
                }
                return false;
            }
            overlapping[found++] = static_cast<u32>(index);
        }
    }
    if (scanned == pending_writes.size()) {
        static const bool trace = std::getenv("EDEN_NX2_OVERLAY_TRACE") != nullptr;
        static std::atomic<u32> traced{};
        if (trace && traced.fetch_add(1, std::memory_order_relaxed) < 8) {
            LOG_INFO(HW_GPU, "overlay: read {:#x} with over {} queued writes", begin,
                     pending_writes.size());
        }
        return false;
    }
    for (u32 i = found; i-- > 0;) {
        const PendingWrite& write = pending_writes[overlapping[i]];
        for (u32 w = 0; w < write.words; ++w) {
            const u64 address = write.begin + u64{w} * sizeof(u32);
            if (address >= begin && address < end) {
                out[(address - begin) / sizeof(u32)] = write.data[w];
            }
        }
    }
    return true;
}

bool GpuSplit::WaitForSync() {
    return WaitFor(std::min(last_sync_position, write_position), DrainSite::ListSync);
}

bool GpuSplit::WaitForSyncedWriters(DrainSite site) {
    return WaitFor(std::min({last_writer_position, last_sync_position, write_position}), site);
}

void GpuSplit::RangeLastWrite(GPUVAddr begin, GPUVAddr end, const u32* data, size_t space) {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_RANGED_WAITS");
        return value == nullptr || value[0] != '0';
    }();
    // Only the method that was just sent: anything after it keeps its unranged mark.
    if (!enabled || begin >= end || last_memory_write_position != write_position) {
        return;
    }
    const u32 method = last_memory_write_method;
    last_memory_write_position = previous_memory_write_position;
    last_memory_write_method = previous_memory_write_method;
    if (last_method_write_position == write_position) {
        last_method_write_position = previous_method_write_position;
    }
    if (write_stamps) [[unlikely]] {
        StampWrite(begin, end, true);
    }
    static const bool forward = [] {
        const char* const value = std::getenv("EDEN_NX2_FORWARD_READS");
        return value == nullptr || value[0] != '0';
    }();
    if (!forward) {
        data = nullptr;
    }
    const size_t words = static_cast<size_t>(end - begin) / sizeof(u32);
    // Consecutive uploads continue one range (each word may use another of the 16 data methods).
    if (!ranged_writes.empty()) {
        RangedWrite& last = ranged_writes.back();
        if (last.end == begin) {
            last.end = end;
            last.position = write_position;
            last.method = method;
            if (data != nullptr && last.space == space && space != 0 &&
                last.data.size() * sizeof(u32) == static_cast<size_t>(begin - last.begin)) {
                last.data.insert(last.data.end(), data, data + words);
            } else {
                last.space = 0;
                last.data.clear();
            }
            return;
        }
    }
    if (ranged_writes.size() >= MaxRangedWrites()) {
        // Too many in flight: the oldest counts for every read again.
        const RangedWrite& oldest = ranged_writes.front();
        if (oldest.position > last_memory_write_position) {
            last_memory_write_position = oldest.position;
            last_memory_write_method = oldest.method;
        }
        last_method_write_position = (std::max)(last_method_write_position, oldest.position);
        ranged_writes.pop_front();
    }
    RangedWrite& write = ranged_writes.push_back(write_position, begin, end, method);
    if (data != nullptr && space != 0 && (end - begin) % sizeof(u32) == 0) {
        write.space = space;
        write.data.assign(data, data + words);
    }
}

bool GpuSplit::ForwardRead(size_t space, GPUVAddr begin, u32 words, u32* out) {
    const GPUVAddr end = begin + GPUVAddr{words} * sizeof(u32);
    const u64 done = consumed.load(std::memory_order_acquire);
    u32 method{};
    const u64 target = MemoryWriterTarget(done, begin, end, method);
    if (target <= done) {
        return false; // Nothing to wait for: memory holds the words
    }
    // The newest overlapping write; the ring is in send order.
    for (size_t index = ranged_writes.size(); index-- > 0;) {
        const RangedWrite* const it = &ranged_writes[index];
        if (!(it->begin < end && begin < it->end)) {
            continue;
        }
        if (it->position != target || it->space != space || space == 0 || begin < it->begin ||
            end > it->end || (begin - it->begin) % sizeof(u32) != 0 ||
            it->data.size() * sizeof(u32) != static_cast<size_t>(it->end - it->begin)) {
            return false;
        }
        std::memcpy(out, it->data.data() + (begin - it->begin) / sizeof(u32),
                    size_t{words} * sizeof(u32));
        return true;
    }
    return false;
}

u64 GpuSplit::MemoryWriterTarget(u64 done, GPUVAddr begin, GPUVAddr end, u32& target_method) {
    while (!ranged_writes.empty() && ranged_writes.front().position <= done) {
        ranged_writes.pop_front();
    }
    u64 target = std::max({last_writer_position, last_sync_position, last_memory_write_position});
    target_method = last_memory_write_method;
    // Newest first (positions rise along the ring): the first overlap is the target, and the
    // writes before `target` can't raise it.
    for (size_t index = ranged_writes.size();
         index-- > 0 && ranged_writes[index].position > target;) {
        const RangedWrite* const it = &ranged_writes[index];
        if (it->begin < end && begin < it->end) {
            target = it->position;
            target_method = it->method;
            break;
        }
    }
    return std::min(target, write_position);
}

size_t GpuSplit::MaxRangedWrites() {
    // 256 overflowed in Ocarina of Time (thousands of constant buffer uploads a frame): the
    // oldest then counted for every semaphore acquire and macro memory read, which waited for
    // the back to reach it.
    static const size_t max = [] {
        const char* const value = std::getenv("EDEN_NX2_RANGED_MAX");
        return value != nullptr ? (std::max<size_t>)(std::strtoull(value, nullptr, 10), 1) : 256;
    }();
    return max;
}

void GpuSplit::RangedRing::Grow() {
    // First use: MaxRangedWrites() slots (RangeLastWrite pops the oldest before it holds more),
    // a power of two for the index mask. Past it, doubled, oldest moved to slot 0.
    const size_t capacity = slots.empty() ? std::bit_ceil(MaxRangedWrites()) : slots.size() * 2;
    std::vector<RangedWrite> grown(capacity);
    for (size_t i = 0; i < count; ++i) {
        grown[i] = std::move((*this)[i]);
    }
    slots = std::move(grown);
    mask = capacity - 1;
    head = 0;
}

GpuSplit::GuestWriters GpuSplit::QueuedGuestWriters(u64 done, GPUVAddr address, u64 device,
                                                     u64 size, std::span<const u64> skip) const {
    GuestWriters found{};
    using Cause = GuestWriterCause;
    const auto unbounded = [&](u64 position, Cause cause, const char* what = nullptr) {
        if (position > done && position > found.unbounded) {
            found.unbounded = position;
            found.unbounded_cause = cause;
            found.what = what;
        }
    };
    const auto noted = [&](u64 position, Cause cause) {
        if (position > found.noted) {
            found.noted = position;
            found.noted_cause = cause;
        }
    };
    unbounded(last_unbounded_writer_position, Cause::UnboundedWriter);
    unbounded(last_unranged_position, Cause::Unranged);
    unbounded(last_function_position, Cause::Function);
    unbounded(last_method_write_position, Cause::MethodWrite);
    const u64 end = device + size;
    // NoteMemoryWrite and NoteMemoryWriteData (puller releases, 3D reports and uploads), newest
    // first; an unknown range may write anything.
    const u64 remembered = (std::min<u64>)(pending_write_count, pending_writes.size());
    u64 scanned = 0;
    for (; scanned < remembered; ++scanned) {
        const PendingWrite& write =
            pending_writes[(pending_write_count - 1 - scanned) % pending_writes.size()];
        if (write.position <= done) {
            break;
        }
        if (write.end == ~u64{0}) {
            unbounded(write.position, Cause::UnknownPending, write.what);
        } else if (write.begin < end && device < write.end) {
            noted(write.position, Cause::Pending);
            break;
        }
    }
    if (scanned == pending_writes.size()) {
        unbounded(write_position, Cause::PendingScanLimit);
    }
    // NoteWriterRange and NoteReleaseWriter (compute and copy releases, uploads, copies).
    const u64 oldest =
        writer_range_count - (std::min<u64>)(writer_range_count, writer_ranges.size());
    bool reached = false;
    scanned = 0;
    for (u64 index = writer_range_count; index > oldest && scanned < MaxWriterScan;
         --index, ++scanned) {
        const WriterRange& range = writer_ranges[(index - 1) % writer_ranges.size()];
        if (range.position <= done) {
            reached = true;
            break;
        }
        if (range.begin < end && device < range.end &&
            std::find(skip.begin(), skip.end(), range.position) == skip.end()) {
            if (range.position > found.noted) {
                found.range_kind = range.kind;
                found.range_begin = range.begin;
                found.range_end = range.end;
            }
            noted(range.position, Cause::WriterRange);
            reached = true;
            break;
        }
    }
    if (!reached && (scanned == MaxWriterScan || oldest != 0)) {
        unbounded(write_position, Cause::WriterScanLimit);
    }
    // RangeLastWrite (3D constant buffer uploads and the like), by GPU address.
    for (size_t index = ranged_writes.size();
         index-- > 0 && ranged_writes[index].position > done;) {
        const RangedWrite* const it = &ranged_writes[index];
        if (it->begin < address + size && address < it->end) {
            noted(it->position, Cause::Ranged);
            break;
        }
    }
    found.noted = (std::min)(found.noted, write_position);
    found.unbounded = (std::min)(found.unbounded, write_position);
    return found;
}

namespace {
// Device and GPU addresses share the tables (a collision only makes a stamp newer).
constexpr u64 GpuStampSalt = 0x9e3779b97f4a7c15ULL;
size_t StampIndex(u64 key, size_t size) {
    return static_cast<size_t>((key * 0xff51afd7ed558ccdULL) >> 40) & (size - 1);
}
} // Anonymous namespace

void GpuSplit::EnableWriteStamps() {
    if (write_stamps) {
        return;
    }
    block_stamps = std::make_unique<StampSlot[]>(BlockStamps);
    page_stamps = std::make_unique<StampSlot[]>(PageStamps);
    // Writes sent so far were never stamped: they all count.
    write_stamp_floor = write_position;
    write_stamps = true;
}

void GpuSplit::StampWrite(u64 begin, u64 end, bool gpu_address) {
    if (end <= begin) {
        return;
    }
    if (end - begin > (u64{1} << 20)) {
        write_stamp_floor = write_position;
        return;
    }
    const u64 salt = gpu_address ? GpuStampSalt : 0;
    const bool pages = end - begin > (u64{16} << 10);
    const u32 shift = pages ? 12 : 8;
    StampSlot* const table = pages ? page_stamps.get() : block_stamps.get();
    const size_t size = pages ? PageStamps : BlockStamps;
    for (u64 key = begin >> shift; key <= (end - 1) >> shift; ++key) {
        StampSlot& slot = table[StampIndex(key ^ salt, size)];
        if (slot.last_tag != write_stamp_tag || write_stamp_tag == 0) {
            slot.before = (std::max)(slot.before, slot.last);
            slot.last_tag = write_stamp_tag;
        }
        slot.last = write_position;
    }
}

u64 GpuSplit::StampOf(u64 begin, u64 end, bool gpu_address, u64 skip_tag) const {
    const u64 salt = gpu_address ? GpuStampSalt : 0;
    const auto read = [skip_tag](const StampSlot& slot) {
        return skip_tag != 0 && slot.last_tag == skip_tag ? slot.before : slot.last;
    };
    u64 stamp = 0;
    for (u64 key = begin >> 8; key <= (end - 1) >> 8; ++key) {
        stamp = (std::max)(stamp, read(block_stamps[StampIndex(key ^ salt, BlockStamps)]));
    }
    for (u64 key = begin >> 12; key <= (end - 1) >> 12; ++key) {
        stamp = (std::max)(stamp, read(page_stamps[StampIndex(key ^ salt, PageStamps)]));
    }
    return stamp;
}

u64 GpuSplit::WriteStamp(u64 device, u64 size, GPUVAddr address, u64 skip_tag) const {
    if (!write_stamps) {
        return write_position;
    }
    u64 stamp = (std::max)({write_stamp_floor, unknown_write_position,
                            last_unbounded_writer_position, last_unranged_position,
                            last_function_position, last_method_write_position});
    if (size != 0) {
        stamp = (std::max)(stamp, StampOf(device, device + size, false, skip_tag));
        stamp = (std::max)(stamp, StampOf(address, address + size, true, skip_tag));
    }
    return (std::min)(stamp, write_position);
}

bool GpuSplit::HasQueuedMemoryWriters(GPUVAddr begin, GPUVAddr end) {
    const u64 done = consumed.load(std::memory_order_acquire);
    u32 method{};
    return MemoryWriterTarget(done, begin, end, method) > done;
}

u32 GpuSplit::QueuedWriterKind(GPUVAddr begin, GPUVAddr end, u32& method) {
    const u64 done = consumed.load(std::memory_order_acquire);
    const u64 target = MemoryWriterTarget(done, begin, end, method);
    if (target <= done) {
        return 0;
    }
    return target == last_writer_position         ? 1
           : target == last_sync_position         ? 2
           : target == last_memory_write_position ? 3
                                                  : 4;
}

void GpuSplit::NoteRelease(u64 device, GPUVAddr address, u32 payload) {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_ACQUIRE_QUEUED");
        return value == nullptr || value[0] != '0';
    }();
    if (!enabled) {
        return;
    }
    queued_releases[(device >> 2) % queued_releases.size()] = {write_position, device, payload};
}

std::optional<u32> GpuSplit::QueuedReleasePayload(u64 device, GPUVAddr address, u32* miss) {
    const QueuedRelease& release = queued_releases[(device >> 2) % queued_releases.size()];
    const u64 done = consumed.load(std::memory_order_acquire);
    if (release.device != device || release.position == 0) {
        if (miss) *miss = 1;
        return std::nullopt;
    }
    if (release.position <= done) {
        if (miss) *miss = 2;
        return std::nullopt;
    }
    // Any write sent after the release (an unranged one, or a ranged one to the word) may change
    // the word before the acquire runs.
    u32 method{};
    if (MemoryWriterTarget(done, address, address + sizeof(u32), method) > release.position) {
        if (miss) *miss = 3;
        return std::nullopt;
    }
    return release.payload;
}

bool GpuSplit::WaitForMemoryWriters(DrainSite site, GPUVAddr begin, GPUVAddr end) {
    const u64 done = consumed.load(std::memory_order_acquire);
    u32 target_method{};
    const u64 target = MemoryWriterTarget(done, begin, end, target_method);
    const u64 unranged =
        std::max({last_writer_position, last_sync_position, last_memory_write_position});
    if (Nx2Profile::Enabled()) {
        // What the read waits for: the newest writer it depends on.
        using Cause = Nx2Profile::DmaWaitCause;
        const u64 all_writes =
            ranged_writes.empty() ? unranged : std::max(unranged, ranged_writes.back().position);
        if (done < target) {
            Cause cause = Cause::HostSync;
            if (target != last_writer_position && target != last_sync_position) {
                const u32 method = target_method;
                cause = method == DeferredWrite               ? Cause::Deferred
                        : method == OtherEngineWrite          ? Cause::OtherEngine
                        : method >= 0x8e4 && method < 0x8f4   ? Cause::ConstBufferUpload
                        : method == 0x6d                      ? Cause::InlineUpload
                        : method == 0x6c3                     ? Cause::Report
                        : method == 0x6c                      ? Cause::DmaLaunch
                        : method == 0xb2                      ? Cause::SyncInfo
                        : method == 0x44                      ? Cause::WaitForIdle
                        : method >= 0xe00                     ? Cause::Macro3D
                                                              : Cause::Other3D;
            } else if (target == last_writer_position) {
                cause = Cause::ComputeCopy;
            }
            Nx2Profile::AddDmaWaitCause(cause);
        } else if (done < std::min(all_writes, write_position)) {
            Nx2Profile::AddDmaWaitCause(Cause::AvoidedByRange);
        }
    }
    return WaitFor(target, site);
}

bool GpuSplit::WaitFor(u64 target, DrainSite site) {
    consumed_cache = consumed.load(std::memory_order_acquire);
    if (consumed_cache >= target) {
        return false;
    }
    Nx2Profile::Scope scope{Nx2Profile::Zone::WriterWait};
    Common::Nx2Timeline::ArgSpan timeline_span{drain_site_names[static_cast<size_t>(site)],
                                               {"behind", target - consumed_cache}};
    wait_target.store(target, std::memory_order_seq_cst);
    Publish();
    const u64 start = Nx2Profile::Enabled() ? Nx2Profile::Ticks() : 0;
    const u64 begin_ns = Common::Nx2Hitch::NowNs();
    const u64 behind = target - consumed_cache;
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (consumed_cache >= target) {
            wait_target.store(~u64{0}, std::memory_order_relaxed);
            Nx2Profile::AddDrain(site, start);
            NoteSiteWait(site, behind, Common::Nx2Hitch::NowNs() - begin_ns);
            return true;
        }
        Backoff(spins, target);
    }
}

void GpuSplit::Backoff(u32 spins, u64 target) {
    if (spins < 16384) {
        Pause();
        return;
    }
    if (!Nx2Perf::On(Nx2Perf::Opt::FrontSleep) || !back_running.load(std::memory_order_acquire)) {
        std::this_thread::yield();
        return;
    }
    // A long wait sleeps until the back stores a new position (a port of the original front_wake): the
    // back checks front_waiting after each store, the front checks the position after setting it.
    front_waiting.store(true, std::memory_order_seq_cst);
    const u32 seen = front_wake.load(std::memory_order_seq_cst);
    if (consumed.load(std::memory_order_seq_cst) < target) {
        front_wake.wait(seen, std::memory_order_seq_cst);
    }
    front_waiting.store(false, std::memory_order_relaxed);
}

void GpuSplit::SwitchEngine(Tegra::Engines::EngineInterface* engine) {
    front_engine = engine;
    u64* const entry = Reserve(2);
    entry[0] = static_cast<u64>(Kind::Engine);
    entry[1] = reinterpret_cast<u64>(engine);
    Commit(2, Effect::None);
}

void GpuSplit::MultiMethod(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                           u32 amount, u32 methods_pending, const GPUVAddr* segment,
                           Effect effect) {
    SelectEngine(engine);
    // Long runs (constant buffer and inline uploads) go in pieces, as when a method's data
    // continues in the next GPFIFO entry.
    while (amount > 0) {
        const u32 piece = std::min(amount, MaxMultiWords);
        const size_t head = segment != nullptr ? 3 : 2;
        const size_t size = head + (piece + 1) / 2;
        u64* const entry = Reserve(size);
        entry[0] = static_cast<u64>(Kind::Multi) | (segment != nullptr ? HasSegment : 0) |
                   (u64{method & 0xffff} << 8) | (u64{piece} << 32);
        entry[1] = methods_pending;
        if (segment != nullptr) {
            entry[2] = *segment;
        }
        entry[size - 1] = 0;
        std::memcpy(entry + head, words, size_t{piece} * sizeof(u32));
        Commit(size, effect);
        words += piece;
        amount -= piece;
        methods_pending -= piece;
    }
    if (engine->memory_write_mask[method]) {
        NoteMethodWrite(engine, method);
    }
}

void GpuSplit::DeferFunction(std::function<void()>&& func) {
    auto* const heap = new std::function<void()>(std::move(func));
    Defer([heap] {
        (*heap)();
        delete heap;
    });
    last_function_position = write_position;
}

void GpuSplit::NoteWrite(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size) {
    if (size == 0 || !Nx2Perf::On(Nx2Perf::Opt::SplitOverlap)) {
        return;
    }
    // While a queued function may still change the GPU mappings, the back may translate the
    // address differently. Sparse pages are written through their own backing.
    const bool sparse = address < Tegra::g_nx2_sparse_high.load(std::memory_order_relaxed) &&
                        Tegra::g_nx2_sparse_low.load(std::memory_order_relaxed) < address + size;
    if (last_function_position > consumed_cache || sparse) {
        writes.Entry(Effect::Unknown, write_position);
        return;
    }
    constexpr GPUVAddr page_mask = ~GPUVAddr{0xfff};
    for (GPUVAddr page = address & page_mask; page < address + size; page += 0x1000) {
        if (&memory != note_memory || page != note_page || note_function != last_function_position) {
            const std::optional<DAddr> device = memory.GpuToCpuAddress(page);
            if (!device) {
                note_memory = nullptr;
                writes.Entry(Effect::Unknown, write_position);
                return;
            }
            note_memory = &memory;
            note_page = page;
            note_device = *device & page_mask;
            note_function = last_function_position;
        }
        writes.Write(note_device, note_device + 0x1000, write_position);
    }
}

void GpuSplit::NoteWriterRange(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size,
                               WriterKind kind) {
    if (size != 0 && TryNoteWriterRange(memory, address, size, kind) != ReleaseNote::Ranged) {
        NoteUnboundedWriter(kind);
    }
}

void GpuSplit::NoteReleaseWriter(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size,
                                 bool at_run) {
    ReleaseNote note = TryNoteWriterRange(memory, address, size, WriterKind::Release);
    if (note == ReleaseNote::Unmapped && !at_run) {
        // A fence callback translates it later, after functions queued meanwhile may map it.
        note = ReleaseNote::LateUnmapped;
    }
    CountRelease(note);
    if (note != ReleaseNote::Ranged && note != ReleaseNote::Unmapped) {
        NoteUnboundedWriter(WriterKind::Release);
    }
}

GpuSplit::ReleaseNote GpuSplit::TryNoteWriterRange(const Tegra::MemoryManager& memory,
                                                   GPUVAddr address, u64 size,
                                                   WriterKind kind) {
    // As NoteWrite: a queued function may still remap the range, sparse pages are written through
    // their own backing. One device range only, as the reads compare device addresses.
    if (last_function_position > consumed_cache) {
        consumed_cache = consumed.load(std::memory_order_acquire);
    }
    if (writer_range_count > writer_range_sealed && &memory == writer_range_memory &&
        address == writer_range_address && size == writer_range_size &&
        writer_range_function == last_function_position &&
        last_function_position <= consumed_cache) {
        WriterRange& last = writer_ranges[(writer_range_count - 1) % writer_ranges.size()];
        last.position = write_position;
        last.kind = kind;
        if (watched_pages != 0) {
            NoteWatchedWrite(last.begin, last.end, last.position);
        }
        if (write_stamps) [[unlikely]] {
            StampWrite(last.begin, last.end, false);
        }
        return ReleaseNote::Ranged;
    }
    writer_range_memory = nullptr;
    if (last_function_position > consumed_cache) {
        return ReleaseNote::PendingMap;
    }
    if (address < Tegra::g_nx2_sparse_high.load(std::memory_order_relaxed) &&
        Tegra::g_nx2_sparse_low.load(std::memory_order_relaxed) < address + size) {
        return ReleaseNote::Sparse;
    }
    if (size > (u64{64} << 20)) {
        return ReleaseNote::Split;
    }
    const std::optional<DAddr> device = memory.GpuToCpuAddress(address);
    if (!device) {
        return ReleaseNote::Unmapped;
    }
    if (memory.MaxContinuousRange(address, size) < size) {
        return ReleaseNote::Split;
    }
    writer_ranges[writer_range_count++ % writer_ranges.size()] = {write_position, *device,
                                                                  *device + size, kind};
    if (watched_pages != 0) {
        NoteWatchedWrite(*device, *device + size, write_position);
    }
    if (write_stamps) [[unlikely]] {
        StampWrite(*device, *device + size, false);
    }
    writer_range_memory = &memory;
    writer_range_address = address;
    writer_range_size = size;
    writer_range_function = last_function_position;
    return ReleaseNote::Ranged;
}

// EXP122 page watch (fable_answer_long_drains 8.2 (a'), fable52_answer_kills 3.4).
//
// WritersMayWriteSince scans the noted writer ranges newest first and gives up after
// MaxWriterScan. At ~11,500 ranges a second (compute inline uploads) a page whose snapshot is a
// few frames old was always past the limit: every such "may write" was a give-up, never an
// overlap, and each cost the front a drain. The page watch keeps, for the few device pages the
// front's macro DMA reads hold a snapshot of, the noted ranges that touched each since its
// snapshot, so the answer for such a page is a few compares and has no limit.
//
// Threads: the ranges are noted (TryNoteWriterRange, from KeplerCompute and MaxwellDMA front
// methods) and asked about (Maxwell3D::FrontDmaRead) on the front thread, as is the ring itself
// (writer_ranges and writer_range_count are plain front-side fields). The table is front-only
// state too, so a range is in a watched page's list before any later check runs, in program
// order: the same guarantee the ring gives, with no atomics needed.
//
// Exactness: an entry started at writer range count C (sealed: the last range below C can no
// longer move its position) holds every range noted from C on that touches the page, with its
// newest position (an inline upload's repeat of the same range only moves the position, and is
// recorded again). A check whose ranges below `known` were already found to miss (or listed)
// may use the entry when C <= known; one whose `since` is at or past the entry's start position
// also may, as every range below C was noted at or before it. Otherwise (another channel's
// snapshot of the same page started the entry later, an evicted entry, a range over several
// pages) the ring scan answers as before. Past 8 entries a page's further ranges are folded into
// one union with the newest position: never smaller than what was written, so never a missed
// write, only a coarser byte list.
//
// The table is set associative (64 sets of 4 pages, least recently used way replaced): two tag
// probes for a range of one or two pages. A range over more than 16 pages walks the 256 tags
// instead, after a test against the lowest and highest watched page. EXP122=2 runs the old scan
// without a limit after every "no write" and counts each overlap it finds that the watch did not
// list (the "verify mismatches" of EXP10's log line, which must stay 0).
struct GpuSplit::PageWatch {
    static constexpr u32 Sets = 64;
    static constexpr u32 Ways = 4;
    static constexpr u32 Slots = Sets * Ways;
    static constexpr u32 Capacity = 8;
    static constexpr u64 Empty = ~u64{0};
    struct Write {
        u64 position;
        u16 begin; ///< Byte offsets in the page
        u16 end;
    };
    struct Entry {
        u64 start_count;
        u64 start_position;
        u64 last_use;
        u32 count;
        bool folded; ///< `fold` holds the ranges past Capacity
        Write fold;
        std::array<Write, Capacity> writes;
    };
    std::array<u64, Slots> pages;
    std::array<Entry, Slots> entries{};
    u64 min_page = Empty;
    u64 max_page = 0;
    u64 clock = 0;
    // Stats: mutable through the pointer from const WritersMayWriteSince.
    u64 answered = 0;
    u64 answered_write = 0;
    u64 overflows = 0;
    u64 verified = 0;
    u64 mismatches = 0;

    PageWatch() {
        pages.fill(Empty);
    }
    static u32 Set(u64 page) {
        return static_cast<u32>((page ^ (page >> 6) ^ (page >> 12)) & (Sets - 1));
    }
    [[nodiscard]] s32 Find(u64 page) const {
        const u32 base = Set(page) * Ways;
        for (u32 way = 0; way < Ways; ++way) {
            if (pages[base + way] == page) {
                return static_cast<s32>(base + way);
            }
        }
        return -1;
    }
    void Record(Entry& entry, u64 page, u64 begin, u64 end, u64 position) {
        const u64 base = page << 12;
        const u16 first = static_cast<u16>((std::max)(begin, base) - base);
        const u16 last = static_cast<u16>((std::min)(end, base + 0x1000) - base);
        for (u32 i = 0; i < entry.count; ++i) {
            Write& write = entry.writes[i];
            if (write.begin == first && write.end == last) {
                write.position = (std::max)(write.position, position);
                return;
            }
        }
        if (entry.count < Capacity) {
            entry.writes[entry.count++] = {position, first, last};
            return;
        }
        if (!entry.folded) {
            entry.folded = true;
            entry.fold = {position, first, last};
            ++overflows;
            return;
        }
        entry.fold.position = (std::max)(entry.fold.position, position);
        entry.fold.begin = (std::min)(entry.fold.begin, first);
        entry.fold.end = (std::max)(entry.fold.end, last);
    }
};

void GpuSplit::ClearPageWatch() {
    // Front only (or with the back stopped).
    if (page_watch) {
        page_watch->pages.fill(PageWatch::Empty);
        page_watch->min_page = PageWatch::Empty;
        page_watch->max_page = 0;
    }
    watched_pages = 0;
}

void GpuSplit::WatchPage(u64 page) {
    if (page_watch_mode == 0) {
        return;
    }
    if (!page_watch) {
        page_watch = std::make_unique<PageWatch>();
    }
    PageWatch& watch = *page_watch;
    s32 slot = watch.Find(page);
    if (slot < 0) {
        const u32 base = PageWatch::Set(page) * PageWatch::Ways;
        u32 pick = base;
        for (u32 way = 0; way < PageWatch::Ways; ++way) {
            const u32 index = base + way;
            if (watch.pages[index] == PageWatch::Empty) {
                pick = index;
                break;
            }
            if (watch.entries[index].last_use < watch.entries[pick].last_use) {
                pick = index;
            }
        }
        if (watch.pages[pick] == PageWatch::Empty) {
            ++watched_pages;
        }
        watch.pages[pick] = page;
        watch.min_page = (std::min)(watch.min_page, page);
        watch.max_page = (std::max)(watch.max_page, page);
        slot = static_cast<s32>(pick);
    }
    PageWatch::Entry& entry = watch.entries[static_cast<u32>(slot)];
    // Sealed: the newest range can't move its position past the start any more.
    entry.start_count = WriterRangeCount();
    entry.start_position = write_position;
    entry.last_use = ++watch.clock;
    entry.count = 0;
    entry.folded = false;
}

void GpuSplit::NoteWatchedWrite(u64 begin, u64 end, u64 position) {
    if (end <= begin) {
        return;
    }
    PageWatch& watch = *page_watch;
    const u64 first_page = begin >> 12;
    const u64 last_page = (end - 1) >> 12;
    if (last_page < watch.min_page || first_page > watch.max_page) {
        return;
    }
    if (last_page - first_page < 16) {
        for (u64 page = first_page; page <= last_page; ++page) {
            if (const s32 slot = watch.Find(page); slot >= 0) {
                watch.Record(watch.entries[static_cast<u32>(slot)], page, begin, end, position);
            }
        }
        return;
    }
    for (u32 slot = 0; slot < PageWatch::Slots; ++slot) {
        const u64 page = watch.pages[slot];
        if (page != PageWatch::Empty && first_page <= page && page <= last_page) {
            watch.Record(watch.entries[slot], page, begin, end, position);
        }
    }
}

GpuSplit::WatchAnswer GpuSplit::WatchedWritesSince(u64 since, u64 begin, u64 size, u64 known,
                                                   ListedWrites* found) const {
    if (size == 0 || page_watch_mode == 0) {
        return WatchAnswer::Unwatched;
    }
    const u64 page = begin >> 12;
    if (((begin + size - 1) >> 12) != page) {
        return WatchAnswer::Unwatched;
    }
    PageWatch& watch = *page_watch;
    const s32 slot = watch.Find(page);
    if (slot < 0) {
        return WatchAnswer::Unwatched;
    }
    PageWatch::Entry& entry = watch.entries[static_cast<u32>(slot)];
    if (entry.start_count > known && entry.start_position > since) {
        // Started after this check's snapshot (another snapshot of the page): the ring answers.
        return WatchAnswer::Unwatched;
    }
    entry.last_use = ++watch.clock;
    const u64 base = page << 12;
    const u64 end = begin + size;
    bool write = false;
    const auto test = [&](const PageWatch::Write& noted) {
        if (noted.position > since && base + noted.begin < end && begin < base + noted.end) {
            if (found == nullptr) {
                write = true;
            } else {
                found->Add(begin, size, base + noted.begin, base + noted.end);
            }
        }
    };
    for (u32 i = 0; i < entry.count && !write; ++i) {
        test(entry.writes[i]);
    }
    if (entry.folded && !write) {
        test(entry.fold);
    }
    ++watch.answered;
    if (write) {
        ++watch.answered_write;
        return WatchAnswer::Write;
    }
    if (page_watch_mode == 2) [[unlikely]] {
        // The old scan without its limit: every range since `since` that overlaps must be in
        // the answer (listed, or "may write" without a list).
        ++watch.verified;
        const u64 oldest =
            writer_range_count - (std::min<u64>)(writer_range_count, writer_ranges.size());
        for (u64 index = writer_range_count; index > oldest; --index) {
            const WriterRange& range = writer_ranges[(index - 1) % writer_ranges.size()];
            if (range.position <= since) {
                break;
            }
            if (!(range.begin < end && begin < range.end)) {
                continue;
            }
            const u64 clip_begin = (std::max)(range.begin, begin);
            const u64 clip_end = (std::min)(range.end, end);
            bool covered = false;
            if (found != nullptr) {
                for (u32 i = 0; i < found->count && !covered; ++i) {
                    covered = begin + found->ranges[i].first <= clip_begin &&
                              clip_end <= begin + found->ranges[i].second;
                }
            }
            if (!covered) {
                if (watch.mismatches++ < 16) {
                    LOG_WARNING(HW_GPU,
                                "Writer page watch (EXP122=2): page {:#x} answered no write since "
                                "{}, but range {:#x}-{:#x} noted at {} overlaps (entry from range "
                                "{}, position {}; {} listed, folded {})",
                                page, since, range.begin, range.end, range.position,
                                entry.start_count, entry.start_position, entry.count,
                                entry.folded);
                }
                break;
            }
        }
    }
    return WatchAnswer::NoWrite;
}

GpuSplit::PageWatchStats GpuSplit::GetPageWatchStats() const {
    if (!page_watch) {
        return {0, 0, watched_pages, 0, 0, 0};
    }
    const PageWatch& watch = *page_watch;
    return {watch.answered, watch.answered_write, watched_pages,
            watch.overflows, watch.verified,       watch.mismatches};
}

void GpuSplit::NoteDispatchWriter() {
    last_dispatch_position = write_position;
    ++dispatches_sent;
    // The back records from the position it is told to, in ring order: the dispatch just sent
    // ran before that. Live switching EXP10 on or off counts it as unrecorded.
    const bool records = VideoCore::Nx2Debug::Experiment(10) != 0;
    if (records != dispatch_records) {
        dispatch_records = records;
        last_unrecorded_dispatch_position = write_position;
        Defer([this, records] { back_dispatch_records = records; }, Effect::None);
    } else if (!records) {
        last_unrecorded_dispatch_position = write_position;
    }
}

GpuSplit::DispatchCheck GpuSplit::DispatchesMayWriteSince(u64 since, u64 begin, u64 size, bool wait,
                                                          u64& checked, bool& waited,
                                                          ListedWrites* listed) {
    if (last_dispatch_position <= since) {
        return DispatchCheck::Miss;
    }
    if (last_unrecorded_dispatch_position > since) {
        return DispatchCheck::Unbounded;
    }
    if (consumed_cache < last_dispatch_position) {
        consumed_cache = consumed.load(std::memory_order_acquire);
    }
    if (consumed_cache < last_dispatch_position) {
        if (!wait) {
            return DispatchCheck::Queued;
        }
        WaitFor(last_dispatch_position, DrainSite::DispatchWait);
        waited = true;
    }
    // The back recorded every dispatch sent since before storing a position past it (acquired
    // by WaitFor or above). Records start at the entry that ran them: since and later is after
    // the snapshot.
    const u64 end = begin + size;
    const u64 count = dispatch_write_count.load(std::memory_order_acquire);
    const u64 oldest = count - (std::min)(count, DispatchWriteSlots);
    ListedWrites found = listed != nullptr ? *listed : ListedWrites{};
    u64 index = count;
    bool reached = false;
    for (u64 scanned = 0; index > (std::max)(checked, oldest); --index, ++scanned) {
        if (scanned == MaxDispatchScan) {
            return DispatchCheck::ScanLimit;
        }
        const DispatchWrite& write = dispatch_writes[(index - 1) % DispatchWriteSlots];
        if (write.position.load(std::memory_order_relaxed) < since) {
            reached = true;
            break;
        }
        const u64 write_begin = write.begin.load(std::memory_order_relaxed);
        const u64 write_end = write.end.load(std::memory_order_relaxed);
        if (write_begin < end && begin < write_end) {
            if (write_end == ~u64{0} && write_begin == 0) {
                return DispatchCheck::Unbounded;
            }
            if (listed == nullptr) {
                return DispatchCheck::Overlap;
            }
            // A slot reused meanwhile lists a stray range; the reuse check below then misses.
            found.Add(begin, size, write_begin, write_end);
        }
    }
    // Writes between `checked` and the oldest remembered were forgotten; the back may have
    // reused the slots read meanwhile (it only appends for dispatches sent after the newest).
    if ((!reached && oldest > checked) ||
        dispatch_write_count.load(std::memory_order_acquire) >= index + DispatchWriteSlots) {
        return DispatchCheck::ScanLimit;
    }
    if (listed != nullptr) {
        *listed = found;
    }
    checked = count;
    return DispatchCheck::Miss;
}

GpuSplit::DispatchStats GpuSplit::GetDispatchStats() const {
    DispatchStats stats{};
    stats.sent = dispatches_sent;
    stats.bounded = dispatches_bounded.load(std::memory_order_relaxed);
    for (size_t i = 0; i < stats.unbounded.size(); ++i) {
        stats.unbounded[i] = dispatches_unbounded[i].load(std::memory_order_relaxed);
    }
    return stats;
}

void GpuSplit::AppendDispatchWrite(u64 begin, u64 end) {
    const u64 count = dispatch_write_count.load(std::memory_order_relaxed);
    DispatchWrite& write = dispatch_writes[count % DispatchWriteSlots];
    write.position.store(back_entry_position, std::memory_order_relaxed);
    write.begin.store(begin, std::memory_order_relaxed);
    write.end.store(end, std::memory_order_relaxed);
    dispatch_write_count.store(count + 1, std::memory_order_release);
}

void GpuSplit::RecordDispatchUnbounded(DispatchUnbounded reason) {
    AppendDispatchWrite(0, ~u64{0});
    if (!dispatch_unbounded) {
        dispatch_unbounded = true;
        auto& counter = dispatches_unbounded[static_cast<size_t>(reason)];
        counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    }
}

void GpuSplit::FinishDispatch(bool ran, bool replaced) {
    if (replaced) {
        RecordDispatchUnbounded(DispatchUnbounded::Replaced);
    } else if (ran && !dispatch_confirmed) {
        RecordDispatchUnbounded(DispatchUnbounded::Unconfirmed);
    }
    if (!dispatch_unbounded) {
        dispatches_bounded.store(dispatches_bounded.load(std::memory_order_relaxed) + 1,
                                 std::memory_order_relaxed);
    }
}

u64 GpuSplit::SettleWrites() {
    WaitFor((std::min)(last_function_position, write_position), DrainSite::Map);
    consumed_cache = consumed.load(std::memory_order_acquire);
    writes.Retire(consumed_cache);
    return consumed_cache;
}

size_t GpuSplit::Execute(const u64* entry, size_t words) {
    const u64 head = entry[0];
    switch (static_cast<Kind>(head & 0xf)) {
    case Kind::Wrap:
        return 0;
    case Kind::Engine:
        back_engine = reinterpret_cast<Tegra::Engines::EngineInterface*>(entry[1]);
        return 2;
    case Kind::Method: {
        const u32 method = static_cast<u32>((head >> 8) & 0xffff);
        const u32 argument = static_cast<u32>(head >> 32);
        auto* const engine = back_engine;
        if (engine->fast_maxwell3d) {
            // Nx2Perf: as below without the profile zones and the virtual call. A profiled run
            // takes this path too: Maxwell3D::CallMethod and CallConstBufferWords sample their
            // own time (Nx2Profile::SampledScope), and ConsumeSink counts as "other".
            using Tegra::Engines::Maxwell3D;
            auto* const maxwell3d = static_cast<Maxwell3D*>(engine);
            constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
            if (method - cb_data < 16 && (head & HasSegment) == 0 &&
                Nx2Perf::On(Nx2Perf::Opt::CbBatch)) {
                // Single constant buffer words in a row (macros write them one by one) go up in
                // one write. They are executable: each entry would only consume the sink first.
                std::array<u32, 64> methods;
                std::array<u32, 64> arguments;
                methods[0] = method;
                arguments[0] = argument;
                u32 count = 1;
                while (count < methods.size() && count < words) {
                    const u64 next = entry[count];
                    const u32 next_method = static_cast<u32>((next >> 8) & 0xffff);
                    if (static_cast<Kind>(next & 0xf) != Kind::Method || (next & HasSegment) != 0 ||
                        next_method - cb_data >= 16) {
                        break;
                    }
                    methods[count] = next_method;
                    arguments[count] = static_cast<u32>(next >> 32);
                    ++count;
                }
                maxwell3d->ConsumeSink(*system);
                if (count == 1) {
                    maxwell3d->CallMethod(*system, method, argument, true);
                } else {
                    maxwell3d->CallConstBufferWords(*system, methods.data(), arguments.data(),
                                                    count);
                }
                return count;
            }
            if ((head & Direct) != 0) {
                maxwell3d->ConsumeSink(*system);
                maxwell3d->CallMethod(*system, method, argument, true);
            } else if (!maxwell3d->execution_mask[method]) {
                maxwell3d->method_sink.emplace_back(method, argument);
            } else {
                maxwell3d->ConsumeSink(*system);
                if ((head & HasSegment) != 0) {
                    maxwell3d->current_dma_segment = entry[1];
                }
                maxwell3d->CallMethod(*system, method, argument, (head & IsLast) != 0);
            }
            return (head & HasSegment) != 0 ? 2 : 1;
        }
        // EDEN_NX2_FASTDRAW_COUNT: any other engine's method ends delta draw reuse.
        if (Nx2FastDrawCount::g_on && !engine->counts_fast_draw_breaks) [[unlikely]] {
            Nx2FastDrawCount::Break(Nx2FastDrawCount::Reason::OtherEngine);
        }
        if ((head & Direct) != 0) {
            Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
            engine->ConsumeSink(*system);
            engine->CallMethod(*system, method, argument, true);
        } else if ((head & Sink) != 0) {
            engine->method_sink.emplace_back(
                method | Tegra::Engines::EngineInterface::SplitPlainTag, argument);
        } else if (!engine->execution_mask[method]) {
            engine->method_sink.emplace_back(method, argument);
        } else {
            Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
            engine->ConsumeSink(*system);
            if ((head & HasSegment) != 0) {
                engine->current_dma_segment = entry[1];
            }
            engine->CallMethod(*system, method, argument, (head & IsLast) != 0);
        }
        return (head & HasSegment) != 0 ? 2 : 1;
    }
    case Kind::Multi: {
        const u32 method = static_cast<u32>((head >> 8) & 0xffff);
        const u32 amount = static_cast<u32>(head >> 32);
        const size_t data = (head & HasSegment) != 0 ? 3 : 2;
        auto* const engine = back_engine;
        if (Nx2FastDrawCount::g_on && !engine->counts_fast_draw_breaks) [[unlikely]] {
            Nx2FastDrawCount::Break(Nx2FastDrawCount::Reason::OtherEngine);
        }
        Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
        engine->ConsumeSink(*system);
        if ((head & HasSegment) != 0) {
            engine->current_dma_segment = entry[2];
        }
        engine->CallMultiMethod(*system, method, reinterpret_cast<const u32*>(entry + data),
                                amount, static_cast<u32>(entry[1]));
        return data + (amount + 1) / 2;
    }
    case Kind::Call: {
        // EDEN_NX2_FASTDRAW_COUNT: deferred calls end reuse unless marked neutral.
        if (Nx2FastDrawCount::g_on && (head & Nx2FastDrawCount::NeutralCallBit) == 0) [[unlikely]] {
            NoteFastDrawCallBreak(entry[1]);
        }
        const size_t payload = static_cast<size_t>((head >> 8) & 0xffffff);
        reinterpret_cast<void (*)(const u64*)>(entry[1])(entry + 2);
        return 2 + payload;
    }
    case Kind::Pairs: {
        // EDEN_NX2_RING_PAIRS: one profiler scope and one call for the block; the sink first,
        // as for any other write, to keep the order.
        const u32 count = static_cast<u32>(head >> 32);
        auto* const engine = back_engine;
        if (Nx2FastDrawCount::g_on && !engine->counts_fast_draw_breaks) [[unlikely]] {
            Nx2FastDrawCount::Break(Nx2FastDrawCount::Reason::OtherEngine);
        }
        Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
        engine->ConsumeSink(*system);
        engine->ApplyPairs(*system, entry + 1, count);
        return 1 + size_t{count};
    }
    case Kind::StateDelta: {
        // EXP22: only Maxwell3D sends StateDelta and DrawBatch entries (Maxwell3D::FrontWrite).
        const u32 count = static_cast<u32>(head >> 32);
        Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
        static_cast<Tegra::Engines::Maxwell3D*>(back_engine)
            ->ApplyFrontDelta(*system, entry + 1, count);
        return 1 + size_t{count};
    }
    case Kind::DrawBatch: {
        const u32 count = static_cast<u32>((head >> 8) & 0xff);
        const u32 repeats = static_cast<u32>(head >> 32);
        Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
        static_cast<Tegra::Engines::Maxwell3D*>(back_engine)
            ->RunDrawBatch(*system, entry + 1, count, repeats);
        return 1 + size_t{count};
    }
    }
    return 1;
}

void GpuSplit::CheckFencesQuiet(u64 position) {
    const int mode = VideoCore::Nx2Debug::Experiment(3);
    if (mode == 0 || fence_quiet_check == nullptr) {
        return;
    }
    u64 epoch{};
    if (fence_quiet_check(fence_quiet_argument, mode == 1, &epoch)) {
        fence_quiet_epoch.store(epoch, std::memory_order_relaxed);
        fence_quiet_position.store(position, std::memory_order_release);
    } else {
        fence_quiet_position.store(~u64{0}, std::memory_order_relaxed);
        fence_busy_position.store(position, std::memory_order_relaxed);
    }
}

void GpuSplit::BackLoop(std::stop_token stop_token) {
    const Common::DkbCpuTrace::Registration cpu_trace{Common::DkbCpuTrace::Role::GpuBack};
    Common::SetCurrentThreadName("GPUBack");
    on_back_thread = true;
    back_running.store(true, std::memory_order_release);
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::PinCurrentThread(Common::PinnedRole::GpuBack);
    Common::SetCurrentThreadToPerformanceCores();
    system->RegisterHostThread();
#ifdef _WIN32
    g_back_thread_id.store(GetCurrentThreadId(), std::memory_order_relaxed);
#endif
    u64 read_position = 0;
    while (!stop_token.stop_requested()) {
        Nx2Profile::Report("GPU back thread");
        u64 available = published.load(std::memory_order_acquire);
        if (available == read_position) {
            // On the timeline (not through the zone) with the phase the wait reached: 0 = spin,
            // 1 = the idle hook ran, 2 = slept on `wake` (tools/nx2/critical_path.py).
            Nx2Profile::Scope scope{Nx2Profile::Zone::SplitWait, Nx2Profile::Enabled(), false};
            Common::Nx2Timeline::ArgSpan timeline_span{
                Nx2Profile::zone_names[static_cast<size_t>(Nx2Profile::Zone::SplitWait)],
                {"phase", 0}};
            // EXP39 & 1: the front publishes what it holds at its next commit (a few words) from
            // now on. Published positions only grow, so the front never matches a stale one.
            back_idle_at.store(read_position, std::memory_order_relaxed);
            // EDEN_NX2_SPLIT_STATS: this wait's length, deepest phase and the words it woke to.
            const bool split_stats = Nx2SplitStats::Enabled();
            const auto wait_start = split_stats ? std::chrono::steady_clock::now()
                                                : std::chrono::steady_clock::time_point{};
            u32 wait_phase = 0;
            SCOPE_EXIT {
                if (split_stats) [[unlikely]] {
                    const u64 ns = static_cast<u64>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - wait_start)
                            .count());
                    Nx2SplitStats::NoteWait(ns, wait_phase,
                                            published.load(std::memory_order_relaxed) -
                                                read_position);
                }
            };
            // The front usually sends more within microseconds: spin briefly before sleeping.
            for (u32 spins = 0; spins < 8192 && available == read_position &&
                                !hook_requested.load(std::memory_order_relaxed);
                 ++spins) {
                Pause();
                available = published.load(std::memory_order_acquire);
            }
            if (available == read_position && idle_hook != nullptr) {
                hook_requested.store(false, std::memory_order_seq_cst);
                ForgetFencesQuiet();
                timeline_span.Set(0, {"phase", 1});
                wait_phase = 1;
                in_idle_wait = true;
                idle_hook(idle_argument);
                in_idle_wait = false;
                available = published.load(std::memory_order_acquire);
                if (available == read_position) {
                    CheckFencesQuiet(read_position);
                }
            }
            if (available == read_position) {
                // Publish bumps `wake` when it sees back_waiting: read it before the last look
                // at `published`, so a publish in between ends the wait at once.
                back_waiting.store(true, std::memory_order_seq_cst);
                const u32 seen = wake.load(std::memory_order_seq_cst);
                available = published.load(std::memory_order_seq_cst);
                // A request made after the hook ran (RequestIdleHook) runs it before sleeping.
                if (available == read_position && !stop_token.stop_requested() &&
                    !hook_requested.load(std::memory_order_seq_cst)) {
                    timeline_span.Set(0, {"phase", 2});
                    wait_phase = 2;
                    wake.wait(seen, std::memory_order_seq_cst);
                }
                back_waiting.store(false, std::memory_order_relaxed);
                continue;
            }
        }
        u64 last_store = read_position;
        // EDEN_NX2_EXP27 & 4 (the original's EDEN_NX2_RING_PREFETCH): prefetch the ring
        // lines ahead of the entry, up to a line short of what the front published (a line it
        // may still be writing would only bounce between the two cores).
        const bool ring_prefetch = (VideoCore::Nx2Debug::Experiment(27) & 4) != 0;
        u64 prefetched = read_position;
        while (read_position != available) {
            const size_t index = read_position & RingMask;
            if (ring_prefetch) {
                const u64 ahead = (std::min)(read_position + RingPrefetchWords,
                                             available > 8 ? available - 8 : u64{0});
                prefetched = (std::max)(prefetched, read_position);
                while (prefetched + 8 <= ahead) {
                    prefetched += 8;
                    PrefetchRingLine(&ring[prefetched & RingMask]);
                }
            }
            back_entry_position = read_position;
            // EDEN_NX2_RUNAHEAD: chunks of CPU writes due before this entry (nx2_runahead.h).
            if (read_position >= Runahead::g_apply_at.load(std::memory_order_relaxed))
                [[unlikely]] {
                Runahead::ApplyDue(read_position);
            }
            const size_t size =
                Execute(&ring[index], std::min<u64>(available - read_position, RingWords - index));
            read_position += size != 0 ? size : RingWords - index;
            if (hook_requested.load(std::memory_order_relaxed) &&
                idle_hook != nullptr) [[unlikely]] {
                hook_requested.store(false, std::memory_order_seq_cst);
                ForgetFencesQuiet();
                idle_hook(idle_argument);
            }
            // Hand ring space back now and then while working through a long batch, and at once
            // when the front waits for this position (WaitForWriters).
            if ((read_position - last_store >= 4096 ||
                 read_position >= wait_target.load(std::memory_order_relaxed)) &&
                read_position != available) {
                consumed.store(read_position, std::memory_order_seq_cst);
                last_store = read_position;
                NotifyFront();
            }
        }
        CheckFencesQuiet(read_position);
        consumed.store(read_position, std::memory_order_seq_cst);
        NotifyFront();
    }
    back_running.store(false, std::memory_order_seq_cst);
    front_wake.fetch_add(1, std::memory_order_seq_cst);
    front_wake.notify_all();
}

} // namespace VideoCommon
