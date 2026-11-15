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

#include "common/assert.h"
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
#include "video_core/fritchii_e_split.h"
#include "video_core/fritchii_price.h"
#include "video_core/nx2_fastdraw_count.h"
#include "video_core/gpu_split.h"
#include "video_core/memstage_shadow.h"
#include "video_core/memory_stage_link.h"
#include "video_core/nx2_split_stats.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/nx2_runahead.h"
#include "video_core/one_follower.h"

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
    /// The delay tests' totals (g_delay_count, g_delay_ns) at the last line, by DelayTest.
    std::array<u64, static_cast<size_t>(GpuSplit::DelayTest::Count)> delay_count{};
    std::array<u64, static_cast<size_t>(GpuSplit::DelayTest::Count)> delay_ns{};
    /// FRITCHII_ONEF_WAITS_E: waits that ended on E's position with R not there yet.
    u64 engine_early{};
    u64 since{};
};
SiteWaitStats g_site_waits;

/// One follower's hang watchdog: the front starts a wait (GpuSplit::FrontWaitNow).
void NoteFrontWait(GpuSplit::FrontWait& wait, DrainSite site, u64 target, bool on_engine) {
    if (!OneFollower::On()) {
        return;
    }
    wait.site.store(static_cast<u32>(site), std::memory_order_relaxed);
    wait.target.store(target, std::memory_order_relaxed);
    wait.on_engine.store(on_engine, std::memory_order_relaxed);
    wait.seq.fetch_add(1, std::memory_order_relaxed);
    wait.begin_ns.store(Common::Nx2Hitch::NowNs(), std::memory_order_release);
}

// The delay tests' spins (GpuSplit::SpinDelay), by GpuSplit::DelayTest, since the start: the
// front's and the back's (Dispatch, Draw), so atomic. Only a spin adds to them.
std::array<std::atomic<u64>, static_cast<size_t>(GpuSplit::DelayTest::Count)> g_delay_count{};
std::array<std::atomic<u64>, static_cast<size_t>(GpuSplit::DelayTest::Count)> g_delay_ns{};

// The switch of a delay test (its microseconds; 0 or less: off).
int DelaySwitch(GpuSplit::DelayTest test) {
    namespace Debug = VideoCore::Nx2Debug;
    using Test = GpuSplit::DelayTest;
    switch (test) {
    case Test::MacroDrain:
        return Debug::SwitchValue(Debug::g_delay_macro_drain_us, "FRITCHII_DELAY_MACRO_DRAIN_US");
    case Test::PollDrain:
        return Debug::SwitchValue(Debug::g_delay_poll_drain_us, "FRITCHII_DELAY_POLL_DRAIN_US");
    case Test::E2c:
        return Debug::SwitchValue(Debug::g_delay_e2c_us, "FRITCHII_DELAY_E2C_US");
    case Test::Dispatch:
        return Debug::SwitchValue(Debug::g_delay_dispatch_us, "FRITCHII_DELAY_DISPATCH_US");
    case Test::Draw:
        return Debug::SwitchValue(Debug::g_delay_draw_us, "FRITCHII_DELAY_DRAW_US");
    case Test::Back:
        return Debug::SwitchValue(Debug::g_delay_back_us, "FRITCHII_DELAY_BACK_US");
    case Test::Front:
        return Debug::SwitchValue(Debug::g_delay_front_us, "FRITCHII_DELAY_FRONT_US");
    case Test::Worker:
        return Debug::SwitchValue(Debug::g_delay_worker_us, "FRITCHII_DELAY_WORKER_US");
    default:
        return 0;
    }
}

// The delay tests' part of the line, the spins since the last line: only while a switch is set
// or spins were counted.
std::string DelayTestText(SiteWaitStats& stats, double window) {
    namespace Debug = VideoCore::Nx2Debug;
    using Test = GpuSplit::DelayTest;
    constexpr size_t tests = static_cast<size_t>(Test::Count);
    std::array<u64, tests> count{};
    std::array<u64, tests> ns{};
    bool any = false;
    for (size_t i = 0; i < tests; ++i) {
        const u64 total_count = g_delay_count[i].load(std::memory_order_relaxed);
        const u64 total_ns = g_delay_ns[i].load(std::memory_order_relaxed);
        count[i] = total_count - stats.delay_count[i];
        ns[i] = total_ns - stats.delay_ns[i];
        stats.delay_count[i] = total_count;
        stats.delay_ns[i] = total_ns;
        any |= count[i] != 0 || DelaySwitch(static_cast<Test>(i)) > 0;
    }
    if (!any) {
        return {};
    }
    const auto test = [&](Test which) {
        const auto i = static_cast<size_t>(which);
        return fmt::format("={} {:.0f}/s {:.1f} ms/s", (std::max)(DelaySwitch(which), 0),
                           static_cast<double>(count[i]) / window,
                           static_cast<double>(ns[i]) / 1e6 / window);
    };
    const bool before = Debug::SwitchValue(Debug::g_delay_before, "FRITCHII_DELAY_BEFORE") > 0;
    return fmt::format("; delay tests added ({} the drain, FRITCHII_DELAY_BEFORE={}): "
                       "FRITCHII_DELAY_MACRO_DRAIN_US{}, FRITCHII_DELAY_POLL_DRAIN_US{}, "
                       "FRITCHII_DELAY_E2C_US{} (front), FRITCHII_DELAY_DISPATCH_US{} (back), "
                       "FRITCHII_DELAY_DRAW_US{} (back), FRITCHII_DELAY_BACK_US{} KIND={} (back), "
                       "FRITCHII_DELAY_FRONT_US{} KIND={} (front), FRITCHII_DELAY_WORKER_US{} "
                       "(Vulkan worker, per chunk)",
                       before ? "before" : "after", before ? 1 : 0, test(Test::MacroDrain),
                       test(Test::PollDrain), test(Test::E2c), test(Test::Dispatch),
                       test(Test::Draw), test(Test::Back),
                       Debug::SwitchValue(Debug::g_delay_back_kind, "FRITCHII_DELAY_BACK_KIND"),
                       test(Test::Front),
                       Debug::SwitchValue(Debug::g_delay_front_kind, "FRITCHII_DELAY_FRONT_KIND"),
                       test(Test::Worker));
}

void NoteSiteWait(DrainSite site, u64 behind, u64 ns) {
    FrontSplit::NoteWait(ns); // FRITCHII_FRONT_SPLIT_COUNT: out of the timed entry's stage
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
    // FRITCHII_ONEF_WAITS_E: the sites that read E's position are marked "(E)".
    const bool waits_e = OneFollower::WaitsE();
    for (const size_t s : order) {
        const bool on_e = waits_e && GpuSplit::SiteOnEngine(static_cast<DrainSite>(s));
        sites += fmt::format("{}{}{} {:.0f}/s {:.1f} ms/s behind {:.0f}", sites.empty() ? "" : "; ",
                             drain_site_names[s], on_e ? " (E)" : "",
                             static_cast<double>(stats.count[s]) / window,
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
             "front {:.1f} ms/s, GPU memory {:.1f} ms/s ({:.0f}/s), other threads {:.1f} ms/s{}",
             sites, lock_ms[1], lock_rate[1], lock_ms[3], lock_rate[3], lock_ms[2], lock_ms[4],
             lock_rate[4], lock_ms[0], DelayTestText(stats, window));
    if (waits_e) {
        LOG_INFO(HW_GPU,
                 "One follower: front waits on E's position {:.0f}/s ended before R got there "
                 "(FRITCHII_ONEF_WAITS_E=1)",
                 static_cast<double>(stats.engine_early) / window);
    }
    stats.engine_early = 0;
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
    // One follower: the scheduler is R's. R passes; E passes only with R parked or drained
    // (OneFollower's "owner on E", must be 0); any other thread is reported below as before.
    if (OneFollower::OnRenderThread()) {
        return;
    }
    if (OneFollower::OnEngineThread()) {
        OneFollower::CheckRenderOwned(where);
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

u32 GpuSplit::DelayBeforeWait(DelayTest test) {
    namespace Debug = VideoCore::Nx2Debug;
    const int us = DelaySwitch(test);
    if (us <= 0) [[likely]] {
        return 0;
    }
    // Only drains that wait (the site line counts those): the back has not run everything sent.
    if (PlayedNow(std::memory_order_acquire) == write_position) {
        return 0;
    }
    if (Debug::SwitchValue(Debug::g_delay_before, "FRITCHII_DELAY_BEFORE") > 0) {
        SpinDelay(test, static_cast<u32>(us));
        return 0;
    }
    return static_cast<u32>(us);
}

void GpuSplit::DelayWork(DelayTest test) {
    const int us = DelaySwitch(test);
    if (us <= 0) [[likely]] {
        return;
    }
    // D and K are the back's: not GPUMem (the memory stage), not the single GPU thread.
    if (test != DelayTest::E2c && !on_back_thread) {
        return;
    }
    SpinDelay(test, static_cast<u32>(us));
}

void GpuSplit::DelayBack(BackDelayKind kind, u32 amount) {
    namespace Debug = VideoCore::Nx2Debug;
    const int us = DelaySwitch(DelayTest::Back);
    if (us <= 0 || !on_back_thread ||
        Debug::SwitchValue(Debug::g_delay_back_kind, "FRITCHII_DELAY_BACK_KIND") !=
            static_cast<int>(kind)) {
        return;
    }
    // Every 64 method-zone entries, every 4th stage bind, every made fence.
    static constexpr std::array<u32, 4> period{1, 64, 4, 1};
    static thread_local std::array<u32, 4> items{};
    const auto index = static_cast<size_t>(kind);
    items[index] += amount;
    while (items[index] >= period[index]) {
        items[index] -= period[index];
        SpinDelay(DelayTest::Back, static_cast<u32>(us));
    }
}

int GpuSplit::DelayTestUs(DelayTest test) {
    return DelaySwitch(test);
}

void GpuSplit::DelayFront(FrontDelayKind kind, bool before) {
    namespace Debug = VideoCore::Nx2Debug;
    const int us = DelaySwitch(DelayTest::Front);
    if (us <= 0 || on_back_thread ||
        Debug::SwitchValue(Debug::g_delay_front_kind, "FRITCHII_DELAY_FRONT_KIND") !=
            static_cast<int>(kind)) {
        return;
    }
    if (kind == FrontDelayKind::Macros || kind == FrontDelayKind::Entries) {
        // Kind 3: 1 call in 64; kind 4 (FS0, the front split's per-entry site): 1 in 16.
        static thread_local u32 calls{};
        if ((++calls & (kind == FrontDelayKind::Macros ? 63U : 15U)) != 0) {
            return;
        }
    } else if (before != (Debug::SwitchValue(Debug::g_delay_before, "FRITCHII_DELAY_BEFORE") > 0)) {
        return;
    }
    SpinDelay(DelayTest::Front, static_cast<u32>(us));
}

void GpuSplit::SpinDelay(DelayTest test, u32 us) {
    const auto start = std::chrono::steady_clock::now();
    const auto until = start + std::chrono::microseconds{us};
    auto now = start;
    while (now < until) {
        Pause();
        now = std::chrono::steady_clock::now();
    }
    const auto i = static_cast<size_t>(test);
    g_delay_count[i].fetch_add(1, std::memory_order_relaxed);
    g_delay_ns[i].fetch_add(
        static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count()),
        std::memory_order_relaxed);
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
    // FRITCHII_MEMSTAGE_SHADOW (memstage_shadow.cpp): before the back thread exists.
    MemStageShadow::ReadSwitch();
    // FRITCHII_ONEF (boot): the render thread R starts before the back, so E's first lend has it.
    OneFollower::ReadSwitches();
    // Its owner asserts ride on the scheduler's owner check: always on with it.
    check_owner = check_owner || OneFollower::On();
    // 8.2: the front's waits read R's position (`played`), E keeps `consumed`.
    two_positions = OneFollower::On();
    ring = std::make_unique<u64[]>(RingWords);
    ++session;
    // EDEN_NX2_RUNAHEAD: Ocarina of Time only.
    Runahead::Reset();
    Runahead::SetProgramId(system_.GetApplicationProcessProgramID());
    MemoryStageLink::Get().Start(system_);
    OneFollower::Start(system_);
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
    // After the back: every lent packet it pushed has been played.
    OneFollower::Stop();
    MemoryStageLink::Get().Stop();
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
    noted_summary.live = false; // FRITCHII_NOTED_SUMMARY: positions start again at 0
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
    // Positions start over: stamps of the old session would refuse every read.
    for (auto& stamp : dispatch_stamps) {
        stamp.store(0, std::memory_order_relaxed);
    }
    dispatch_wide_stamp.store(0, std::memory_order_relaxed);
    back_entry_position = 0;
    async_writes.clear();
    last_sync_position = 0;
    last_memory_write_position = 0;
    previous_memory_write_position = 0;
    last_method_write_position = 0;
    previous_method_write_position = 0;
    ranged_overflow_position = 0;
    ranged_device_stamps = false;
    ranged_device_since = 0;
    stamp_memory = nullptr;
    stamp_page = ~GPUVAddr{0};
    stamp_read_pages.clear();
    stamp_read_cache_page = ~u64{0};
    stamp_skip_floor = 0;
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
    cb_header = nullptr;
    cb_count = 0;
    cb_verify = false;
    cb_end = 0;
    pending_flush = nullptr;
    pending_flush_argument = nullptr;
    fold_count = 0;
    fold_flush = nullptr;
    fold_context = nullptr;
    back_engine = nullptr;
    front_macros = false;
    published.store(0, std::memory_order_relaxed);
    consumed.store(0, std::memory_order_relaxed);
    played.store(0, std::memory_order_relaxed);
    staged.store(0, std::memory_order_relaxed);
    lent_from.store(0, std::memory_order_relaxed);
    lent_until.store(0, std::memory_order_relaxed);
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
        consumed_cache = PlayedNow(std::memory_order_acquire);
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
    cb_header = nullptr;
    if (published_local == write_position) {
        return;
    }
    // fable_stage_levelpoints 15.7: the front's "push" doubles with the stage on. The publishes
    // that publish (their count: the zone's calls) and their wakes (PublishWake, in
    // MemoryStageLink::NotePublished too) are zones of their own (profile 2).
    const Nx2Profile::Scope zone{Nx2Profile::Zone::Publish, Nx2Profile::Fine(), false};
    if (Nx2SplitStats::Enabled()) [[unlikely]] {
        Nx2SplitStats::NotePublish(write_position - published_local);
    }
    published_local = write_position;
    publish_every = std::min<size_t>(publish_every * 2, PublishEvery);
    publish_waiting = (VideoCore::Nx2Debug::Experiment(39) & 1) != 0;
    published.store(write_position, std::memory_order_seq_cst);
    if (back_waiting.load(std::memory_order_seq_cst)) {
        const Nx2Profile::Scope wake_zone{Nx2Profile::Zone::PublishWake, Nx2Profile::Fine(),
                                          false};
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }
    // FRITCHII_MEMSTAGE_THREAD=2: the stage is the next reader (spec 1.2).
    MemoryStageLink::Get().NotePublished();
    if ((++publishes & 1023) == 0) [[unlikely]] {
        ReportBatched();
        ReportQmdFloor();
        ReportCbWords();
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

bool GpuSplit::SiteOnEngine(DrainSite site) noexcept {
    switch (site) {
    case DrainSite::MacroDmaRead:
    case DrainSite::MacroDmaWrites:
    case DrainSite::QmdRead:
    case DrainSite::SemaphoreFlush:
    case DrainSite::ListWriters:
    case DrainSite::ListSync:
    case DrainSite::ListCopies:
    case DrainSite::ListFlush:
    case DrainSite::DispatchWait:
    case DrainSite::LaggedCheckEngine:
        return true;
    default:
        return false;
    }
}

bool GpuSplit::UsesEngine(DrainSite site) const {
    // LaggedCheckEngine is chosen by FRITCHII_ONEF_SEM_E itself, not WAITS_E.
    return two_positions && SiteOnEngine(site) &&
           (site == DrainSite::LaggedCheckEngine || OneFollower::WaitsE());
}

bool GpuSplit::Reached(u64 target, bool on_engine, std::memory_order order) const {
    if (!on_engine) {
        return PlayedNow(order) >= target;
    }
    // Game memory, the buffer cache and the lists are E's: everything R does that the front
    // could read there is lent (E parked until R ran it), except the plain reports
    // (FRITCHII_ONEF_QUERY_OPS), which land on R. So: E ran `target`, and R played the newest
    // report op from before it. The mark is read after `consumed`: a report in an entry E ran
    // is in it. A newer mark (at `target` or later) may hide an older one: R's position then.
    if (consumed.load(order) < target) {
        return false;
    }
    const u64 mark = OneFollower::GuestWriteOpMark();
    if (mark == 0) {
        return true;
    }
    if (mark - 1 < target) {
        return played.load(order) > mark - 1;
    }
    return played.load(order) >= target;
}

void GpuSplit::Drain(DrainSite site) {
    RunPendingFlush(); // Everything sent so far includes the held-back work
    const bool on_engine = UsesEngine(site);
    consumed_cache = PlayedNow(std::memory_order_acquire);
    if (consumed_cache == write_position || Reached(write_position, on_engine,
                                                    std::memory_order_acquire)) {
        return;
    }
    // Profile on: GPUMem's wait for this publish counts as the front's start of a wait (15.7).
    Nx2Profile::FrontWaitBegin(site);
    Publish();
    NoteFrontWait(front_wait, site, write_position, on_engine);
    SCOPE_EXIT {
        front_wait.begin_ns.store(0, std::memory_order_relaxed);
    };
    Nx2Profile::Scope scope{Nx2Profile::Zone::SplitDrain};
    // "behind": the ring words the back had left to run, which tells a slow back from a front
    // far ahead (tools/nx2/critical_path.py).
    Common::Nx2Timeline::ArgSpan timeline_span{drain_site_names[static_cast<size_t>(site)],
                                               {"behind", write_position - consumed_cache}};
    const u64 start = Nx2Profile::Enabled() ? Nx2Profile::Ticks() : 0;
    const u64 begin_ns = Common::Nx2Hitch::NowNs();
    const u64 behind = write_position - consumed_cache;
    for (u32 spins = 0;; ++spins) {
        consumed_cache = PlayedNow(std::memory_order_acquire);
        if (consumed_cache == write_position ||
            Reached(write_position, on_engine, std::memory_order_acquire)) {
            if (on_engine && consumed_cache != write_position) {
                ++g_site_waits.engine_early;
            }
            publish_every = 8;
            if (start != 0) [[unlikely]] {
                // fable_stage_levelpoints 18.4 rows 7 and 8: until the back reached it, after.
                FritchiiPrice::NoteFrontWait(static_cast<size_t>(site), write_position, start);
            }
            Nx2Profile::AddDrain(site, start);
            NoteSiteWait(site, behind, Common::Nx2Hitch::NowNs() - begin_ns);
            return;
        }
        Backoff(spins, write_position, on_engine);
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
    consumed_cache = PlayedNow(std::memory_order_acquire);
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
    consumed_cache = PlayedNow(std::memory_order_acquire);
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

void GpuSplit::RangeLastWrite(GPUVAddr begin, GPUVAddr end, const u32* data, size_t space,
                              const Tegra::MemoryManager* memory) {
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
        // FRITCHII_STAMP_READ_PAGES: the stamps only on pages a reader asked about (2), or
        // counted (1). Needs the device stamps (EXP126) and a translation; else as before.
        const int read_pages = StampReadPagesMode();
        if (read_pages == 0 || memory == nullptr || !RangedDeviceStamps() ||
            !StampRangedRead(*memory, begin, end, read_pages == 2)) {
            StampWrite(begin, end, true);
            if (RangedDeviceStamps()) {
                if (memory != nullptr) {
                    StampRangedDevice(*memory, begin, end);
                } else {
                    write_stamp_floor = write_position; // No translation: it may write anything
                    ++floor_stats.device_floors;
                }
            }
        }
        if (read_pages != 0) [[unlikely]] {
            MaybeLogStampRead(read_pages);
        }
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
        // Too many in flight: the oldest counts for every read again (the ring scans no longer
        // see its range; its bytes stay stamped).
        const RangedWrite& oldest = ranged_writes.front();
        if (oldest.position > last_memory_write_position) {
            last_memory_write_position = oldest.position;
            last_memory_write_method = oldest.method;
        }
        ranged_overflow_position = (std::max)(ranged_overflow_position, oldest.position);
        ++floor_stats.overflows;
        ++ranged_overflow_count;
        ranged_writes.pop_front();
    }
    RangedWrite& write = ranged_writes.push_back(write_position, begin, end, method);
    if (data != nullptr && space != 0 && (end - begin) % sizeof(u32) == 0) {
        write.space = space;
        write.data.assign(data, data + words);
    }
}

bool GpuSplit::RangedDeviceStamps() {
    const bool on = OverflowFloorMode() != 0;
    if (on != ranged_device_stamps) {
        ranged_device_stamps = on;
        // Turned on: the ranged writes sent so far were stamped by GPU address only.
        ranged_device_since = on ? write_position : 0;
    }
    return on;
}

void GpuSplit::StampRangedDevice(const Tegra::MemoryManager& memory, GPUVAddr begin,
                                 GPUVAddr end) {
    // As TryNoteWriterRange: while a queued function may still remap the range the back may
    // translate it differently, and sparse pages are written through their own backing. Then
    // the write counts for every address until the back ran it.
    if (last_function_position > consumed_cache) {
        consumed_cache = PlayedNow(std::memory_order_acquire);
    }
    const bool sparse = begin < Tegra::g_nx2_sparse_high.load(std::memory_order_relaxed) &&
                        Tegra::g_nx2_sparse_low.load(std::memory_order_relaxed) < end;
    if (last_function_position > consumed_cache || sparse) {
        write_stamp_floor = write_position;
        ++floor_stats.device_floors;
        return;
    }
    constexpr GPUVAddr page_size = 0x1000;
    for (GPUVAddr page = begin & ~(page_size - 1); page < end; page += page_size) {
        if (&memory != stamp_memory || page != stamp_page ||
            stamp_function != last_function_position) {
            const std::optional<DAddr> device = memory.GpuToCpuAddress(page);
            stamp_memory = &memory;
            stamp_page = page;
            stamp_function = last_function_position;
            stamp_mapped = device.has_value();
            stamp_device = device.value_or(0);
        }
        if (!stamp_mapped) {
            continue; // Unmapped: the back writes nothing there
        }
        const GPUVAddr from = (std::max)(begin, page);
        const GPUVAddr to = (std::min)(end, page + page_size);
        StampWrite(stamp_device + (from - page), stamp_device + (to - page), false);
    }
    ++floor_stats.device_stamped;
}

bool GpuSplit::ForwardRead(size_t space, GPUVAddr begin, u32 words, u32* out) {
    const GPUVAddr end = begin + GPUVAddr{words} * sizeof(u32);
    const u64 done = PlayedNow(std::memory_order_acquire);
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
                                                     u64 size, std::span<const u64> skip,
                                                     bool overflow_stamped) const {
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
    // EXP127 (the header): the dropped writes are at or before the overflow position, and those
    // not stamped by device address before ranged_device_since.
    unbounded(overflow_stamped ? (std::min)(ranged_overflow_position, FloorlessOverflowTerm())
                               : ranged_overflow_position,
              Cause::RangedOverflow);
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
    // EXP126 = 1 (fable_qmd_wait_brief piece 2): without the ranged overflow. RangeLastWrite
    // stamps a ranged write's bytes before it touches the ring, so a dropped one is still found
    // over its own bytes below; only its count for every other address goes. Its device bytes
    // too (RangeLastWrite stamps them with EXP126 on), so a read through another GPU address
    // of the same memory finds it; the writes sent before EXP126 turned on count until they ran.
    const u64 overflow =
        OverflowFloorOff() ? FloorlessOverflowTerm() : ranged_overflow_position;
    u64 stamp = (std::max)({write_stamp_floor, unknown_write_position,
                            last_unbounded_writer_position, last_unranged_position,
                            last_function_position, last_method_write_position, overflow});
    if (size != 0) {
        stamp = (std::max)(stamp, StampOf(device, device + size, false, skip_tag));
        stamp = (std::max)(stamp, StampOf(address, address + size, true, skip_tag));
        stamp = (std::max)(stamp, StampReadFloor(device, size)); // FRITCHII_STAMP_READ_PAGES
    }
    return (std::min)(stamp, write_position);
}

int GpuSplit::StampReadPagesMode() {
    return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_stamp_read_pages,
                                            "FRITCHII_STAMP_READ_PAGES");
}

bool GpuSplit::StampPageRead(u64 page) {
    if (page != stamp_read_cache_page) {
        stamp_read_cache_page = page;
        stamp_read_cache_hit = stamp_read_pages.contains(page);
    }
    return stamp_read_cache_hit;
}

bool GpuSplit::StampRangedRead(const Tegra::MemoryManager& memory, GPUVAddr begin, GPUVAddr end,
                               bool skip) {
    auto& s = stamp_read_stats;
    ++s.writes;
    // Over a page, block stamps per page would not be StampWrite's page stamps: as before.
    if (end - begin > 0x1000) {
        ++s.whole;
        return false;
    }
    // StampRangedDevice's translation rule: a queued function may remap the range, sparse pages
    // write through their own backing. Then as before (it floors).
    if (last_function_position > consumed_cache) {
        consumed_cache = PlayedNow(std::memory_order_acquire);
    }
    const bool sparse = begin < Tegra::g_nx2_sparse_high.load(std::memory_order_relaxed) &&
                        Tegra::g_nx2_sparse_low.load(std::memory_order_relaxed) < end;
    if (last_function_position > consumed_cache || sparse) {
        ++s.whole;
        return false;
    }
    // Per page: the GPU stamp and the device stamp of the bytes in it (StampWrite of the whole
    // range and StampRangedDevice, split at the page: the same block slots, as the range is under
    // 16 KiB), or neither where no reader ever asked about the device page.
    constexpr GPUVAddr page_size = 0x1000;
    for (GPUVAddr page = begin & ~(page_size - 1); page < end; page += page_size) {
        if (&memory != stamp_memory || page != stamp_page ||
            stamp_function != last_function_position) {
            const std::optional<DAddr> device = memory.GpuToCpuAddress(page);
            stamp_memory = &memory;
            stamp_page = page;
            stamp_function = last_function_position;
            stamp_mapped = device.has_value();
            stamp_device = device.value_or(0);
        }
        const GPUVAddr from = (std::max)(begin, page);
        const GPUVAddr to = (std::min)(end, page + page_size);
        if (!stamp_mapped) {
            StampWrite(from, to, true); // As before: the GPU stamp only (the back writes nothing)
            continue;
        }
        if (StampPageRead(stamp_device >> 12)) {
            ++s.pages_read;
        } else {
            ++s.pages_skipped;
            if (skip) {
                stamp_skip_floor = write_position;
                continue;
            }
        }
        StampWrite(from, to, true);
        StampWrite(stamp_device + (from - page), stamp_device + (to - page), false);
    }
    ++floor_stats.device_stamped;
    return true;
}

u64 GpuSplit::StampReadFloor(u64 device, u64 size) const {
    if (size == 0 || (stamp_skip_floor == 0 && StampReadPagesMode() == 0)) {
        return 0;
    }
    if (stamp_read_pages.size() >= StampReadPagesMax) {
        // A page asked about again takes the newest skipped write as its floor: still exact.
        stamp_read_pages.clear();
        ++stamp_read_stats.clears;
    }
    u64 floor = 0;
    for (u64 page = device >> 12; page <= (device + size - 1) >> 12; ++page) {
        const auto [it, inserted] = stamp_read_pages.try_emplace(page, stamp_skip_floor);
        if (inserted) {
            ++stamp_read_stats.added;
            stamp_read_cache_page = ~u64{0};
        }
        floor = (std::max)(floor, it->second);
    }
    stamp_read_stats.floors += floor != 0 ? 1 : 0;
    return floor;
}

void GpuSplit::MaybeLogStampRead(int mode) {
    auto& s = stamp_read_stats;
    if ((++s.ticks & 0xffff) != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (s.since == std::chrono::steady_clock::time_point{}) {
        s.since = now;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - s.since).count();
    if (seconds < 10.0) {
        return;
    }
    const auto rate = [seconds](u64 count) { return static_cast<double>(count) / seconds; };
    LOG_INFO(HW_GPU,
             "Stamp read pages (FRITCHII_STAMP_READ_PAGES={}) {:.0f} s: ranged writes {:.0f}/s, "
             "their pages on read pages {:.0f}/s (stamped), on never-read pages {:.0f}/s ({}), "
             "stamped whole as before {:.0f}/s; pages asked about {} (new {:.0f}/s), reader "
             "answers raised by an add floor {:.0f}/s, table clears {}",
             mode, seconds, rate(s.writes), rate(s.pages_read), rate(s.pages_skipped),
             mode == 2 ? "skipped" : "would skip", rate(s.whole), stamp_read_pages.size(),
             rate(s.added), rate(s.floors), s.clears);
    s = {};
    s.since = now;
}

int GpuSplit::OverflowFloorMode() {
    return VideoCore::Nx2Debug::Experiment(126);
}

GpuSplit::StampWhy GpuSplit::WriteStampWhy(u64 device, u64 size, GPUVAddr address,
                                           u64 skip_tag) const {
    if (!write_stamps) {
        return {write_position, write_position, write_position, StampTerm::Floor};
    }
    // WriteStamp's terms one by one, the tables split into blocks and pages.
    u64 block = 0;
    u64 page = 0;
    if (size != 0) {
        const auto read = [skip_tag](const StampSlot& slot) {
            return skip_tag != 0 && slot.last_tag == skip_tag ? slot.before : slot.last;
        };
        for (const bool gpu : {false, true}) {
            const u64 begin = gpu ? address : device;
            const u64 end = begin + size;
            const u64 salt = gpu ? GpuStampSalt : 0;
            for (u64 key = begin >> 8; key <= (end - 1) >> 8; ++key) {
                block = (std::max)(block, read(block_stamps[StampIndex(key ^ salt, BlockStamps)]));
            }
            for (u64 key = begin >> 12; key <= (end - 1) >> 12; ++key) {
                page = (std::max)(page, read(page_stamps[StampIndex(key ^ salt, PageStamps)]));
            }
        }
        // FRITCHII_STAMP_READ_PAGES: the add floors stand for skipped stamps (a page term).
        page = (std::max)(page, StampReadFloor(device, size));
    }
    const std::array<std::pair<u64, StampTerm>, 9> terms{{
        {block, StampTerm::Block},
        {page, StampTerm::Page},
        {write_stamp_floor, StampTerm::Floor},
        {unknown_write_position, StampTerm::Unknown},
        {last_unbounded_writer_position, StampTerm::Unbounded},
        {last_unranged_position, StampTerm::Unranged},
        {last_function_position, StampTerm::Function},
        {last_method_write_position, StampTerm::MethodWrite},
        // As WriteStamp: with EXP126 = 1 the writes not stamped by device address stand for it.
        {OverflowFloorOff() ? FloorlessOverflowTerm() : ranged_overflow_position,
         StampTerm::RangedOverflow},
    }};
    // `floorless` (what 1 answers): with device stamps on (EXP126 not 0) the writes sent before
    // they started; with 0 (counters only) nothing for the overflow.
    StampWhy why{0, ranged_device_stamps ? ranged_device_since : 0, 0, StampTerm::None};
    for (const auto& [position, term] : terms) {
        if (term != StampTerm::RangedOverflow) {
            why.floorless = (std::max)(why.floorless, position);
        }
        if (position > why.term_position) {
            why.term_position = position;
            why.term = term;
        }
    }
    why.stamp = (std::min)(why.term_position, write_position);
    why.floorless = (std::min)(why.floorless, write_position);
    return why;
}

u8 GpuSplit::ExactWriterCheck(u64 done, GPUVAddr address, u64 device, u64 size,
                              u64 term_position, std::span<const u64> skip) const {
    const GuestWriters writers = QueuedGuestWriters(done, address, device, size, skip);
    if (writers.noted > done && writers.noted >= (std::min)(term_position, write_position)) {
        return 0;
    }
    return (std::max)(writers.noted, writers.unbounded) > done ? 1 : 2;
}

void GpuSplit::NoteQmdPatchRefused(const StampWhy& why, u64 done, u8 exact) {
    ++floor_stats.patches[static_cast<size_t>(QmdPatchOutcome::Stamp)];
    ++floor_stats.patch_refused[static_cast<size_t>(why.term)];
    if (why.floorless <= done) {
        ++floor_stats.patch_overflow_alone;
    }
    if (exact < 3) {
        ++floor_stats.exact[0][exact];
    }
}

void GpuSplit::NoteQmdRead(u64 done) {
    ++floor_stats.qmd_reads;
    if (ranged_overflow_position > done) {
        const u64 words = write_position - ranged_overflow_position;
        ++floor_stats.reads_floor_up;
        floor_stats.floor_words += words;
        floor_stats.floor_words_max = (std::max)(floor_stats.floor_words_max, words);
    }
}

void GpuSplit::NoteQmdReadWait(const StampWhy& why, u64 done, u64 target, bool evicted,
                               u8 exact) {
    const size_t term = evicted ? static_cast<size_t>(StampTerm::Count)
                                : static_cast<size_t>(why.term);
    ++floor_stats.read_waits[term];
    floor_stats.read_wait_words[term] += target - done;
    if (!evicted && why.floorless <= done) {
        ++floor_stats.read_overflow_alone;
    }
    if (exact < 3) {
        ++floor_stats.exact[1][exact];
    }
}

void GpuSplit::ReportQmdFloor() {
    const auto now = std::chrono::steady_clock::now();
    if (floor_since == std::chrono::steady_clock::time_point{}) {
        floor_since = now;
        floor_stats = {};
        return;
    }
    const std::chrono::duration<double> elapsed = now - floor_since;
    if (elapsed.count() < 5.0) {
        return;
    }
    const FloorStats s = floor_stats;
    floor_stats = {};
    floor_since = now;
    const int mode = OverflowFloorMode();
    if (!Nx2Profile::Enabled() && mode == 0) {
        return;
    }
    const auto rate = [&](u64 count) { return static_cast<double>(count) / elapsed.count(); };
    // StampTerm's order, then the evicted inline QMD (read waits only).
    constexpr std::array<const char*, static_cast<size_t>(StampTerm::Count) + 1> term_names{
        "none",       "block stamp",     "page stamp",      "stamps on since/over 1 MiB",
        "unknown write", "unbounded writer", "unranged writer", "queued function",
        "3D method write", "ranged overflow", "evicted inline QMD"};
    std::string refused;
    std::string waits;
    for (size_t i = 0; i < term_names.size(); ++i) {
        if (i < s.patch_refused.size() && s.patch_refused[i] != 0) {
            refused += fmt::format("{}{} {:.0f}", refused.empty() ? "" : ", ", term_names[i],
                                   rate(s.patch_refused[i]));
        }
        if (s.read_waits[i] != 0) {
            waits += fmt::format("{}{} {:.0f} ({:.0f} words each)", waits.empty() ? "" : ", ",
                                 term_names[i], rate(s.read_waits[i]),
                                 static_cast<double>(s.read_wait_words[i]) /
                                     static_cast<double>(s.read_waits[i]));
        }
    }
    const auto patch = [&](QmdPatchOutcome outcome) {
        return rate(s.patches[static_cast<size_t>(outcome)]);
    };
    LOG_INFO(HW_GPU,
             "QMD wait floor (EXP126={}): ranged ring overflows {:.0f}/s (ring of {}); QMD reads "
             "{:.0f}/s, {:.0f}/s with the overflow queued, {:.0f} ring words from it to the write "
             "position on average, {} most; small patches/s: {:.0f} a holder took, {:.0f} overlay "
             "from memory made ({:.0f} with the overflow queued), refused {:.0f} unmapped or "
             "GPU-written, {:.0f} overlay queued, {:.0f} evicted inline QMD, {:.0f} by the "
             "stamp ({}), {:.0f} of them by the overflow alone; QMD read waits/s by term: {}; "
             "{:.0f} of them by the overflow alone; block/page stamps against the exact scan "
             "(writer found, older or unbounded only, nothing): patches {}/{}/{}, reads "
             "{}/{}/{}; macro DMA read waits {:.0f}/s, {:.0f}/s on the overflow ({:.0f} words "
             "each); ranged writes stamped by device address {:.0f}/s, {:.0f}/s counted for every "
             "address instead; trial (EXP126=2): reads where 1 serves otherwise compared, 1 "
             "serving guest memory {}, an overlay {}, an upload shadow block {}; {} differed; {} "
             "trial overlays and blocks built; {} reads not compared (1's copy not followed), {} "
             "trial entries lost",
             mode, rate(s.overflows), MaxRangedWrites(),
             rate(s.qmd_reads), rate(s.reads_floor_up),
             s.reads_floor_up != 0 ? static_cast<double>(s.floor_words) /
                                         static_cast<double>(s.reads_floor_up)
                                   : 0.0,
             s.floor_words_max, patch(QmdPatchOutcome::Holder), patch(QmdPatchOutcome::Made),
             patch(QmdPatchOutcome::MadeFloorUp), patch(QmdPatchOutcome::Unmapped),
             patch(QmdPatchOutcome::InlineQueued), patch(QmdPatchOutcome::Evicted),
             patch(QmdPatchOutcome::Stamp), refused.empty() ? "none" : refused,
             rate(s.patch_overflow_alone), waits.empty() ? "none" : waits,
             rate(s.read_overflow_alone), s.exact[0][0], s.exact[0][1], s.exact[0][2],
             s.exact[1][0], s.exact[1][1], s.exact[1][2], rate(s.dma_waits),
             rate(s.dma_overflow_waits),
             s.dma_overflow_waits != 0 ? static_cast<double>(s.dma_overflow_words) /
                                             static_cast<double>(s.dma_overflow_waits)
                                       : 0.0,
             rate(s.device_stamped), rate(s.device_floors), s.trial_compared[0],
             s.trial_compared[1], s.trial_compared[2], s.trial_differed, s.trial_built,
             s.trial_skipped, s.trial_lost);
    if (s.trial_differed != 0) {
        LOG_ERROR(HW_GPU,
                  "QMD wait floor (EXP126=2): {} QMD reads differed from what EXP126=1 would "
                  "have served in the last {:.0f} s",
                  s.trial_differed, elapsed.count());
    }
}

bool GpuSplit::HasQueuedMemoryWriters(GPUVAddr begin, GPUVAddr end) {
    const u64 done = PlayedNow(std::memory_order_acquire);
    u32 method{};
    return MemoryWriterTarget(done, begin, end, method) > done;
}

u32 GpuSplit::QueuedWriterKind(GPUVAddr begin, GPUVAddr end, u32& method) {
    const u64 done = PlayedNow(std::memory_order_acquire);
    const u64 target = MemoryWriterTarget(done, begin, end, method);
    if (target <= done) {
        return 0;
    }
    return target == last_writer_position         ? 1
           : target == last_sync_position         ? 2
           : target == last_memory_write_position ? 3
                                                  : 4;
}

void GpuSplit::NoteRelease(u64 device, GPUVAddr address, u32 payload, ReleaseKind kind) {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_ACQUIRE_QUEUED");
        return value == nullptr || value[0] != '0';
    }();
    if (!enabled) {
        return;
    }
    queued_releases[(device >> 2) % queued_releases.size()] = {write_position, device, payload,
                                                               kind};
}

std::optional<u32> GpuSplit::QueuedReleasePayload(u64 device, GPUVAddr address, u32* miss) {
    const QueuedRelease& release = queued_releases[(device >> 2) % queued_releases.size()];
    const u64 done = PlayedNow(std::memory_order_acquire);
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
    const u64 done = PlayedNow(std::memory_order_acquire);
    u32 target_method{};
    const u64 target = MemoryWriterTarget(done, begin, end, target_method);
    const u64 unranged =
        std::max({last_writer_position, last_sync_position, last_memory_write_position});
    if (target > done) {
        // The QMD wait floor counters: waits whose target is the ranged ring's overflow
        // (last_memory_write_position took the dropped write's position).
        ++floor_stats.dma_waits;
        if (target == ranged_overflow_position) {
            ++floor_stats.dma_overflow_waits;
            floor_stats.dma_overflow_words += target - done;
        }
    }
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
    const bool on_engine = UsesEngine(site);
    consumed_cache = PlayedNow(std::memory_order_acquire);
    if (consumed_cache >= target || Reached(target, on_engine, std::memory_order_acquire)) {
        return false;
    }
    Nx2Profile::Scope scope{Nx2Profile::Zone::WriterWait};
    Common::Nx2Timeline::ArgSpan timeline_span{drain_site_names[static_cast<size_t>(site)],
                                               {"behind", target - consumed_cache}};
    wait_target.store(target, std::memory_order_seq_cst);
    Nx2Profile::FrontWaitBegin(site);
    Publish();
    NoteFrontWait(front_wait, site, target, on_engine);
    SCOPE_EXIT {
        front_wait.begin_ns.store(0, std::memory_order_relaxed);
    };
    const u64 start = Nx2Profile::Enabled() ? Nx2Profile::Ticks() : 0;
    const u64 begin_ns = Common::Nx2Hitch::NowNs();
    const u64 behind = target - consumed_cache;
    for (u32 spins = 0;; ++spins) {
        consumed_cache = PlayedNow(std::memory_order_acquire);
        if (consumed_cache >= target || Reached(target, on_engine, std::memory_order_acquire)) {
            if (on_engine && consumed_cache < target) {
                ++g_site_waits.engine_early;
            }
            wait_target.store(~u64{0}, std::memory_order_relaxed);
            if (start != 0) [[unlikely]] {
                // fable_stage_levelpoints 18.4 rows 7 and 8: until the back reached it, after.
                FritchiiPrice::NoteFrontWait(static_cast<size_t>(site), target, start);
            }
            Nx2Profile::AddDrain(site, start);
            NoteSiteWait(site, behind, Common::Nx2Hitch::NowNs() - begin_ns);
            return true;
        }
        Backoff(spins, target, on_engine);
    }
}

void GpuSplit::Backoff(u32 spins, u64 target, bool on_engine) {
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
    // E's `consumed` store and R's `played` raise both wake the front (RaisePlayed).
    if (!Reached(target, on_engine, std::memory_order_seq_cst)) {
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

void GpuSplit::MacroWords(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                          u32 amount, Effect effect) {
    // Not for the 3D engine: the memory stage applies a Multi entry to its 3D copy as a
    // multi-method and ignores the Direct bit (memstage_shadow.cpp, Kind::Multi), while the
    // back calls every word as a Direct write. A 3D engine's words go one MacroWrite each.
    if (dynamic_cast<const Tegra::Engines::Maxwell3D*>(engine) != nullptr) [[unlikely]] {
        ASSERT_MSG(false, "GpuSplit::MacroWords is not for the 3D engine");
        for (u32 i = 0; i < amount; ++i) {
            MacroWrite(engine, method, words[i], effect);
        }
        return;
    }
    SelectEngine(engine);
    // As MultiMethod, with no segment (a MacroWrite has none), marked Direct.
    u32 pending = amount;
    while (pending > 0) {
        const u32 piece = std::min(pending, MaxMultiWords);
        constexpr size_t head = 2;
        const size_t size = head + (piece + 1) / 2;
        u64* const entry = Reserve(size);
        entry[0] = static_cast<u64>(Kind::Multi) | Direct | (u64{method & 0xffff} << 8) |
                   (u64{piece} << 32);
        entry[1] = pending;
        entry[size - 1] = 0;
        std::memcpy(entry + head, words, size_t{piece} * sizeof(u32));
        Commit(size, effect);
        words += piece;
        pending -= piece;
    }
    if (amount != 0 && engine->memory_write_mask[method]) {
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

// FRITCHII_NOTED_SUMMARY (live, default 0; F4 of work\FRONT_SPLIT_DESIGN.md 2d, the same shape as
// FRITCHII_DISPATCH_WALK_SUMMARY): each 0xe2c read check walked pending_writes from the newest
// note down to `since` (37 ms/s sampled). The summary keeps, per page bucket (4,096 by page
// hash), the position of the newest note over the page, and one "wide" position for notes over
// more than 16 pages or of unknown range.
//
// Exact: the buckets answer "no noted write" only when since >= floor and every bucket of the
// read's pages and the wide position are at or before `since`. Positions only grow while the
// session runs (the reset turns the summary off), and every note made after `floor` (the write
// position when it went live; it stayed live since, or it would have a newer floor) set the
// buckets of all its pages, or the wide position, to its own position, which is the position
// the walk compares. So a note after `since` that overlaps the read shares a page with it (an
// empty note at b counts as over b's page: the walk's overlap test can take it inside the
// read) and keeps that bucket above `since`. Collisions only make a bucket newer (a walk).
// The walk can also answer "write" because its table forgot notes (Overflow); the buckets
// forget none, so their "no" is the exact answer there too (counted apart, not a difference).
//
// 1 = on: answered checks skip the walk; 1 in 64 checks walk anyway (timed, and compared).
// 2 = verify: the walk runs after every check and answers; NotedSummaryDiffer counts "the
//     buckets said no, the walk found an overlapping note" (must be 0, first 16 logged).
// 3 = dry: as 2, read for the counts and the timed saving. Line every 10 s "Noted summary".
void GpuSplit::NoteSummaryWrite(u64 begin, u64 size) {
    auto& sum = noted_summary;
    if (VideoCore::Nx2Debug::g_noted_summary.load(std::memory_order_relaxed) <= 0) {
        sum.live = false; // Switched off: the next turn-on starts a new floor.
        return;
    }
    const u64 position = write_position;
    if (size == ~u64{0} || begin + size < begin) {
        sum.wide = position;
        return;
    }
    const u64 first = begin >> 12;
    const u64 last = (begin + (std::max<u64>)(size, 1) - 1) >> 12;
    if (last - first >= NotedMaxPages) {
        sum.wide = position;
        return;
    }
    for (u64 page = first; page <= last; ++page) {
        sum.buckets[NotedBucket(page)] = position;
    }
}

bool GpuSplit::NotedWriteSinceSummary(u64 since, u64 begin, u64 size) const {
    namespace Debug = VideoCore::Nx2Debug;
    const int mode = Debug::SwitchValue(Debug::g_noted_summary, "FRITCHII_NOTED_SUMMARY");
    u64 scanned = 0;
    if (mode < 1 || mode > 3) {
        return NotedWalk(since, begin, size, scanned) != NotedWalkResult::None;
    }
    auto& sum = noted_summary;
    if (!sum.live) {
        // Notes sent so far were never bucketed: only reads since now or later can be answered.
        sum.buckets.fill(0);
        sum.wide = 0;
        sum.floor = write_position;
        sum.live = true;
        sum.window_ns = Common::Nx2Hitch::NowNs();
        sum.window_ticks = Nx2Profile::Ticks();
        sum.logged = sum.stats;
    }
    auto& stats = sum.stats;
    if ((++stats.checks & 1023) == 0) {
        LogNotedSummary(mode);
    }
    const bool timed = (++sum.calls & 63) == 0;
    const u64 t0 = timed ? Nx2Profile::Ticks() : 0;
    enum class Answer : u8 { Miss, Bucket, Wide, Floor, Long };
    Answer answer = Answer::Miss;
    if (since < sum.floor) {
        answer = Answer::Floor;
    } else if (sum.wide > since) {
        answer = Answer::Wide;
    } else {
        const u64 first = begin >> 12;
        const u64 last = (begin + (std::max<u64>)(size, 1) - 1) >> 12;
        if (begin + size < begin || last - first >= NotedMaxPages) {
            answer = Answer::Long;
        } else {
            for (u64 page = first; page <= last; ++page) {
                if (sum.buckets[NotedBucket(page)] > since) {
                    answer = Answer::Bucket;
                    break;
                }
            }
        }
    }
    const u64 t1 = timed ? Nx2Profile::Ticks() : 0;
    switch (answer) {
    case Answer::Miss:
        ++stats.answered;
        break;
    case Answer::Bucket:
        ++stats.refused_bucket;
        break;
    case Answer::Wide:
        ++stats.refused_wide;
        break;
    case Answer::Floor:
        ++stats.refused_floor;
        break;
    case Answer::Long:
        ++stats.refused_long;
        break;
    }
    if (answer == Answer::Miss && mode == 1 && !timed) {
        return false;
    }
    const NotedWalkResult walk = NotedWalk(since, begin, size, scanned);
    const u64 t2 = timed ? Nx2Profile::Ticks() : 0;
    if (timed) {
        ++stats.timed;
        stats.summary_ticks += t1 - t0;
    }
    if (answer == Answer::Miss) {
        stats.avoided_notes += scanned;
        if (timed) {
            ++stats.timed_answered;
            stats.walk_ticks_answered += t2 - t1;
        }
        if (walk == NotedWalkResult::Overlap) [[unlikely]] {
            if (stats.differ++ < 16) {
                LOG_WARNING(HW_GPU,
                            "NotedSummaryDiffer (FRITCHII_NOTED_SUMMARY={}, must be 0): the "
                            "buckets found no note over {:#x}+{:#x} since {} (floor {}, wide {}), "
                            "the walk found one {} notes back",
                            mode, begin, size, since, sum.floor, sum.wide, scanned);
            }
        } else if (walk == NotedWalkResult::Overflow) {
            ++stats.overflow_answered;
        }
        if (mode == 1) {
            return walk == NotedWalkResult::Overlap; // a difference takes the walk's answer
        }
    }
    return walk != NotedWalkResult::None;
}

void GpuSplit::LogNotedSummary(int mode) const {
    auto& sum = noted_summary;
    const u64 now = Common::Nx2Hitch::NowNs();
    if (now - sum.window_ns < 10'000'000'000ULL) {
        return;
    }
    const u64 ticks = Nx2Profile::Ticks();
    const double window = static_cast<double>(now - sum.window_ns) / 1e9;
    const double ns_per_tick =
        static_cast<double>(now - sum.window_ns) /
        static_cast<double>((std::max)(ticks - sum.window_ticks, u64{1}));
    const NotedSummaryStats& s = sum.stats;
    const NotedSummaryStats& l = sum.logged;
    const auto rate = [window](u64 count) { return static_cast<double>(count) / window; };
    const u64 checks = s.checks - l.checks;
    const u64 answered = s.answered - l.answered;
    const u64 timed = s.timed - l.timed;
    const u64 timed_answered = s.timed_answered - l.timed_answered;
    const u64 summary_ticks = s.summary_ticks - l.summary_ticks;
    const u64 walk_ticks = s.walk_ticks_answered - l.walk_ticks_answered;
    // At 1 only the timed answered checks walked: their notes scaled to all answered ones.
    const double avoided =
        mode == 1 ? (timed_answered != 0 ? static_cast<double>(s.avoided_notes - l.avoided_notes) *
                                               static_cast<double>(answered) /
                                               static_cast<double>(timed_answered)
                                         : 0.0)
                  : static_cast<double>(s.avoided_notes - l.avoided_notes);
    const double summary_ns =
        timed != 0 ? static_cast<double>(summary_ticks) * ns_per_tick / static_cast<double>(timed)
                   : 0.0;
    const double walk_ns = timed_answered != 0 ? static_cast<double>(walk_ticks) * ns_per_tick /
                                                     static_cast<double>(timed_answered)
                                               : 0.0;
    // The walks the answered checks skip (would skip, at 2 and 3) minus the summary on every
    // check, from the timed 1 in 64, scaled to all checks.
    const double saving_ms =
        timed != 0 ? (static_cast<double>(walk_ticks) - static_cast<double>(summary_ticks)) *
                         ns_per_tick * static_cast<double>(checks) / static_cast<double>(timed) /
                         1e6 / window
                   : 0.0;
    LOG_INFO(HW_GPU,
             "Noted summary (FRITCHII_NOTED_SUMMARY={}) {:.0f} s: checks {:.0f}/s, answered by "
             "the buckets {:.0f}/s ({:.0f}%; {}); walked: page bucket newer {:.0f}/s, wide note "
             "{:.0f}/s, read older than the summary {:.0f}/s, read over 16 pages {:.0f}/s; walk "
             "notes on answered checks {:.0f}/s; timed 1 in 64: summary {:.0f} ns a check, walk "
             "{:.0f} ns an answered check, saving {:.2f} ms/s; NotedSummaryDiffer {} this "
             "session (must be 0), walk overflows the buckets answered {}",
             mode, window, rate(checks), rate(answered),
             checks != 0 ? static_cast<double>(answered) * 100.0 / static_cast<double>(checks)
                         : 0.0,
             mode == 1 ? "walk skipped" : "the walk ran and answered",
             rate(s.refused_bucket - l.refused_bucket),
             rate(s.refused_wide - l.refused_wide), rate(s.refused_floor - l.refused_floor),
             rate(s.refused_long - l.refused_long), avoided / window, summary_ns, walk_ns,
             saving_ms, s.differ, s.overflow_answered);
    sum.logged = s;
    sum.window_ns = now;
    sum.window_ticks = ticks;
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
        consumed_cache = PlayedNow(std::memory_order_acquire);
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
// The table is set associative (256 sets of 8 pages, least recently used way replaced): two tag
// probes for a range of one or two pages. A range over more than 16 pages walks the occupied
// slots instead, after a test against the lowest and highest watched page. 2,048 slots: OoT
// snapshots ~400 distinct pages, and with 256 slots the LRU rule evicted exactly the rarely read
// pages (hot pages refresh their last use with every answer), the ones whose ring scans run past
// the limit: the watch answered 96% of the checks but none of the "past the scan" drains
// (fable52_answer_exp122). EXP122=2 runs the old scan
// without a limit after every "no write" and counts each overlap it finds that the watch did not
// list (the "verify mismatches" of EXP10's log line, which must stay 0).
struct GpuSplit::PageWatch {
    static constexpr u32 Sets = 256;
    static constexpr u32 Ways = 8;
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
    /// The slots holding a page, in the order they were first filled (a wide write walks these).
    std::array<u16, Slots> occupied{};
    u32 occupied_count = 0;
    u64 min_page = Empty;
    u64 max_page = 0;
    u64 clock = 0;
    // Stats: mutable through the pointer from const WritersMayWriteSince.
    u64 answered = 0;
    u64 answered_write = 0;
    u64 overflows = 0;
    u64 verified = 0;
    u64 mismatches = 0;
    u64 unwatched_absent = 0; ///< Checks of a page not in the table: the ring answers
    u64 unwatched_newer = 0;  ///< Checks of a page watched since after their snapshot
    u64 unwatched_spans = 0;  ///< Checks over two pages
    u64 replaced = 0;         ///< Watched pages pushed out by a new one

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
        page_watch->occupied_count = 0;
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
            watch.occupied[watch.occupied_count++] = static_cast<u16>(pick);
        } else {
            ++watch.replaced;
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
    for (u32 i = 0; i < watch.occupied_count; ++i) {
        const u32 slot = watch.occupied[i];
        const u64 page = watch.pages[slot];
        if (first_page <= page && page <= last_page) {
            watch.Record(watch.entries[slot], page, begin, end, position);
        }
    }
}

GpuSplit::WatchAnswer GpuSplit::WatchedWritesSince(u64 since, u64 begin, u64 size, u64 known,
                                                   ListedWrites* found) const {
    if (size == 0 || page_watch_mode == 0) {
        return WatchAnswer::Unwatched;
    }
    // Callers test watched_pages first: the table exists.
    PageWatch& watch = *page_watch;
    const u64 page = begin >> 12;
    if (((begin + size - 1) >> 12) != page) {
        ++watch.unwatched_spans;
        return WatchAnswer::Unwatched;
    }
    const s32 slot = watch.Find(page);
    if (slot < 0) {
        ++watch.unwatched_absent;
        return WatchAnswer::Unwatched;
    }
    PageWatch::Entry& entry = watch.entries[static_cast<u32>(slot)];
    if (entry.start_count > known && entry.start_position > since) {
        // Started after this check's snapshot (another snapshot of the page): the ring answers.
        ++watch.unwatched_newer;
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
        return {0, 0, watched_pages, 0, 0, 0, 0, 0, 0, 0};
    }
    const PageWatch& watch = *page_watch;
    return {watch.answered,         watch.answered_write,  watched_pages,
            watch.overflows,        watch.verified,        watch.mismatches,
            watch.unwatched_absent, watch.unwatched_newer, watch.unwatched_spans,
            watch.replaced};
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
        Defer([this, records] { back_dispatch_records = records; }, Effect::None,
              StageTagBits(StageTag::Pass));
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
        consumed_cache = PlayedNow(std::memory_order_acquire);
    }
    if (consumed_cache < last_dispatch_position) {
        if (!wait) {
            return DispatchCheck::Queued;
        }
        // FRITCHII_DELAY_FRONT_KIND=2 ("Q"): after this wait returned, or before it
        // (FRITCHII_DELAY_BEFORE=1). Off: one relaxed load and a branch each side.
        DelayFrontIf(FrontDelayKind::DispatchWait, true);
        WaitFor(last_dispatch_position, DrainSite::DispatchWait);
        DelayFrontIf(FrontDelayKind::DispatchWait, false);
        waited = true;
    }
    // FRITCHII_DISPATCH_WALK_SUMMARY (live, default 0 = the walk alone; fable_b2 2.4): with
    // bounded records (FRITCHII_ONEF_DISPATCH_RANGES) a page's snapshot is never renewed, so
    // each read walked every record made since its page's last read. The dispatch stamps answer
    // first: every dispatch sent since `since` ran and recorded its writes (above), which is
    // what DispatchStampsMissSince needs, and its count is loaded before the stamps, so a miss
    // says no record below that count at or after `since` touches the page's 64 KB block (nor
    // stamped everything). That is the walk's own miss over the same records, or stricter (the
    // walk stops at the first record before `since`; collisions only make a stamp newer): the
    // cursor moves to the count, nothing to list. A refusal walks as before. 2 also walks after
    // a stamp miss and answers with the walk, counting a write it found (must be 0).
    const int summary_mode = VideoCore::Nx2Debug::SwitchValue(
        VideoCore::Nx2Debug::g_dispatch_walk_summary, "FRITCHII_DISPATCH_WALK_SUMMARY");
    bool summary_miss = false;
    if (summary_mode != 0) {
        u64 summary_count = 0;
        bool wide = false;
        if (DispatchStampsMissSince(since, begin, size, summary_count, wide)) {
            ++dispatch_walk.summary_misses;
            if (summary_mode != 2) {
                checked = (std::max)(checked, summary_count);
                return DispatchCheck::Miss;
            }
            summary_miss = true;
        } else {
            ++(wide ? dispatch_walk.refused_wide : dispatch_walk.refused_block);
        }
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
    // FRITCHII_DISPATCH_WALK_SUMMARY=2: a write over the bytes after a stamp miss. A record
    // whose slot the back reused while it was read is a stray range, not a disagreement.
    const auto disagree = [&](u64 write_begin, u64 write_end) {
        if (!summary_miss ||
            dispatch_write_count.load(std::memory_order_acquire) >= index + DispatchWriteSlots) {
            return;
        }
        if (dispatch_walk.verify_disagreed++ < 16) {
            LOG_WARNING(HW_GPU,
                        "Dispatch walk summary (FRITCHII_DISPATCH_WALK_SUMMARY=2, must be 0): "
                        "the stamps missed {:#x}+{:#x} since {}, but record {} ({:#x}-{:#x}) "
                        "writes it",
                        begin, size, since, index - 1, write_begin, write_end);
        }
    };
    u64 walked = 0;
    for (u64 scanned = 0; index > (std::max)(checked, oldest); --index, ++scanned) {
        if (scanned == MaxDispatchScan) {
            dispatch_walk.walked += walked;
            return DispatchCheck::ScanLimit;
        }
        ++walked;
        const DispatchWrite& write = dispatch_writes[(index - 1) % DispatchWriteSlots];
        if (write.position.load(std::memory_order_relaxed) < since) {
            reached = true;
            break;
        }
        const u64 write_begin = write.begin.load(std::memory_order_relaxed);
        const u64 write_end = write.end.load(std::memory_order_relaxed);
        if (write_begin < end && begin < write_end) {
            if (summary_miss) [[unlikely]] {
                disagree(write_begin, write_end);
            }
            if (write_end == ~u64{0} && write_begin == 0) {
                dispatch_walk.walked += walked;
                return DispatchCheck::Unbounded;
            }
            if (listed == nullptr) {
                dispatch_walk.walked += walked;
                return DispatchCheck::Overlap;
            }
            // A slot reused meanwhile lists a stray range; the reuse check below then misses.
            found.Add(begin, size, write_begin, write_end);
        }
    }
    dispatch_walk.walked += walked;
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
    // DispatchStampsMissSince: the stamps before the count, so a front that loaded the count
    // sees the stamps of every record below it. A max that holds even with two writers (the
    // back, and GPUMem in a fill it runs for the back).
    const u64 position = back_entry_position;
    const auto raise = [position](std::atomic<u64>& stamp) {
        u64 current = stamp.load(std::memory_order_relaxed);
        while (current < position &&
               !stamp.compare_exchange_weak(current, position, std::memory_order_release,
                                            std::memory_order_relaxed)) {
        }
    };
    if (begin < end) {
        const u64 first = begin >> DispatchStampBits;
        const u64 last = (end - 1) >> DispatchStampBits;
        if ((begin == 0 && end == ~u64{0}) || last - first >= DispatchStampSlots) {
            raise(dispatch_wide_stamp);
        } else {
            for (u64 block = first; block <= last; ++block) {
                raise(dispatch_stamps[block % DispatchStampSlots]);
            }
        }
    }
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
    consumed_cache = PlayedNow(std::memory_order_acquire);
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
                // FRITCHII_ONEF_CB_LEAN & 1 (fable_b 23.1 cut 3, on E): a single word too goes
                // through the block write (CallConstBufferWords takes every check CallMethod's
                // path would, and falls back to it where it cannot write the run in one).
                if (count == 1 && (!OneFollower::OnEngineThread() ||
                                   (OneFollower::CbLean() & 1) == 0)) {
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
        if ((head & Direct) != 0) {
            // EXP124: a macro's words (MacroWords), each called as its MacroWrite was.
            engine->CallMacroWords(*system, method, reinterpret_cast<const u32*>(entry + data),
                                   amount);
        } else {
            engine->CallMultiMethod(*system, method, reinterpret_cast<const u32*>(entry + data),
                                    amount, static_cast<u32>(entry[1]));
        }
        return data + (amount + 1) / 2;
    }
    case Kind::Call: {
        // EDEN_NX2_FASTDRAW_COUNT: deferred calls end reuse unless marked neutral.
        if (Nx2FastDrawCount::g_on && (head & Nx2FastDrawCount::NeutralCallBit) == 0) [[unlikely]] {
            NoteFastDrawCallBreak(entry[1]);
        }
        const size_t payload = static_cast<size_t>((head >> 8) & 0xffffff);
        // FRITCHII_MEMSTAGE_RELEASE: GPUMem, level at this entry, had this thread run the puller's
        // semaphore release already (memstage_shadow.cpp, StageRelease); skipped here, also when
        // the record outlived the ring by this entry (GPUMem failed in the request), so it never
        // runs twice. The test is on the header first: other calls pay one compare.
        // FRITCHII_MEMSTAGE_PASS (fable_stage_p3 1): a release GPUMem passed has a pass record,
        // asked for first: the call runs here, and BackTakeRelease (which counts a served one
        // without a request record) is not asked.
        if (StageCacheCallOf(head) == StageCacheCall::SemaphoreRelease &&
            StageTagOf(head) == StageTag::Cache &&
            (MemoryStageLink::RingActive() || MemoryStageLink::Get().ReleasePending() ||
             MemoryStageLink::Get().PassPending())) [[unlikely]] {
            MemoryStageLink& link = MemoryStageLink::Get();
            const bool passed =
                link.PassPending() && link.BackPassRecord(back_entry_position) != nullptr;
            if (!passed && (MemoryStageLink::RingActive() || link.ReleasePending()) &&
                link.BackTakeRelease(back_entry_position)) {
                return 2 + payload;
            }
        }
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
    case Kind::CbWords: {
        // FRITCHII_CB_WORDS_BLOCK: one entry for what were `count` Method entries in a row.
        const u32 count = static_cast<u32>(head >> 32);
        const bool verify = (head & CbVerify) != 0;
        const size_t first = verify ? 2 : 1;
        ExecuteCbWords(entry + first, count, verify ? entry + 1 : nullptr);
        return first + size_t{count};
    }
    }
    return 1;
}

namespace {
/// A counter with one writer (the back): a plain load and store.
void CbBump(std::atomic<u64>& counter, u64 amount = 1) {
    counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}
} // Anonymous namespace

void GpuSplit::ExecuteCbWords(const u64* words, u32 count, const u64* verify_address) {
    // Must be 0: a word that is not what CbWord writes (a Method word for a constant buffer data
    // register, without a segment or the sink tag), or a block longer than CbWord makes one.
    using Tegra::Engines::Maxwell3D;
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    bool well_formed = count != 0 && count <= MaxCbWords;
    for (u32 i = 0; i < count && well_formed; ++i) {
        const u64 word = words[i];
        well_formed = static_cast<Kind>(word & 0xf) == Kind::Method &&
                      (word & (HasSegment | Sink)) == 0 &&
                      static_cast<u32>((word >> 8) & 0xffff) - cb_data < 16;
    }
    CbBump(cb_back.blocks);
    CbBump(cb_back.words, count);
    if (!well_formed) [[unlikely]] {
        CbBump(cb_back.bad);
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
            LOG_ERROR(HW_GPU, "CB words block: a block of {} words holds other words (first {:#x})",
                      count, count != 0 ? words[0] : 0);
        }
        // Each Method word on its own, as its entry would run (none has a segment word to
        // read); anything else is dropped.
        for (u32 i = 0; i < count; ++i) {
            if (static_cast<Kind>(words[i] & 0xf) == Kind::Method &&
                (words[i] & HasSegment) == 0) {
                Execute(words + i, 1);
            }
        }
        return;
    }
    if (verify_address != nullptr) [[unlikely]] {
        VerifyCbWords(words, count, *verify_address);
        return;
    }
    // As the entries in a row ran: Execute's Method case takes the constant buffer words that
    // follow it (up to 64, the block's limit) in one CallConstBufferWords, or runs each alone
    // where that gather is off (Nx2Perf cb_batch, not fast_maxwell3d).
    for (u32 done = 0; done < count;) {
        done += static_cast<u32>(Execute(words + done, count - done));
    }
}

void GpuSplit::VerifyCbWords(const u64* words, u32 count, u64 front_address) {
    using Tegra::Engines::Maxwell3D;
    const auto run_block = [&] {
        for (u32 done = 0; done < count;) {
            done += static_cast<u32>(Execute(words + done, count - done));
        }
    };
    auto* const maxwell3d = dynamic_cast<Maxwell3D*>(back_engine);
    if (maxwell3d == nullptr) {
        CbBump(cb_back.apply_skipped);
        run_block();
        return;
    }
    // The sink first, as the block's first word does (the offset may wait there): from here
    // the registers are those the words see.
    maxwell3d->ConsumeSink(*system);
    auto& regs = maxwell3d->regs;
    // 1. Where the back writes the first word against where the front noted it (its NoteWrite,
    // RangeLastWrite and macro reads used this address).
    const GPUVAddr back_address = regs.const_buffer.Address() + regs.const_buffer.offset;
    CbBump(cb_back.address_checked);
    if (back_address != front_address) {
        CbBump(cb_back.address_differ);
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_ERROR(HW_GPU,
                      "CB words block (FRITCHII_CB_WORDS_BLOCK=2): {} words go to {:#x} on the "
                      "back, the front noted {:#x}",
                      count, back_address, front_address);
        }
    }
    // 2. Applied as the block, then from the same state word by word (each as its own entry
    // ran: Execute with one word). Not when the first word would draw a deferred draw (it can
    // run once), and not where both are one path (one word, the gather off).
    if (count < 2 || !maxwell3d->fast_maxwell3d || !Nx2Perf::On(Nx2Perf::Opt::CbBatch) ||
        maxwell3d->draw_manager.HasDeferredDraw()) {
        CbBump(cb_back.apply_skipped);
        run_block();
        return;
    }
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    struct State {
        u32 offset;
        std::array<u32, 16> data;
        std::array<u32, 16> shadow;
        Maxwell3D::DirtyState::Flags flags;
    };
    const auto take = [&] {
        State state{};
        state.offset = regs.const_buffer.offset;
        for (u32 i = 0; i < 16; ++i) {
            state.data[i] = regs.reg_array[cb_data + i];
            state.shadow[i] = maxwell3d->shadow_state.reg_array[cb_data + i];
        }
        state.flags = maxwell3d->dirty.flags;
        return state;
    };
    const auto put = [&](const State& state) {
        regs.const_buffer.offset = state.offset;
        for (u32 i = 0; i < 16; ++i) {
            regs.reg_array[cb_data + i] = state.data[i];
            maxwell3d->shadow_state.reg_array[cb_data + i] = state.shadow[i];
        }
        maxwell3d->dirty.flags = state.flags;
    };
    const State before = take();
    // The words' values as ProcessCBMultiData writes them, one at a time (the shadow RAM's mode
    // stays for the block: shadow_ram_control is another method), and the words that fit the
    // buffer (its room is checked word by word there).
    const bool replay =
        maxwell3d->shadow_state.shadow_ram_control == Maxwell3D::Regs::ShadowRamControl::Replay;
    std::array<u32, MaxCbWords> values{};
    for (u32 i = 0; i < count; ++i) {
        const u32 method = static_cast<u32>((words[i] >> 8) & 0xffff);
        values[i] = replay ? before.shadow[method - cb_data] : static_cast<u32>(words[i] >> 32);
    }
    const GPUVAddr address = regs.const_buffer.Address();
    const u32 size = regs.const_buffer.size;
    u32 fitting = 0;
    while (fitting < count && u64{before.offset} + (u64{fitting} + 1) * sizeof(u32) <= size) {
        ++fitting;
    }
    run_block();
    const State as_block = take();
    bool differs = false;
    // Memory: only where the back writes it itself (not served by the memory stage, whose
    // thread may already have written later words over these).
    const u64 bytes = u64{fitting} * sizeof(u32);
    auto& memory = maxwell3d->GetMemoryManager();
    if (bytes != 0 && address != 0 && !MemoryStageLink::BackServed() &&
        !MemoryStageLink::RingActive() &&
        memory.IsFullyMappedRange(address + before.offset, bytes)) {
        std::array<u32, MaxCbWords> written{};
        memory.ReadBlockUnsafe(address + before.offset, written.data(), bytes);
        if (std::memcmp(written.data(), values.data(), bytes) != 0) {
            differs = true;
        }
    }
    put(before);
    for (u32 i = 0; i < count; ++i) {
        Execute(words + i, 1);
    }
    const State by_word = take();
    differs |= as_block.offset != by_word.offset || as_block.data != by_word.data ||
               as_block.shadow != by_word.shadow || as_block.flags != by_word.flags;
    CbBump(cb_back.applied_twice);
    if (differs) {
        CbBump(cb_back.apply_differ);
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_ERROR(HW_GPU,
                      "CB words block (FRITCHII_CB_WORDS_BLOCK=2): {} words at {:#x}+{:#x}: "
                      "offset {:#x} as a block, {:#x} word by word; data {}, shadow {}, dirty "
                      "flags {}, memory checked {}",
                      count, address, before.offset, as_block.offset, by_word.offset,
                      as_block.data == by_word.data ? "same" : "differ",
                      as_block.shadow == by_word.shadow ? "same" : "differ",
                      as_block.flags == by_word.flags ? "same" : "differ", bytes);
        }
    }
}

void GpuSplit::ReportCbWords() {
    const auto now = std::chrono::steady_clock::now();
    if (cb_since == std::chrono::steady_clock::time_point{}) {
        cb_since = now;
        cb_reported = cb_stats;
        cb_reported_overflows = ranged_overflow_count;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - cb_since).count();
    if (seconds < 10.0) {
        return;
    }
    const int mode = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_cb_words_block,
                                                      "FRITCHII_CB_WORDS_BLOCK");
    const u64 blocks = cb_stats.blocks - cb_reported.blocks;
    const u64 words = cb_stats.words - cb_reported.words;
    const u64 overflows = ranged_overflow_count - cb_reported_overflows;
    // The back's counts are totals since the start (one writer, read here relaxed).
    const auto back = [](const std::atomic<u64>& counter) {
        return counter.load(std::memory_order_relaxed);
    };
    if (mode != 0 || blocks != 0) {
        LOG_INFO(HW_GPU,
                 "CB words block (FRITCHII_CB_WORDS_BLOCK={}) {:.0f} s: front blocks {:.0f}/s "
                 "with {:.0f} words/s ({:.2f} a block), ring entries saved {:.0f}/s, blocks "
                 "ended full {:.0f}/s; ranged write overflows {:.0f}/s; back totals: blocks {}, "
                 "words {}; must be 0: broken blocks {}, bad blocks {}; verify (=2): addresses "
                 "checked {}, differ {}; applied both ways {}, differ {}, not applied twice {}",
                 mode, seconds, static_cast<double>(blocks) / seconds,
                 static_cast<double>(words) / seconds,
                 blocks != 0 ? static_cast<double>(words) / static_cast<double>(blocks) : 0.0,
                 static_cast<double>(words - blocks) / seconds,
                 static_cast<double>(cb_stats.full - cb_reported.full) / seconds,
                 static_cast<double>(overflows) / seconds, back(cb_back.blocks),
                 back(cb_back.words), cb_stats.broken, back(cb_back.bad),
                 back(cb_back.address_checked), back(cb_back.address_differ),
                 back(cb_back.applied_twice), back(cb_back.apply_differ),
                 back(cb_back.apply_skipped));
    }
    cb_since = now;
    cb_reported = cb_stats;
    cb_reported_overflows = ranged_overflow_count;
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

void GpuSplit::BackWaitForStage(std::stop_token& stop_token, u64 read_position) {
    // FRITCHII_MEMSTAGE_THREAD=2: the "split wait" span's phase 3, waiting for the stage. The
    // back serves the stage's sync requests meanwhile (memory_stage_link.cpp).
    MemoryStageLink& link = MemoryStageLink::Get();
    Nx2Profile::Scope scope{Nx2Profile::Zone::SplitWait, Nx2Profile::Enabled(), false};
    Common::Nx2Timeline::ArgSpan timeline_span{
        Nx2Profile::zone_names[static_cast<size_t>(Nx2Profile::Zone::SplitWait)], {"phase", 3}};
    const u64 begin_ns = Common::Nx2Hitch::NowNs();
    const auto waiting = [&] {
        return MemoryStageLink::RingActive() &&
               staged.load(std::memory_order_acquire) == read_position &&
               published.load(std::memory_order_acquire) != read_position &&
               !stop_token.stop_requested();
    };
    for (u32 spins = 0;;) {
        if (link.BackServeRequests()) {
            spins = 0;
            continue;
        }
        if (!waiting()) {
            break;
        }
        if (spins < 8192) {
            ++spins;
            Pause();
            continue;
        }
        back_waiting.store(true, std::memory_order_seq_cst);
        const u32 seen = wake.load(std::memory_order_seq_cst);
        if (waiting() && !link.BackRequestPending()) {
            wake.wait(seen, std::memory_order_seq_cst);
        }
        back_waiting.store(false, std::memory_order_relaxed);
    }
    link.NoteBackWait(Common::Nx2Hitch::NowNs() - begin_ns);
}

void GpuSplit::StoreConsumed(u64 position) {
    if (OneFollower::OnEngineThread()) [[unlikely]] {
        // FRITCHII_ONEF_CARRY_OPS: ops E kept for its next packet go now, before the front can
        // see E past them (a kept report's GuestWriteOpMark and its packet's floor come first).
        OneFollower::FlushCarried();
    }
    consumed.store(position, std::memory_order_seq_cst);
    if (OneFollower::OnEngineThread()) [[unlikely]] {
        // `played`: this position clamped to R's oldest unplayed packet (RaisePlayed wakes the
        // front). Below step 4 R is never behind here, so it is this position.
        OneFollower::EngineConsumed(position);
        return;
    }
    NotifyFront();
}

void GpuSplit::BackLoop(std::stop_token stop_token) {
    const Common::DkbCpuTrace::Registration cpu_trace{Common::DkbCpuTrace::Role::GpuBack};
    Common::SetCurrentThreadName("GPUBack");
    on_back_thread = true;
    // FRITCHII_ONEF=1: this thread is E; the renderer's calls from it are lent to R.
    OneFollower::MarkEngineThread();
    back_running.store(true, std::memory_order_release);
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::PinCurrentThread(Common::PinnedRole::GpuBack);
    Common::SetCurrentThreadToPerformanceCores();
    system->RegisterHostThread();
#ifdef _WIN32
    g_back_thread_id.store(GetCurrentThreadId(), std::memory_order_relaxed);
#endif
    u64 read_position = 0;
    if (MemStageShadow::g_on.load(std::memory_order_relaxed)) [[unlikely]] {
        MemStageShadow::Start();
    }
    // FRITCHII_MEMSTAGE_THREAD=2 (memory_stage_link.cpp): the stage reads the ring before the
    // back; the back's limit is the stage's `staged`, and each entry is lent or served.
    MemoryStageLink& link = MemoryStageLink::Get();
    while (!stop_token.stop_requested()) {
        Nx2Profile::Report("GPU back thread");
        // The ring starts and ends here, where everything before read_position is consumed.
        const bool ring_on = link.BackRingTop(read_position);
        MemoryStageLink::BackLimits limits =
            link.BackLoad(published.load(std::memory_order_acquire));
        const auto limit = [&] {
            limits = link.BackLoad(published.load(std::memory_order_acquire));
            return limits.available;
        };
        u64 available = limits.available;
        if (available == read_position) {
            if (ring_on) {
                // A sync of the stage's fill (Serve): the back is parked at the stage's position.
                if (link.BackServeRequests()) {
                    continue;
                }
                if (published.load(std::memory_order_acquire) != read_position) {
                    // Phase 3 of the split wait: the front published, the stage has not passed
                    // it on yet. No idle hook, no fence quiet check.
                    BackWaitForStage(stop_token, read_position);
                    continue;
                }
            }
            // On the timeline (not through the zone) with the phase the wait reached: 0 = spin,
            // 1 = the idle hook ran, 2 = slept on `wake`, 3 = waited for the memory stage
            // (tools/nx2/critical_path.py).
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
                                !hook_requested.load(std::memory_order_relaxed) &&
                                !(ring_on && link.BackRequestPending());
                 ++spins) {
                Pause();
                available = limit();
            }
            if (ring_on && link.BackRequestPending()) {
                continue;
            }
            // THREAD=2: not while the stage works (BackEnterHook's handshake).
            if (available == read_position && idle_hook != nullptr && link.BackEnterHook()) {
                hook_requested.store(false, std::memory_order_seq_cst);
                ForgetFencesQuiet();
                timeline_span.Set(0, {"phase", 1});
                wait_phase = 1;
                in_idle_wait = true;
                idle_hook(idle_argument);
                in_idle_wait = false;
                link.BackLeaveHook();
                available = limit();
                if (available == read_position) {
                    CheckFencesQuiet(read_position);
                }
            } else if (available == read_position && idle_hook != nullptr && ring_on &&
                       MemoryStageLink::Ahead() != 0 &&
                       hook_requested.load(std::memory_order_seq_cst)) [[unlikely]] {
                // THREAD=2 with the window open (step 7), the stage busy: a waiter asked for the
                // hook. Run it as the mid-batch
                // path below does (its flush needs nothing of the stage; the idle-only parts skip
                // outside the idle wait), or the back could not sleep while the request stands.
                hook_requested.store(false, std::memory_order_seq_cst);
                ForgetFencesQuiet();
                idle_hook(idle_argument);
                link.NoteHookFlush();
                available = limit();
            }
            if (available == read_position) {
                // Publish bumps `wake` when it sees back_waiting: read it before the last look
                // at `published`, so a publish in between ends the wait at once. (The stage bumps
                // it too, for `staged` and for its requests.)
                back_waiting.store(true, std::memory_order_seq_cst);
                const u32 seen = wake.load(std::memory_order_seq_cst);
                available = limit();
                // A request made after the hook ran (RequestIdleHook) runs it before sleeping.
                if (available == read_position && !stop_token.stop_requested() &&
                    !hook_requested.load(std::memory_order_seq_cst) &&
                    MemoryStageLink::RingActive() == ring_on &&
                    !(ring_on && link.BackRequestPending()) && !link.BackRingStartWanted()) {
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
        // Step 7 (fps fix 3): with the window open the ring's per-entry calls run only at the
        // batch's first entry and where a run starts or ends. The limits stay the batch's, so
        // between their edges BackEntry would do what it did at the edge, and BackAfter can end
        // a run only at or past lent_until (every entry there while the run stays open: the
        // stage may extend it). Level mode (AHEAD = 0) calls them for every entry.
        const bool ring_edges = ring_on && MemoryStageLink::Ahead() != 0;
        u64 ring_next_edge = read_position;
        bool ring_run = false;
        // FRITCHII_DELAY_BACK_KIND=1 ("M", fable_stage_levelpoints 18.3): the method zone's
        // entries are counted for the spin. Read once a batch: with the switch at 0 an entry pays
        // a test of this local.
        const bool entry_delay =
            VideoCore::Nx2Debug::g_delay_back_us.load(std::memory_order_relaxed) != 0;
        // fable_stage_levelpoints 18.4 row 4 and the waits' split (GPU profile on): each entry's
        // kind and the "method" zone time it added; the position stores stamped.
        const bool entry_price = FritchiiPrice::On();
        const bool entry_hook = entry_delay || entry_price;
        if (entry_price) [[unlikely]] {
            FritchiiPrice::MaybeReport();
        }
        // Row 4's kind of an entry (`head`, `size` words run).
        const auto entry_kind = [this](u64 head, size_t size, u32& entries) {
            using Entry = FritchiiPrice::Entry;
            entries = 1;
            switch (static_cast<Kind>(head & 0xf)) {
            case Kind::Method: {
                using Tegra::Engines::Maxwell3D;
                const u32 method = static_cast<u32>((head >> 8) & 0xffff);
                constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
                if (back_engine == nullptr || !back_engine->fast_maxwell3d) {
                    return Entry::MethodOther;
                }
                if (method - cb_data < 16 && (head & HasSegment) == 0) {
                    entries = static_cast<u32>(size);
                    return Entry::CbRun;
                }
                return Entry::Method3D;
            }
            case Kind::Multi:
                return Entry::Multi;
            case Kind::Call:
                return Entry::Call;
            case Kind::Pairs:
                entries = static_cast<u32>(head >> 32);
                return Entry::Pairs;
            case Kind::StateDelta:
                return Entry::StateDelta;
            case Kind::DrawBatch:
                return Entry::DrawBatch;
            case Kind::CbWords:
                entries = static_cast<u32>(head >> 32);
                return Entry::CbWords;
            default:
                return Entry::Other;
            }
        };
        // The ring entries an entry of the method zone stands for (Execute's cases with a
        // "method" scope): a run of constant buffer words or a words block counts each word, a
        // pairs block each pair; calls, engine switches and wraps count none.
        const auto method_entries = [](u64 head, size_t size) -> u32 {
            switch (static_cast<Kind>(head & 0xf)) {
            case Kind::Method:
                return (head & HasSegment) != 0 ? 1U : static_cast<u32>(size);
            case Kind::Multi:
            case Kind::StateDelta:
            case Kind::DrawBatch:
                return 1;
            case Kind::Pairs:
            case Kind::CbWords:
                return static_cast<u32>(head >> 32);
            default:
                return 0;
            }
        };
        // FRITCHII_ESPLIT_COUNT (fritchii_e_split.h, the E split's dry step): read once a batch;
        // off, each entry tests this local. The decode classes an entry by kind, engine, method.
        const int esplit = ESplit::BatchLevel();
        const auto esplit_decode = [this](u64 head) {
            using ESplit::Class;
            using ESplit::EngineKind;
            using ESplit::JoinCause;
            using Tegra::Engines::Maxwell3D;
            ESplit::EntryInfo info{};
            const auto join = [&info](JoinCause cause) {
                info.cls = Class::Join;
                info.cause = cause;
            };
            constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
            const Kind kind = static_cast<Kind>(head & 0xf);
            switch (kind) {
            case Kind::Wrap:
            case Kind::Engine:
                break;
            case Kind::Pairs:
            case Kind::StateDelta:
            case Kind::DrawBatch:
                // Register writes and the draws they end (whose fills are carved out as E2's).
                // A compute engine's registers are E2's with the rest of it.
                if (ESplit::KindOf(back_engine) == EngineKind::Compute) {
                    info.cls = Class::E2;
                }
                break;
            case Kind::CbWords:
                info.cls = Class::E2;
                info.cb_entry = true;
                info.cb_words = static_cast<u32>(head >> 32);
                info.sink_first = ESplit::KindOf(back_engine) == EngineKind::Maxwell3D;
                break;
            case Kind::Call:
                switch (StageTagOf(head)) {
                case StageTag::Cache:
                    info.cls = Class::E2;
                    break;
                case StageTag::Channel:
                    // E1's (GPU::Impl's bound id): E2 takes the bind in order; a real switch's
                    // bind lends, and the lend parks both stages (fritchii_e_split.h).
                    info.channel = true;
                    break;
                case StageTag::Level:
                    join(JoinCause::UntaggedCall); // no tag: unclassified
                    break;
                default:
                    break; // Writes, FinishMacro, DrawLoop, Pass, Ops: register and draw work
                }
                break;
            case Kind::Method:
            case Kind::Multi: {
                const EngineKind engine_kind = ESplit::KindOf(back_engine);
                if (engine_kind == EngineKind::Compute) {
                    info.cls = Class::E2; // forwarded whole (dispatches, QMD reads, its fences)
                    break;
                }
                const u32 method = static_cast<u32>((head >> 8) & 0xffff);
                const bool multi = kind == Kind::Multi;
                // A register store (sunk until the next executed method).
                const bool store = !multi && ((head & Sink) != 0 ||
                                              ((head & Direct) == 0 && back_engine != nullptr &&
                                               !back_engine->execution_mask[method]));
                if (engine_kind != EngineKind::Maxwell3D) {
                    if (!store) {
                        join(JoinCause::OtherEngine); // 2D, DMA, inline-to-memory
                    }
                    break;
                }
                if (method - cb_data < 16 && (head & Sink) == 0) {
                    info.cls = Class::E2;
                    info.cb_entry = true;
                    info.sink_first = true;
                    info.cb_words = multi                         ? static_cast<u32>(head >> 32)
                                    : (head & HasSegment) != 0 ? 1U
                                                               : 0U; // the run Execute takes
                    break;
                }
                if (store) {
                    break;
                }
                switch (method) {
                case MAXWELL3D_REG_INDEX(report_semaphore.query):
                case MAXWELL3D_REG_INDEX(sync_info):
                case MAXWELL3D_REG_INDEX(clear_report_value):
                case MAXWELL3D_REG_INDEX(wait_for_idle):
                case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
                case MAXWELL3D_REG_INDEX(fragment_barrier):
                case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
                    if (!multi) {
                        info.cls = Class::E2; // guest-visible or the caches' order
                        info.sink_first = true;
                    }
                    break;
                case MAXWELL3D_REG_INDEX(launch_dma):
                case MAXWELL3D_REG_INDEX(inline_data):
                    join(JoinCause::Upload);
                    break;
                default:
                    break; // draws, binds' register half, conditions, macros, clears: E1
                }
                break;
            }
            }
            return info;
        };
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
            // The entries Execute may take with this one (constant buffer words) stop at a level
            // run's edge, so each entry runs lent or served as a whole.
            u64 end = available;
            if (ring_edges) {
                if (read_position >= ring_next_edge) [[unlikely]] {
                    link.BackEntry(read_position, limits);
                    ring_run = MemoryStageLink::BackInRun();
                    ring_next_edge = read_position < limits.lent_from    ? limits.lent_from
                                     : read_position < limits.lent_until ? limits.lent_until
                                                                         : ~u64{0};
                }
                end = (std::min)(end, ring_next_edge);
            } else if (ring_on) [[unlikely]] {
                link.BackEntry(read_position, limits);
                if (read_position < limits.lent_from) {
                    end = (std::min)(end, limits.lent_from);
                } else if (read_position < limits.lent_until) {
                    end = (std::min)(end, limits.lent_until);
                }
            } else if (MemStageShadow::g_on.load(std::memory_order_relaxed)) [[unlikely]] {
                // FRITCHII_MEMSTAGE_SHADOW: the memory stage's inputs for this entry, compared at
                // the back's draws (memstage_shadow.cpp). Off: this one test per entry.
                MemStageShadow::Before(read_position);
            }
            u64 method_before = 0;
            size_t sink_before = 0;
            if (entry_price) [[unlikely]] {
                method_before = Nx2Profile::t_data
                                    .exclusive_ns[static_cast<size_t>(Nx2Profile::Zone::Method)];
                // Row 4b: the state writes a 3D method entry finds sunk (it consumes them first).
                if (static_cast<Kind>(ring[index] & 0xf) == Kind::Method &&
                    back_engine != nullptr && back_engine->fast_maxwell3d) {
                    sink_before = static_cast<Tegra::Engines::Maxwell3D*>(back_engine)
                                      ->method_sink.size();
                }
            }
            if (esplit != 0) [[unlikely]] {
                const ESplit::EntryInfo info = esplit_decode(ring[index]);
                // At 2 and 3 E1 applies the sunk register writes before it forwards an E2 entry,
                // as the split would; Execute consumes the sink first thing for these entries, so
                // the order and the values stay the same.
                if (esplit >= 2 && info.sink_first && back_engine != nullptr) {
                    back_engine->ConsumeSink(*system);
                }
                ESplit::BeforeEntry(info);
            }
            const size_t size =
                Execute(&ring[index], std::min<u64>(end - read_position, RingWords - index));
            if (esplit != 0) [[unlikely]] {
                ESplit::AfterEntry(size);
            }
            if (entry_hook) [[unlikely]] {
                const u64 head = ring[index];
                if (entry_price) {
                    u32 entries = 1;
                    const FritchiiPrice::Entry kind = entry_kind(head, size, entries);
                    const u64 added =
                        Nx2Profile::t_data
                            .exclusive_ns[static_cast<size_t>(Nx2Profile::Zone::Method)] -
                        method_before;
                    FritchiiPrice::g_entries.Note(static_cast<size_t>(kind), entries, added);
                    if (kind == FritchiiPrice::Entry::Method3D) {
                        const size_t sink_after =
                            static_cast<Tegra::Engines::Maxwell3D*>(back_engine)
                                ->method_sink.size();
                        FritchiiPrice::NoteMethod3D(static_cast<u32>((head >> 8) & 0xffff), added,
                                                    sink_after < sink_before ? sink_before : 0);
                    }
                }
                if (entry_delay) {
                    if (const u32 entries = method_entries(head, size); entries != 0) {
                        DelayBackIf(BackDelayKind::Methods, entries);
                    }
                }
            }
            read_position += size != 0 ? size : RingWords - index;
            if (ring_on && (!ring_edges || (ring_run && read_position >= limits.lent_until)) &&
                link.BackAfter(read_position, limits)) [[unlikely]] {
                ring_run = false;
                // The run's last entry ran (spec 1.5, plan 5.2): no register write still waits in
                // the sink, then the run's collected ranges; then `consumed` below.
                if (back_engine != nullptr) {
                    back_engine->ConsumeSink(*system);
                }
                link.BackEndRun();
            }
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
                if (entry_price) [[unlikely]] {
                    FritchiiPrice::BackReached(read_position);
                }
                StoreConsumed(read_position);
                last_store = read_position;
                if (ring_on) {
                    link.NoteConsumed(read_position);
                }
            }
            if (!ring_on && link.BackRingStartWanted()) [[unlikely]] {
                break; // THREAD=2 starts at the loop top
            }
        }
        CheckFencesQuiet(read_position);
        if (entry_price) [[unlikely]] {
            FritchiiPrice::BackReached(read_position);
        }
        StoreConsumed(read_position);
        if (ring_on) {
            link.NoteConsumed(read_position);
        }
    }
    back_running.store(false, std::memory_order_seq_cst);
    front_wake.fetch_add(1, std::memory_order_seq_cst);
    front_wake.notify_all();
}

} // namespace VideoCommon
