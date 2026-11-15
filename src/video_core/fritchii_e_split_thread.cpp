// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// FRITCHII_ESPLIT: the E split's thread (fritchii_e_split_thread.h; work\E_SPLIT_DESIGN.md 2b).
//
// S1, lockstep (FRITCHII_ESPLIT_DEPTH=0). Who owns what:
//   E1 (GpuSplit's back thread): ring A, engine A (registers, dirty flags, draw manager, sink),
//      the pipeline lookup, the records (fritchii_e_split.cpp's pending sets), the hand-over ring
//      below (it writes jobs and `posted`).
//   E2 (this file's thread): E2's view, every job's run, `done`.
// A hand-over: E1 takes a job slot, builds the record (when the work reads E2's view), moves its
// per-call thread state into the job (one follower's, the renderer's fill state), hands a lock it
// holds over, publishes `posted` and waits for `done` to reach the job. E2 takes the job, installs
// the state, applies the record to its view (1 job in 64: the view is compared with the engine),
// runs the work, takes the state back into the job, publishes `done`. E1 installs the state the
// work left and goes on. One of the two threads runs at a time and each piece of work runs on the
// same objects in the same order as on one thread: the result is today's.
//
// Waits: both sides poll first (the haltpoll rule, Common::NextPollWindow) and then sleep on the
// position (std::atomic::wait), with a "sleeping" flag each so a publish wakes only a sleeper.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

#include <fmt/format.h>

#include "common/logging.h"
#include "common/spin_recursive_mutex.h"
#include "common/thread.h"
#include "core/core.h"
#include "video_core/back_read_audit.h"
#include "video_core/fritchii_e_split.h"
#include "video_core/fritchii_e_split_thread.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_profile.h"
#include "video_core/one_follower.h"
#include "video_core/rasterizer_interface.h"

namespace VideoCommon::ESplit {

namespace {

namespace OneF = VideoCommon::OneFollower;
namespace Profile = VideoCommon::Nx2Profile;
using Maxwell3D = Tegra::Engines::Maxwell3D;

void Pause() noexcept {
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

constexpr u64 Slots = 256; ///< hand-over ring (lockstep uses one at a time)
constexpr u64 SampleMask = 63; ///< 1 hand-over in 64 compared
constexpr size_t HandCount = static_cast<size_t>(Hand::Count);
constexpr size_t JoinCount = static_cast<size_t>(Join::Count);
constexpr size_t TouchCount = static_cast<size_t>(Touch::Count);
constexpr std::array<const char*, HandCount> HandNames{"entries", "flushes", "draw tails", "binds",
                                                       "stores"};
constexpr std::array<const char*, JoinCount> JoinNames{"entries", "decisions", "conditions",
                                                       "idle",    "pushes",    "lockstep"};
constexpr std::array<const char*, TouchCount> TouchNames{"cache lock", "memory read",
                                                         "memory write"};

struct Job {
    void (*function)(void*);
    void* argument;
    u64 seq;
    u64 position; ///< ring A position of the entry the work belongs to
    HandInfo info;
    RecordKind record_kind;
    u32 record_bytes;
    bool sample;  ///< compare E2's view after the record (lockstep: the engine is at E2's place)
    bool view_ok; ///< E2: the compare found the view equal (the fill may read it)
    OneF::EngineTls tls;
    alignas(16) std::array<u8, FillStateBytes> fill_state;
};

/// Records: one per job slot would be 64 KiB each; lockstep needs one at a time.
struct RecordBuffer {
    std::unique_ptr<u8[]> bytes;
    size_t size{};
};

/// Per-window counters. E1's: plain (E1 writes and reports). E2's: single writer, relaxed.
struct E1Counts {
    std::array<u64, HandCount> hands{};
    std::array<u64, JoinCount> joins{};
    std::array<u64, TouchCount> touches{};
    u64 wait_ticks{};  ///< E1 from the publish to `done`
    u64 seeds{};
    u64 lost{};
    u64 samples{};
};
struct E2Counts {
    std::atomic<u64> job_ticks{};
    std::atomic<u64> wait_ticks{};
    std::atomic<u64> view_differs{};
    std::atomic<u64> compares{};
    std::atomic<u64> view_fills{};
    std::atomic<u64> after_checks{};
    std::atomic<u64> after_differs{};
    std::atomic<u64> order{};
};

void Bump(std::atomic<u64>& counter, u64 amount = 1) noexcept {
    counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}
u64 Take(std::atomic<u64>& counter) noexcept {
    return counter.exchange(0, std::memory_order_relaxed);
}

struct Shared {
    // E1 -> E2
    alignas(128) std::atomic<u64> posted{};
    std::atomic<bool> e2_sleeping{};
    std::atomic<u32> e2_wake{};
    std::atomic<bool> stop{};
    // E2 -> E1
    alignas(128) std::atomic<u64> done{};
    std::atomic<bool> e1_sleeping{};
    std::atomic<u32> e1_wake{};
    alignas(128) std::array<Job, Slots> jobs{};
    RecordBuffer record;
    Core::System* system{};
    void (*adopt)(void*){};
    void* adopt_argument{};
    std::thread::id e2_id{};
    std::thread::id e1_id{};
    std::jthread thread;
    std::atomic<bool> e2_ready{};
};

Shared g;
E1Counts g_e1;
E2Counts g_e2;
// E1's own
u64 g_next_seq = 1;
u32 g_rng = 0x9e3779b9u;
u64 g_batches{};
u64 g_since_ns{};
u64 g_since_ticks{};
u64 g_touch_logged{};
std::chrono::nanoseconds g_e1_poll{std::chrono::microseconds{50}};
// E2's own
u64 g_e2_last_seq{};
u64 g_e2_last_position{};
u64 g_order_logged{};
Job* g_current{}; ///< E2: the job it runs
std::chrono::nanoseconds g_e2_poll{std::chrono::microseconds{20}};

/// E1 polls up to 200 us for a lockstep hand-over (the work is a few us; a sleep and a wake
/// would cost more than the work), E2 up to 50 us for the next job.
constexpr std::chrono::nanoseconds E1PollLimit{std::chrono::microseconds{200}};
constexpr std::chrono::nanoseconds E2PollLimit{std::chrono::microseconds{50}};

[[nodiscard]] bool Sample() noexcept {
    u32 x = g_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rng = x;
    return (x & SampleMask) == 0;
}

/// Waits until `ready()`: polls for `poll`, then sleeps on `wake` with `sleeping` set (the other
/// side bumps `wake` after its store when it sees `sleeping`; both seq_cst, so one of the two
/// sees the other). The poll adapts (haltpoll). Returns the ticks waited.
template <typename Ready>
u64 WaitFor(Ready&& ready, std::atomic<u32>& wake, std::atomic<bool>& sleeping,
            std::chrono::nanoseconds& poll, std::chrono::nanoseconds limit) {
    if (ready()) {
        return 0;
    }
    const u64 start_ticks = Profile::Ticks();
    const auto start = std::chrono::steady_clock::now();
    const auto poll_end = start + poll;
    u32 spins = 0;
    while (!ready()) {
        Pause();
        if ((++spins & 63) == 0 && std::chrono::steady_clock::now() >= poll_end) {
            while (!ready()) {
                const u32 seen = wake.load(std::memory_order_seq_cst);
                sleeping.store(true, std::memory_order_seq_cst);
                if (!ready()) {
                    wake.wait(seen, std::memory_order_seq_cst);
                }
                sleeping.store(false, std::memory_order_relaxed);
            }
            break;
        }
    }
    const auto waited = std::chrono::steady_clock::now() - start;
    poll = Common::NextPollWindow(poll, std::chrono::duration_cast<std::chrono::nanoseconds>(waited),
                                  limit);
    return Profile::Ticks() - start_ticks;
}

std::string Site(const void* address) {
#ifdef _WIN32
    if (address != nullptr) {
        HMODULE module{};
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               static_cast<LPCSTR>(address), &module)) {
            char name[MAX_PATH] = "?";
            GetModuleFileNameA(module, name, MAX_PATH);
            const char* base = std::strrchr(name, '\\');
            return fmt::format("{}+{:#x}", base != nullptr ? base + 1 : name,
                               reinterpret_cast<uintptr_t>(address) -
                                   reinterpret_cast<uintptr_t>(module));
        }
    }
#endif
    return fmt::format("{}", address);
}

std::string Stack() {
#ifdef _WIN32
    void* frames[10];
    const USHORT count = RtlCaptureStackBackTrace(2, 10, frames, nullptr);
    std::string text;
    for (USHORT i = 0; i < count; ++i) {
        text += (i == 0 ? "" : " < ") + Site(frames[i]);
    }
    return text;
#else
    return "(no stack)";
#endif
}

/// BackReadAudit's watcher: E's game memory reads.
void ReadTouch(const void* caller) {
    if (t_role == 1 && t_join == 0 && g_thread) [[unlikely]] {
        NoteTouchImpl(Touch::MemoryRead, caller);
    }
}

// --- E2 --------------------------------------------------------------------------------------------

void RunJob(Job& job) {
    const Profile::Scope zone{Profile::Zone::ESplitJob, Profile::Enabled(), false};
    const u64 start = Profile::Ticks();
    // E2Order: hand-overs run in the order E1 made them, at non-decreasing ring positions.
    if (job.seq != g_e2_last_seq + 1 || job.position < g_e2_last_position) [[unlikely]] {
        Bump(g_e2.order);
        if (++g_order_logged <= 16) {
            LOG_ERROR(HW_GPU,
                      "E split thread: E2Order: job {} at ring position {:#x} after job {} at "
                      "{:#x} ({} since boot)",
                      job.seq, job.position, g_e2_last_seq, g_e2_last_position, g_order_logged);
        }
    }
    g_e2_last_seq = job.seq;
    g_e2_last_position = job.position;
    OneF::PutEngineTls(job.tls);
    if (g_fill_state.install != nullptr) {
        g_fill_state.install(job.fill_state.data(), false);
    }
    job.view_ok = false;
    if (job.record_kind != RecordKind::None) {
        if (const Maxwell3D* const engine = OneF::Engine()) {
            const u64 differs =
                ThreadConsume(job.record_kind, g.record.bytes.get(), *engine, job.sample,
                              job.info.draw, job.info.list_args);
            if (job.sample && job.record_kind == RecordKind::Record &&
                ThreadView().seeded) {
                Bump(g_e2.compares);
                Bump(g_e2.view_differs, differs);
                job.view_ok = differs == 0;
            }
        }
    }
    g_current = &job;
    job.function(job.argument);
    g_current = nullptr;
    if (job.info.cb_entry) {
        const u64 words = job.info.cb_words != 0 ? job.info.cb_words
                          : job.info.cb_size != nullptr ? *job.info.cb_size
                                                        : 0;
        ThreadCbAdvance(words);
    }
    OneF::TakeEngineTls(job.tls);
    if (g_fill_state.capture != nullptr) {
        g_fill_state.capture(job.fill_state.data(), false);
    }
    Bump(g_e2.job_ticks, Profile::Ticks() - start);
}

void E2Main(std::stop_token) {
    Common::SetCurrentThreadName("GPUBackE2");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();
    g.system->RegisterHostThread();
    if (g.adopt != nullptr) {
        g.adopt(g.adopt_argument); // GpuSplit: OnBackThread
    }
    OneF::MarkEngineHelperThread();
    ThreadMark();
    t_role = 2;
    g.e2_id = std::this_thread::get_id();
    g.e2_ready.store(true, std::memory_order_release);
    g.e2_ready.notify_all();
    u64 next = 1;
    while (true) {
        if (g.posted.load(std::memory_order_acquire) < next) {
            if (g.stop.load(std::memory_order_acquire)) {
                break;
            }
            const Profile::Scope zone{Profile::Zone::ESplitWait, Profile::Enabled(), false};
            Bump(g_e2.wait_ticks,
                 WaitFor(
                     [next] {
                         return g.posted.load(std::memory_order_acquire) >= next ||
                                g.stop.load(std::memory_order_acquire);
                     },
                     g.e2_wake, g.e2_sleeping, g_e2_poll, E2PollLimit));
            continue;
        }
        RunJob(g.jobs[next % Slots]);
        g.done.store(next, std::memory_order_seq_cst);
        if (g.e1_sleeping.load(std::memory_order_seq_cst)) {
            g.e1_wake.fetch_add(1, std::memory_order_seq_cst);
            g.e1_wake.notify_all();
        }
        ++next;
        Profile::Report("GPU E2 thread");
    }
}

// --- E1 --------------------------------------------------------------------------------------------

void Publish(u64 seq) {
    g.posted.store(seq, std::memory_order_seq_cst);
    if (g.e2_sleeping.load(std::memory_order_seq_cst)) {
        g.e2_wake.fetch_add(1, std::memory_order_seq_cst);
        g.e2_wake.notify_all();
    }
}

void WaitDone(u64 seq) {
    g_e1.wait_ticks += WaitFor([seq] { return g.done.load(std::memory_order_acquire) >= seq; },
                               g.e1_wake, g.e1_sleeping, g_e1_poll, E1PollLimit);
}

void Report(u64 now) {
    const double window = static_cast<double>(now - g_since_ns) / 1e9;
    const double ns_per_tick =
        static_cast<double>(now - g_since_ns) /
        static_cast<double>((std::max)(Profile::Ticks() - g_since_ticks, u64{1}));
    const auto rate = [window](u64 count) { return static_cast<double>(count) / window; };
    const auto ms = [&](u64 ticks) {
        return static_cast<double>(ticks) * ns_per_tick / 1e6 / window;
    };
    const E1Counts c = g_e1;
    g_e1 = {};
    const u64 job_ticks = Take(g_e2.job_ticks);
    const u64 e2_wait = Take(g_e2.wait_ticks);
    const u64 view_differs = Take(g_e2.view_differs);
    const u64 compares = Take(g_e2.compares);
    const u64 view_fills = Take(g_e2.view_fills);
    const u64 after_checks = Take(g_e2.after_checks);
    const u64 after_differs = Take(g_e2.after_differs);
    const u64 order = Take(g_e2.order);
    u64 hands = 0;
    std::string hand_text;
    for (size_t i = 0; i < HandCount; ++i) {
        hands += c.hands[i];
        hand_text += fmt::format("{}{} {:.0f}", i == 0 ? "" : ", ", HandNames[i], rate(c.hands[i]));
    }
    u64 joins = 0;
    std::string join_text;
    for (size_t i = 0; i < JoinCount; ++i) {
        joins += c.joins[i];
        join_text += fmt::format("{}{} {:.0f}", i == 0 ? "" : ", ", JoinNames[i], rate(c.joins[i]));
    }
    const double e1_wait = ms(c.wait_ticks);
    const double e2_busy = ms(job_ticks);
    const double cost = e1_wait - e2_busy;
    LOG_INFO(HW_GPU,
             "E split thread (FRITCHII_ESPLIT=1, depth {}) 10 s: hand-overs {:.0f}/s ({}); E2 busy "
             "{:.1f} ms/s, E2 waiting {:.1f} ms/s; E1 in hand-overs {:.1f} ms/s = E2's runs + "
             "hand-over cost {:.1f} ms/s ({:.2f} us each); joins {:.0f}/s ({})",
             g_async ? "async" : "0 (lockstep)", rate(hands), hand_text, e2_busy, ms(e2_wait),
             e1_wait, cost, hands != 0 ? cost * 1000.0 * window / static_cast<double>(hands) : 0.0,
             rate(joins), join_text);
    const u64 touches = c.touches[0] + c.touches[1] + c.touches[2];
    LOG_INFO(HW_GPU,
             "E split thread 10 s: checks (must be 0): ESplitView {} in {} compares (1 hand-over "
             "in 64), ESplitInputs {} in {} fills from the view, E1Touch {} ({} {}, {} {}, {} {}), "
             "E2Order {}; seeds {}, untracked hand-overs {}",
             view_differs, compares, after_differs, after_checks, touches, TouchNames[0],
             c.touches[0], TouchNames[1], c.touches[1], TouchNames[2], c.touches[2], order, c.seeds,
             c.lost);
    LOG_INFO(HW_GPU,
             "E split thread 10 s: fills from E2's view {:.0f}/s (the sampled draws whose compare "
             "was clean), records sampled {:.0f}/s",
             rate(view_fills), rate(c.samples));
}

} // Anonymous namespace

bool ThreadWanted() noexcept {
    static const bool wanted =
        VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_esplit, "FRITCHII_ESPLIT") == 1;
    return wanted;
}

void StartE2(Core::System& system, void (*adopt)(void*), void* adopt_argument) {
    if (!OneF::OnEngineThread()) {
        LOG_ERROR(HW_GPU, "E split thread: FRITCHII_ESPLIT=1 needs FRITCHII_ONEF=1; E2 not started");
        return;
    }
    g.system = &system;
    g.adopt = adopt;
    g.adopt_argument = adopt_argument;
    g.e1_id = std::this_thread::get_id();
    g.record.size = RecordRoom();
    g.record.bytes = std::make_unique<u8[]>(g.record.size);
    t_role = 1;
    ThreadMark();
    ThreadRecordsOn();
    // E1's reads of game memory, through the back read audit's funnel (its own tables count only
    // with its switch).
    BackReadAudit::g_touch_hook = &ReadTouch;
    BackReadAudit::enabled = true;
    g.thread = std::jthread{E2Main};
    g.e2_ready.wait(false, std::memory_order_acquire);
    g_since_ns = Profile::Now();
    g_since_ticks = Profile::Ticks();
    g_thread = true;
    LOG_INFO(HW_GPU,
             "E split thread: FRITCHII_ESPLIT=1, E2 (GPUBackE2) started; FRITCHII_ESPLIT_DEPTH=0 "
             "runs lockstep (E1 waits for each hand-over)");
}

void StopE2() {
    if (!g_thread) {
        return;
    }
    g_thread = false;
    g.stop.store(true, std::memory_order_seq_cst);
    g.e2_wake.fetch_add(1, std::memory_order_seq_cst);
    g.e2_wake.notify_all();
    g.thread = {};
    BackReadAudit::g_touch_hook = nullptr;
}

void ThreadBatch() noexcept {
    if (!g_thread) {
        return;
    }
    if ((++g_batches & 63) == 0) [[unlikely]] {
        const u64 now = Profile::Now();
        if (now - g_since_ns >= 10'000'000'000ULL) {
            Report(now);
            g_since_ns = now;
            g_since_ticks = Profile::Ticks();
        }
    }
}

void Run(const HandInfo& info, void (*function)(void*), void* argument) noexcept {
    const Profile::Scope zone{Profile::Zone::ESplitHandOver, Profile::Enabled(), false};
    const u64 seq = g_next_seq++;
    Job& job = g.jobs[seq % Slots];
    job.function = function;
    job.argument = argument;
    job.seq = seq;
    job.position = GpuSplit::Get().BackEntryPosition();
    job.info = info;
    job.record_kind = RecordKind::None;
    job.record_bytes = 0;
    job.sample = false;
    ++g_e1.hands[static_cast<size_t>(info.hand)];
    if (info.record) {
        if (const Maxwell3D* const engine = OneF::Engine()) {
            job.record_kind = ThreadBuild(*engine, info.draw, info.list_args, g.record.bytes.get(),
                                          job.record_bytes);
            job.sample = Sample();
            g_e1.samples += job.sample ? 1 : 0;
            g_e1.seeds += job.record_kind == RecordKind::Seed ? 1 : 0;
            g_e1.lost += job.record_kind == RecordKind::Lost ? 1 : 0;
        }
    }
    OneF::TakeEngineTls(job.tls);
    if (g_fill_state.capture != nullptr) {
        g_fill_state.capture(job.fill_state.data(), true);
    }
    if (info.held != nullptr) {
        info.held->HandOver(g.e2_id);
    }
    Publish(seq);
    WaitDone(seq);
    if (info.held != nullptr) {
        info.held->HandOver(g.e1_id);
    }
    OneF::PutEngineTls(job.tls);
    if (g_fill_state.install != nullptr) {
        g_fill_state.install(job.fill_state.data(), true);
    }
}

void HandBind(VideoCore::RasterizerInterface& rasterizer, std::size_t stage, u32 slot,
              bool enabled, u64 address, u32 size) noexcept {
    struct Bind {
        VideoCore::RasterizerInterface* rasterizer;
        std::size_t stage;
        u64 address;
        u32 slot;
        u32 size;
        bool enabled;
    } bind{&rasterizer, stage, address, slot, size, enabled};
    Run(HandInfo{.hand = Hand::Bind}, [](void* raw) {
        const Bind& b = *static_cast<const Bind*>(raw);
        if (!b.enabled) {
            b.rasterizer->DisableGraphicsUniformBuffer(b.stage, b.slot);
        } else {
            b.rasterizer->BindGraphicsUniformBuffer(b.stage, b.slot, b.address, b.size);
        }
    }, &bind);
}

void BeginJoin(Join why) noexcept {
    if (t_join++ == 0) {
        ++g_e1.joins[static_cast<size_t>(why)];
    }
}

View* FillView(const Maxwell3D& engine) noexcept {
    if (t_role != 2 || g_current == nullptr || g_current->info.hand != Hand::DrawTail ||
        !g_current->view_ok) {
        return nullptr;
    }
    if (!engine.draw_manager.draw_state.inline_index_draw_indexes.empty()) {
        return nullptr;
    }
    Bump(g_e2.view_fills);
    return &ThreadView();
}

void FillDone(const Maxwell3D& engine, const View* used) noexcept {
    if (t_role == 2 || (t_role == 1 && t_join != 0)) {
        const u64 before = ThreadAfterDiffers();
        AfterFill(engine, used);
        if (used != nullptr && t_role == 2) {
            Bump(g_e2.after_checks);
            Bump(g_e2.after_differs, ThreadAfterDiffers() - before);
        }
    }
}

void NoteTouchImpl(Touch kind, const void* caller) noexcept {
    ++g_e1.touches[static_cast<size_t>(kind)];
    if (++g_touch_logged <= 16) {
        LOG_ERROR(HW_GPU, "E split thread: E1Touch ({}, {} since boot) at {}: {}",
                  TouchNames[static_cast<size_t>(kind)], g_touch_logged,
                  GpuSplit::Get().BackEntryPosition(),
                  caller != nullptr ? Site(caller) : Stack());
    }
}

} // namespace VideoCommon::ESplit
