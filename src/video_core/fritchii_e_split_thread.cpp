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
// S2, async (FRITCHII_ESPLIT_DEPTH=N > 0). E1 posts draw tails and binds and goes on, up to N
// draw tails ahead of E2. The rules that keep the two threads off each other's state:
//   - Everything that is not an async job is still a lockstep hand-over or a join, and both first
//     drain E2 (E1 waits for `done` to reach `posted`).
//   - The buffer flags (IndexBuffer, VertexBuffers, VertexBuffer0..31) are E2's: E1 takes the
//     ones raised on its engine at each async draw record and clears them there (the record
//     carries them, E2 ORs them into its view: the take of fritchii_e_split.h). Wherever E2's
//     work runs on the engine (a lockstep hand-over, a join, E1's own lend or push) the view's
//     flags are copied to the engine first and back after (SyncFlagsToEngine / ToView).
//   - An async fill reads only E2's view (SetInputsOverride, the bindings, a list draw's indirect
//     parameters, the render target formats); the repack layout's VertexInput raise goes to E1
//     (DeferVertexInput, raised at E1's next draw, join or push).
//   - R's draw packet is packed by E1 at the record (OneFollower::PackDraw: the spans are E1's
//     registers at the draw) and pushed by E2 after the fill (PushDrawPacked): R's packet queue
//     keeps one producer, E2, while E2 is behind; E1 drains E2 before any lend or push of its own.
//   - A lend E2 makes while E1 runs ahead holds E1 at its next safe point (each ring entry, and
//     every wait for E2 counts as held): R's lent call then sees no E1 work in flight. Such a lend
//     packs no new spans for R from E1 (its view stays at E2's place: OneFollower::Lend skips the
//     pack on E2 in S2) and the view keeps its own flags after it (EndLend's sync skipped).
//   - E2's position: async jobs carry the ring position of their entry; GpuSplit::
//     BackEntryPosition returns it on E2; `consumed` is stored by an async job (the front's waits
//     and ring A's room then follow E2), and at a batch's end after the drain.
// Async draws need: rule 3's pipeline (FRITCHII_ONEF_E_PIPELINE), no texel buffers, ring B by
// value, no marks ahead, no inline index bytes, no handle program keep on a pipeline with handle
// programs, no price timers, a record (no seed due). The rest go lockstep.
//
// Waits: both sides poll first (the haltpoll rule, Common::NextPollWindow) and then sleep on a
// wake word (std::atomic::wait), with a "sleeping" flag each so a publish wakes only a sleeper.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
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

constexpr u64 Slots = 64;          ///< hand-over ring
constexpr int MaxDepth = 48;       ///< FRITCHII_ESPLIT_DEPTH's cap (draw tails ahead)
constexpr u64 SampleMask = 63;     ///< 1 hand-over in 64 compared (or hashed)
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
    void* argument; ///< lockstep: E1's; async: the payload below
    u64 seq;
    u64 position; ///< ring A position of the entry the work belongs to
    u64 hash;     ///< async draw, sampled: E1's hash of what the fill reads
    HandInfo info;
    RecordKind record_kind;
    u32 record_bytes;
    bool async;
    bool sample;  ///< lockstep: compare the view; async draw: compare the hash
    bool view_ok; ///< E2: the fill may read the view
    OneF::EngineTls tls;
    alignas(16) std::array<u8, FillStateBytes> fill_state;
    alignas(16) std::array<u8, AsyncPayloadBytes> payload;
};

/// Per-window counters. E1's: plain (E1 writes and reports). E2's: single writer, relaxed.
struct E1Counts {
    std::array<u64, HandCount> hands{};
    std::array<u64, JoinCount> joins{};
    std::array<u64, TouchCount> touches{};
    u64 wait_ticks{}; ///< E1 waiting for E2 (lockstep runs, drains, room)
    u64 hold_ticks{}; ///< E1 held for E2's lends
    u64 seeds{};
    u64 lost{};
    u64 samples{};
    u64 async_draws{};
    u64 visible{};
    u64 front_ahead{};
    u64 depth_sum{}; ///< draw tails outstanding at each async post
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
    std::atomic<u64> hash_checks{};
    std::atomic<u64> hash_differs{};
    std::atomic<u64> holds{};
    std::atomic<u64> hold_spin_ticks{};
    std::atomic<u64> room_waits{};
    std::atomic<u64> late_flushes{};
    std::atomic<u64> refused{};
};

void Bump(std::atomic<u64>& counter, u64 amount = 1) noexcept {
    counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}
u64 TakeCount(std::atomic<u64>& counter) noexcept {
    return counter.exchange(0, std::memory_order_relaxed);
}

struct SlotBuffers {
    std::unique_ptr<u8[]> record;
    std::unique_ptr<u8[]> packet;
};

struct Shared {
    // E1 -> E2
    alignas(128) std::atomic<u64> posted{};
    std::atomic<bool> e2_sleeping{};
    std::atomic<u32> e2_wake{};
    std::atomic<bool> stop{};
    // E2 -> E1
    alignas(128) std::atomic<u64> done{};
    std::atomic<u64> draws_done{}; ///< async draw tails E2 finished
    std::atomic<bool> e1_sleeping{};
    std::atomic<u32> e1_wake{};
    /// E1 is at a safe point (waiting for E2, or held): an E2 lend may run.
    alignas(128) std::atomic<bool> e1_parked{};
    /// E2's async fills raised this flag (index + 1) for E1's engine (VertexInput).
    std::atomic<u32> back_raise{};
    alignas(128) std::array<Job, Slots> jobs{};
    std::array<SlotBuffers, Slots> buffers;
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
u64 g_draws_posted{};
int g_depth{};
Job* g_open{}; ///< a slot BeginAsync* gave, not posted yet
u32 g_rng = 0x9e3779b9u;
u64 g_batches{};
u64 g_since_ns{};
u64 g_since_ticks{};
u64 g_touch_logged{};
u64 g_visible_logged{};
u64 g_front_logged{};
std::chrono::nanoseconds g_e1_poll{std::chrono::microseconds{50}};
std::chrono::nanoseconds g_hold_poll{std::chrono::microseconds{20}};
// E2's own
u64 g_e2_last_seq{};
u64 g_e2_last_position{};
u64 g_order_logged{};
u64 g_hash_logged{};
u64 g_refused_logged{};
Job* g_current{}; ///< E2: the job it runs
std::chrono::nanoseconds g_e2_poll{std::chrono::microseconds{20}};

/// E1 polls up to 200 us for a lockstep hand-over (the work is a few us; a sleep and a wake
/// would cost more than the work), up to 20 us in S2 (where it waits only for room or a drain:
/// sleeping waits, so the rest of the machine does not grow). E2 polls up to 50 us for the next
/// job in S1, 20 us in S2.
constexpr std::chrono::nanoseconds E1PollLimit{std::chrono::microseconds{200}};
constexpr std::chrono::nanoseconds AsyncPollLimit{std::chrono::microseconds{20}};
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

void WakeE1() noexcept {
    if (g.e1_sleeping.load(std::memory_order_seq_cst)) {
        g.e1_wake.fetch_add(1, std::memory_order_seq_cst);
        g.e1_wake.notify_all();
    }
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
    t_e2_position = job.position;
    OneF::PutEngineTls(job.tls);
    if (g_fill_state.install != nullptr) {
        g_fill_state.install(job.fill_state.data(), false);
    }
    job.view_ok = false;
    if (job.record_kind != RecordKind::None) {
        if (const Maxwell3D* const engine = OneF::Engine()) {
            const u8* const record = g.buffers[job.seq % Slots].record.get();
            // Async: the engine is past E2's place, nothing to compare with (the hash instead).
            const bool compare = job.sample && !job.async;
            const u64 differs = ThreadConsume(job.record_kind, record, *engine, compare,
                                              job.info.draw, job.info.list_args);
            if (compare && job.record_kind == RecordKind::Record && ThreadView().seeded) {
                Bump(g_e2.compares);
                Bump(g_e2.view_differs, differs);
                job.view_ok = differs == 0;
            }
            if (job.async && job.info.draw) {
                job.view_ok = ThreadView().seeded && ThreadView().engine == engine;
                if (job.sample) {
                    Bump(g_e2.hash_checks);
                    const u64 hash = HashView(job.info.list_args);
                    if (hash != job.hash) [[unlikely]] {
                        Bump(g_e2.hash_differs);
                        if (++g_hash_logged <= 16) {
                            LOG_ERROR(HW_GPU,
                                      "E split thread: ESplitHash: the view's hash {:#x} differs "
                                      "from E1's {:#x} at ring position {:#x} ({} since boot)",
                                      hash, job.hash, job.position, g_hash_logged);
                        }
                    }
                }
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
    // Lockstep: the state goes back to E1 with the job. Async: E1 moved on; E2's copy is dropped
    // (the next job installs its own).
    OneF::TakeEngineTls(job.tls);
    if (g_fill_state.capture != nullptr) {
        g_fill_state.capture(job.fill_state.data(), false);
    }
    if (job.async && job.info.hand == Hand::DrawTail) {
        g.draws_done.store(g.draws_done.load(std::memory_order_relaxed) + 1,
                           std::memory_order_release);
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
                     g.e2_wake, g.e2_sleeping, g_e2_poll,
                     g_async ? AsyncPollLimit : E2PollLimit));
            continue;
        }
        RunJob(g.jobs[next % Slots]);
        g.done.store(next, std::memory_order_seq_cst);
        WakeE1();
        ++next;
        Profile::Report("GPU E2 thread");
    }
}

/// E2, S2: E1 held at a safe point for an E2 lend.
void HoldE1() noexcept {
    g_hold.store(1, std::memory_order_seq_cst);
    const u64 start = Profile::Ticks();
    u32 spins = 0;
    while (!g.e1_parked.load(std::memory_order_seq_cst)) {
        if (++spins < 4096) {
            Pause();
        } else {
            std::this_thread::yield();
        }
    }
    Bump(g_e2.holds);
    Bump(g_e2.hold_spin_ticks, Profile::Ticks() - start);
}

void ReleaseE1() noexcept {
    g_hold.store(0, std::memory_order_seq_cst);
    g.e1_wake.fetch_add(1, std::memory_order_seq_cst);
    g.e1_wake.notify_all();
}

// --- E1 --------------------------------------------------------------------------------------------

void Publish(u64 seq) {
    g.posted.store(seq, std::memory_order_seq_cst);
    if (g.e2_sleeping.load(std::memory_order_seq_cst)) {
        g.e2_wake.fetch_add(1, std::memory_order_seq_cst);
        g.e2_wake.notify_all();
    }
}

/// E1 waits for E2 (a safe point: an E2 lend may run meanwhile), then holds if E2 asks.
template <typename Ready>
void E1Wait(Ready&& ready) {
    if (ready()) {
        return;
    }
    const Profile::Scope zone{Profile::Zone::ESplitHandOver, Profile::Enabled(), false};
    g.e1_parked.store(true, std::memory_order_seq_cst);
    g_e1.wait_ticks += WaitFor(ready, g.e1_wake, g.e1_sleeping, g_e1_poll,
                               g_async ? AsyncPollLimit : E1PollLimit);
    g.e1_parked.store(false, std::memory_order_seq_cst);
    if (g_hold.load(std::memory_order_seq_cst) != 0) [[unlikely]] {
        HoldHereImpl();
    }
}

void WaitDone(u64 seq) {
    E1Wait([seq] { return g.done.load(std::memory_order_acquire) >= seq; });
}

void WaitDrained() {
    WaitDone(g_next_seq - 1);
}

/// E1 in S2 before E2's work runs on the engine (E2 drained): VertexInput raises, the flags.
void ToEngine() {
    if (Maxwell3D* const engine = OneF::Engine()) {
        ApplyBackRaises(*engine);
        SyncFlagsToEngine(*engine);
    }
}

void ToView() {
    if (const Maxwell3D* const engine = OneF::Engine()) {
        SyncFlagsToView(*engine);
    }
}

Job& TakeSlot(bool draw_tail) {
    E1Wait([draw_tail] {
        const u64 done = g.done.load(std::memory_order_acquire);
        if (g_next_seq - 1 - done >= Slots - 1) {
            return false;
        }
        return !draw_tail ||
               g_draws_posted - g.draws_done.load(std::memory_order_acquire) <
                   static_cast<u64>(g_depth);
    });
    const u64 seq = g_next_seq++;
    Job& job = g.jobs[seq % Slots];
    job.seq = seq;
    job.position = GpuSplit::Get().BackEntryPosition();
    job.record_kind = RecordKind::None;
    job.record_bytes = 0;
    job.sample = false;
    job.async = false;
    job.hash = 0;
    return job;
}

void SetDepth(int depth) {
    const bool async = depth > 0;
    if (async != g_async) {
        WaitDrained();
        if (!async) {
            ToEngine(); // the lockstep fills read the engine's flags again
        }
        g_async = async;
        LOG_INFO(HW_GPU, "E split thread: FRITCHII_ESPLIT_DEPTH={} ({})", depth,
                 async ? "async: E1 posts draw tails and binds and runs ahead"
                       : "lockstep: E1 waits for each hand-over");
    }
    g_depth = depth;
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
    const u64 job_ticks = TakeCount(g_e2.job_ticks);
    const u64 e2_wait = TakeCount(g_e2.wait_ticks);
    const u64 view_differs = TakeCount(g_e2.view_differs);
    const u64 compares = TakeCount(g_e2.compares);
    const u64 view_fills = TakeCount(g_e2.view_fills);
    const u64 after_checks = TakeCount(g_e2.after_checks);
    const u64 after_differs = TakeCount(g_e2.after_differs);
    const u64 order = TakeCount(g_e2.order);
    const u64 hash_checks = TakeCount(g_e2.hash_checks);
    const u64 hash_differs = TakeCount(g_e2.hash_differs);
    const u64 holds = TakeCount(g_e2.holds);
    const u64 hold_spin = TakeCount(g_e2.hold_spin_ticks);
    const u64 room_waits = TakeCount(g_e2.room_waits);
    const u64 late_flushes = TakeCount(g_e2.late_flushes);
    const u64 refused = TakeCount(g_e2.refused);
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
    const std::string depth = g_async ? fmt::format("{}, async", g_depth) : "0, lockstep";
    LOG_INFO(HW_GPU,
             "E split thread (FRITCHII_ESPLIT=1, depth {}) 10 s: hand-overs {:.0f}/s ({}); E2 "
             "busy {:.1f} ms/s, E2 waiting {:.1f} ms/s; E1 waiting for E2 {:.1f} ms/s{}; joins "
             "{:.0f}/s ({})",
             depth, rate(hands), hand_text, e2_busy, ms(e2_wait), e1_wait,
             g_async ? std::string{}
                     : fmt::format(" = E2's runs + hand-over cost {:.1f} ms/s ({:.2f} us each)",
                                   cost,
                                   hands != 0 ? cost * 1000.0 * window / static_cast<double>(hands)
                                              : 0.0),
             rate(joins), join_text);
    const u64 touches = c.touches[0] + c.touches[1] + c.touches[2];
    LOG_INFO(HW_GPU,
             "E split thread 10 s: checks (must be 0): ESplitView {} in {} compares (1 lockstep "
             "hand-over in 64), ESplitInputs {} in {} fills from the view, E1Touch {} ({} {}, {} "
             "{}, {} {}), E2Order {}, E1Visible {}, FrontAheadOfE2 {}, ESplitHash {} in {} async "
             "draws hashed, E2Refused {}; seeds {}, untracked hand-overs {}",
             view_differs, compares, after_differs, after_checks, touches, TouchNames[0],
             c.touches[0], TouchNames[1], c.touches[1], TouchNames[2], c.touches[2], order,
             c.visible, c.front_ahead, hash_differs, hash_checks, refused, c.seeds, c.lost);
    LOG_INFO(HW_GPU,
             "E split thread 10 s: async draw tails {:.0f}/s (mean {:.1f} ahead of E2 when "
             "posted), E2 lends holding E1 {:.0f}/s (E2 waited {:.2f} ms/s for E1 to park, E1 "
             "held {:.1f} ms/s), ring B room waits {:.0f}/s, flushes after E1's lookup with work "
             "{:.0f}/s; fills from E2's view {:.0f}/s",
             rate(c.async_draws),
             c.async_draws != 0 ? static_cast<double>(c.depth_sum) / static_cast<double>(c.async_draws)
                                : 0.0,
             rate(holds), ms(hold_spin), ms(c.hold_ticks), rate(room_waits), rate(late_flushes),
             rate(view_fills));
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
    for (SlotBuffers& buffers : g.buffers) {
        buffers.record = std::make_unique<u8[]>(RecordRoom());
        buffers.packet = std::make_unique<u8[]>(OneF::PackedDrawRoom());
    }
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
    LOG_INFO(HW_GPU, "E split thread: FRITCHII_ESPLIT=1, E2 (GPUBackE2) started; "
                     "FRITCHII_ESPLIT_DEPTH=0 runs lockstep, N > 0 lets E1 run N draws ahead");
}

void StopE2() {
    if (!g_thread) {
        return;
    }
    WaitDrained();
    g_thread = false;
    g_async = false;
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
    int depth = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_esplit_depth,
                                                 "FRITCHII_ESPLIT_DEPTH");
    depth = std::clamp(depth, 0, MaxDepth);
    if (depth != g_depth) [[unlikely]] {
        SetDepth(depth);
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
    if (g_async) {
        // A lockstep hand-over in S2: E2 drained, its flags on the engine for the run.
        WaitDrained();
        ToEngine();
        ++g_e1.joins[static_cast<size_t>(Join::Lockstep)];
    }
    Job& job = TakeSlot(false);
    job.function = function;
    job.argument = argument;
    job.info = info;
    ++g_e1.hands[static_cast<size_t>(info.hand)];
    if (info.record) {
        if (const Maxwell3D* const engine = OneF::Engine()) {
            job.record_kind = ThreadBuild(*engine, info.draw, info.list_args,
                                          g.buffers[job.seq % Slots].record.get(),
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
    Publish(job.seq);
    WaitDone(job.seq);
    if (info.held != nullptr) {
        info.held->HandOver(g.e1_id);
    }
    OneF::PutEngineTls(job.tls);
    if (g_fill_state.install != nullptr) {
        g_fill_state.install(job.fill_state.data(), true);
    }
    if (g_async) {
        ToView();
    }
}

bool E2Idle() noexcept {
    return g.done.load(std::memory_order_acquire) >= g_next_seq - 1;
}

bool BeginAsyncDraw(Maxwell3D& engine, bool list_args, AsyncSlot& slot) noexcept {
    if (!g_async || t_role != 1 || t_join != 0 || !ThreadRecordReady(engine)) {
        return false;
    }
    const u64 ahead = g_draws_posted - g.draws_done.load(std::memory_order_acquire);
    Job& job = TakeSlot(true);
    job.async = true;
    job.info = HandInfo{.hand = Hand::DrawTail, .record = true, .draw = true, .list_args = list_args};
    job.record_kind = RecordKind::Record;
    job.record_bytes = ThreadBuildAsync(engine, list_args, g.buffers[job.seq % Slots].record.get());
    job.sample = Sample();
    if (job.sample) {
        job.hash = HashEngine(engine, list_args);
    }
    slot.payload = job.payload.data();
    slot.packet = g.buffers[job.seq % Slots].packet.get();
    g_open = &job;
    ++g_draws_posted;
    ++g_e1.async_draws;
    g_e1.depth_sum += ahead;
    ++g_e1.hands[static_cast<size_t>(Hand::DrawTail)];
    return true;
}

void* BeginAsync(Hand hand) noexcept {
    if (!g_async || t_role != 1 || t_join != 0) {
        return nullptr;
    }
    Job& job = TakeSlot(false);
    job.async = true;
    job.info = HandInfo{.hand = hand};
    g_open = &job;
    ++g_e1.hands[static_cast<size_t>(hand)];
    return job.payload.data();
}

void PostAsync(void (*function)(void*)) noexcept {
    Job& job = *std::exchange(g_open, nullptr);
    job.function = function;
    job.argument = job.payload.data();
    if (job.info.hand == Hand::DrawTail) {
        OneF::TakeEngineTls(job.tls); // the draw values go with the draw (PushPacked takes them)
    } else {
        job.tls = OneF::EngineTls{};
        job.tls.query_kind = OneF::QueryKind::Count;
    }
    if (g_fill_state.capture != nullptr) {
        g_fill_state.capture(job.fill_state.data(), true);
    }
    Publish(job.seq);
}

bool FillAsync() noexcept {
    return t_role == 2 && g_current != nullptr && g_current->async;
}

void ApplyBackRaises(Maxwell3D& engine) noexcept {
    if (const u32 raise = g.back_raise.exchange(0, std::memory_order_acquire); raise != 0)
        [[unlikely]] {
        engine.dirty.flags[raise - 1] = true;
    }
}

void DeferFlag(u32 flag) noexcept {
    g.back_raise.store(flag + 1, std::memory_order_release);
}

void NoteRoomWait() noexcept {
    Bump(g_e2.room_waits);
}

void NoteLateFlush() noexcept {
    Bump(g_e2.late_flushes);
}

void NoteAsyncRefused() noexcept {
    Bump(g_e2.refused);
    if (++g_refused_logged <= 16) {
        LOG_ERROR(HW_GPU,
                  "E split thread: E2Refused: an async draw's fill refused at ring position {:#x} "
                  "after E1 moved on; the draw is dropped ({} since boot)",
                  t_e2_position, g_refused_logged);
    }
}

void CheckStoreOnE1() noexcept {
    if (g_thread && g_async && t_role == 1 && !E2Idle()) [[unlikely]] {
        ++g_e1.front_ahead;
        if (++g_front_logged <= 16) {
            LOG_ERROR(HW_GPU,
                      "E split thread: FrontAheadOfE2: E1 stored `consumed` with E2 behind ({} "
                      "since boot)",
                      g_front_logged);
        }
    }
}

void HoldHereImpl() noexcept {
    const u64 start = Profile::Ticks();
    while (g_hold.load(std::memory_order_seq_cst) != 0) {
        g.e1_parked.store(true, std::memory_order_seq_cst);
        (void)WaitFor([] { return g_hold.load(std::memory_order_seq_cst) == 0; }, g.e1_wake,
                      g.e1_sleeping, g_hold_poll, AsyncPollLimit);
        g.e1_parked.store(false, std::memory_order_seq_cst);
    }
    g_e1.hold_ticks += Profile::Ticks() - start;
}

bool BeforeEnginePushImpl(bool lend, bool visible) noexcept {
    if (t_role == 1 && t_join == 0) {
        if (visible) [[unlikely]] {
            ++g_e1.visible;
            if (++g_visible_logged <= 16) {
                LOG_ERROR(HW_GPU,
                          "E split thread: E1Visible: a fence or query op from E1 outside a join "
                          "at ring position {:#x} ({} since boot): {}",
                          GpuSplit::Get().BackEntryPosition(), g_visible_logged, Stack());
            }
        }
        if (g_async) {
            WaitDrained();
            ToEngine();
            ++g_e1.joins[static_cast<size_t>(Join::Push)];
            ++t_join; // E2's work on the engine until the push is over
            return true;
        }
        return false;
    }
    if (t_role == 2 && lend && g_async) {
        HoldE1();
        return true;
    }
    return false;
}

void AfterEnginePushImpl(bool lend, bool acted) noexcept {
    (void)lend;
    if (!acted) {
        return;
    }
    if (t_role == 1) {
        --t_join;
        ToView();
    } else {
        ReleaseE1();
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
    static_assert(sizeof(Bind) <= AsyncPayloadBytes);
    constexpr auto run = [](void* raw) {
        Bind b;
        std::memcpy(&b, raw, sizeof(b));
        if (!b.enabled) {
            b.rasterizer->DisableGraphicsUniformBuffer(b.stage, b.slot);
        } else {
            b.rasterizer->BindGraphicsUniformBuffer(b.stage, b.slot, b.address, b.size);
        }
    };
    // S2: by value, async (it reads nothing of the engine).
    if (void* const payload = BeginAsync(Hand::Bind)) {
        std::memcpy(payload, &bind, sizeof(bind));
        PostAsync(run);
        return;
    }
    Run(HandInfo{.hand = Hand::Bind}, run, &bind);
}

void BeginJoin(Join why) noexcept {
    if (t_join++ == 0) {
        ++g_e1.joins[static_cast<size_t>(why)];
        if (g_async) {
            WaitDrained();
            ToEngine();
        }
    }
}

void EndJoin() noexcept {
    if (--t_join == 0 && g_async) {
        ToView();
    }
}

View* FillView(const Maxwell3D& engine) noexcept {
    if (t_role != 2 || g_current == nullptr || g_current->info.hand != Hand::DrawTail ||
        !g_current->view_ok) {
        return nullptr;
    }
    if (!g_current->async && !engine.draw_manager.draw_state.inline_index_draw_indexes.empty()) {
        return nullptr;
    }
    Bump(g_e2.view_fills);
    return &ThreadView();
}

void FillDone(const Maxwell3D& engine, const View* used) noexcept {
    if (FillAsync()) {
        return; // the fill kept the flags in the view, E2's own
    }
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
        LOG_ERROR(HW_GPU, "E split thread: E1Touch ({}, {} since boot) at {:#x}: {}",
                  TouchNames[static_cast<size_t>(kind)], g_touch_logged,
                  GpuSplit::Get().BackEntryPosition(),
                  caller != nullptr ? Site(caller) : Stack());
    }
}

} // namespace VideoCommon::ESplit
