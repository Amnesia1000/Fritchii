// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <exception>
#include <thread>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace VideoCommon {

struct MemStageShadowEngine;

/// The memory stage's thread, GPUMem, and the back's link to it (memory_stage_link.cpp).
class MemoryStageLink {
public:
    using Function = void (*)(void* argument);

    enum class HandOff : u8 { Draw, Dispatch, Idle, Served, Count };

    /// Counters that must read 0 (memory_stage_link.cpp).
    enum class Check : u8 {
        AllocatorUser,
        StagingUser,
        PushUser,
        BudgetUser,
        StageScheduler,
        PushNowhere,
        PushWhilePending,
        PlayerOnStage,
        FrontWaitOnStage,
        HandOffUnlocked,
        BackInCache,
        BackInvalidation,
        PacketMissing,
        PacketUnplayed,
        RingFull,
        DrawCount,
        PacketOrphan,
        StageException,
        PacketMismatch,
        BackHandles,
        Count,
    };

    /// THREAD = 2: what a served call of the back is for (counted, not a must-be-0).
    enum class Served : u8 {
        FlushRegion,
        FlushRegionForRead,
        Gather,
        Invalidation,
        InnerInvalidation,
        MustFlush,
        Fallback,
        Count,
    };

    /// THREAD = 2: what the Vulkan side does for the stage (memory_stage_link.cpp).
    struct RingBackend {
        void* context{};
        /// GPUMem: fills the draw the stage's copy issues and puts its packet in ring B. False:
        /// the stage cannot serve it (no facts, texel buffers): the draw is level.
        bool (*fill_draw)(void* context, MemStageShadowEngine& copy, bool indexed, u64 position,
                          u64 seq, u64 hash,
                          std::vector<std::pair<u64, std::size_t>>& ranges){};
        void (*bind_channel)(void* context, s32 bind_id){};
        void (*bind_uniform)(void* context, size_t stage, u32 slot, u64 gpu_address, u32 size,
                             bool enabled){};
        /// The back at a run's start: the buffer half of the stage's collected ranges.
        void (*apply_ranges)(void* context, const std::pair<u64, std::size_t>* ranges,
                             size_t count){};
        /// GPUMem, the back parked: the buffer cache's idle work.
        void (*idle)(void* context){};
        /// The back at a run's end: FlushCaching.
        void (*end_run)(void* context){};
        /// The back: plays whole the packets of entries before `position`; returns how many.
        u64 (*drop_packets)(void* context, u64 position){};
        /// The bound channel as the back has it (the stage's start, after an unfollowed entry).
        s32 (*back_channel)(void* context){};
    };

    [[nodiscard]] static MemoryStageLink& Get() noexcept {
        return instance;
    }

    void Start(Core::System& system);
    void Stop();

    /// Back, at a frame end: 0 fills on the back, 1 hand-off, 2 the ring-fed stage. Returns the
    /// mode it runs.
    u32 SetMode(u32 wanted);
    [[nodiscard]] u32 Mode() const noexcept {
        return mode;
    }

    void Call(HandOff kind, Function function, void* argument);
    template <typename F>
    void Call(HandOff kind, F& function) {
        Call(kind, [](void* argument) { (*static_cast<F*>(argument))(); }, &function);
    }

    void Serve(Function function, void* argument);
    template <typename F>
    void Serve(F& function) {
        Serve([](void* argument) { (*static_cast<F*>(argument))(); }, &function);
    }

    [[nodiscard]] static bool OnStageThread() noexcept {
        return t_on_stage;
    }
    [[nodiscard]] static bool InCall() noexcept {
        return t_in_call;
    }
    [[nodiscard]] static bool IsCacheUser() noexcept {
        return !g_checking.load(std::memory_order_relaxed) || IsCacheUserSlow();
    }
    static void CheckCacheUser(Check check) noexcept {
        if (g_checking.load(std::memory_order_relaxed)) [[unlikely]] {
            CheckCacheUserSlow(check);
        }
    }
    static void Note(Check check, const char* where) noexcept;
    [[nodiscard]] static bool Checking() noexcept {
        return g_checking.load(std::memory_order_relaxed);
    }

    class AllocatorOp {
    public:
        explicit AllocatorOp(const void* allocator) noexcept
            : watched{allocator != nullptr &&
                      allocator == g_buffer_allocator.load(std::memory_order_relaxed)} {
            if (watched) [[unlikely]] {
                CheckCacheUser(Check::AllocatorUser);
            }
        }
        ~AllocatorOp() {
            if (watched) [[unlikely]] {
                g_buffer_allocator_ops.fetch_add(1, std::memory_order_release);
            }
        }
        AllocatorOp(const AllocatorOp&) = delete;
        AllocatorOp& operator=(const AllocatorOp&) = delete;

    private:
        bool watched;
    };
    static void WatchBufferAllocator(const void* allocator) noexcept {
        g_buffer_allocator.store(allocator, std::memory_order_relaxed);
    }
    [[nodiscard]] static u64 BufferAllocatorOps() noexcept {
        return g_buffer_allocator_ops.load(std::memory_order_acquire);
    }

    [[nodiscard]] static bool FullVertexBinds() noexcept {
        return g_full_vertex_binds.load(std::memory_order_relaxed);
    }
    static void SetFullVertexBinds(bool on) noexcept {
        g_full_vertex_binds.store(on, std::memory_order_relaxed);
    }

    // --- THREAD = 2 ---

    void SetRingBackend(const RingBackend& backend) noexcept {
        ring_backend = backend;
    }
    [[nodiscard]] const RingBackend& Backend() const noexcept {
        return ring_backend;
    }
    /// The back's limit is `staged` (gpu_split.cpp).
    [[nodiscard]] static bool RingActive() noexcept {
        return g_ring_active.load(std::memory_order_acquire);
    }
    /// The back runs an entry the stage serves (not lent, not serving a request).
    [[nodiscard]] static bool BackServed() noexcept {
        return t_back_served && !t_back_serving;
    }
    /// The back runs a level run (the cache is lent to it).
    [[nodiscard]] static bool BackInRun() noexcept {
        return t_back_run;
    }
    /// At a buffer cache entry point: the back in the cache while not lent (must be 0).
    static void CheckBackCache(const char* where) noexcept {
        if (t_back_served && !t_back_serving) [[unlikely]] {
            Note(Check::BackInCache, where);
        }
    }
    /// The back, served: runs `function` on GPUMem, which holds the cache.
    void ServeOnStage(Served cause, Function function, void* argument);
    template <typename F>
    void ServeOnStage(Served cause, F& function) {
        ServeOnStage(cause, [](void* argument) { (*static_cast<F*>(argument))(); }, &function);
    }
    /// Ring B's counters, kept by the Vulkan side.
    void NotePacketSent(u64 bytes) noexcept;
    void NotePacketTaken() noexcept;
    void NoteLevelDraw(u32 kind) noexcept;
    /// The verify of compare b with two threads (FRITCHII_MEMSTAGE_RING_VERIFY).
    void NoteHash(bool same) noexcept;
    [[nodiscard]] static bool RingVerify() noexcept;

    /// The back's loop (gpu_split.cpp).
    struct BackLimits {
        u64 available;
        u64 lent_from;
        u64 lent_until;
    };
    /// At the top of the back's loop: starts or ends ring mode at `position` (everything before
    /// it consumed). True while it is on.
    bool BackRingTop(u64 position);
    /// Whether the back should leave its batch so the ring starts at its loop top.
    [[nodiscard]] bool BackRingStartWanted() const noexcept {
        return ring_wanted.load(std::memory_order_relaxed) &&
               !g_ring_active.load(std::memory_order_relaxed);
    }
    [[nodiscard]] BackLimits BackLoad(u64 published) const noexcept;
    /// Sets the back's thread-locals for the entry at `position`; starts and ends runs.
    void BackEntry(u64 position, const BackLimits& limits);
    /// After an entry: true when the run's last entry ran (the back then consumes its method
    /// sink and calls BackEndRun).
    [[nodiscard]] bool BackAfter(u64 position, const BackLimits& limits);
    void BackEndRun();
    [[nodiscard]] bool BackRequestPending() const noexcept;
    void NoteBackWait(u64 ns) noexcept;
    /// The back stored `consumed`; the front published (wakes a stage that waits for them).
    void NoteConsumed(u64 position) noexcept;
    void NotePublished() noexcept;
    /// The back waits for the stage (phase 3 of the split wait): serves its requests. True if it
    /// served one.
    bool BackServeRequests();
    /// The back's idle hook is about to run: false when the stage is busy (the hook is skipped).
    bool BackEnterHook();
    void BackLeaveHook();
    /// The back drew (DrawManager, every draw, before the render-enable test).
    void NoteBackDraw(const void* engine) noexcept;
    void NoteBackDrawState(const void* engine) noexcept;
    [[nodiscard]] u64 BackDrawSeq() const noexcept {
        return back_draw_seq;
    }

    void LogStats();

private:
    static bool IsCacheUserSlow() noexcept;
    static void CheckCacheUserSlow(Check check) noexcept;

    void StageMain(std::stop_token stop);
    void RunCall(u32 seq);
    void WaitForStage(u32 seq);
    void ServeOne();
    void WakeStage();
    void WakeBack();

    // THREAD = 2: the stage's loop (memory_stage_link.cpp).
    void RingLoop(std::stop_token& stop, u64 start);
    void PublishStaged(u64 position);
    enum class StageWait : u8 { Input, Back, Run, Count };
    bool StageWaitFor(std::stop_token& stop, StageWait reason, u64 target);
    bool StageServeCalls();
    bool BecomeBusy(std::stop_token& stop);
    void StageIdle(std::stop_token& stop);
    void Lend(u64 from, u64 until);
    u64 RingFailed(std::stop_token& stop, const char* what);

    static MemoryStageLink instance;

    static inline std::atomic<bool> g_checking{false};
    static inline std::atomic<const void*> g_buffer_allocator{nullptr};
    static inline std::atomic<u64> g_buffer_allocator_ops{0};
    static inline std::atomic<bool> g_full_vertex_binds{false};
    static inline std::atomic<bool> g_ring_active{false};

    static inline thread_local bool t_on_stage{};
    static inline thread_local bool t_in_call{};
    static inline thread_local bool t_stage_user{};
    static inline thread_local bool t_back_lent{};
    static inline thread_local bool t_back_serving{};
    static inline thread_local bool t_back_run{};
    static inline thread_local bool t_back_served{};

    Core::System* system{};
    u32 mode{};
    bool refused_logged{};

    // Written by the back before `calls`, read by the stage after it.
    Function call_function{};
    void* call_argument{};
    bool call_recording{};
    std::exception_ptr call_error;
    u32 call_sent{};
    u32 calls_seen{};
    // Written by the stage before `requests`, read by the back after it.
    Function request_function{};
    void* request_argument{};
    std::exception_ptr request_error;
    u32 request_sent{};
    u32 answered_local{};

    alignas(64) std::atomic<u32> calls{0};
    alignas(64) std::atomic<u32> done{0};
    alignas(64) std::atomic<u32> requests{0};
    alignas(64) std::atomic<u32> answered{0};
    alignas(64) std::atomic<u32> stage_word{0};
    std::atomic<bool> stage_sleeping{false};
    alignas(64) std::atomic<u32> back_word{0};
    std::atomic<bool> back_sleeping{false};

    // THREAD = 2.
    RingBackend ring_backend{};
    alignas(64) std::atomic<bool> ring_wanted{false};
    /// Stage: inside a batch (between a park of the back and the next publish), or its idle work.
    alignas(64) std::atomic<bool> stage_busy{false};
    /// Back: in its idle hook.
    std::atomic<bool> back_in_hook{false};
    /// Set by the back at the loop top: where the stage starts reading.
    std::atomic<u64> ring_start{~u64{0}};
    std::atomic<bool> ring_running{false};
    std::atomic<bool> ring_failed{false};
    u64 staged_local{};
    /// The stage's ranges handed to the back at a lend (read by the back at the run's start).
    std::vector<std::pair<u64, std::size_t>> lend_ranges;
    /// Back: the run it is in.
    u64 back_run_from{};
    u64 back_draw_seq{};
    std::atomic<const void*> back_drew_in_run{};
    std::atomic<const void*> back_state_in_run{};

    struct Counters {
        std::array<std::atomic<u64>, static_cast<size_t>(HandOff::Count)> calls{};
        std::atomic<u64> call_ns{};
        std::atomic<u64> stage_ns{};
        std::atomic<u64> served{};
        std::atomic<u64> served_ns{};
        std::atomic<u64> back_sleeps{};
        std::atomic<u64> stage_sleeps{};
        std::array<std::atomic<u64>, static_cast<size_t>(Check::Count)> checks{};
        // THREAD = 2.
        std::array<std::atomic<u64>, static_cast<size_t>(Served::Count)> served_calls{};
        std::array<std::atomic<u64>, 4> entries{}; ///< pass, work, level, neutral
        std::atomic<u64> runs{};
        std::atomic<u64> run_entries{};
        std::atomic<u64> run_ns{};       ///< the stage, from the lend to the return
        std::atomic<u64> back_run_ns{};  ///< the back, from the lend seen to the next packet taken
        std::atomic<u64> back_wait{};
        std::atomic<u64> back_wait_ns{};
        std::array<std::atomic<u64>, static_cast<size_t>(StageWait::Count)> stage_wait_ns{};
        std::atomic<u64> stage_busy_ns{};
        std::atomic<u64> fill_ns{};
        std::atomic<u64> packets{};
        std::atomic<u64> packet_bytes{};
        std::atomic<u64> taken{};
        std::atomic<u64> publishes{};
        std::atomic<u64> copies{};
        std::atomic<u64> flag_handovers{};
        std::array<std::atomic<u64>, 8> level_draws{};
        std::atomic<u64> hashes{};
        std::atomic<u64> hash_mismatches{};
        std::atomic<u64> ring_ahead_max{};
    };
    Counters counters;
    struct Snapshot {
        std::array<u64, static_cast<size_t>(HandOff::Count)> calls{};
        u64 call_ns{};
        u64 stage_ns{};
        u64 served{};
        u64 served_ns{};
        u64 back_sleeps{};
        u64 stage_sleeps{};
        std::array<u64, static_cast<size_t>(Check::Count)> checks{};
        std::array<u64, static_cast<size_t>(Served::Count)> served_calls{};
        std::array<u64, 4> entries{};
        u64 runs{};
        u64 run_entries{};
        u64 run_ns{};
        u64 back_run_ns{};
        u64 back_wait{};
        u64 back_wait_ns{};
        std::array<u64, static_cast<size_t>(StageWait::Count)> stage_wait_ns{};
        u64 stage_busy_ns{};
        u64 fill_ns{};
        u64 packets{};
        u64 packet_bytes{};
        u64 taken{};
        u64 publishes{};
        u64 copies{};
        u64 flag_handovers{};
        std::array<u64, 8> level_draws{};
        u64 hashes{};
        u64 hash_mismatches{};
    };
    void Take(Snapshot& out) const noexcept;
    Snapshot reported{};
    u64 last_log_ns{};
    u64 back_run_seen_ns{};

    std::jthread thread;
};

} // namespace VideoCommon
