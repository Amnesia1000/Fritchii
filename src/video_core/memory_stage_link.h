// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <exception>
#include <memory>
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
        SyncAhead,
        ServedAheadJoin,
        FutureWriteUnderImage,
        ServedOverStageWrite,
        Count,
    };

    /// Step 7: why GPUMem came level inside its window (StageLevel, counted).
    enum class LevelReason : u8 {
        Copy,        ///< a full copy of the back's engine (a new or stale copy)
        WritingDraw, ///< a draw whose fill marks buffers GPU-written (storage writes, xfb)
        Claim,       ///< collected constant buffer ranges over GPU-written buffer data
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
        /// Step 7: packets in ring B the back has not taken yet; whether ring B has room.
        u64 (*ring_packets)(void* context){};
        bool (*ring_room)(void* context){};
        /// The back, both threads parked at the ring's start (`end` false) or end: GPUMem's fill
        /// side starts clean (its unplayed mark 0; at the end its pending entries played first).
        void (*reset_stage)(void* context, bool end){};
        /// GPUMem, inside an entry: whether these ranges cover GPU-written buffer data (the page
        /// bits, then the cache's answer under its lock).
        bool (*ranges_gpu_written)(void* context, const std::pair<u64, std::size_t>* ranges,
                                   size_t count){};
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
    /// Ring B's counters, kept by the Vulkan side. `ahead`: packets in ring B after the push.
    void NotePacketSent(u64 bytes, u64 ahead) noexcept;
    void NotePacketTaken() noexcept;
    /// The back took or dropped a packet: wakes a stage that waits for the window or for room.
    void NoteRingFreed() noexcept;

    // --- Step 7: the window (FRITCHII_MEMSTAGE_AHEAD) ---

    /// Values at or above this are unlimited (spec 1.4: 65,536).
    static constexpr u32 AheadUnlimited = 65536;
    /// The back, at a frame end (a level run: GPUMem is parked): the window, in packets. 0 is
    /// level mode (6b). `blocked`: a setting the window does not run with (it runs 0 then).
    void SetAhead(u32 packets, const char* blocked = nullptr);
    /// The window GPUMem uses (read at its loop top; it changes only while GPUMem is parked).
    [[nodiscard]] static u32 Ahead() noexcept {
        return g_ahead.load(std::memory_order_relaxed);
    }
    /// GPUMem, inside an entry at `position`, outside any fill: waits until the back has run every
    /// entry before it (so every packet before it is played), serving the back's calls meanwhile.
    /// Does nothing in level mode (the stage is level before each entry there) or off GPUMem.
    /// False when GPUMem is stopping (the caller stops its work).
    bool StageLevel(u64 position, LevelReason reason);
    /// The staging pool's stream was busy on GPUMem with the window open: it took a new buffer
    /// (the pool never waits for a region, so a stamp not played yet is never waited on).
    static void NoteStagingFallback() noexcept {
        if (t_on_stage && g_ahead.load(std::memory_order_relaxed) != 0) [[unlikely]] {
            auto& counter = instance.counters.staging_fallbacks;
            counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
        }
    }
    /// The entry turned hard at a constant buffer write onto GPU-written data (counted).
    void NoteWriteLevel() noexcept {
        counters.write_levels.store(counters.write_levels.load(std::memory_order_relaxed) + 1,
                                    std::memory_order_relaxed);
    }
    /// The back, at a frame end: the window's verify runs (FRITCHII_MEMSTAGE_READ_LOG or
    /// FRITCHII_MEMSTAGE_RING_VERIFY): GPUMem keeps its own writes, images look for them.
    static void SetVerify(bool on) noexcept {
        g_verify.store(on, std::memory_order_relaxed);
    }
    /// GPUMem keeps its own writes in its ring (the window open and its verify on).
    [[nodiscard]] static bool StageWritesKept() noexcept {
        return g_ahead.load(std::memory_order_relaxed) != 0 &&
               g_verify.load(std::memory_order_relaxed);
    }

    /// GPUMem, around a lock of the buffer cache taken outside any fill and outside a served
    /// call: while it waits for the lock, held by the back, it serves the back's call (step 7).
    class CleanPoint {
    public:
        CleanPoint() noexcept : was{t_stage_clean} {
            t_stage_clean = t_on_stage;
        }
        ~CleanPoint() {
            t_stage_clean = was;
        }
        CleanPoint(const CleanPoint&) = delete;
        CleanPoint& operator=(const CleanPoint&) = delete;

    private:
        bool was;
    };
    /// The buffer cache joined a buffer at [address, address + size) into a new one
    /// (BufferCache::JoinOverlap): with the window open GPUMem keeps where (the entry it fills).
    static void NoteBufferJoin(u64 address, u64 size) {
        if (t_on_stage && g_ahead.load(std::memory_order_relaxed) != 0) [[unlikely]] {
            instance.NoteBufferJoinSlow(address, size);
        }
    }
    /// BufferCache::JoinOverlap, before a join: whether the window needs the join's GPU-written
    /// test (GPUMem's own fill, the window open), and then the catch-up it asks for.
    [[nodiscard]] static bool JoinNeedsTest() noexcept {
        return t_on_stage && !t_serving_back && g_ahead.load(std::memory_order_relaxed) != 0;
    }
    static void JoinOverGpuWritten() {
        instance.StageJoinCatchUpSlow();
    }
    /// GPUMem, in a call it serves for the back at entry `back_position`: whether a buffer over
    /// [address, address + size) was joined by a fill of a later entry (its copy is in a packet
    /// the back has not played).
    enum class Joined : u8 { No, Yes, Unknown };
    [[nodiscard]] Joined JoinedAhead(u64 address, u64 size, u64 back_position) const noexcept;
    /// GPUMem, in a download it serves for the back: the join ring could not answer (every kept
    /// join is ahead of the back). Counted, not a failure.
    void NoteJoinUnknown() noexcept {
        counters.joins_unknown.store(counters.joins_unknown.load(std::memory_order_relaxed) + 1,
                                     std::memory_order_relaxed);
    }
    /// GPUMem processes the entry at `position` (the joins' and the stage writes' position).
    [[nodiscard]] u64 StageEntry() const noexcept {
        return stage_entry;
    }

    // --- Step 7: GPUMem's own writes to game memory (spec 6.4's "short ring of the stage's own
    // stream writes with positions"): constant buffer words, by device range and entry ---

    /// GPUMem, the window open: it wrote [begin, begin + size) of device memory for the entry at
    /// `position`. Positions only grow (the stream's order).
    void NoteStageWrite(u64 begin, u64 size, u64 position) noexcept;
    /// How many writes were noted (the number the next one gets).
    [[nodiscard]] u64 StageWriteCount() const noexcept {
        return stage_write_count.load(std::memory_order_acquire);
    }
    /// Any thread: calls func(begin, end, position) for each write numbered `since` or later,
    /// newest first, until it returns true. False when some of them were overwritten already.
    template <typename F>
    bool ForEachStageWriteSince(u64 since, F&& func) const {
        if (!stage_writes) {
            return true;
        }
        const u64 count = stage_write_count.load(std::memory_order_acquire);
        if (count < since) {
            return true;
        }
        bool complete = count - since <= StageWriteSlots;
        const u64 oldest = complete ? since : count - StageWriteSlots;
        for (u64 n = count; n > oldest; --n) {
            const StageWrite& slot = stage_writes[(n - 1) % StageWriteSlots];
            if (slot.seq.load(std::memory_order_acquire) != n) {
                complete = false;
                continue;
            }
            const u64 begin = slot.begin.load(std::memory_order_relaxed);
            const u64 end = slot.end.load(std::memory_order_relaxed);
            const u64 position = slot.position.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (slot.seq.load(std::memory_order_relaxed) != n) {
                complete = false;
                continue;
            }
            if (func(begin, end, position)) {
                break;
            }
        }
        return complete;
    }
    /// The back registers an image over [address, address + size) (the texture cache): risk 9.1,
    /// a constant buffer write GPUMem made for a later entry under it (must be 0). Only while the
    /// window's verify runs (StageWritesKept): the fps gate measures play without the walk.
    static void NoteImageRegistered(u64 address, u64 size) {
        if (g_ring_active.load(std::memory_order_relaxed) && StageWritesKept()) [[unlikely]] {
            instance.NoteImageRegisteredSlow(address, size);
        }
    }
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
    /// Step 7: the back ran its idle hook from the wait though the stage was busy (a request).
    void NoteHookFlush() noexcept;
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
    bool BackRingTopImpl(u64 position);
    void PublishStaged(u64 position);
    /// Input: the front; Back: level mode; Run: a level run; Window: AHEAD packets in ring B; Room:
    /// ring B full; Level: StageLevel inside the window.
    enum class StageWait : u8 { Input, Back, Run, Window, Room, Level, Count };
    bool StageWaitFor(std::stop_token& stop, StageWait reason, u64 target);
    /// Step 7, GPUMem in its own fill, before a sync goes to the back: the back first plays every
    /// packet before the published position (unless it waits in a call GPUMem must serve first).
    void StageSyncCatchUp();
    void StageJoinCatchUpSlow();
    /// The back runs up to the published position; false when it could not (counted).
    bool StageCatchUp(const char* what);
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
    static inline std::atomic<u32> g_ahead{0};
    static inline std::atomic<bool> g_verify{false};

    static inline thread_local bool t_on_stage{};
    /// GPUMem runs a call the back posted (a served call): its syncs go to the back at once.
    static inline thread_local bool t_serving_back{};
    /// GPUMem is at a lock site that is a clean point (CleanPoint).
    static inline thread_local bool t_stage_clean{};
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
    bool call_served{};
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
    /// Step 7: the lend's flag hand-over waits for the back's run start (the back is not parked
    /// at a lend with the window open). Written by the stage before it publishes the run.
    bool lend_flags_deferred{};
    /// Step 7: the window is refused for the session (a sync went to a back that had not played
    /// the packets ahead of it); the last blocking setting logged.
    std::atomic<bool> ahead_refused{false};
    const char* ahead_blocked_logged{};
    /// GPUMem's loop: its stop token while the ring runs, and the window it read last.
    std::stop_token* ring_stop{};
    u32 ring_window{};
    /// The entry GPUMem processes (GPUMem only).
    u64 stage_entry{};
    /// Buffers joined by GPUMem's fills, with the entry (GPUMem only; served calls run there too).
    struct Join {
        u64 begin;
        u64 end;
        u64 position;
    };
    static constexpr size_t JoinSlots = 4096;
    std::array<Join, JoinSlots> joins{};
    u64 join_count{};
    void NoteBufferJoinSlow(u64 address, u64 size) noexcept;
    /// GPUMem's own writes (NoteStageWrite): one writer, any reader (a sequence word per slot).
    struct StageWrite {
        std::atomic<u64> seq{};
        std::atomic<u64> begin{};
        std::atomic<u64> end{};
        std::atomic<u64> position{};
    };
    static constexpr u64 StageWriteSlots = u64{1} << 17;
    std::unique_ptr<StageWrite[]> stage_writes;
    std::atomic<u64> stage_write_count{0};
    void NoteImageRegisteredSlow(u64 address, u64 size);
    /// GPUMem's thread id (the back serves it while it waits for a lock GPUMem holds).
    std::atomic<std::thread::id> stage_thread{};
    /// Bumped while the back waits for a lock GPUMem holds (StageSyncCatchUp gives up then).
    std::atomic<u64> back_contended{0};
    static bool BackContendedHook(std::thread::id holder, bool first);
    /// The back's thread id (GPUMem serves the back's call while it waits for a lock it holds).
    std::atomic<std::thread::id> back_thread{};
    static bool StageContendedHook(std::thread::id holder, bool first);
    /// Back: the run it is in.
    u64 back_run_from{};
    u64 back_draw_seq{};
    std::atomic<const void*> back_drew_in_run{};
    std::atomic<const void*> back_state_in_run{};

    /// The window's fill histogram: 1, 2-3, 4-7, 8-15, 16-31, 32-63, 64-255, 256 and more.
    static constexpr size_t WindowBuckets = 8;

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
        // Step 7.
        std::array<std::atomic<u64>, WindowBuckets> window{}; ///< packets in ring B at each push
        std::atomic<u64> ahead_sum{};
        std::array<std::atomic<u64>, static_cast<size_t>(LevelReason::Count)> levels{};
        std::atomic<u64> served_ahead{};
        std::atomic<u64> catchups{};
        std::atomic<u64> catchup_ns{};
        std::atomic<u64> deferred_lends{};
        std::atomic<u64> back_lock_waits{};   ///< the back waited for the lock GPUMem held
        std::atomic<u64> stage_lock_serves{}; ///< GPUMem served a call while waiting for it
        std::atomic<u64> join_catchups{};
        std::atomic<u64> joins_unknown{};     ///< a served download past the join ring's reach
        std::atomic<u64> write_levels{};      ///< words onto GPU-written data: the entry hard
        std::atomic<u64> staging_fallbacks{}; ///< GPUMem took a new staging buffer, stream busy
        std::atomic<u64> hook_flushes{};      ///< the back's skipped idle hook, its flush run
        std::atomic<u64> stage_idles{};
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
        std::array<u64, WindowBuckets> window{};
        u64 ahead_sum{};
        std::array<u64, static_cast<size_t>(LevelReason::Count)> levels{};
        u64 served_ahead{};
        u64 catchups{};
        u64 catchup_ns{};
        u64 deferred_lends{};
        u64 back_lock_waits{};
        u64 stage_lock_serves{};
        u64 join_catchups{};
        u64 joins_unknown{};
        u64 write_levels{};
        u64 staging_fallbacks{};
        u64 hook_flushes{};
        u64 stage_idles{};
    };
    void Take(Snapshot& out) const noexcept;
    Snapshot reported{};
    u64 last_log_ns{};
    u64 back_run_seen_ns{};

    std::jthread thread;
};

} // namespace VideoCommon
