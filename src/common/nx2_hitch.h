// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <string>
#include <vector>

#include "common/common_types.h"

// Hitch attribution (2026-11-10, the owner's v5.6 playtest: 0.4-1 s frames with 0 pipelines built,
// the cause unclear from 10-second totals; design fable52_answer_11/12). Every place that can hold
// the game up adds its time here; the game's present (nx2:vi2) calls OnPresent with the gap since
// the previous present, and a gap over 50 ms logs one line with its exact time, what each cause
// took inside it, page faults, memory the kernel zero-filled, and the CPU time of the busiest
// threads. EDEN_NX2_HITCH_LOG=0 turns the line off (the counters stay, a few relaxed adds).
// With the line on, it also says how much CPU the rest of the machine used, and a once-a-second
// "Pulse:" line samples the clock, a speed probe and free memory (FRITCHII_PULSE=0 turns that
// off; nx2_hitch.cpp, machine pulse).
namespace Common::Nx2Hitch {

enum class Cause : u8 {
    FullDrain,       ///< fence releases: the front waited until the back and the host GPU were idle
    ReadBack,        ///< guest reads of GPU-written memory (download + wait), any thread
    SaveData,        ///< save data writes and flushes (fsp)
    MemorySample,    ///< working-set walks for the memory lines (memory probe thread)
    ZeroTrim,        ///< zero-page passes that ended in the gap (their own thread)
    FileThrottle,    ///< IStorage::Read's sleep under 40 MB/s (game files, the one FS thread)
    FileRead,        ///< IStorage::Read's read itself
    HostMap,         ///< host Map of guest memory (placeholder lock included), amount = bytes
    HostUnmap,       ///< host Unmap, amount = bytes
    HostProtect,     ///< host Protect, amount = bytes
    Recompile,       ///< a CPU core's recompiler compiling game code no core had compiled
    RecompileRepeat, ///< the same, for code another core had compiled already (separate JITs)
    WarmCompile,     ///< EDEN_NX2_JIT_WARM: idle cores compiling what another core compiled
    ImageCreate,     ///< Vulkan images created with their memory (MemoryAllocator::CreateImage)
    Count,
};

/// Names in the hitch line, in Cause order.
inline constexpr std::array<const char*, static_cast<size_t>(Cause::Count)> CauseNames{
    "full drains",        "read-backs", "save data", "memory walks", "zero-page passes",
    "file read throttle", "file reads", "host map",  "host unmap",   "host protect",
    "recompiling new game code (summed over cores)",
    "recompiling code another core had (summed over cores)",
    "warm compiles on idle cores", "image creates"};

struct Counters {
    std::array<std::atomic<u64>, static_cast<size_t>(Cause::Count)> ns{};
    std::array<std::atomic<u64>, static_cast<size_t>(Cause::Count)> count{};
    std::array<std::atomic<u64>, static_cast<size_t>(Cause::Count)> amount{};
    /// The longest single one since the last present (reset there)
    std::array<std::atomic<u64>, static_cast<size_t>(Cause::Count)> longest{};
    /// steady_clock ns when the running zero-page pass began, 0 while none runs
    std::atomic<u64> zero_trim_since{};
};

inline Counters g_counters;

[[nodiscard]] inline u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

inline void Add(Cause cause, u64 ns, u64 amount = 0) {
    const auto i = static_cast<size_t>(cause);
    g_counters.ns[i].fetch_add(ns, std::memory_order_relaxed);
    g_counters.count[i].fetch_add(1, std::memory_order_relaxed);
    if (amount != 0) {
        g_counters.amount[i].fetch_add(amount, std::memory_order_relaxed);
    }
    u64 longest = g_counters.longest[i].load(std::memory_order_relaxed);
    while (ns > longest &&
           !g_counters.longest[i].compare_exchange_weak(longest, ns, std::memory_order_relaxed)) {
    }
}

/// Times its scope as `cause`.
class Scope {
public:
    explicit Scope(Cause cause_, u64 amount_ = 0)
        : cause{cause_}, amount{amount_}, begin{NowNs()} {}
    ~Scope() {
        Add(cause, NowNs() - begin, amount);
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Cause cause;
    u64 amount;
    u64 begin;
};

inline void ZeroTrimBegin() {
    g_counters.zero_trim_since.store(NowNs(), std::memory_order_relaxed);
}

inline void ZeroTrimEnd() {
    const u64 since = g_counters.zero_trim_since.exchange(0, std::memory_order_relaxed);
    if (since != 0) {
        Add(Cause::ZeroTrim, NowNs() - since);
    }
}

/// Set on a CPU core's thread while it warm-compiles (PhysicalCore::Idle).
inline thread_local bool t_warming{};

/// Which emulator thread this is, from its name (RegisterThread), for the lock wait counters.
enum class Role : u8 { Other, GuestCore, GpuFront, GpuBack, Count };
inline thread_local Role t_role{};

/// Time spent waiting for a contended SpinRecursiveMutex (the buffer and texture cache locks),
/// per role (fable52_answer_13 section 3: the back's lock waits and the guest's 144 ms/s in
/// GetFlushAreaLocked that nothing showed). Logged with the drain sites line (GpuSplit).
struct LockWaits {
    std::array<std::atomic<u64>, static_cast<size_t>(Role::Count)> ns{};
    std::array<std::atomic<u64>, static_cast<size_t>(Role::Count)> count{};
};
inline LockWaits g_lock_waits;

inline void AddLockWait(u64 ns) {
    const auto i = static_cast<size_t>(t_role);
    g_lock_waits.ns[i].fetch_add(ns, std::memory_order_relaxed);
    g_lock_waits.count[i].fetch_add(1, std::memory_order_relaxed);
}

/// From a CPU core's recompiler before it compiles block `key` (its LocationDescriptor value):
/// the cause to time it as (new code, code another core compiled, a warm compile). Process-wide
/// key -> core mask, never forgets. With EDEN_NX2_JIT_WARM=1, an on-demand compile also queues
/// the key for every other core that hasn't compiled it (fable52_answer_12).
[[nodiscard]] Cause CompileCause(u64 key, u32 core);

/// EDEN_NX2_JIT_WARM=1 (boot): idle cores compile what other cores compiled. Off by default.
[[nodiscard]] bool JitWarmEnabled();

/// Moves up to `max` queued keys for `core` into `out` (cleared first); returns how many.
size_t TakeWarmKeys(u32 core, std::vector<u64>& out, size_t max);

/// From Common::SetCurrentThreadName: the thread's CPU time goes in the hitch line under `name`.
void RegisterThread(const char* name);

/// From the game's present, one thread: `gap_us` since its previous present.
void OnPresent(long long gap_us);

// Stall snapshot (fable52_answer_stall2 section 2): EDEN_NX2_STALL_SNAPSHOT=1 (read as each
// thread is named, so nx2_switches.txt applies). A present gap of 100 ms or more then logs one
// "Stall snapshot:" line after its Hitch line: for the GPU back, GPU front, Vulkan worker, GPU
// fencing and present threads the span each is in now and for how long, and the longest span
// each spent in the gap; the master semaphore's ticks; whether the present thread is between
// vkAcquireNextImageKHR and vkQueuePresentKHR. Spans are the timeline's (Nx2Timeline::Span,
// ArgSpan), the profiler's coarse zones and a few marks of its own (StallScope). Off, a span pays
// one thread_local load (t_stall_slot stays null on every thread).
enum class StallThread : u8 { GpuBack, GpuFront, VulkanWorker, Fencing, Present, Count };

struct StallSlot {
    std::atomic<const char*> zone{}; ///< The innermost span now, nullptr outside every span
    std::atomic<u64> since{};        ///< Wall clock ns it began (or the last span ended); 0 never
    std::atomic<const char*> longest_zone{}; ///< The longest span since the last present
    std::atomic<u64> longest_ns{};
};

inline std::array<StallSlot, static_cast<size_t>(StallThread::Count)> g_stall_slots;
/// Set when the first thread registers with EDEN_NX2_STALL_SNAPSHOT=1.
inline std::atomic<bool> g_stall_on{false};
/// This thread's slot (RegisterThread), null unless the snapshot is on and this is one of the five.
inline thread_local StallSlot* t_stall_slot{};

/// Ns between vkAcquireNextImageKHR returning an image and vkQueuePresentKHR returning; 0 outside.
inline std::atomic<u64> g_stall_acquired_since{};
/// The longest acquire to present since the last present (ns).
inline std::atomic<u64> g_stall_acquire_longest{};

[[nodiscard]] inline bool StallOn() {
    return g_stall_on.load(std::memory_order_relaxed);
}

/// What a span entered on a slot restores on exit. `slot` null: nothing entered. Trivial, so a
/// profiler Scope can hold one without writing it on its off path.
struct StallMark {
    StallSlot* slot;
    const char* prev_zone;
    u64 prev_since;
    u64 since;
};

StallMark StallEnterSlow(StallSlot* slot, const char* zone);
void StallLeaveSlow(const StallMark& mark);

[[nodiscard]] inline StallMark StallEnter(const char* zone) {
    StallSlot* const slot = t_stall_slot;
    if (slot == nullptr) [[likely]] {
        return {};
    }
    return StallEnterSlow(slot, zone);
}

inline void StallLeave(const StallMark& mark) {
    if (mark.slot != nullptr) [[unlikely]] {
        StallLeaveSlow(mark);
    }
}

/// A span for the snapshot only (not on the timeline): waits the timeline has no span for.
class StallScope {
public:
    explicit StallScope(const char* zone) : mark{StallEnter(zone)} {}
    ~StallScope() {
        StallLeave(mark);
    }
    StallScope(const StallScope&) = delete;
    StallScope& operator=(const StallScope&) = delete;

private:
    StallMark mark;
};

/// The present thread got a swapchain image / presented it (Swapchain). Snapshot on only.
void StallAcquired();
void StallPresented();

/// The renderer's tick text for the line (the master semaphore): `reporter(owner)`. A null
/// reporter removes it; it is called under a mutex from the game's present, so the owner clears it
/// in its destructor.
void SetStallTickReporter(std::string (*reporter)(const void*), const void* owner);
/// Removes the reporter if `owner` set it.
void ClearStallTickReporter(const void* owner);

} // namespace Common::Nx2Hitch
