// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>

#include "common/common_types.h"

namespace Common {
class SpinRecursiveMutex;
}
namespace Core {
class System;
}
namespace Tegra::Engines {
class Maxwell3D;
}
namespace VideoCore {
class RasterizerInterface;
}

/// FRITCHII_ESPLIT (boot, default 0): the E split's thread, steps S1 and S2 of
/// work\E_SPLIT_DESIGN.md (2b, 5). The engine thread E (GpuSplit's back) becomes two:
///
///   E1 = the back thread itself: reads ring A, applies the register entries (engine A: the
///        real Maxwell3D, its dirty flags, the draw manager), looks the pipeline up.
///   E2 = "GPUBackE2", new: the memory work. Constant buffer words and every other E2 entry of
///        the dry classifier (fritchii_e_split.h), the draw's flush and tail (FlushCaching,
///        FillRing, the R packet's push, TickWork), a constant buffer bind's buffer cache half.
///        Its own view of the registers a fill reads (fritchii_e_split.h's View) is fed only by
///        E1's records.
///
/// S1, FRITCHII_ESPLIT_DEPTH=0 (live, default 0): lockstep. E1 hands each piece of E2 work over
/// and waits for it: one thread runs at a time, in today's order, on today's objects, so it is
/// exact by construction. E's per-call thread state moves with each hand-over (one follower's
/// draw values, lend site and query kind; the renderer's fill state, g_fill_state), E2 is E for
/// every check that asks (GpuSplit::OnBackThread, OneFollower::OnEngineThread), and a lock E1
/// holds that the work takes is handed over for the run.
///
/// Checks, logged every 10 s ("E split thread", must be 0):
///   ESplitView    E2's view against the engine after the record, 1 hand-over in 64.
///   ESplitInputs  on those draws (view equal), the fill reads the view and every input it can
///                 read is compared after the fill (as FRITCHII_ESPLIT_COUNT=3).
///   E1Touch       E1 held the buffer cache's lock, read or wrote game memory outside a join.
///   E2Order       E2 ran a hand-over out of ring order.
/// Measured: E2's busy time (zone "e split job"), E2's wait ("e split E2 wait"), E1's time in
/// hand-overs ("e split hand-over") and the hand-over cost (E1's wait less E2's run).
///
/// S2, FRITCHII_ESPLIT_DEPTH=N (1, 4, 16): async. See the S2 section of the .cpp.
///
/// FRITCHII_ESPLIT=0: nothing starts; each hand-over site pays one load and a branch (g_thread).
namespace VideoCommon::ESplit {

struct View;

/// What E1 hands to E2.
enum class Hand : u8 {
    Entry,    ///< an E2 entry of ring A (GpuSplit::Execute on E2)
    Flush,    ///< a draw's FlushCaching, before E1's pipeline lookup
    DrawTail, ///< a draw's FillRing, R packet push and TickWork
    Bind,     ///< a constant buffer bind's buffer cache half, by value
    Store,    ///< S2: GpuSplit::StoreConsumed at E2's place
    Count,
};

/// Where E1 runs E2-owned work itself (S1: E2 is idle anyway; S2: E2 drained first).
enum class Join : u8 {
    Entry,     ///< a join entry of the classifier (uploads, other engines, untagged calls, channel)
    Decision,  ///< E1's own decision reads game memory (the indirect reset test)
    Condition, ///< a render condition from game memory
    Idle,      ///< the idle hook and the fence check (E's idle work)
    Push,      ///< S2: E1 lends or pushes to R (R's packet queue has one producer)
    Lockstep,  ///< S2: a hand-over that waits (seed, an entry, a draw that cannot go async)
    Count,
};

/// FRITCHII_ESPLIT=1 and E2 started (written by E1 before E2 exists, read everywhere).
inline bool g_thread{};
/// S2: FRITCHII_ESPLIT_DEPTH > 0 (E1 writes it at a batch's start with E2 drained).
inline bool g_async{};
/// 1 on E1, 2 on E2.
inline thread_local u8 t_role{};
/// E1 inside a join (it runs E2-owned work itself).
inline thread_local u32 t_join{};

/// The boot switch (read once; FRITCHII_ONEF=1 is needed, checked at the start).
[[nodiscard]] bool ThreadWanted() noexcept;
/// GpuSplit::BackLoop at its start (E1): E2 starts; `adopt` runs first on E2 (GpuSplit's own
/// thread marks).
void StartE2(Core::System& system, void (*adopt)(void*), void* adopt_argument);
/// GpuSplit::BackLoop at its end (E1): E2 drains and stops.
void StopE2();
/// E1, once a batch: FRITCHII_ESPLIT_DEPTH, the 10 s lines.
void ThreadBatch() noexcept;

[[nodiscard]] inline bool OnE1() noexcept {
    return t_role == 1;
}
[[nodiscard]] inline bool OnE2() noexcept {
    return t_role == 2;
}
/// E2's work here goes to E2: E1, the thread on, outside a join.
[[nodiscard]] inline bool ShouldHand() noexcept {
    return g_thread && t_role == 1 && t_join == 0;
}

struct HandInfo {
    Hand hand{Hand::Entry};
    bool record{};    ///< E1 builds a record for E2's view first
    bool draw{};      ///< the record carries the draw state
    bool list_args{}; ///< ... and a list draw's indirect parameters
    bool cb_entry{};  ///< a constant buffer entry: E2's view offset advances after it
    u32 cb_words{};   ///< its words, 0: what `cb_size` holds after the run
    const std::size_t* cb_size{};
    /// A lock E1 holds that the work takes: E2 holds it for the run (SpinRecursiveMutex::HandOver).
    Common::SpinRecursiveMutex* held{};
};

/// E1: `function(argument)` runs on E2; E1 waits for it (lockstep, also in S2: E2 drains first).
void Run(const HandInfo& info, void (*function)(void*), void* argument) noexcept;
template <typename F>
void HandTo(const HandInfo& info, F& body) noexcept {
    Run(info, [](void* raw) { (*static_cast<F*>(raw))(); }, &body);
}

/// Maxwell3D::ProcessCBBind's buffer cache half, handed over by value (async in S2).
void HandBind(VideoCore::RasterizerInterface& rasterizer, std::size_t stage, u32 slot,
              bool enabled, u64 address, u32 size) noexcept;

void BeginJoin(Join why) noexcept;
/// E1 runs E2-owned work itself here.
class JoinScope {
public:
    explicit JoinScope(Join why, bool on = true) noexcept {
        if (on && g_thread && t_role == 1) [[unlikely]] {
            BeginJoin(why);
            active = true;
        }
    }
    ~JoinScope() {
        if (active) [[unlikely]] {
            --t_join;
        }
    }
    JoinScope(const JoinScope&) = delete;
    JoinScope& operator=(const JoinScope&) = delete;

private:
    bool active{};
};

/// E2, in a draw tail's FillRing: E2's view when the fill reads it (S1: the draw's sampled
/// compare found it equal; no inline index bytes), else null (the fill reads the engine).
[[nodiscard]] View* FillView(const Tegra::Engines::Maxwell3D& engine) noexcept;
/// After that fill (E2, or E1 in a join): the view's buffer flags, the inputs check.
void FillDone(const Tegra::Engines::Maxwell3D& engine, const View* used) noexcept;

// --- E1Touch ---------------------------------------------------------------------------------------

enum class Touch : u8 { CacheLock, MemoryRead, MemoryWrite, Count };
void NoteTouchImpl(Touch kind, const void* caller) noexcept;
/// The buffer cache's lock released (its release hook, set at the cache's start with the switch).
inline void NoteCacheRelease() noexcept {
    if (t_role == 1 && t_join == 0 && g_thread) [[unlikely]] {
        NoteTouchImpl(Touch::CacheLock, nullptr);
    }
}
/// E's game memory write (OneFollower::GuardStreamWrite).
inline void NoteMemoryWrite() noexcept {
    if (t_role == 1 && t_join == 0 && g_thread) [[unlikely]] {
        NoteTouchImpl(Touch::MemoryWrite, nullptr);
    }
}

// --- The renderer's per-thread fill state (MemStage), moved like one follower's ------------------

struct FillStateHooks {
    /// The calling thread's state into `out` (FillStateBytes); `as_owner`: with the thread's own
    /// identity filled in (E1, for E2's run of its work).
    void (*capture)(void* out, bool as_owner){};
    /// Installs it; `keep_identity`: the calling thread keeps its own identity (E1, back).
    void (*install)(const void* in, bool keep_identity){};
};
inline constexpr std::size_t FillStateBytes = 128;
/// Set by the renderer at static initialization (vk_mem_stage.cpp).
inline FillStateHooks g_fill_state{};

} // namespace VideoCommon::ESplit
