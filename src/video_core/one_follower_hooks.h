// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// One follower (one_follower.h), part (b): the engine side's write tracking, in a header of its
// own so maxwell_3d.h can include it (one_follower.h includes maxwell_3d.h).

#include <atomic>

#include "common/common_types.h"

namespace VideoCommon::OneFollower {

/// Maxwell3D's register write paths test this before NoteRegWrite: set while step 2 or more is on
/// (FRITCHII_ONEF_VIEW). Off: one relaxed load and a branch per register write.
inline std::atomic<bool> g_track_writes{};

/// FRITCHII_ONEF_E_PIPELINE=1 (boot, with FRITCHII_ONEF=1; fable_b_one_follower 10.2 rule 3): E
/// owns the pipeline lookup. The pipeline cache always reads E's engine (R's view never replaces
/// it), E looks a draw's pipeline up before its fill and the packet carries it, R never looks
/// one up outside a lent call. Set once by ReadSwitches.
inline bool g_e_pipeline{};

/// Register word `index` of the engine was written on E (one_follower_view.cpp). Words written on
/// another thread (the front's own copy) are not E's and are ignored there.
void NoteRegWrite(u32 index) noexcept;

namespace Detail {
/// True on E while FRITCHII_ONEF=1 (set once by MarkEngineThread): ShouldLend's one load.
inline thread_local bool t_engine{};
/// True on R (its thread start).
inline thread_local bool t_render{};
} // namespace Detail

/// fable_b 11.3: acquire latches (Nx2SemaphoreWaiters::OnWrite) by the thread that ran the
/// release: R, E, any other. With one follower every release is a lent call or a render op, so
/// R's count is the latch; E's or another's is what the queries line names.
inline std::atomic<u64> g_latches[3]{};
inline void NoteLatch() noexcept {
    g_latches[Detail::t_render ? 0 : Detail::t_engine ? 1 : 2].fetch_add(
        1, std::memory_order_relaxed);
}

/// Owner asserts (section 5), for the caches' entry points: one thread-local load when off.
void CheckRenderOwned(const char* where) noexcept;
void CheckEngineOwned(const char* where) noexcept;
/// An R-owned cache (texture, pipeline, query) is used here: counted on E unless R is parked or
/// drained ("owner on E").
inline void RenderOwned(const char* where) noexcept {
    if (Detail::t_engine) [[unlikely]] {
        CheckRenderOwned(where);
    }
}
/// The buffer cache (E's) is used here: counted on R outside a lent call ("owner on R").
inline void EngineOwned(const char* where) noexcept {
    if (Detail::t_render) [[unlikely]] {
        CheckEngineOwned(where);
    }
}

} // namespace VideoCommon::OneFollower
