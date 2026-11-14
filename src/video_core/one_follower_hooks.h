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

/// Register word `index` of the engine was written on E (one_follower_view.cpp). Words written on
/// another thread (the front's own copy) are not E's and are ignored there.
void NoteRegWrite(u32 index) noexcept;

namespace Detail {
/// True on E while FRITCHII_ONEF=1 (set once by MarkEngineThread): ShouldLend's one load.
inline thread_local bool t_engine{};
/// True on R (its thread start).
inline thread_local bool t_render{};
} // namespace Detail

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
