// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"

namespace Tegra::Engines {
class EngineInterface;
}

/// FRITCHII_ESPLIT_COUNT (live, default 0): step S0 of the E split (work\E_SPLIT_DESIGN.md 4.2),
/// dry. The split would cut E by memory: E1 applies the register entries, runs the draw manager,
/// looks the pipeline up and packs R's spans; E2 writes constant buffer words, fills, owns the
/// buffer cache, fences and compute. Nothing here runs a thread or changes a value.
///
/// 1 = every ring entry E runs is classed E1, E2, join (E1 waits for E2 to drain, then runs it
///     itself) or lend by its kind (GpuSplit::BackLoop decodes it); the E2 parts inside an E1
///     entry (the draw's fill and push, a bind's buffer cache half), the E1 parts inside E2 work
///     (R's span packing, the method sink's register writes) and every lend are carved out by
///     time. 1 entry in 64 is timed
///     (time stamp ticks, picked at random, scaled by 64). Counted: the points where E1 would
///     stall for E2 (joins by cause, E1 decisions that read game memory, a pipeline lookup after
///     a stream write onto shader pages), conditions from memory (forwarded, decided on E2).
/// 2 = 1, and the hand-over is rehearsed on E's own thread: at each E2 entry and each E fill,
///     E1's record (the watched register words written since the last record, the constant
///     buffer bindings and root table words changed, the draw state by value) is built into a
///     ring and consumed at once into E2's view; the view is compared with the engine (must be
///     0; the first 16 differences logged with the register index). Build and consume are timed
///     1 in 64 (the hand-over cost).
/// 3 = 2, and E's draw fill reads the view (BufferCache::SetInputsOverride and the handle values'
///     constant buffer bindings) whenever the compare before it found it equal; every input the
///     fill can read through CacheInputs is compared again after the fill (must be 0).
///
/// Lines every 10 s start "E split count (FRITCHII_ESPLIT_COUNT=N) 10 s:".
///
/// Off: per ring batch one relaxed load (the switch); per entry a test of a local; per filled
/// draw two plain loads of g_state (FillAndPush, PushDraw) and one of g_records (FillRing); per
/// lend, constant buffer bind, root table load, condition, stream write and non-empty method sink
/// one plain load. The register write hook shares the one follower's key-watch byte (no added
/// load).
namespace VideoCommon::ESplit {

using Maxwell3D = Tegra::Engines::Maxwell3D;

enum class Class : u8 {
    E1,   ///< leaves E2's thread: registers, draw manager, pipeline lookup, span packing
    E2,   ///< stays: constant buffer words, fills, buffer cache, fences, compute, guest-visible
    Join, ///< E1 waits for E2 to drain, then runs it itself (exact by construction)
    Lend, ///< R runs it, E parked (both stages park in the split)
    Count,
};

enum class JoinCause : u8 {
    Upload,       ///< 3D inline upload (launch_dma, inline_data)
    OtherEngine,  ///< a 2D, DMA or inline-to-memory engine's executed method
    UntaggedCall, ///< a ring call with no memory stage tag (unclassified)
    Channel,      ///< the rasterizer's channel switch
    Count,
};

enum class StallCause : u8 {
    MemoryDecision, ///< E1's own decision reads game memory (indirect reset test, list on CPU)
    ShaderWrite,    ///< a pipeline lookup after an E2 stream write onto shader pages
    Count,
};

/// GpuSplit::BackLoop's decode of a ring entry (it knows the format).
struct EntryInfo {
    Class cls{Class::E1};
    JoinCause cause{JoinCause::Count};
    /// 3D constant buffer data: the entry advances the write offset (E2 does that itself).
    bool cb_entry{};
    /// Its words when the decode knows them (Multi, CbWords, a single word with a segment);
    /// 0 with cb_entry: the words Execute ran (a run of single words).
    u32 cb_words{};
    /// An E2 entry of the 3D engine whose Execute consumes the method sink first (an executed
    /// method, a multi write, a words block): at 2 and 3 E1 applies the sink before the hand-over,
    /// as the split would (the same order: Execute would do it first thing).
    bool sink_first{};
};

inline constexpr u8 StateOn = 1;       ///< FRITCHII_ESPLIT_COUNT >= 1
inline constexpr u8 StateSampling = 2; ///< E is inside a timed entry
/// Written by E only (BatchLevel, BeforeEntry, AfterEntry). Every hook tests it first.
inline u8 g_state{};
/// FRITCHII_ESPLIT_COUNT >= 2: the E1 side of the records is fed (bindings, root words; the
/// register words through OneFollower::g_write_watch).
inline bool g_records{};

/// GpuSplit::BackLoop, once a batch: the switch's value (0 to 3); the state follows it and the
/// lines are written from here.
[[nodiscard]] int BatchLevel() noexcept;

enum class EngineKind : u8 { Maxwell3D, Compute, Other, None };
/// The engine's kind (the last one asked is remembered: the decode asks per entry).
[[nodiscard]] EngineKind KindOf(const Tegra::Engines::EngineInterface* engine) noexcept;

/// Around GpuSplit::Execute of each entry (FRITCHII_ESPLIT_COUNT >= 1). `size`: what it returned.
void BeforeEntry(const EntryInfo& info) noexcept;
void AfterEntry(std::size_t size) noexcept;

// --- Carves: a part of an entry whose time goes to another class ----------------------------------

[[nodiscard]] bool PushCarve(Class cls) noexcept;
void PopCarve() noexcept;

class Carve {
public:
    explicit Carve(Class cls) noexcept : Carve(cls, g_state) {}
    /// With g_state read once by the caller.
    Carve(Class cls, u8 state) noexcept {
        if ((state & StateSampling) != 0) [[unlikely]] {
            pushed = PushCarve(cls);
        }
    }
    ~Carve() {
        if (pushed) [[unlikely]] {
            PopCarve();
        }
    }
    Carve(const Carve&) = delete;
    Carve& operator=(const Carve&) = delete;

private:
    bool pushed{};
};

[[nodiscard]] bool BeginLend() noexcept;
void EndLend() noexcept;

/// OneFollower::Lend on E: the lend (pack, park, R's run) is class Lend, counted; at 2 and 3 the
/// view takes the engine's buffer flags after it (both stages are parked in a lend).
class LendScope {
public:
    LendScope() noexcept {
        if (g_state != 0) [[unlikely]] {
            active = BeginLend();
        }
    }
    ~LendScope() {
        if (active) [[unlikely]] {
            EndLend();
        }
    }
    LendScope(const LendScope&) = delete;
    LendScope& operator=(const LendScope&) = delete;

private:
    bool active{};
};

// --- Counters (E) --------------------------------------------------------------------------------

void NoteStallImpl(StallCause cause) noexcept;
void NoteConditionImpl() noexcept;
void NoteStreamWriteImpl(bool shader_pages) noexcept;
void NoteLookupImpl() noexcept;
void NoteBackRaiseImpl() noexcept;
void NoteDrawFilledImpl() noexcept;
void SetEntryClassImpl(Class cls) noexcept;

/// A stall cause inside an E1 entry.
inline void NoteStall(StallCause cause) noexcept {
    if (g_state != 0) [[unlikely]] {
        NoteStallImpl(cause);
    }
}
/// A render condition decided from game memory (E1 forwards the draws, E2 decides).
inline void NoteCondition() noexcept {
    if (g_state != 0) [[unlikely]] {
        NoteConditionImpl();
    }
}
/// E's stream write (OneFollower::GuardStreamWrite): `shader_pages` when it lands on them.
inline void NoteStreamWrite(bool shader_pages) noexcept {
    if (g_state != 0) [[unlikely]] {
        NoteStreamWriteImpl(shader_pages);
    }
}
/// E's pipeline lookup for a draw it fills (E1's lookup in the split).
inline void NoteLookup(u8 state) noexcept {
    if (state != 0) [[unlikely]] {
        NoteLookupImpl();
    }
}
/// The fill raised a flag on the engine for R (the repack layout): an E2 to E1 write.
inline void NoteBackRaise() noexcept {
    if (g_state != 0) [[unlikely]] {
        NoteBackRaiseImpl();
    }
}
/// E filled and pushed a draw.
inline void NoteDrawFilled(u8 state) noexcept {
    if (state != 0) [[unlikely]] {
        NoteDrawFilledImpl();
    }
}
/// The running entry is of class `cls` after all (a call the decode could not tell apart).
inline void SetEntryClass(Class cls) noexcept {
    if (g_state != 0) [[unlikely]] {
        SetEntryClassImpl(cls);
    }
}

// --- Records (2 and 3), the E1 side ---------------------------------------------------------------

/// Register word `index` was written on E (or by R in a lent call, E parked): from
/// OneFollower::NoteRegWrite while g_write_watch has WriteWatchSplit.
void NoteWord(u32 index) noexcept;
void NoteCbBindImpl(std::size_t stage, u32 slot) noexcept;
void NoteRootWordImpl(u32 table, u32 index) noexcept;

/// Maxwell3D::ProcessCBBind: binding (stage, slot) changed.
inline void NoteCbBind(std::size_t stage, u32 slot) noexcept {
    if (g_records) [[unlikely]] {
        NoteCbBindImpl(stage, slot);
    }
}
/// Maxwell3D's root table load: word `index` of table `table`.
inline void NoteRootWord(u32 table, u32 index) noexcept {
    if (g_records) [[unlikely]] {
        NoteRootWordImpl(table, index);
    }
}

// --- E2's view (2 and 3) ---------------------------------------------------------------------------

/// E2's copy of what the fill reads, fed only by records (seeded from the engine at the first
/// hand-over, a channel change, or when the write tracking was off).
struct View {
    Maxwell3D::Regs regs{};
    Maxwell3D::State state{};
    Maxwell3D::RootTables root_tables{};
    /// The draw manager's state by value (its inline index bytes stay empty).
    Maxwell3D::DrawManager::State draw_state{};
    Maxwell3D::DrawManager::IndirectParams indirect{};
    /// The buffer flags (IndexBuffer, VertexBuffers, VertexBuffer0..31): E2's in the split.
    Maxwell3D::DirtyState::Flags flags{};
    const Maxwell3D* engine{};
    bool seeded{};
};

/// FillRing, the buffer cache's lock held, before the fill reads anything (only while
/// g_records): E1's draw record built and consumed, the view compared. At 3 it returns the view
/// when the fill may read it (equal to the engine, no inline index bytes), else null.
[[nodiscard]] const View* DrawHandOver(const Maxwell3D& engine, bool list_args) noexcept;
/// After that fill: the view takes the buffer flags the fill left; with `used` (3) every input
/// the fill can read is compared again.
void AfterFill(const Maxwell3D& engine, const View* used) noexcept;

} // namespace VideoCommon::ESplit
