// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace Tegra {
class MemoryManager;
}
namespace Tegra::Engines {
class Maxwell3D;
}

namespace VideoCommon {

class PageFilter;
struct MemStageShadowEngine;

/// Memory stage (fable_memstage_step6_spec.md 3.3): what a ring Call entry does, as the stage
/// reads it, in bits 32-39 of the entry's header (GpuSplit::Defer's and CallWithPayload's
/// `flags`; Execute ignores them). Level (0) means unknown: the back runs it and the stage copies
/// its registers afterwards, which is always correct. A tag other than Level and Pass promises a
/// fixed payload layout, given below; the closures that carry one are plain structs (a lambda's
/// captures are no layout to rely on), with static_asserts on the words at their call site.
enum class StageTag : u8 {
    Level = 0,
    /// The rasterizer switches channel. Payload word 0: the bind id (control/scheduler.cpp).
    Channel = 1,
    /// Maxwell3D::ApplyDrawPacket: [engine, system, count | checks << 32, then count writes as
    /// method | argument << 32, then the checks]. Each write as a Direct method entry.
    Writes = 2,
    /// Maxwell3D::FinishFrontMacro: [engine, system]. ConsumeSink, then DrawDeferred.
    FinishMacro = 3,
    /// A draw manager change, StageDrawLoop in header bits 40-47 (StageDrawLoopBits). Payload:
    /// FoldFirstInstance [engine]; AddLoopInstances [engine, count in the low 32 bits].
    DrawLoop = 4,
    /// Touches no register, no buffer cache state and no game memory the stage reads.
    Pass = 5,
    /// Uses the buffer cache or another back-side object; writes no 3D state (memstage_shadow.cpp).
    Cache = 6,
};

/// StageTag::DrawLoop's kind, in header bits 40-47 (the payloads stay as small as the lambdas
/// they replaced: one word and two).
enum class StageDrawLoop : u8 {
    FoldFirstInstance = 0, ///< DrawManager::FoldFirstInstance; payload [engine]
    AddLoopInstances = 1,  ///< DrawManager::AddLoopInstances(count); payload [engine, count]
    ListDraw = 2,          ///< the list draw's address and keep id; payload [engine, address, id]
};

/// The bits a tag takes in a Call entry's header.
[[nodiscard]] constexpr u64 StageTagBits(StageTag tag) noexcept {
    return u64{static_cast<u8>(tag)} << 32;
}

[[nodiscard]] constexpr StageTag StageTagOf(u64 header) noexcept {
    return static_cast<StageTag>((header >> 32) & 0xff);
}

/// StageTag::DrawLoop with its kind, for a Call entry's header.
[[nodiscard]] constexpr u64 StageDrawLoopBits(StageDrawLoop which) noexcept {
    return StageTagBits(StageTag::DrawLoop) | (u64{static_cast<u8>(which)} << 40);
}

[[nodiscard]] constexpr StageDrawLoop StageDrawLoopOf(u64 header) noexcept {
    return static_cast<StageDrawLoop>((header >> 40) & 0xff);
}

/// Memory stage piece 5b (spec 6.2), FRITCHII_MEMSTAGE_SHADOW=1: the stage's classify, register
/// copy, draw manager copy and handlers run on the back thread on each ring entry just before the
/// back executes it, and produce nothing; every draw the back issues is compared with the
/// shadow's. memstage_shadow.cpp has the details. Off, each hook below is one test of g_on.
namespace MemStageShadow {

/// FRITCHII_MEMSTAGE_SHADOW (boot): read by ReadSwitch before the back thread starts. Atomic
/// (relaxed): FlushRegion and FlushCaching test it on CPU threads too, while a new session's
/// ReadSwitch may write it.
inline std::atomic<bool> g_on{};

/// Raised by the state tracker's flag invalidations while on (compare b of piece 5c).
inline std::atomic<u64> g_flag_invalidations{};
inline void NoteFlagInvalidation() noexcept {
    if (g_on.load(std::memory_order_relaxed)) [[unlikely]] {
        g_flag_invalidations.fetch_add(1, std::memory_order_relaxed);
    }
}

/// Reads the switch (GpuSplit::MaybeStart, before the back thread exists).
void ReadSwitch();
/// FRITCHII_MEMSTAGE_SHADOW's value (0 when off).
[[nodiscard]] int Mode();
/// The back thread starts reading the ring at position 0 (GpuSplit::BackLoop).
void Start();
/// The back is about to execute the entry at ring position `position` (GpuSplit::BackLoop).
void Before(u64 position);

/// The back's draw manager issues a draw (DrawManager::ProcessDraw / ProcessDrawIndirect, before
/// the render-enable test): compares a and b.
void BackDraw(Tegra::Engines::Maxwell3D& maxwell3d, bool indexed, u32 instances, bool indirect);
/// The back bound channel `bind_id` (the Channel-tagged closure).
void BackChannel(s32 bind_id);
/// The back's render-enable decision: true when host conditional rendering took it.
void BackCondition(bool host);
/// MemoryManager::FlushCaching on the back: the device ranges it hands InnerInvalidation
/// (compare d).
void BackFlush(const Tegra::MemoryManager& memory,
               std::span<const std::pair<DAddr, std::size_t>> ranges);

/// What the rasterizer knows of a draw once it has its pipeline (compare c, the draw kinds).
enum PipelineKind : u32 {
    PipelineTexel = 1,      ///< texel buffers: LateFill, a level draw
    PipelineDiagnostic = 2, ///< a diagnostic reads the bound buffers on the back
};
/// The pipeline of the draw the back is preparing: its program set and the digest of its facts.
void BackPipeline(const std::array<u64, 6>& hashes, u64 facts_digest, u32 kinds);

/// PrepareDraw: true after the draw's FillDraw, false before its record (draw_func). Read
/// flushes and collected-write applies in between are counted (the buffer half there would make
/// the draw level in step 6).
void BackDrawWindow(bool open);
/// PrepareDraw, before the vertex repack and the fill (piece 5c). SHADOW=3: the stage's copy the
/// fill reads instead of the engine, or null.
MemStageShadowEngine* BackFill(Tegra::Engines::Maxwell3D& maxwell3d, bool indirect);
/// The rasterizer wrote a GPU count into the back's draw state.
void BackDrawStateWritten();
/// The OoT trackers' record in the draw packet against today's query (mismatching slots, a mask).
void BackTrackerRecord(bool had_record, u32 mismatches);
/// RasterizerVulkan::FlushRegion (and FlushRegionForRead: all three) on the back: which caches'
/// halves run.
void BackFlushRegion(bool texture, bool buffer, bool query);
/// RasterizerVulkan::OnCacheInvalidations (`batch`, with its range count) or OnCacheInvalidation
/// (one range; the batch's fallback path calls it per range too) on the back.
void BackCacheInvalidation(bool batch, u64 ranges);

/// The texture cache's "pages that ever held an image" filter: constant buffer words onto those
/// pages make their entry level (join 2, spec 3.2). Set by the rasterizer at creation.
void SetWatchFilter(const PageFilter* filter);

// Piece 6b, FRITCHII_MEMSTAGE_THREAD=2 (memstage_shadow.cpp, memory_stage_link.cpp).

/// Set by the back at a frame end: every draw is level (a diagnostic, host conditional rendering).
inline std::atomic<bool> g_ring_level_draws{};

struct StageResult {
    size_t size;
    u8 cls; ///< pass, work, level, neutral
    bool hard;
    u32 packets;
};
/// The back: ring mode starts (its hooks off) or ended.
void SetRingBack(bool on);
/// GPUMem: the stage's state starts over (every copy taken from the back at its first use).
void StageStart(s32 channel);
StageResult StageProcess(u64 position, bool in_run);
[[nodiscard]] bool StageExtendsRun(u64 position);
void StageLend();
/// Step 7, the window open: StageLend's flags are kept, and the back sets them at the run's first
/// entry (BackTakeLend, on the back; the publish of the run orders the two).
void StageLendDeferred();
void BackTakeLend();
/// Step 7: whether processing the entry at `position` may take a full copy from the back (a copy
/// that is new or stale): GPUMem must be level there first.
[[nodiscard]] bool StageMayCopy(u64 position);
/// Returns the full copies it made.
u64 StageReturn(const void* drew_engine, const void* state_engine, s32 back_channel);
void StageTick();
[[nodiscard]] std::vector<std::pair<u64, std::size_t>>& StageRanges();
/// Sets the stage's draw count to `back_seq`; returns what it was.
u64 StageDrawSeq(u64 back_seq);
[[nodiscard]] u64 BackStateHash(const Tegra::Engines::Maxwell3D& maxwell3d);
/// The registers that select the programs (the six pipeline slots, then the program region), as
/// 8-byte words, and their hash: the facts table's key. The table compares the words on a key
/// match, so a hash collision is a miss, never another program's facts.
inline constexpr std::size_t ProgramWordCount = 49;
struct ProgramId {
    u64 key{};
    std::array<u64, ProgramWordCount> words{};
};
void BackProgramId(const Tegra::Engines::Maxwell3D& maxwell3d, ProgramId& out);
void StageProgramId(const MemStageShadowEngine& copy, ProgramId& out);

} // namespace MemStageShadow

} // namespace VideoCommon
