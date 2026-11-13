// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <span>
#include <utility>

#include "common/common_types.h"

namespace Tegra {
class MemoryManager;
}
namespace Tegra::Engines {
class Maxwell3D;
}

namespace VideoCommon {

class PageFilter;

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
    /// A draw manager change: [engine, StageDrawLoop, value].
    DrawLoop = 4,
    /// Touches no register, no buffer cache state and no game memory the stage reads.
    Pass = 5,
};

/// StageTag::DrawLoop's payload word 1.
enum class StageDrawLoop : u64 {
    FoldFirstInstance = 0, ///< DrawManager::FoldFirstInstance; no value
    AddLoopInstances = 1,  ///< DrawManager::AddLoopInstances(value)
};

/// The bits a tag takes in a Call entry's header.
[[nodiscard]] constexpr u64 StageTagBits(StageTag tag) noexcept {
    return u64{static_cast<u8>(tag)} << 32;
}

[[nodiscard]] constexpr StageTag StageTagOf(u64 header) noexcept {
    return static_cast<StageTag>((header >> 32) & 0xff);
}

/// Memory stage piece 5b (spec 6.2), FRITCHII_MEMSTAGE_SHADOW=1: the stage's classify, register
/// copy, draw manager copy and handlers run on the back thread on each ring entry just before the
/// back executes it, and produce nothing; every draw the back issues is compared with the
/// shadow's. memstage_shadow.cpp has the details. Off, each hook below is one test of g_on.
namespace MemStageShadow {

/// FRITCHII_MEMSTAGE_SHADOW (boot): read by ReadSwitch before the back thread starts.
inline bool g_on{};

/// Reads the switch (GpuSplit::MaybeStart, before the back thread exists).
void ReadSwitch();
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
/// RasterizerVulkan::FlushRegion (and FlushRegionForRead: all three) on the back: which caches'
/// halves run.
void BackFlushRegion(bool texture, bool buffer, bool query);
/// RasterizerVulkan::OnCacheInvalidations (`batch`, with its range count) or OnCacheInvalidation
/// (one range; the batch's fallback path calls it per range too) on the back.
void BackCacheInvalidation(bool batch, u64 ranges);

/// The texture cache's "pages that ever held an image" filter: constant buffer words onto those
/// pages make their entry level (join 2, spec 3.2). Set by the rasterizer at creation.
void SetWatchFilter(const PageFilter* filter);

} // namespace MemStageShadow

} // namespace VideoCommon
