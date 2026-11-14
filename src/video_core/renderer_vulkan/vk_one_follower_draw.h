// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// One follower, part c (work\fable_b_one_follower.md 3, 5 and 8.1): R's draws from the packet and
// the register view, and the verify compares (steps 3 and 4).
//
// The view engine. R's code reads `maxwell3d->regs`, `maxwell3d->dirty.flags` and
// `maxwell3d->draw_manager.draw_state` at about 680 sites. R keeps a Maxwell3D of its own that no
// method ever reaches (no CallMethod, no macros, no shadow RAM): the packets' spans write its
// registers, raise its flags and set its draw state, and a lent packet (E parked) gives it what
// deltas do not carry (the constant buffer bindings, the engine hint, the macro mode). While R
// plays an unlent draw, the R-owned objects (the rasterizer's own engine pointer, the pipeline,
// texture and query caches, the state tracker's flags) point at it; the buffer cache keeps E's
// engine. So the draw runs today's code, PrepareDraw included, on the values the packet carries:
// exact by construction where the view equals the engine, which steps 2 and 3 verify.
//
// The two engines' dirty flags. A lent draw reads E's engine and clears its flags; an unlent draw
// reads the view and clears the view's. A side that did not consume a flag keeps it raised, so
// its next build redoes that part from registers equal to the other side's (conservative, exact).
// After every lent draw this file runs (RunLent), the view's R-owned flags take the engine's.
//
// Draw kinds R plays at step 4 (UnlendKind): PlainIndexed, PlainArrays and InstanceLoop. Indirect,
// List, Mesh, GpuCount and InlineIndex keep their bits but stay lent here (KindNotInMask): their
// R side needs values E owns (indirect words, the GPU count's buffer, the staging pool of mesh
// emulation, the inline index vector).

#include <array>
#include <atomic>
#include <memory>
#include <vector>

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/one_follower.h"
#include "video_core/renderer_vulkan/vk_mem_stage.h"

namespace Vulkan {

class GraphicsPipeline;
class RasterizerVulkan;

class OneFollowerDraw final : public VideoCommon::OneFollower::DrawPlayer {
public:
    using Maxwell3D = Tegra::Engines::Maxwell3D;
    using PacketHeader = VideoCommon::OneFollower::PacketHeader;
    using LentReason = VideoCommon::OneFollower::LentReason;

    explicit OneFollowerDraw(RasterizerVulkan& rasterizer);
    ~OneFollowerDraw();

    OneFollowerDraw(const OneFollowerDraw&) = delete;
    OneFollowerDraw& operator=(const OneFollowerDraw&) = delete;

    /// RasterizerVulkan::Draw's first step. True: the draw is done here (on E: lent and run on R,
    /// or filled into ring B and pushed; on R inside a foreign lent call: run with the verify).
    /// False: go on with Draw (steps 0 to 2, or R inside this file's own call).
    bool RouteDraw(bool is_indexed, u32 instance_count);

    // DrawPlayer (R).
    void OnPacket(const PacketHeader& header, const u8* spans) override;
    void PlayDraw(const PacketHeader& header) override;
    void Condition(bool engine_execute_on) override;

    /// R, at GraphicsPipeline::ConfigureDraw's descriptor count mismatch on an unlent draw: logs
    /// once, at Critical, what E's fill used for the packet (program words and facts) against R's
    /// pipeline (its programs, the view's program words, descriptors per stage), and the packet.
    static void ReportDescriptorMismatch(const GraphicsPipeline& pipeline, u32 expected,
                                         u32 got);

    /// R, in PrepareDraw: the header of the unlent draw being played, or null.
    [[nodiscard]] const PacketHeader* Playing() const noexcept {
        return playing;
    }

    /// Played, dropped (0xe2c False) and verified draws since boot (part d's lines).
    struct Counts {
        std::atomic<u64> played{};
        std::atomic<u64> dropped{};
        std::atomic<u64> verified{};
        std::atomic<u64> pushed{}; ///< E
    };
    [[nodiscard]] const Counts& Totals() const noexcept {
        return counts;
    }

private:
    /// The R-owned objects read the view engine for one played draw, and E's after.
    class ViewScope;

    // E.
    [[nodiscard]] u32 Classify(Maxwell3D& engine, bool is_indexed, u32 instance_count) const;
    /// The first lend reason that applies to a draw of `kind` whose bit is in the mask, before
    /// the fill (Verify, StreamWrite, Diagnostic, Transient); Count when none.
    [[nodiscard]] LentReason UnlentBlocker(Maxwell3D& engine, bool is_indexed, u32 instance_count,
                                           u32 kind);
    /// Fills ring B for an unlent draw from E's engine (as GPUMem's StageFillDraw does from its
    /// copy). False with `reason` when the draw must be lent.
    bool FillRing(Maxwell3D& engine, bool is_indexed, u64 position, u64 seq, u64& buffer_number,
                  LentReason& reason);
    void LendDraw(bool is_indexed, u32 instance_count, LentReason reason, bool verify);

    // R.
    /// A lent draw, E parked: the compares when `verify`, Draw, then the view's flags re-synced.
    void RunLent(bool is_indexed, u32 instance_count, bool verify);
    void Compare(Maxwell3D& engine, bool with_flags);
    void ResyncFlags(const Maxwell3D& engine);
    /// The view takes the spans of one packet.
    void ApplyViewSpans(const PacketHeader& header, const u8* spans);
    /// At a lent packet (E parked): the view takes from E's engine what deltas do not carry, or
    /// everything (`full`).
    void SyncFromEngine(Maxwell3D& engine, bool full);
    /// A dropped draw's packet: its sections played, nothing recorded.
    void DropDraw(const PacketHeader& header);
    void UpdateDiagnostic();
    /// The 0xe2c mode as Maxwell3D::ProcessQueryCondition decides it (no host condition), on
    /// `engine`'s registers and the compare words in game memory.
    [[nodiscard]] static bool ConditionMode(Maxwell3D& engine);

    RasterizerVulkan& rasterizer;

    // R only (E reads the synced_* fields below only after a lent packet released it).
    std::unique_ptr<Maxwell3D> view;
    const Maxwell3D* view_source{}; ///< the engine the view follows (its channel)
    const PacketHeader* playing{};
    bool nested{};                  ///< inside RunLent's own Draw call
    bool render_mode_on{true};      ///< R's 0xe2c mode
    /// Lent packets since the view's flags last took the engine's (RunLent, a seed): the step-3
    /// flags compare holds only at 1 (the draw's own packet).
    u32 lent_since_resync{2};
    VideoCommon::OneFollower::DrawValues last_draw{};
    bool have_draw{};
    /// The R-owned and shared dirty flags (OwnerOf(flag) != FlagOwner::E).
    Maxwell3D::DirtyState::Flags render_flags{};

    // Written by R at lent packets (E parked), read by E.
    bool view_ready{};
    const Maxwell3D* synced_engine{};
    Maxwell3D::EngineHint synced_engine_state{};
    bool synced_native{};
    VideoCommon::OneFollower::StepLevel seeded_step{};

    /// What E's fill used for a pushed draw (E writes before the push, R reads at a mismatch).
    struct FillRecord {
        u64 seq;
        u64 key;
        std::array<u64, VideoCommon::MemStageShadow::ProgramWordCount> words;
        std::array<u64, 2> hashes;
        std::array<u32, 5> uniform_masks;
        std::array<u8, 5> ssbo_counts;
        u32 stages;
        u64 buffer_number;
    };
    static constexpr size_t FillRecords = 64;
    std::array<FillRecord, FillRecords> fill_records{};
    std::atomic<bool> mismatch_reported{};

    // E only.
    u64 e_draw_seq{};
    u64 unlent_candidates{};
    MemStage::StagedFacts facts{};
    std::vector<u32> handle_values;

    /// Diagnostics that read the draw on R (frame dump, render trace, flash watch): set by R.
    std::atomic<bool> diagnostic{};
    bool env_traces{}; ///< debug environment traces in Draw's head (read once)

    Counts counts;
};

} // namespace Vulkan
