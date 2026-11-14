// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// One follower, part c: R's draws from the packet and the view, and the verify compares
// (vk_one_follower_draw.h has the model; work\fable_b_one_follower.md 3, 5, 8.1).
//
// Where the design and the code disagree, the code wins (written down here):
// - "R plays the buffer sections (MemPlayer)": R takes the draw's ring B packet with TakeRingDraw
//   and PrepareDraw plays it as a served draw (the THREAD = 2 path), so Configure binds nothing in
//   the buffer cache. E fills it with FillRingDraw, as GPUMem does, from its own engine.
// - "E finds the pipeline's layout": E looks the facts up under BackProgramId's key (LookupFacts);
//   PrepareDraw publishes them at step 4. A miss is LentReason::PipelineMiss (R has not seen the
//   programs yet), so the first draw of a pipeline is always lent.
// - The view engine's constant buffer bindings (Maxwell3D::state) come only with lent packets: the
//   R-side readers of them (EDEN_NX2_TEX_BANK_GEN, EXP87's template, the handle verify) make every
//   draw lent while they are on. Handle words always come in the packet (ValuesByValue required).
// - The engine hint (engine_state) and the macro mode (native_turing) are not registers: E lends
//   the draw when either differs from what the last lent packet gave the view.
// - TickWork and the stream's invalidations (FlushCaching) stay E's: E runs them around the push,
//   R skips them in PrepareDraw. FlushWork (the scheduler's) is R's.
// - 0xe2c: in this branch E still decides every condition (lent) and skips False draws itself;
//   R tracks its own mode at each condition (NoteCondition) and compares it (BuildCondition), and
//   drops PacketConditional draws while it is False. Step 5 makes E send those.

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <span>
#include <string_view>
#include <type_traits>

#include "common/dkb_frame_trace.h"
#include "common/scope_exit.h"
#include "video_core/fritchii_delta_path.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/memory_stage_link.h"
#include "video_core/memstage_shadow.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_fastdraw_count.h"
#include "video_core/nx2_indirect_grid.h"
#include "video_core/render_trace.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/vk_one_follower_draw.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/texture_handle_values.h"

namespace Vulkan {

namespace OneF = VideoCommon::OneFollower;
using OneF::Check;
using OneF::LentOp;
using OneF::StepLevel;
using OneF::UnlendKind;
using Maxwell = Tegra::Engines::Maxwell3D::Regs;

namespace {

/// The kinds R has a path for at step 4; the other bits stay lent (vk_one_follower_draw.h).
constexpr u32 PlayableKinds = OneF::PlainIndexed | OneF::PlainArrays | OneF::InstanceLoop;

template <typename T>
[[nodiscard]] bool SameBytes(const T& a, const T& b) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

[[nodiscard]] constexpr size_t Padded(size_t bytes) noexcept {
    return (bytes + 7) & ~size_t{7};
}

/// EDEN_NX2_INVALIDATE_BEFORE_DISPATCH's draw form (vk_rasterizer.cpp's InvalidateBeforeUse).
[[nodiscard]] bool InvalidateBeforeUseOn() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_INVALIDATE_BEFORE_DISPATCH");
        return value != nullptr && value[0] != '0' && std::string_view{value} != "list";
    }();
    return enabled;
}

/// The key FixedPipelineState::Refresh builds from `engine` with every refresh step forced
/// (VertexInput, ViewportSwizzles; blending is always rebuilt without dynamic blend state). The
/// engine's flags are given back as they were.
FixedPipelineState FullKey(Tegra::Engines::Maxwell3D& engine, DynamicFeatures features) {
    static_assert(std::is_trivially_copyable_v<FixedPipelineState>);
    FixedPipelineState key;
    std::memset(&key, 0, sizeof(key));
    const auto saved = engine.dirty.flags;
    engine.dirty.flags[Dirty::VertexInput] = true;
    engine.dirty.flags[Dirty::ViewportSwizzles] = true;
    engine.dirty.flags[Dirty::Blending] = true;
    key.Refresh(engine, features);
    engine.dirty.flags = saved;
    return key;
}

/// The flags each build part consumes (attribution of a missing flag; Compare).
enum class FlagPart : u8 { Pipeline, RenderTargets, ImageViews, Dynamic, Other };

[[nodiscard]] FlagPart PartOf(size_t flag) noexcept {
    namespace CommonDirty = VideoCommon::Dirty;
    if (flag == CommonDirty::Shaders || flag == Dirty::ViewportSwizzles ||
        flag == Dirty::Blending) {
        return FlagPart::Pipeline;
    }
    if (flag >= CommonDirty::RenderTargets && flag <= CommonDirty::RescaleScissors) {
        return FlagPart::RenderTargets;
    }
    if (flag == CommonDirty::Descriptors) {
        return FlagPart::ImageViews;
    }
    if (flag > Dirty::First && flag < Dirty::Last) {
        return FlagPart::Dynamic;
    }
    return FlagPart::Other;
}

} // Anonymous namespace

// --- The view engine's scope ---------------------------------------------------------------------

class OneFollowerDraw::ViewScope {
public:
    ViewScope(RasterizerVulkan& r_, Maxwell3D& engine) : r{r_} {
        rasterizer_engine = r.ExchangeMaxwell3D(&engine);
        pipeline_engine = r.pipeline_cache.ExchangeMaxwell3D(&engine);
        texture_engine = r.texture_cache.ExchangeMaxwell3D(&engine);
        query_engine = r.query_cache.ExchangeMaxwell3D(&engine);
        tracker_flags = r.state_tracker.ExchangeFlags(&engine.dirty.flags);
    }
    ~ViewScope() {
        r.state_tracker.ExchangeFlags(tracker_flags);
        r.query_cache.ExchangeMaxwell3D(query_engine);
        r.texture_cache.ExchangeMaxwell3D(texture_engine);
        r.pipeline_cache.ExchangeMaxwell3D(pipeline_engine);
        r.ExchangeMaxwell3D(rasterizer_engine);
    }
    ViewScope(const ViewScope&) = delete;
    ViewScope& operator=(const ViewScope&) = delete;

private:
    RasterizerVulkan& r;
    Maxwell3D* rasterizer_engine{};
    Maxwell3D* pipeline_engine{};
    Maxwell3D* texture_engine{};
    Maxwell3D* query_engine{};
    Maxwell3D::DirtyState::Flags* tracker_flags{};
};

// --- Construction ----------------------------------------------------------------------------------

OneFollowerDraw::OneFollowerDraw(RasterizerVulkan& rasterizer_) : rasterizer{rasterizer_} {
    for (size_t flag = 0; flag < render_flags.size(); ++flag) {
        render_flags[flag] = OneF::OwnerOf(static_cast<u8>(flag)) != OneF::FlagOwner::E;
    }
    env_traces = std::getenv("EDEN_NX2_CB24_TRACE") != nullptr ||
                 std::getenv("EDEN_NX2_TRACE_VS") != nullptr ||
                 std::getenv("EDEN_NX2_RT_TRACE") != nullptr ||
                 std::getenv("EDEN_NX2_WINDOW_TRACE") != nullptr ||
                 std::getenv("EDEN_NATIVE_DRAW_CAPTURE") != nullptr ||
                 std::getenv("EDEN_DKB_INDIRECT_READBACK") != nullptr;
}

OneFollowerDraw::~OneFollowerDraw() = default;

// --- E: routing a draw -----------------------------------------------------------------------------

u32 OneFollowerDraw::Classify(Maxwell3D& engine, bool is_indexed, u32 instance_count) const {
    using DrawMode = Maxwell3D::DrawManager::DrawMode;
    const auto& draw = engine.draw_manager.draw_state;
    if (draw.draw_mode == DrawMode::InlineIndex || !draw.inline_index_draw_indexes.empty()) {
        return OneF::InlineIndex;
    }
    if (draw.list_draw_address != 0) {
        return OneF::List;
    }
    if (!is_indexed && Tegra::Nx2IndirectGrid::AnyLive() &&
        Tegra::Nx2IndirectGrid::LooksLikeDrawPlaceholder(draw.vertex_buffer.count)) {
        return OneF::GpuCount;
    }
    // Task/mesh pipelines (RefreshStages: native programs with the mesh stages enabled).
    if (engine.GetNativeMacroEngine().native_turing && engine.regs.IsMeshEnabled()) {
        return OneF::Mesh;
    }
    if (draw.draw_mode == DrawMode::Instance || engine.draw_manager.first_folded) {
        return OneF::InstanceLoop;
    }
    (void)instance_count;
    return is_indexed ? OneF::PlainIndexed : OneF::PlainArrays;
}

OneF::LentReason OneFollowerDraw::UnlentBlocker(Maxwell3D& engine, bool is_indexed,
                                                u32 instance_count, u32 kind) {
    namespace Debug = VideoCore::Nx2Debug;
    // Verify (8.1 point 5): every Nth draw that would run unlent is lent and compared.
    const int every =
        Debug::SwitchValue(Debug::g_onef_verify_every, "FRITCHII_ONEF_VERIFY_EVERY",
                           OneF::DefaultVerifyEvery);
    const bool verify = every > 0 && (++unlent_candidates % static_cast<u64>(every)) == 0;
    // Read every draw, so a hit never waits for a later draw (the next draw after it is lent).
    const bool stream_write = OneF::TakeStreamWriteLend();
    if (verify) {
        return LentReason::Verify;
    }
    if (stream_write) {
        return LentReason::StreamWrite;
    }
    if (diagnostic.load(std::memory_order_relaxed) || env_traces ||
        Common::DkbFrameTrace::Enabled()) {
        return LentReason::Diagnostic;
    }
    // Transient: what R's view or R's path cannot carry yet.
    const auto& draw = engine.draw_manager.draw_state;
    const bool native = engine.GetNativeMacroEngine().native_turing;
    if (!view_ready || synced_engine != &engine || synced_engine_state != engine.engine_state ||
        synced_native != native || seeded_step != OneF::Step()) {
        return LentReason::Transient; // the lent packet seeds or syncs the view
    }
    // Readers of the constant buffer bindings on R (the view has them only from lent packets).
    if (Debug::SwitchValue(Debug::g_tex_bank_gen, "EDEN_NX2_TEX_BANK_GEN") > 0 ||
        Debug::Experiment(87) != 0 || (MemStage::VerifyBits() & MemStage::VerifyHandles) != 0) {
        return LentReason::Transient;
    }
    // Paths that touch the buffer cache or its lock on R, or change who fills.
    if (InvalidateBeforeUseOn() || engine.regs.transform_feedback_enabled != 0 ||
        VideoCommon::Nx2FastDrawCount::g_count ||
        (VideoCommon::Nx2FastDrawCount::g_on &&
         Debug::Experiment(VideoCommon::FritchiiDeltaPath::Switch) != 0) ||
        VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed) ||
        VideoCommon::MemoryStageLink::RingActive()) {
        return LentReason::Transient;
    }
    // Draw's own early outs: skipped past 16M vertices, the experimental indirect reset.
    const u64 first = is_indexed ? draw.index_buffer.first : draw.vertex_buffer.first;
    const u64 count = is_indexed ? draw.index_buffer.count : draw.vertex_buffer.count;
    if (first + count > (u64{1} << 24)) {
        return LentReason::Transient;
    }
    if (!is_indexed && instance_count == 1 && native &&
        draw.topology == Maxwell::PrimitiveTopology::Points) {
        return LentReason::Transient; // a candidate for IsNx2IndirectReset
    }
    (void)kind;
    return LentReason::Count;
}

bool OneFollowerDraw::FillRing(Maxwell3D& engine, bool is_indexed, u64 position, u64 seq,
                               u64& buffer_number, LentReason& reason) {
    auto& r = rasterizer;
    reason = LentReason::Transient;
    // Join 1 (spec 4.4) with E's own registers: the facts R published for these programs.
    VideoCommon::MemStageShadow::ProgramId program_id;
    VideoCommon::MemStageShadow::BackProgramId(engine, program_id);
    if (!r.mem_stage.LookupFacts(program_id, facts)) {
        reason = LentReason::PipelineMiss;
        return false;
    }
    if (facts.has_texel_buffers || facts.infos == nullptr || !r.mem_stage.Packets() ||
        !r.mem_stage.ValuesByValue() || !r.mem_stage.RingRoom() || r.buffer_cache.MarksAhead()) {
        return false; // texel buffers need the texture cache (LateFill); no room or no values
    }
    std::scoped_lock lock{r.buffer_cache.mutex};
    SCOPE_EXIT {
        r.mem_stage.EndStageFill();
    };
    // The repack layout and the cache's flags, as PrepareDraw's own fill sets them.
    const u32 repack_dirty = r.buffer_cache_runtime.SetVertexRepackLayout(
        r.buffer_cache.Inputs().RepackLayoutRegs(), facts.dynamic_vertex_input);
    r.mem_stage.BeginDraw(r.buffer_cache_runtime.RepackMask(), repack_dirty);
    if (repack_dirty) {
        r.buffer_cache.Inputs().MarkVertexBuffersDirty(repack_dirty);
        // The lent draws read E's engine: raise it there too (R raises its own from the packet).
        engine.dirty.flags[Dirty::VertexInput] = true;
    }
    const MemStage::DrawFacts draw_facts{
        .uniform_masks = &facts.uniform_masks,
        .uniform_sizes = &facts.uniform_sizes,
        .stage_infos = nullptr,
        .stages = facts.stages,
        .has_texel_buffers = false,
        .uses_descriptor_buffer = facts.uses_descriptor_buffer,
        .hashes = facts.hashes,
        .staged = &facts,
    };
    // The draw's raw handles at E's position (SEAM 3): R's Configure takes them from the packet.
    const bool null_pointer_handles = VideoCore::Nx2Debug::Experiment(21) != 0;
    handle_values.clear();
    const auto& stages = engine.state.shader_stages;
    using Cbufs = std::remove_cvref_t<decltype(stages[0].const_buffers)>;
    VideoCommon::FillGraphicsHandleValues(
        std::span<const Shader::Info>{*facts.infos}, engine.GetMemoryManager(),
        [&stages](size_t stage) {
            return VideoCommon::GraphicsHandleCbufs<Cbufs>{stages[stage].const_buffers};
        },
        r.device.GetStorageBufferAlignment(), null_pointer_handles,
        [this](u32 raw) { handle_values.push_back(raw); });
    buffer_number = r.mem_stage.NextRingPacketIndex();
    const bool filled =
        r.mem_stage.FillRingDraw(r.buffer_cache, draw_facts, is_indexed, position, seq, 0, nullptr,
                                 nullptr, &handle_values, null_pointer_handles, false);
    return filled;
}

void OneFollowerDraw::LendDraw(bool is_indexed, u32 instance_count, LentReason reason,
                               bool verify) {
    auto body = [this, is_indexed, instance_count, verify] {
        RunLent(is_indexed, instance_count, verify);
    };
    OneF::Lend(LentOp::Draw, reason, body);
}

bool OneFollowerDraw::RouteDraw(bool is_indexed, u32 instance_count) {
    const StepLevel step = OneF::Step();
    if (step < StepLevel::Build) {
        return false; // steps 0 to 2: today's path (part a lends it)
    }
    if (OneF::OnRenderThread()) {
        if (nested || playing != nullptr) {
            return false; // RunLent's or PlayDraw's own call of Draw
        }
        // A lent call from elsewhere (part a) runs this draw on R with E parked.
        RunLent(is_indexed, instance_count, step == StepLevel::Build);
        return true;
    }
    // E. The engine is the channel's (the rasterizer's own pointer names the view while R plays).
    Maxwell3D& engine = rasterizer.channel_state->maxwell3d;
    const u64 seq = ++e_draw_seq;
    if (step < StepLevel::Unlend) {
        LendDraw(is_indexed, instance_count, LentReason::StepBelowUnlend, true);
        return true;
    }
    const u32 kind = Classify(engine, is_indexed, instance_count);
    const u32 mask = static_cast<u32>(VideoCore::Nx2Debug::SwitchValue(
        VideoCore::Nx2Debug::g_onef_unlend, "FRITCHII_ONEF_UNLEND"));
    if ((mask & kind & PlayableKinds) == 0) {
        LendDraw(is_indexed, instance_count, LentReason::KindNotInMask, false);
        return true;
    }
    LentReason reason = UnlentBlocker(engine, is_indexed, instance_count, kind);
    if (reason == LentReason::Count) {
        // Today's PrepareDraw head, E's half: the stream's invalidations before the fill.
        engine.GetMemoryManager().FlushCaching();
        const u64 position = VideoCommon::GpuSplit::Get().BackEntryPosition();
        u64 buffer_number = 0;
        if (FillRing(engine, is_indexed, position, seq, buffer_number, reason)) {
            OneF::PushDraw(engine, seq, buffer_number, 0);
            counts.pushed.fetch_add(1, std::memory_order_relaxed);
            // PrepareDraw's tail, E's half (read-backs wait on E's work).
            rasterizer.gpu.TickWork();
            return true;
        }
    }
    LendDraw(is_indexed, instance_count, reason, reason == LentReason::Verify);
    return true;
}

// --- R: lent draws and the compares ----------------------------------------------------------------

void OneFollowerDraw::RunLent(bool is_indexed, u32 instance_count, bool verify) {
    Maxwell3D& engine = rasterizer.channel_state->maxwell3d;
    nested = true;
    SCOPE_EXIT {
        nested = false;
        if (view != nullptr && view_source == &engine) {
            ResyncFlags(engine);
        }
    };
    if (verify && view != nullptr && view_source == &engine) {
        // The flags compare needs the view's flags re-synced at the last lent operation: a lent
        // clear, query or dispatch in between ran R's code on E's engine, which may raise flags
        // there that the view never sees (not a forwarding miss).
        Compare(engine, OneF::Step() == StepLevel::Build && lent_since_resync == 1);
    }
    rasterizer.Draw(is_indexed, instance_count);
}

void OneFollowerDraw::Compare(Maxwell3D& engine, bool with_flags) {
    counts.verified.fetch_add(1, std::memory_order_relaxed);
    const Maxwell& e = engine.regs;
    const Maxwell& v = view->regs;
    // Pipeline: the key each engine gives (pure, full refresh) and what RefreshStages reads.
    const DynamicFeatures& features = rasterizer.pipeline_cache.Features();
    if (!(FullKey(engine, features) == FullKey(*view, features))) {
        OneF::Note(Check::BuildPipeline, "key");
    } else if (!SameBytes(e.pipelines, v.pipelines) ||
               !SameBytes(e.program_region, v.program_region) ||
               e.rasterize_enable != v.rasterize_enable) {
        OneF::Note(Check::BuildPipeline, "programs");
    }
    // Render targets: the registers UpdateRenderTargets and the framebuffer read.
    if (!SameBytes(e.rt, v.rt) || !SameBytes(e.rt_control, v.rt_control) ||
        !SameBytes(e.zeta, v.zeta) || e.zeta_enable != v.zeta_enable ||
        !SameBytes(e.zeta_size, v.zeta_size) || !SameBytes(e.surface_clip, v.surface_clip) ||
        e.anti_alias_samples_mode != v.anti_alias_samples_mode) {
        OneF::Note(Check::BuildRenderTargets, "registers");
    }
    // Image views and descriptors: the pools and the sampler binding the handles resolve in (the
    // handles themselves come in the packet).
    if (!SameBytes(e.tex_header, v.tex_header)) {
        OneF::Note(Check::BuildImageViews, "TIC pool");
    }
    if (!SameBytes(e.tex_sampler, v.tex_sampler) || e.sampler_binding != v.sampler_binding) {
        OneF::Note(Check::BuildDescriptors, "TSC pool or sampler binding");
    }
    // Draw parameters: what MakeDrawParams and the key's topology read.
    const auto& ed = engine.draw_manager.draw_state;
    const auto& vd = view->draw_manager.draw_state;
    if (ed.topology != vd.topology || ed.draw_mode != vd.draw_mode ||
        ed.base_instance != vd.base_instance || ed.base_index != vd.base_index ||
        ed.instance_count != vd.instance_count || ed.drawn_instances != vd.drawn_instances ||
        !SameBytes(ed.vertex_buffer, vd.vertex_buffer) ||
        !SameBytes(ed.index_buffer, vd.index_buffer)) {
        OneF::Note(Check::BuildDrawParams, "draw state");
    }
    if (!with_flags) {
        return; // step 4: the engine's R-owned flags were not consumed by the unlent draws
    }
    // Step 3: a flag the lent draw consumes now that R's view would not have had (both re-synced
    // after the last lent draw, so the difference is what the packets since forwarded).
    const auto missing = engine.dirty.flags & ~view->dirty.flags & render_flags;
    if (missing.none()) {
        return;
    }
    bool parts[5]{};
    for (size_t flag = 0; flag < missing.size(); ++flag) {
        if (missing[flag]) {
            parts[static_cast<size_t>(PartOf(flag))] = true;
        }
    }
    if (parts[static_cast<size_t>(FlagPart::Pipeline)]) {
        OneF::Note(Check::BuildPipeline, "flag not forwarded");
    }
    if (parts[static_cast<size_t>(FlagPart::RenderTargets)]) {
        OneF::Note(Check::BuildRenderTargets, "flag not forwarded");
    }
    if (parts[static_cast<size_t>(FlagPart::ImageViews)]) {
        OneF::Note(Check::BuildImageViews, "flag not forwarded");
    }
    if (parts[static_cast<size_t>(FlagPart::Dynamic)] ||
        parts[static_cast<size_t>(FlagPart::Other)]) {
        OneF::Note(Check::BuildDynamic, "flag not forwarded");
    }
}

void OneFollowerDraw::ResyncFlags(const Maxwell3D& engine) {
    view->dirty.flags = (view->dirty.flags & ~render_flags) | (engine.dirty.flags & render_flags);
    lent_since_resync = 0;
}

// --- R: the packets --------------------------------------------------------------------------------

void OneFollowerDraw::ApplyViewSpans(const PacketHeader& header, const u8* spans) {
    if (spans == nullptr || view == nullptr) {
        return;
    }
    const u8* cursor = spans;
    if ((header.spans & OneF::SpanDelta) != 0) {
        for (u32 i = 0; i < header.delta_count; ++i) {
            OneF::RegPair pair;
            std::memcpy(&pair, cursor + size_t{i} * sizeof(pair), sizeof(pair));
            if (pair.index < Maxwell::NUM_REGS) {
                view->regs.reg_array[pair.index] = pair.value;
            }
        }
        cursor += Padded(size_t{header.delta_count} * sizeof(OneF::RegPair));
    }
    if ((header.spans & OneF::SpanFlags) != 0) {
        static_assert(std::is_trivially_copyable_v<OneF::FlagBits>);
        OneF::FlagBits raised;
        std::memcpy(&raised, cursor, sizeof(raised));
        view->dirty.flags |= raised;
        cursor += Padded(sizeof(OneF::FlagBits));
    }
    if ((header.spans & OneF::SpanDraw) != 0) {
        std::memcpy(&last_draw, cursor, sizeof(last_draw));
        have_draw = true;
        auto& draw = view->draw_manager.draw_state;
        draw.topology = last_draw.topology;
        draw.draw_mode = static_cast<Maxwell3D::DrawManager::DrawMode>(last_draw.draw_mode);
        draw.draw_indexed = last_draw.draw_indexed != 0;
        draw.draw_element_count = last_draw.draw_element_count;
        draw.base_index = last_draw.base_index;
        draw.vertex_buffer = last_draw.vertex_buffer;
        draw.index_buffer = last_draw.index_buffer;
        draw.base_instance = last_draw.base_instance;
        draw.instance_count = last_draw.instance_count;
        draw.drawn_instances = last_draw.drawn_instances;
        cursor += Padded(sizeof(OneF::DrawValues));
    }
    // Programs and Pools: R computes both from the registers itself (RefreshStages, the pools'
    // registers); part b's view keeps them for its compares.
}

void OneFollowerDraw::SyncFromEngine(Maxwell3D& engine, bool full) {
    if (view == nullptr || view_source != &engine) {
        // A new channel: the view follows its memory manager (a reference in Maxwell3D).
        view = std::make_unique<Maxwell3D>(engine.GetMemoryManager());
        view_source = &engine;
        full = true;
    }
    if (full) {
        view->regs = engine.regs;
        view->dirty.flags = engine.dirty.flags;
        auto& draw = view->draw_manager.draw_state;
        const auto& from = engine.draw_manager.draw_state;
        draw.topology = from.topology;
        draw.draw_mode = from.draw_mode;
        draw.draw_indexed = from.draw_indexed;
        draw.draw_element_count = from.draw_element_count;
        draw.base_index = from.base_index;
        draw.vertex_buffer = from.vertex_buffer;
        draw.index_buffer = from.index_buffer;
        draw.base_instance = from.base_instance;
        draw.instance_count = from.instance_count;
        draw.drawn_instances = from.drawn_instances;
        render_mode_on = engine.ShouldExecute();
        seeded_step = OneF::Step();
    }
    // What no delta carries (taken at every lent packet).
    view->state = engine.state;
    view->engine_state = engine.engine_state;
    view->GetNativeMacroEngine().native_turing = engine.GetNativeMacroEngine().native_turing;
    synced_engine = &engine;
    synced_engine_state = engine.engine_state;
    synced_native = engine.GetNativeMacroEngine().native_turing;
    view_ready = true;
}

void OneFollowerDraw::UpdateDiagnostic() {
    diagnostic.store(rasterizer.frame_dump.Capturing() || VideoCommon::RenderTrace::Enabled() ||
                         rasterizer.flash_watch.WatchesPasses(),
                     std::memory_order_relaxed);
}

void OneFollowerDraw::OnPacket(const PacketHeader& header, const u8* spans) {
    if (header.kind == OneF::PacketKind::Lent) {
        // E is parked: take what deltas do not carry before the spans, which are newer.
        Maxwell3D& engine = rasterizer.channel_state->maxwell3d;
        const bool full = (header.flags & OneF::PacketSeed) != 0 || seeded_step != OneF::Step();
        SyncFromEngine(engine, full);
        UpdateDiagnostic();
        // This packet's own operation; a seed took the flags too, so it starts the count again.
        lent_since_resync = full ? 1 : lent_since_resync + 1;
        if (full) {
            // The seed is the engine at this lent packet, spans included.
            return;
        }
    }
    ApplyViewSpans(header, spans);
}

void OneFollowerDraw::DropDraw(const PacketHeader& header) {
    auto& stage = rasterizer.mem_stage;
    const MemStage::DrawPacket* const packet =
        stage.TakeRingDraw(header.draw_seq, header.position, 0, false);
    if (packet == nullptr) {
        OneF::Note(Check::RenderPacketMissing, "dropped draw");
        return;
    }
    // Section 5's guard: a dropped packet only brings E's buffers up to date.
    if (packet->gpu_count.buffer != VK_NULL_HANDLE || packet->indirect.buffer != VK_NULL_HANDLE ||
        packet->indirect_count.buffer != VK_NULL_HANDLE ||
        (packet->flags & MemStage::ListArgs) != 0) {
        OneF::Note(Check::DropNotPlain, "indirect or count buffers");
    }
    const bool barrier = (packet->flags & MemStage::PostCopyBarrier) != 0;
    stage.PlayRest();
    if (barrier) {
        stage.PlayPostCopyBarrier();
    }
    counts.dropped.fetch_add(1, std::memory_order_relaxed);
}

void OneFollowerDraw::PlayDraw(const PacketHeader& header) {
    UpdateDiagnostic();
    if (view == nullptr || !view_ready || !have_draw ||
        (header.spans & OneF::SpanDraw) == 0) [[unlikely]] {
        // E unlends only after a lent packet seeded the view, and every draw carries its values.
        OneF::Note(Check::RenderPacketMissing, "no view or no draw values");
        DropDraw(header);
        return;
    }
    if ((header.flags & OneF::PacketConditional) != 0 && !render_mode_on) {
        DropDraw(header);
        return;
    }
    const OneF::DrawValues values = last_draw;
    ViewScope scope{rasterizer, *view};
    playing = &header;
    SCOPE_EXIT {
        playing = nullptr;
    };
    rasterizer.Draw(values.indexed != 0, values.instance_arg);
    counts.played.fetch_add(1, std::memory_order_relaxed);
}

// --- 0xe2c -------------------------------------------------------------------------------------------

bool OneFollowerDraw::ConditionMode(Maxwell3D& engine) {
    const auto& regs = engine.regs;
    using RenderEnable = Maxwell::RenderEnable;
    switch (regs.render_enable_override) {
    case RenderEnable::Override::AlwaysRender:
        return true;
    case RenderEnable::Override::NeverRender:
        return false;
    case RenderEnable::Override::UseRenderEnable:
        break;
    default:
        return true;
    }
    Maxwell::ReportSemaphore::Compare cmp{};
    const auto read = [&] {
        // ProcessQueryCondition read these with E parked at this place (its ReadBlock flushed
        // what was pending), so the bytes are the ones it saw.
        engine.GetMemoryManager().ReadBlockUnsafe(regs.render_enable.Address(), &cmp, sizeof(cmp));
    };
    switch (regs.render_enable.mode) {
    case RenderEnable::Mode::True:
        return true;
    case RenderEnable::Mode::False:
        return false;
    case RenderEnable::Mode::Conditional:
        read();
        return cmp.initial_sequence != 0U && cmp.initial_mode != 0U;
    case RenderEnable::Mode::IfEqual:
        read();
        return cmp.initial_sequence == cmp.current_sequence &&
               cmp.initial_mode == cmp.current_mode;
    case RenderEnable::Mode::IfNotEqual:
        read();
        return cmp.initial_sequence != cmp.current_sequence ||
               cmp.initial_mode != cmp.current_mode;
    default:
        return true;
    }
}

void OneFollowerDraw::Condition(bool engine_execute_on) {
    Maxwell3D& engine = rasterizer.channel_state->maxwell3d;
    if (view == nullptr || view_source != &engine) {
        render_mode_on = engine_execute_on; // not seeded yet: nothing of R's to compare
        return;
    }
    render_mode_on = ConditionMode(*view);
    if (render_mode_on != engine_execute_on) {
        OneF::Note(Check::BuildCondition, engine_execute_on ? "R says False" : "R says True");
    }
}

} // namespace Vulkan
