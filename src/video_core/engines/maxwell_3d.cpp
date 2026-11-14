// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "common/assert.h"
#include "common/bit_util.h"
#include "common/fritchii_write_ring.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/fritchii_draw_kinds.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_qmd_prediction.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/memory_manager.h"
#include "video_core/memory_stage_link.h"
#include "video_core/memstage_shadow.h"
#include "video_core/memstage_stats.h"
#include "video_core/nx2.h"
#include "video_core/nx2_cb_path.h"
#include "video_core/nx2_cmdpage_diag.h"
#include "video_core/nx2_indirect_grid.h"
#include "video_core/nx2_list_dispatch.h"
#include "video_core/nx2_list_draw.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/nx2_report_census.h"
#include "video_core/nx2_runahead.h"
#include "video_core/query_cache/frame_trace.h"
#include "video_core/query_dma.h"
#include "video_core/rasterizer_download_area.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/textures/texture.h"
#include "video_core/macro_hle_oot.h"
#ifdef ARCHITECTURE_x86_64
#include "video_core/macro_turing_jit.h"
#endif

namespace Tegra {
/// Why EXP120 = 2 left blit launches unranged (dma_pusher.cpp).
std::string FritchiiBlitMissText();
} // namespace Tegra

namespace Tegra::Engines {

/// First register id that is actually a Macro call.
constexpr u32 MacroRegistersStart = 0xE00;

// EDEN_NX2_3D_METHODS: log the first write to each 3D register (finds methods the emulator doesn't model).
static const bool g_trace_first_method_writes = std::getenv("EDEN_NX2_3D_METHODS") != nullptr;

// The front's closures that trigger or shape draws, as structs with the payload layout the
// memory stage reads (StageTag, memstage_shadow.h): the engine in word 0, then the tag's words.
struct Maxwell3D::StageCall {
    /// StageTag::FinishMacro: [engine, system].
    struct FinishMacro {
        Maxwell3D* engine;
        Core::System* system;
        void operator()() const {
            engine->FinishFrontMacro(*system);
        }
    };
    /// StageTag::DrawLoop, FoldFirstInstance (header bits 40-47): [engine].
    struct FoldFirstInstance {
        Maxwell3D* engine;
        void operator()() const {
            engine->draw_manager.FoldFirstInstance();
        }
    };
    /// StageTag::DrawLoop, AddLoopInstances (header bits 40-47): [engine, count].
    struct AddLoopInstances {
        Maxwell3D* engine;
        u32 count;
        void operator()() const {
            engine->draw_manager.AddLoopInstances(*engine, count);
        }
    };
    /// StageTag::DrawLoop, ListDraw (header bits 40-47): [engine, address, keep id]. The next
    /// draw.end draws the list whose two words are at `address` (FrontTakeListDraw).
    struct ListDraw {
        Maxwell3D* engine;
        GPUVAddr address;
        u64 keep_id;
        void operator()() const {
            engine->draw_manager.draw_state.list_draw_address = address;
            engine->draw_manager.draw_state.list_keep_id = keep_id;
        }
    };
    static constexpr u64 FinishMacroFlags =
        VideoCommon::StageTagBits(VideoCommon::StageTag::FinishMacro);
    static constexpr u64 FoldFirstInstanceFlags =
        VideoCommon::StageDrawLoopBits(VideoCommon::StageDrawLoop::FoldFirstInstance);
    static constexpr u64 AddLoopInstancesFlags =
        VideoCommon::StageDrawLoopBits(VideoCommon::StageDrawLoop::AddLoopInstances);
    static constexpr u64 ListDrawFlags =
        VideoCommon::StageDrawLoopBits(VideoCommon::StageDrawLoop::ListDraw);
    // The words the stage reads (memstage_shadow.cpp), and ring entries as large as the lambdas
    // these replaced ([this, &system], [this], [self, count], [this, address, keep_id]).
    static void CheckLayout() {
        static_assert(offsetof(FinishMacro, engine) == 0 && offsetof(FinishMacro, system) == 8 &&
                      sizeof(FinishMacro) == 16);
        static_assert(offsetof(FoldFirstInstance, engine) == 0 && sizeof(FoldFirstInstance) == 8);
        static_assert(offsetof(AddLoopInstances, engine) == 0 &&
                      offsetof(AddLoopInstances, count) == 8 && sizeof(AddLoopInstances) == 16);
        static_assert(offsetof(ListDraw, engine) == 0 && offsetof(ListDraw, address) == 8 &&
                      offsetof(ListDraw, keep_id) == 16 && sizeof(ListDraw) == 24);
    }
};

Maxwell3D::Maxwell3D(MemoryManager& memory_manager_)
    : draw_manager()
    , memory_manager{memory_manager_}
#ifdef ARCHITECTURE_x86_64
    , macro_engine(bool(Settings::values.disable_macro_jit))
#else
    , macro_engine(true)
#endif
    , upload_state{memory_manager, regs.upload}
{
    dirty.flags.flip();
    InitializeRegisterDefaults();
    execution_mask.reset();
    for (size_t i = 0; i < execution_mask.size(); i++)
        execution_mask[i] = IsMethodExecutable(u32(i));
    memory_write_mask.reset();
    for (u32 i = 0; i < 16; ++i) {
        memory_write_mask[MAXWELL3D_REG_INDEX(const_buffer.buffer) + i] = true;
    }
    for (const auto method : {MAXWELL3D_REG_INDEX(launch_dma), MAXWELL3D_REG_INDEX(inline_data),
                             MAXWELL3D_REG_INDEX(report_semaphore.query),
                             MAXWELL3D_REG_INDEX(sync_info), MAXWELL3D_REG_INDEX(wait_for_idle)}) {
        memory_write_mask[method] = true;
    }
    for (size_t i = MacroRegistersStart; i < memory_write_mask.size(); ++i) {
        memory_write_mask[i] = true;
    }
    // Profiled runs take the same path: CallMethod counts and times its calls itself.
    fast_maxwell3d = VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::Method) && !record_write_history &&
                     !g_trace_first_method_writes;
    const char* skip_same = std::getenv("EDEN_NX2_SKIP_SAME");
    skip_same_value_writes = skip_same == nullptr || skip_same[0] != '0';
    perf_verify = VideoCommon::Nx2Perf::Verify();
    VideoCommon::Nx2FastDrawCount::Init();
    counts_fast_draw_breaks = true;
}

Maxwell3D::~Maxwell3D() {
    // Its held-back draw batch (EXP22, FrontFlushPending) goes with it.
    VideoCommon::GpuSplit::Get().DropPendingFlush(this);
}

void Maxwell3D::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
    upload_state.BindRasterizer(rasterizer_);
}

void Maxwell3D::InitializeRegisterDefaults() {
    // Initializes registers to their default values - what games expect them to be at boot. This is
    // for certain registers that may not be explicitly set by games.

    // Reset all registers to zero
    std::memset(&regs, 0, sizeof(regs));

    // Depth range near/far is not always set, but is expected to be the default 0.0f, 1.0f. This is
    // needed for ARMS.
    for (auto& viewport : regs.viewports) {
        viewport.depth_range_near = 0.0f;
        viewport.depth_range_far = 1.0f;
    }
    for (auto& viewport : regs.viewport_transform) {
        viewport.swizzle.x.Assign(Regs::ViewportSwizzle::PositiveX);
        viewport.swizzle.y.Assign(Regs::ViewportSwizzle::PositiveY);
        viewport.swizzle.z.Assign(Regs::ViewportSwizzle::PositiveZ);
        viewport.swizzle.w.Assign(Regs::ViewportSwizzle::PositiveW);
    }

    // Doom and Bomberman seems to use the uninitialized registers and just enable blend
    // so initialize blend registers with sane values
    regs.blend.color_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.color_source = Regs::Blend::Factor::One_D3D;
    regs.blend.color_dest = Regs::Blend::Factor::Zero_D3D;
    regs.blend.alpha_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.alpha_source = Regs::Blend::Factor::One_D3D;
    regs.blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    for (auto& blend : regs.blend_per_target) {
        blend.color_op = Regs::Blend::Equation::Add_D3D;
        blend.color_source = Regs::Blend::Factor::One_D3D;
        blend.color_dest = Regs::Blend::Factor::Zero_D3D;
        blend.alpha_op = Regs::Blend::Equation::Add_D3D;
        blend.alpha_source = Regs::Blend::Factor::One_D3D;
        blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    }
    regs.stencil_front_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_front_func_mask = 0xFFFFFFFF;
    regs.stencil_front_mask = 0xFFFFFFFF;
    regs.stencil_two_side_enable = 1;
    regs.stencil_back_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_back_func_mask = 0xFFFFFFFF;
    regs.stencil_back_mask = 0xFFFFFFFF;

    regs.depth_test_func = Regs::ComparisonOp::Always_GL;
    regs.gl_front_face = Regs::FrontFace::CounterClockWise;
    regs.gl_cull_face = Regs::CullFace::Back;

    // TODO(Rodrigo): Most games do not set a point size. I think this is a case of a
    // register carrying a default value. Assume it's OpenGL's default (1).
    regs.point_size = 1.0f;

    // TODO(bunnei): Some games do not initialize the color masks (e.g. Sonic Mania). Assuming a
    // default of enabled fixes rendering here.
    for (auto& color_mask : regs.color_mask) {
        color_mask.R.Assign(1);
        color_mask.G.Assign(1);
        color_mask.B.Assign(1);
        color_mask.A.Assign(1);
    }

    for (auto& format : regs.vertex_attrib_format) {
        format.constant.Assign(1);
    }

    // NVN games expect these values to be enabled at boot
    regs.rasterize_enable = 1;
    regs.color_target_mrt_enable = 1;
    regs.framebuffer_srgb = 1;
    regs.line_width_aliased = 1.0f;
    regs.line_width_smooth = 1.0f;
    regs.gl_front_face = Maxwell3D::Regs::FrontFace::ClockWise;
    regs.polygon_mode_back = Maxwell3D::Regs::PolygonMode::Fill;
    regs.polygon_mode_front = Maxwell3D::Regs::PolygonMode::Fill;

    shadow_state = regs;
}

bool Maxwell3D::IsMethodExecutable(u32 method) {
    if (method >= MacroRegistersStart) {
        return true;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(draw.begin):
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(index_buffer.count):
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_first):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent):
    case MAXWELL3D_REG_INDEX(draw_texture.src_y0):
    case RootTables::SelectMethod:
    case RootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(topology_override):
    case MAXWELL3D_REG_INDEX(clear_surface):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true;
    default:
        return false;
    }
}

void Maxwell3D::ProcessMacro(Core::System& system, u32 method, const u32* base_start, u32 amount, bool is_last_call) {
    if (executing_macro == 0) {
        // A macro call must begin by writing the macro method's register, not its argument.
        ASSERT((method % 2) == 0 && "Can't start macro execution by writing to the ARGS register");
        executing_macro = method;
    }

    macro_segments.emplace_back(current_dma_segment, amount);
    current_macro_dirty |= current_dirty;
    current_dirty = false;

    // Nearly every call arrives whole in one method run (title 1 makes ~1,000,000 a second):
    // copy it to the stack instead of growing the parameter vector. Still a copy: the command
    // list is often guest memory read in place, which the guest may reuse while the macro runs
    // (and the macro memo stores the parameters after it ran). Only accurate GPU mode rewrites
    // the parameters in place (RefreshParameters), so it keeps the vector.
    // EDEN_NX2_MACRO_COPY=1: always use the vector (for comparisons).
    static const bool always_copy = std::getenv("EDEN_NX2_MACRO_COPY") != nullptr;
    if (is_last_call && macro_params.empty() && amount <= 64 && !always_copy &&
        !Settings::IsGPULevelHigh()) {
        std::array<u32, 64> parameters;
        std::copy_n(base_start, amount, parameters.begin());
        ConsumeSink(system);
        CallMacroMethod(system, executing_macro, std::span<const u32>{parameters.data(), amount});
        macro_segments.clear();
        current_macro_dirty = false;
        return;
    }
    macro_params.insert(macro_params.end(), base_start, base_start + amount);

    // Call the macro when there are no more parameters in the command buffer
    if (is_last_call) {
        ConsumeSink(system);
        CallMacroMethod(system, executing_macro, macro_params);
        macro_params.clear();
        macro_segments.clear();
        current_macro_dirty = false;
    }
}

void Maxwell3D::RefreshParametersImpl() {
    if (!Settings::IsGPULevelHigh()) {
        return;
    }
    size_t current_index = 0;
    for (auto& segment : macro_segments) {
        if (segment.first == 0) {
            current_index += segment.second;
            continue;
        }
        memory_manager.ReadBlock(segment.first, &macro_params[current_index],
                                 sizeof(u32) * segment.second);
        current_index += segment.second;
    }
}

u32 Maxwell3D::GetMaxCurrentVertices() {
    u32 num_vertices = 0;
    for (size_t index = 0; index < Regs::NumVertexArrays; ++index) {
        const auto& array = regs.vertex_streams[index];
        if (array.enable == 0) {
            continue;
        }
        const auto& attribute = regs.vertex_attrib_format[index];
        if (attribute.constant) {
            num_vertices = (std::max)(num_vertices, 1U);
            continue;
        }
        const auto& limit = regs.vertex_stream_limits[index];
        const GPUVAddr gpu_addr_begin = array.Address();
        const GPUVAddr gpu_addr_end = limit.Address() + 1;
        const u64 stream_size = GetNativeMacroEngine().native_turing
                                    ? regs.NativeVertexStreamSize(index)
                                    : gpu_addr_end - gpu_addr_begin;
        const u32 address_size = static_cast<u32>((std::min<u64>)(stream_size, (std::numeric_limits<u32>::max)()));
        num_vertices = (std::max)(
            num_vertices, address_size / (std::max)(attribute.SizeInBytes(), array.stride.Value()));
        break;
    }
    return num_vertices;
}

size_t Maxwell3D::EstimateIndexBufferSize() {
    GPUVAddr start_address = regs.index_buffer.StartAddress();
    GPUVAddr end_address = regs.index_buffer.EndAddress();
    auto const byte_size = regs.index_buffer.FormatSizeInBytes();
    auto const max_size = 1ull << (byte_size * CHAR_BIT);
    auto const upper_cap = GetMaxCurrentVertices() * 4 * byte_size;
    auto const lower_cap = std::min<size_t>(size_t(end_address - start_address), upper_cap);
    return std::min<size_t>(memory_manager.GetMemoryLayoutSize(start_address, byte_size * max_size) / byte_size, lower_cap);
}

void Maxwell3D::ConsumeSinkImpl(Core::System& system) {
    const VideoCommon::Nx2Profile::FineScope profile_scope{VideoCommon::Nx2Profile::Zone::Sink};
    if (VideoCommon::FritchiiDrawKinds::g_on) [[unlikely]] {
        VideoCommon::FritchiiDrawKinds::NoteWrites(method_sink.size());
    }
    const auto control = shadow_state.shadow_ram_control;
    // Sunk methods are state writes only (not in execution_mask): skip unchanged values, as
    // CallMethod does.
    const bool skip_same = SkipSameValueWrites();
    const auto process = [&](u32 method, u32 value) {
        if (!skip_same || regs.reg_array[method] != value) {
            ProcessDirtyRegisters(method, value);
        }
    };
    if (control == Regs::ShadowRamControl::Track || control == Regs::ShadowRamControl::TrackWithFilter) {
        for (auto [method, value] : method_sink) {
            shadow_state.reg_array[method] = value;
            process(method, value);
        }
    } else if (control == Regs::ShadowRamControl::Replay) {
        for (auto [method, value] : method_sink)
            process(method, shadow_state.reg_array[method]);
    } else {
        for (auto [method, value] : method_sink)
            process(method, value);
    }
    method_sink.clear();
}

void Maxwell3D::ApplyPairs(Core::System& system, const u64* pairs, u32 count) {
    // EDEN_NX2_RING_PAIRS: state writes the front would have sunk (never in execution_mask),
    // applied as ConsumeSinkImpl applies the sink. The shadow RAM mode cannot change inside a
    // block: shadow_ram_control is executable, and any other entry ends the block.
    if (VideoCommon::FritchiiDrawKinds::g_on) [[unlikely]] {
        VideoCommon::FritchiiDrawKinds::NoteWrites(count);
    }
    const auto control = shadow_state.shadow_ram_control;
    const bool track = control == Regs::ShadowRamControl::Track ||
                       control == Regs::ShadowRamControl::TrackWithFilter;
    const bool replay = control == Regs::ShadowRamControl::Replay;
    const bool skip_same = SkipSameValueWrites();
    for (u32 i = 0; i < count; ++i) {
        const u32 method = static_cast<u32>(pairs[i] & 0xffff);
        u32 value = static_cast<u32>(pairs[i] >> 32);
        if (track) {
            shadow_state.reg_array[method] = value;
        } else if (replay) {
            value = shadow_state.reg_array[method];
        }
        if (!skip_same || regs.reg_array[method] != value) {
            ProcessDirtyRegisters(method, value);
        }
    }
}

// EDEN_NX2_FASTDRAW_COUNT: the ported change's lists (its maxwell_3d.cpp), counting
// only.
bool Maxwell3D::IsFastDrawParameter(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(const_buffer.size):
    case MAXWELL3D_REG_INDEX(const_buffer.address_high):
    case MAXWELL3D_REG_INDEX(const_buffer.address_low):
    case MAXWELL3D_REG_INDEX(const_buffer.offset):
    case MAXWELL3D_REG_INDEX(global_base_vertex_index):
    case MAXWELL3D_REG_INDEX(global_base_instance_index):
    // Written with the first vertex before each draw; nothing a draw resolves reads it.
    case MAXWELL3D_REG_INDEX(vertex_id_base):
    case MAXWELL3D_REG_INDEX(index_buffer.start_addr_high):
    case MAXWELL3D_REG_INDEX(index_buffer.start_addr_low):
    case MAXWELL3D_REG_INDEX(index_buffer.limit_addr_high):
    case MAXWELL3D_REG_INDEX(index_buffer.limit_addr_low):
    case MAXWELL3D_REG_INDEX(index_buffer.format):
        return true;
    default:
        return DrawManager::IsInstanceLoopMethod(method) || IsFastDrawDeltaMethod(method) ||
               IsFastDrawVertexBuffer(method) || IsFastDrawNeutralMethod(method) ||
               IsFastDrawReportOperand(method) || IsFastDrawVertexFormat(method);
    }
}

bool Maxwell3D::IsFastDrawVertexFormat(u32 method) {
    constexpr u32 formats = MAXWELL3D_REG_INDEX(vertex_attrib_format);
    return VideoCommon::Nx2FastDrawCount::g_vertex_streams_dynamic && method >= formats &&
           method < formats + Regs::NumVertexAttributes;
}

bool Maxwell3D::IsFastDrawNeutralMethod(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
        return true;
    default:
        return false;
    }
}

bool Maxwell3D::IsFastDrawReportOperand(u32 method) {
    // SET_MME_MEM_ADDRESS_A/B and the data RAM address.
    constexpr u32 MmeMemAddressA = 0x154;
    constexpr u32 MmeDataRamAddress = 0x156;
    if (method >= MmeMemAddressA && method <= MmeDataRamAddress) {
        return true;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(report_semaphore.address_high):
    case MAXWELL3D_REG_INDEX(report_semaphore.address_low):
    case MAXWELL3D_REG_INDEX(report_semaphore.payload):
        return true;
    default:
        return false;
    }
}

bool Maxwell3D::IsFastDrawVertexBuffer(u32 method) {
    constexpr u32 streams = MAXWELL3D_REG_INDEX(vertex_streams);
    constexpr u32 stream_words = sizeof(Regs::VertexStream) / sizeof(u32);
    if (method >= streams && method < streams + Regs::NumVertexArrays * stream_words) {
        const u32 word = (method - streams) % stream_words;
        return word == 1 || word == 2 ||
               (word == 0 && VideoCommon::Nx2FastDrawCount::g_vertex_streams_dynamic);
    }
    constexpr u32 sizes = 0x600 / sizeof(u32); // SET_VERTEX_STREAM_SIZE_A/B (NativeVertexStreamSize)
    constexpr u32 limits = MAXWELL3D_REG_INDEX(vertex_stream_limits);
    constexpr u32 limit_words = sizeof(Regs::VertexStreamLimit) / sizeof(u32);
    return (method >= sizes && method < sizes + Regs::NumVertexArrays * 2) ||
           (method >= limits && method < limits + Regs::NumVertexArrays * limit_words);
}

bool Maxwell3D::IsFastDrawDeltaMethod(u32 method) {
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method >= cb_data && method < cb_data + Regs::NumCBData) {
        return true;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return true;
    default:
        return false;
    }
}

u32 Maxwell3D::FastDrawCountGroup(u32 method) {
    namespace Count = VideoCommon::Nx2FastDrawCount;
    if (method == MAXWELL3D_REG_INDEX(vertex_id_base)) {
        return Count::VertexIdBase;
    }
    constexpr u32 streams = MAXWELL3D_REG_INDEX(vertex_streams);
    constexpr u32 stream_words = sizeof(Regs::VertexStream) / sizeof(u32);
    constexpr u32 formats = MAXWELL3D_REG_INDEX(vertex_attrib_format);
    if ((method >= streams && method < streams + Regs::NumVertexArrays * stream_words &&
         (method - streams) % stream_words == 0) ||
        (method >= formats && method < formats + Regs::NumVertexAttributes)) {
        return Count::VertexLayout;
    }
    if (method == MAXWELL3D_REG_INDEX(clear_report_value) ||
        method == MAXWELL3D_REG_INDEX(render_enable.mode) ||
        method == MAXWELL3D_REG_INDEX(report_semaphore.query) ||
        method == MAXWELL3D_REG_INDEX(report_semaphore.address_high) ||
        method == MAXWELL3D_REG_INDEX(report_semaphore.address_low) ||
        method == MAXWELL3D_REG_INDEX(report_semaphore.payload) ||
        (method >= 0x154 && method <= 0x156)) {
        return Count::QueryMethods;
    }
    return Count::RealState;
}

void Maxwell3D::NoteFastDrawCountWrite(u32 method, u32 argument) {
    if (regs.reg_array[method] != argument && !IsFastDrawParameter(method)) {
        VideoCommon::Nx2FastDrawCount::BreakOn(VideoCommon::Nx2FastDrawCount::Reason::Register,
                                               FastDrawCountGroup(method), method);
    }
}

Maxwell3D::FastDrawCountPools Maxwell3D::FastDrawCountPoolRanges() const {
    constexpr u64 HeaderSize = 0x20;
    // As TextureCache::SynchronizeDescriptors sizes the sampler table.
    return {
        .tic = regs.tex_header.Address(),
        .tic_size = (u64{regs.tex_header.limit} + 1) * HeaderSize,
        .tsc = regs.tex_sampler.Address(),
        .tsc_size = (u64{regs.sampler_binding == Regs::SamplerBinding::ViaHeaderBinding
                             ? regs.tex_header.limit
                             : regs.tex_sampler.limit} +
                     1) *
                    HeaderSize,
    };
}

void Maxwell3D::UpdateFastDrawCountPools() {
    namespace Count = VideoCommon::Nx2FastDrawCount;
    const FastDrawCountPools pools = FastDrawCountPoolRanges();
    const bool tic_changed =
        fast_draw_count_tic_spans.Update(memory_manager, pools.tic, pools.tic_size);
    const bool tsc_changed =
        fast_draw_count_tsc_spans.Update(memory_manager, pools.tsc, pools.tsc_size);
    if (tic_changed || tsc_changed || Count::g_pools_owner != this) {
        Count::PublishPools(this, fast_draw_count_tic_spans, fast_draw_count_tsc_spans);
    }
}

void Maxwell3D::NoteFastDrawCountData(GPUVAddr begin, GPUVAddr end, bool unchanged) {
    namespace Count = VideoCommon::Nx2FastDrawCount;
    // The original NoteFastDrawConstantData: texture and sampler headers are read when image views are
    // resolved, which delta draws skip. Header writes end reuse even when they change nothing
    // (FastDrawHeaderWrite comes first).
    const FastDrawCountPools pools = FastDrawCountPoolRanges();
    if ((begin < pools.tic + pools.tic_size && pools.tic < end) ||
        (begin < pools.tsc + pools.tsc_size && pools.tsc < end)) {
        Count::BreakOn(Count::Reason::TextureHeaders, Count::Other);
        return;
    }
    // EXP107: words memory held already change nothing a draw reads.
    if (unchanged) {
        Count::g_unchanged_writes.fetch_add(1, std::memory_order_relaxed);
    } else {
        Count::NoteConstantData(begin, end);
    }
    // The same device memory can be mapped at other GPU addresses: compare where it lands (as the
    // ported change does; Fable plan D1, row 5).
    UpdateFastDrawCountPools();
    constexpr u64 PageSize = 0x1000; // the smallest GPU page
    for (GPUVAddr page = begin; page < end;) {
        const GPUVAddr page_end = (std::min)(end, (page & ~(PageSize - 1)) + PageSize);
        if (const std::optional<DAddr> device = memory_manager.GpuToCpuAddress(page)) {
            const DAddr device_end = *device + (page_end - page);
            if (fast_draw_count_tic_spans.Intersects(*device, device_end) ||
                fast_draw_count_tsc_spans.Intersects(*device, device_end)) {
                Count::BreakOn(Count::Reason::PoolAlias, Count::Other);
                return;
            }
        }
        page = page_end;
    }
}

Maxwell3D::ConstantWrite Maxwell3D::CompareConstantData(GPUVAddr begin, const void* data,
                                                       u64 bytes) {
    // EXP107: OoT rewrites ~258k constant buffer blocks a
    // second with the words memory already holds. This runs on the thread that writes them, in
    // order, after everything sent before (the split's back), so memory is what the write lands on.
    if (VideoCore::Nx2Debug::Experiment(107) == 0 || bytes == 0) {
        return ConstantWrite::Changed;
    }
    // Within one GPU page the words are contiguous in host memory. Switch 2 sparse memory goes
    // the general path, as EDEN_NX2_CB_DIRECT's.
    constexpr u64 PageSize = 0x1000; // the smallest GPU page
    if ((begin & (PageSize - 1)) + bytes > PageSize ||
        (begin + bytes > Tegra::g_nx2_sparse_low.load(std::memory_order_relaxed) &&
         begin < Tegra::g_nx2_sparse_high.load(std::memory_order_relaxed))) {
        return ConstantWrite::Changed;
    }
    const u8* const held = memory_manager.GetPointer(begin);
    if (held == nullptr || std::memcmp(held, data, bytes) != 0) {
        return ConstantWrite::Changed;
    }
    // Writing them again changes nothing either, unless the GPU caches may hold the words newer
    // than memory or a download for a guest read may still land over them: the write must win
    // then (the original conditions). MayHoldGpuWrites is true for every page outside Switch 2 titles.
    const std::optional<DAddr> device = memory_manager.GpuToCpuAddress(begin);
    if (!device || VideoCore::MayHoldGpuWrites(*device, bytes) ||
        VideoCore::ReadDownloadsMayCover(*device, bytes)) {
        return ConstantWrite::Unchanged;
    }
    // Run-ahead (EDEN_NX2_RUNAHEAD) lets guest memory hold the CPU's next frame while the back
    // draws the current one from the host buffers: equal memory says nothing about those then.
    if (VideoCommon::Runahead::g_exact_used.load(std::memory_order_relaxed) ||
        VideoCommon::Runahead::g_words_written.load(std::memory_order_relaxed)) {
        return ConstantWrite::Unchanged;
    }
    // Texture and sampler header writes still take the write (the original never skips them), also
    // through another GPU address mapping the pools' device memory.
    const FastDrawCountPools pools = FastDrawCountPoolRanges();
    const GPUVAddr end = begin + bytes;
    if ((begin < pools.tic + pools.tic_size && pools.tic < end) ||
        (begin < pools.tsc + pools.tsc_size && pools.tsc < end)) {
        return ConstantWrite::Unchanged;
    }
    unchanged_write_tic_spans.Update(memory_manager, pools.tic, pools.tic_size);
    unchanged_write_tsc_spans.Update(memory_manager, pools.tsc, pools.tsc_size);
    if (unchanged_write_tic_spans.Intersects(*device, *device + bytes) ||
        unchanged_write_tsc_spans.Intersects(*device, *device + bytes)) {
        return ConstantWrite::Unchanged;
    }
    if (VideoCommon::Nx2FastDrawCount::g_on) [[unlikely]] {
        VideoCommon::Nx2FastDrawCount::g_skipped_writes.fetch_add(1, std::memory_order_relaxed);
    }
    return ConstantWrite::Skip;
}

void Maxwell3D::ProcessMethodCall(u32 method, u32 argument, u32 nonshadow_argument, bool is_last_call) {
    if (VideoCommon::Nx2FastDrawCount::g_on) [[unlikely]] {
        if (!DrawManager::IsInstanceLoopMethod(method) && !IsFastDrawDeltaMethod(method) &&
            !IsFastDrawNeutralMethod(method)) {
            VideoCommon::Nx2FastDrawCount::BreakOn(VideoCommon::Nx2FastDrawCount::Reason::Method,
                                                   FastDrawCountGroup(method), method);
        }
    }
    if (GetNativeMacroEngine().native_turing && method == MAXWELL3D_REG_INDEX(clear_surface)) {
        LOG_DEBUG(HW_GPU, "Native GPU surface clear: value={:#x}", argument);
    }
    switch (method) {
    case RootTables::SelectMethod:
        root_tables.selected = argument & 7;
        root_tables.offset = (argument >> 8) & 0xff;
        return;
    case RootTables::LoadMethod:
        if (root_tables.offset < 256 && root_tables.offset % 4 == 0) {
            root_tables.tables[root_tables.selected][root_tables.offset / 4] = argument;
            root_tables.loaded = true;
        }
        root_tables.offset += 4;
        return;
    case MAXWELL3D_REG_INDEX(wait_for_idle):
        // FRITCHII_MEMSTAGE_PASS (fable_stage_p3 1.4): the back runs a wait for idle in a served
        // entry only when GPUMem passed it, with a record of the entry (must be 0 otherwise).
        if (VideoCommon::MemoryStageLink::BackServed()) [[unlikely]] {
            auto& link = VideoCommon::MemoryStageLink::Get();
            if (link.BackPassRecord(VideoCommon::GpuSplit::Get().BackEntryPosition()) ==
                nullptr) {
                VideoCommon::MemoryStageLink::Note(
                    VideoCommon::MemoryStageLink::Check::PassWaitMissing,
                    "Maxwell3D::ProcessMethodCall (wait_for_idle)");
            }
        }
        return rasterizer->WaitForIdle();
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
        shadow_state.shadow_ram_control = static_cast<Regs::ShadowRamControl>(nonshadow_argument);
        return;
    // With macros on the split GPU thread's front (FrontWrite), the front loads and binds them.
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
        if (front_macros) return;
        return GetNativeMacroEngine().ClearCode(regs.load_mme.instruction_ptr);
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
        if (front_macros) return;
        return GetNativeMacroEngine().AddCode(regs.load_mme.instruction_ptr, argument);
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
        if (front_macros) {
            ++regs.load_mme.start_address_ptr;
            return;
        }
        return ProcessMacroBind(argument);
    case MAXWELL3D_REG_INDEX(falcon[4]):
        return ProcessFirmwareCall4();
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
        return ProcessCBData(argument);
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
        return ProcessCBBind(0);
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
        return ProcessCBBind(1);
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
        return ProcessCBBind(2);
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
        return ProcessCBBind(3);
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return ProcessCBBind(4);
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
        // FRITCHII_MEMSTAGE_PASS bit 2 (fable_stage_p3 1.2 and 1.4): a fence report (a Release
        // with short_query) runs served only when GPUMem passed it, with a record of the entry.
        // The back asks for it here, at the entry, as for the other passed kinds: an unmapped one
        // reaches nothing else that asks (CounterReport returns at once), and a record the back
        // passed without asking is applied late (PassLate). No record: must be 0.
        if (VideoCommon::MemoryStageLink::BackServed() &&
            regs.report_semaphore.query.operation.Value() ==
                Regs::ReportSemaphore::Operation::Release &&
            regs.report_semaphore.query.short_query != 0) [[unlikely]] {
            auto& link = VideoCommon::MemoryStageLink::Get();
            if (link.BackPassRecord(VideoCommon::GpuSplit::Get().BackEntryPosition()) ==
                nullptr) {
                VideoCommon::MemoryStageLink::Note(
                    VideoCommon::MemoryStageLink::Check::PassFenceMissing,
                    "Maxwell3D::ProcessMethodCall (a fence report)");
            }
        }
        return ProcessQueryGet();
    case MAXWELL3D_REG_INDEX(render_enable.mode):
        return ProcessQueryCondition();
    case MAXWELL3D_REG_INDEX(clear_report_value):
        return ProcessCounterReset();
    case MAXWELL3D_REG_INDEX(sync_info):
        return ProcessSyncPoint();
    case MAXWELL3D_REG_INDEX(launch_dma):
        return upload_state.ProcessExec(regs.launch_dma.memory_layout.Value() == Regs::LaunchDMA::Layout::Pitch);
    case MAXWELL3D_REG_INDEX(inline_data):
        upload_state.ProcessData(argument, is_last_call);
        return;
    case MAXWELL3D_REG_INDEX(fragment_barrier):
        return rasterizer->FragmentBarrier();
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
        // InvalidateGPUCache, then WaitForIdle (RasterizerInterface's default).
        return rasterizer->InvalidateTextureDataCache();
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return rasterizer->TiledCacheBarrier();
    default:
        draw_manager.ProcessMethodCall(*this, method, argument);
        break;
    }
}

void Maxwell3D::CallMacroMethod(Core::System& system, u32 method, std::span<const u32> parameters) {
    // Reset the current macro.
    executing_macro = 0;

    // Lookup the macro offset
    const u32 entry =
        ((method - MacroRegistersStart) >> 1) % static_cast<u32>(GetNativeMacroPositions().size());

    // Execute the current macro.
    GetNativeMacroEngine().Execute(system, *this, GetNativeMacroPositions()[entry], parameters);
    draw_manager.DrawDeferred(*this);
}

void Maxwell3D::TraceFirstMethodWrite(u32 method, u32 argument) {
    static std::array<bool, Regs::NUM_REGS> seen{};
    if (method < seen.size() && !seen[method]) {
        seen[method] = true;
        LOG_INFO(HW_GPU, "3D method first write: {:#x} (byte {:#x}) = {:#x}", method, method * 4, argument);
    }
}

void Maxwell3D::CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    if (VideoCommon::FritchiiDrawKinds::g_on) [[unlikely]] {
        VideoCommon::FritchiiDrawKinds::NoteWrites(1);
    }
    if (VideoCommon::Nx2Profile::Enabled()) [[unlikely]] {
        return ProfiledCallMethod(system, method, method_argument, is_last_call);
    }
    // About half of the split back's 3D writes (~5 million a second in Ocarina of Time) are state
    // registers: CallMethodFast's state write, here without the calls to it. A register not in
    // execution_mask is not SET_SHADOW_RAM_CONTROL either (CallMethodSlow's). The verify switch
    // takes the longer path through DispatchMethod; the write history and the first-write trace
    // leave fast_maxwell3d off. Profiled runs go through DispatchMethod to be counted.
    if (fast_maxwell3d && executing_macro == 0 && method < MacroRegistersStart &&
        !execution_mask[method] && !perf_verify) {
        return WriteStateRegister(method, method_argument);
    }
    DispatchMethod(system, method, method_argument, is_last_call);
}

void Maxwell3D::DispatchMethod(Core::System& system, u32 method, u32 method_argument,
                               bool is_last_call) {
    if (!fast_maxwell3d || executing_macro != 0 || method >= MacroRegistersStart ||
        method == MAXWELL3D_REG_INDEX(shadow_ram_control)) {
        return CallMethodSlow(system, method, method_argument, is_last_call);
    }
    if (perf_verify) {
        return VerifyCallMethod(system, method, method_argument, is_last_call);
    }
    CallMethodFast(method, method_argument, is_last_call);
}

// EDEN_NX2_GPU_PROFILE: the path unprofiled runs take, counted. GpuSplit's back calls the 3D engine
// outside every zone (fast_maxwell3d), millions of times a second: a sample of those calls is
// timed (SampledScope). Called inside another zone (a back macro, a run of words, the unsplit GPU
// thread's method zone), the call is that zone's time, as before.
void Maxwell3D::ProfiledCallMethod(Core::System& system, u32 method, u32 method_argument,
                                   bool is_last_call) {
    const VideoCommon::Nx2Profile::SampledScope scope{VideoCommon::Nx2Profile::Zone::Method};
    if (method < MacroRegistersStart) {
        VideoCommon::Nx2Profile::CountRegisterWrite(method,
                                                    regs.reg_array[method] == method_argument);
    }
    DispatchMethod(system, method, method_argument, is_last_call);
}

namespace {

// Executable methods the draw manager does nothing for (ProcessDirtyRegisters is all they do).
bool IsStoreOnlyExecutable(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(topology_override):
        return true;
    default:
        return false;
    }
}

} // Anonymous namespace

// CallMethodSlow without the checks that cannot apply (CallMethod): no macro collects parameters,
// no write history, trace or profile, not a macro call or SET_SHADOW_RAM_CONTROL. The hot
// executable methods go straight to their handler.
void Maxwell3D::CallMethodFast(u32 method, u32 method_argument, bool is_last_call) {
    if (!execution_mask[method]) {
        return WriteStateRegister(method, method_argument);
    }
    const u32 argument = ProcessShadowRam(method, method_argument);
    ProcessDirtyRegisters(method, argument);
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method - cb_data < 16) {
        return ProcessCBData(argument);
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.begin):
        return draw_manager.DrawBegin(*this);
    case MAXWELL3D_REG_INDEX(draw.end):
        return draw_manager.DrawEnd(*this);
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(topology_override):
        return;
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
        return ProcessCBBind(0);
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
        return ProcessCBBind(1);
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
        return ProcessCBBind(2);
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
        return ProcessCBBind(3);
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return ProcessCBBind(4);
    default:
        return ProcessMethodCall(method, argument, method_argument, is_last_call);
    }
}

void Maxwell3D::VerifyCallMethod(Core::System& system, u32 method, u32 method_argument,
                                 bool is_last_call) {
    // Only register writes can run twice: handlers draw, upload and bind, and a deferred draw
    // ProcessDirtyRegisters may issue is a draw too.
    if ((execution_mask[method] && !IsStoreOnlyExecutable(method)) ||
        draw_manager.HasDeferredDraw()) {
        return CallMethodSlow(system, method, method_argument, is_last_call);
    }
    const u32 reg = regs.reg_array[method];
    const u32 shadow = shadow_state.reg_array[method];
    const DirtyState::Flags flags = dirty.flags;
    CallMethodFast(method, method_argument, is_last_call);
    const u32 fast_reg = regs.reg_array[method];
    const u32 fast_shadow = shadow_state.reg_array[method];
    const DirtyState::Flags fast_flags = dirty.flags;
    regs.reg_array[method] = reg;
    shadow_state.reg_array[method] = shadow;
    dirty.flags = flags;
    CallMethodSlow(system, method, method_argument, is_last_call);
    if (fast_reg == regs.reg_array[method] && fast_shadow == shadow_state.reg_array[method] &&
        fast_flags == dirty.flags) {
        return;
    }
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1, std::memory_order_relaxed) >= 64) {
        return;
    }
    std::string flag_differences;
    const DirtyState::Flags differing = fast_flags ^ dirty.flags;
    for (size_t i = 0; i < differing.size(); ++i) {
        if (differing[i]) {
            flag_differences += fmt::format(" {}={}", i, fast_flags[i] ? 1 : 0);
        }
    }
    LOG_ERROR(HW_GPU,
              "Nx2Perf method: {:#x} = {:#x}: register {:#x} (old {:#x}), shadow {:#x} (old {:#x}), "
              "fast dirty flags differ:{}",
              method, method_argument, fast_reg, regs.reg_array[method], fast_shadow,
              shadow_state.reg_array[method], flag_differences);
}

void Maxwell3D::CallConstBufferWords(Core::System& system, const u32* methods,
                                     const u32* arguments, u32 count) {
    const u32 offset = regs.const_buffer.offset;
    const u64 bytes = u64{count} * sizeof(u32);
    const GPUVAddr address = regs.const_buffer.Address();
    // Keep per-word translation across device pages and per-word clipping at buffer limits.
    if (!fast_maxwell3d || executing_macro != 0 || address == 0 ||
        offset > regs.const_buffer.size || bytes > regs.const_buffer.size - offset ||
        bytes > Core::DEVICE_PAGESIZE - ((address + offset) & Core::DEVICE_PAGEMASK)) {
        for (u32 i = 0; i < count; ++i) {
            CallMethod(system, methods[i], arguments[i], true);
        }
        return;
    }
    std::array<u32, 64> values;
    const bool replay = shadow_state.shadow_ram_control == Regs::ShadowRamControl::Replay;
    for (u32 i = 0; i < count; ++i) {
        values[i] = replay ? shadow_state.reg_array[methods[i]] : arguments[i];
    }
    if (perf_verify) {
        for (u32 i = 0; i < count; ++i) {
            CallMethod(system, methods[i], arguments[i], true);
        }
        std::array<u32, 64> written;
        const bool mapped = memory_manager.IsFullyMappedRange(address + offset, bytes);
        if (mapped) {
            memory_manager.ReadBlockUnsafe(address + offset, written.data(), bytes);
        }
        if (regs.const_buffer.offset != offset + bytes ||
            (mapped && std::memcmp(written.data(), values.data(), bytes) != 0)) {
            static std::atomic<u32> reports{};
            if (reports.fetch_add(1, std::memory_order_relaxed) < 64) {
                LOG_ERROR(HW_GPU,
                          "Nx2Perf method: {} constant buffer words at {:#x}+{:#x}: offset {:#x}, "
                          "expected {:#x}, memory {}",
                          count, address, offset, regs.const_buffer.offset, offset + bytes,
                          mapped && std::memcmp(written.data(), values.data(), bytes) != 0
                              ? "differs"
                              : "matches");
            }
        }
        return;
    }
    // Counted and timed as CallMethod counts and times each word (EDEN_NX2_GPU_PROFILE).
    const VideoCommon::Nx2Profile::SampledScope scope{VideoCommon::Nx2Profile::Zone::Method, count};
    if (VideoCommon::FritchiiDrawKinds::g_on) [[unlikely]] {
        VideoCommon::FritchiiDrawKinds::NoteWrites(count);
    }
    const bool profile = VideoCommon::Nx2Profile::Enabled();
    // As CallMethodFast for each word: the first may issue a deferred draw, before any upload.
    for (u32 i = 0; i < count; ++i) {
        if (profile) [[unlikely]] {
            VideoCommon::Nx2Profile::CountRegisterWrite(methods[i],
                                                        regs.reg_array[methods[i]] == arguments[i]);
        }
        ProcessDirtyRegisters(methods[i], ProcessShadowRam(methods[i], arguments[i]));
    }
    if (VideoCommon::MemoryStageLink::BackServed()) [[unlikely]] {
        // FRITCHII_MEMSTAGE_THREAD=2: the memory stage wrote the words (ProcessCBMultiData).
        regs.const_buffer.offset += static_cast<u32>(bytes);
        return;
    }
    const VideoCommon::Nx2Profile::FineScope upload_scope{VideoCommon::Nx2Profile::Zone::CbUpload};
    const GPUVAddr destination = address + offset;
    // EDEN_NX2_MEMSTAGE_STATS: constant buffer words (the slow paths above count each word in
    // ProcessCBMultiData).
    if (VideoCommon::MemStageStats::Enabled()) [[unlikely]] {
        VideoCommon::MemStageStats::StreamWrite(
            VideoCommon::MemStageStats::Write::CbWords,
            memory_manager.GpuToCpuAddress(destination).value_or(0), bytes);
    }
    // EXP107: words memory holds already, where writing them again changes
    // nothing, are not written (no invalidation either); unchanged ones are neutral for the count.
    const ConstantWrite compared = CompareConstantData(destination, values.data(), bytes);
    if (VideoCommon::Nx2FastDrawCount::g_on) [[unlikely]] {
        NoteFastDrawCountData(destination, destination + bytes,
                              compared != ConstantWrite::Changed);
    }
    if (compared == ConstantWrite::Skip) {
        regs.const_buffer.offset += static_cast<u32>(bytes);
        return;
    }
    // EXP88 & 2: through the bound buffer's device address, the device range logged for
    // FlushCaching, in place of CB_DIRECT's translation and the word-by-word accumulation.
    if (WriteBoundConstantBuffer(offset, values.data(), static_cast<u32>(bytes))) {
        regs.const_buffer.offset += static_cast<u32>(bytes);
        return;
    }
    // EDEN_NX2_CB_DIRECT (ported, its maxwell_3d.cpp CallConstBufferWords):
    // what WriteWordsCached does for words within one device page (proven above) outside Switch 2
    // sparse memory, with one translation (as MemoryManager::Write) instead of WriteBlockImpl's
    // page walk: the words, then the invalidation accumulated word by word, as WriteWordsCached
    // accumulates it. While an EXP49 read-back download is in flight, the walk, so its FinishRead
    // check still runs first. Unmapped: the walk.
    u8* const pointer =
        VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_cb_direct, "EDEN_NX2_CB_DIRECT",
                                         1) != 0 &&
                VideoCore::g_read_downloads.load(std::memory_order_relaxed) == 0 &&
                (destination + bytes <= Tegra::g_nx2_sparse_low.load(std::memory_order_relaxed) ||
                 destination >= Tegra::g_nx2_sparse_high.load(std::memory_order_relaxed))
            ? memory_manager.GetPointer(destination)
            : nullptr;
    if (pointer != nullptr) [[likely]] {
        std::memcpy(pointer, values.data(), bytes);
        for (u32 i = 0; i < count; ++i) {
            memory_manager.AccumulateInvalidation(destination + u64{i} * sizeof(u32), sizeof(u32));
        }
    } else {
        memory_manager.WriteWordsCached(destination, values.data(), count);
    }
    regs.const_buffer.offset += static_cast<u32>(bytes);
}

void Maxwell3D::CallMethodSlow(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    // It is an error to write to a register other than the current macro's ARG register before
    // it has finished execution.
    if (executing_macro != 0 && method != executing_macro + 1) {
        DumpRecentWrites(method);
    }
    if (executing_macro != 0) {
        ASSERT(method == executing_macro + 1);
    }
    RecordWrite(method, method_argument, 1, is_last_call ? 1 : 2);

    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    if (method >= MacroRegistersStart) {
        ProcessMacro(system, method, &method_argument, 1, is_last_call);
        return;
    }
    if (g_trace_first_method_writes) [[unlikely]] {
        TraceFirstMethodWrite(method, method_argument);
    }

    ASSERT(method < Regs::NUM_REGS && "Invalid Maxwell3D register, increase the size of the Regs structure");
    // The profile's write counters: ProfiledCallMethod, for this path and CallMethodFast.
    if (method == MAXWELL3D_REG_INDEX(shadow_ram_control) && front_macros) [[unlikely]] {
        back_control_writes[back_control_count++ % back_control_writes.size()] = {
            method_argument, static_cast<u32>(shadow_state.shadow_ram_control), is_last_call ? 1U : 0U};
    }
    const u32 argument = ProcessShadowRam(method, method_argument);
    // A state register written with the value it holds changes nothing: no dirty flags (they
    // would make the next draw re-check that state), no method processing. Title 1 makes ~6
    // million 3D register writes a second, ~60% of them same-value, most from macro replays.
    // Methods with side effects (execution_mask: draws, clears, constant buffer data and
    // binds, semaphores, ...) always run. EDEN_NX2_SKIP_SAME=0 turns it off.
    if (!execution_mask[method]) {
        // ProcessMethodCall (and the draw manager behind it) acts only on executable methods.
        if (!SkipSameValueWrites() || regs.reg_array[method] != argument) {
            ProcessDirtyRegisters(method, argument);
        }
        return;
    }
    ProcessDirtyRegisters(method, argument);
    ProcessMethodCall(method, argument, method_argument, is_last_call);
}

void Maxwell3D::DumpRecentWrites(u32 method) {
    static u32 dumps{};
    if (dumps++ >= 4) {
        return;
    }
    LOG_ERROR(HW_GPU, "Write to {:#x} while macro {:#x} collects parameters ({} so far); last writes:",
              method, executing_macro, macro_params.size());
    if (!record_write_history) {
        LOG_ERROR(HW_GPU, "Register write history disabled; EDEN_NX2_3D_HISTORY=1 enables it");
        return;
    }
    for (size_t i = 0; i < recent_writes.size(); ++i) {
        const auto& w = recent_writes[(recent_write_index + i) % recent_writes.size()];
        if (w.amount != 0) {
            LOG_ERROR(HW_GPU, "  method {:#x} arg {:#x} amount {} pending {}", w.method, w.argument,
                      w.amount, w.pending);
        }
    }
}

void Maxwell3D::CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) {
    RecordWrite(method, amount ? base_start[0] : 0, amount, methods_pending);
    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    // FRITCHII_DRAW_KINDS: the words written here (the default path counts them in CallMethod).
    if (VideoCommon::FritchiiDrawKinds::g_on) [[unlikely]] {
        constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
        const bool per_word =
            method < MacroRegistersStart && method - cb_data >= 16 &&
            method != MAXWELL3D_REG_INDEX(inline_data) &&
            ((method != MAXWELL3D_REG_INDEX(draw_inline_index) &&
              method != MAXWELL3D_REG_INDEX(inline_index_2x16.even) &&
              method != MAXWELL3D_REG_INDEX(inline_index_4x8.index0)) ||
             shadow_state.shadow_ram_control == Regs::ShadowRamControl::Replay);
        if (!per_word) {
            VideoCommon::FritchiiDrawKinds::NoteWrites(amount);
        }
    }
    if (method >= MacroRegistersStart) {
        ProcessMacro(system, method, base_start, amount, amount == methods_pending);
        return;
    }
    // As in ProcessDirtyRegisters: the paths below write without it.
    if (draw_manager.HasDeferredDraw() && !DrawManager::IsInstanceLoopMethod(method)) [[unlikely]] {
        draw_manager.DrawDeferred(*this, true);
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
        ProcessCBMultiData(base_start, amount);
        break;
    case MAXWELL3D_REG_INDEX(inline_data): {
        ASSERT(methods_pending == amount);
        VideoCommon::Nx2FastDrawCount::Break(VideoCommon::Nx2FastDrawCount::Reason::InlineData);
        upload_state.ProcessData(base_start, amount);
        return;
    }
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        if (shadow_state.shadow_ram_control != Regs::ShadowRamControl::Replay) {
            ProcessInlineIndexMultiData(method, base_start, amount);
            break;
        }
        [[fallthrough]];
    default:
        for (u32 i = 0; i < amount; i++) {
            CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        }
        break;
    }
}

void Maxwell3D::ProcessMacroUpload(u32 data) {
    GetNativeMacroEngine().AddCode(regs.load_mme.instruction_ptr++, data);
}

void Maxwell3D::ProcessMacroBind(u32 data) {
    const auto positions = GetNativeMacroPositions();
    const u32 index = regs.load_mme.start_address_ptr++;
    if (index < positions.size()) {
        positions[index] = data;
    }
}

void Maxwell3D::ProcessFirmwareCall4() {
    LOG_DEBUG(HW_GPU, "(STUBBED) called");

    // Firmware call 4 is a blob that changes some registers depending on its parameters.
    // These registers don't affect emulation and so are stubbed by setting 0xd00 to 1.
    regs.shadow_scratch[0] = 1;
}

void Maxwell3D::StampQueryResult(Core::System& system, u64 payload, bool long_query) {
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    if (long_query) {
        memory_manager.Write<u64>(sequence_address + sizeof(u64), system.GPU().GetTicks());
        memory_manager.Write<u64>(sequence_address, payload);
    } else {
        memory_manager.Write<u32>(sequence_address, static_cast<u32>(payload));
    }
    if (Common::Nx2GpuWrite::g_listener.load(std::memory_order_relaxed) != nullptr) {
        Common::Nx2GpuWrite::Notify(Common::Nx2GpuWrite::Path::Stamp3D,
                                    memory_manager.GpuToCpuAddress(sequence_address).value_or(0),
                                    sequence_address, memory_manager.GetPointer(sequence_address),
                                    static_cast<u32>(payload));
    }
    if (const u64 trace = VideoCommon::QueryTrace::Sample(VideoCommon::QueryTrace::Event::Write)) {
        LOG_INFO(HW_GPU,
                 "DKB_QUERY write n={} path=stamp as={} gpu={:#x} bytes={} value={:#x}",
                 trace, memory_manager.GetID(), sequence_address, long_query ? 8 : 4,
                 long_query ? payload : static_cast<u32>(payload));
    }
}

void Maxwell3D::ProcessQueryGet() {
    const VideoCommon::Nx2Profile::FineScope profile_scope{
        VideoCommon::Nx2Profile::Zone::QueryReport};
    VideoCommon::QueryPropertiesFlags flags{};
    if (regs.report_semaphore.query.short_query == 0) {
        flags |= VideoCommon::QueryPropertiesFlags::HasTimeout;
    }
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    const VideoCommon::QueryType query_type =
        static_cast<VideoCommon::QueryType>(regs.report_semaphore.query.report.Value());
    const u32 payload = regs.report_semaphore.payload;
    const u32 subreport = regs.report_semaphore.query.sub_report;
    // FRITCHII_DRAW_KINDS: a sample count report marks the draw before it.
    if (VideoCommon::FritchiiDrawKinds::g_on) [[unlikely]] {
        using Report = Regs::ReportSemaphore::Report;
        using Operation = Regs::ReportSemaphore::Operation;
        const Report report = regs.report_semaphore.query.report;
        const Operation operation = regs.report_semaphore.query.operation;
        if ((report == Report::ZPassPixelCount || report == Report::ZPassPixelCount64) &&
            (operation == Operation::Release || operation == Operation::ReportOnly)) {
            VideoCommon::FritchiiDrawKinds::NoteZPassReport();
        }
    }
    if (VideoCommon::Nx2FastDrawCount::g_on) [[unlikely]] {
        // Ported fix: the release lands in guest memory (a short one the payload, a long one the
        // 16-byte report), noted as constant data.
        const u64 bytes = regs.report_semaphore.query.short_query != 0 ? 4 : 16;
        NoteFastDrawCountData(sequence_address, sequence_address + bytes);
    }
    switch (regs.report_semaphore.query.operation) {
    case Regs::ReportSemaphore::Operation::Release:
        if (regs.report_semaphore.query.short_query != 0) {
            flags |= VideoCommon::QueryPropertiesFlags::IsAFence;
        }
        rasterizer->Query(sequence_address, query_type, flags, payload, subreport);
        break;
    case Regs::ReportSemaphore::Operation::Acquire:
        // TODO(Blinkhawk): Under this operation, the GPU waits for the CPU to write a value that
        // matches the current payload.
        UNIMPLEMENTED_MSG("Unimplemented query operation ACQUIRE");
        break;
    case Regs::ReportSemaphore::Operation::ReportOnly:
        rasterizer->Query(sequence_address, query_type, flags, payload, subreport);
        break;
    case Regs::ReportSemaphore::Operation::Trap:
        UNIMPLEMENTED_MSG("Unimplemented query operation TRAP");
        break;
    default:
        UNIMPLEMENTED_MSG("Unknown query operation");
        break;
    }
}

void Maxwell3D::ProcessQueryCondition() {
    const VideoCommon::Nx2Profile::FineScope profile_scope{
        VideoCommon::Nx2Profile::Zone::RenderEnable};
    const bool host_condition = rasterizer->AccelerateConditionalRendering();
    // FRITCHII_MEMSTAGE_SHADOW: draws under host conditional rendering are level (spec 3.4).
    if (VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed)) [[unlikely]] {
        VideoCommon::MemStageShadow::BackCondition(host_condition);
    }
    if (host_condition) {
        execute_on = true;
        // EDEN_NX2_MEMSTAGE_STATS: decided by the host (the draws run predicated).
        VideoCommon::MemStageStats::Condition(true, true);
        return;
    }
    // EDEN_NX2_MEMSTAGE_STATS: decided here (the end of this function sets execute_on).
    SCOPE_EXIT {
        VideoCommon::MemStageStats::Condition(false, execute_on);
    };
    const GPUVAddr condition_address{regs.render_enable.Address()};
    if (VideoCore::Nx2Debug::Experiment(53) != 0 &&
        regs.render_enable.mode >= Regs::RenderEnable::Mode::Conditional) {
        // EXP53: the front left the Compare words to this thread. Report words wait in the query
        // cache until a fence, and the caches belong to the thread running this (the split's
        // back): write them out before they are compared.
        memory_manager.FlushRegion(condition_address, sizeof(Regs::ReportSemaphore::Compare),
                                   VideoCommon::CacheType::QueryCache);
    }
    Regs::ReportSemaphore::Compare cmp{};
    bool operands_read = false;
    switch (regs.render_enable_override) {
    case Regs::RenderEnable::Override::AlwaysRender:
        execute_on = true;
        break;
    case Regs::RenderEnable::Override::NeverRender:
        execute_on = false;
        break;
    case Regs::RenderEnable::Override::UseRenderEnable: {
        switch (regs.render_enable.mode) {
        case Regs::RenderEnable::Mode::True: {
            execute_on = true;
            break;
        }
        case Regs::RenderEnable::Mode::False: {
            execute_on = false;
            break;
        }
        case Regs::RenderEnable::Mode::Conditional: {
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            operands_read = true;
            execute_on = cmp.initial_sequence != 0U && cmp.initial_mode != 0U;
            break;
        }
        case Regs::RenderEnable::Mode::IfEqual: {
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            operands_read = true;
            execute_on = cmp.initial_sequence == cmp.current_sequence &&
                         cmp.initial_mode == cmp.current_mode;
            break;
        }
        case Regs::RenderEnable::Mode::IfNotEqual: {
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            operands_read = true;
            execute_on = cmp.initial_sequence != cmp.current_sequence ||
                         cmp.initial_mode != cmp.current_mode;
            break;
        }
        default: {
            UNIMPLEMENTED_MSG("Uninplemented Condition Mode!");
            execute_on = true;
            break;
        }
        }
        break;
    }
    }
    if (const u64 trace =
            VideoCommon::QueryTrace::Sample(VideoCommon::QueryTrace::Event::CpuCondition)) {
        LOG_INFO(HW_GPU,
                 "DKB_CONDITION cpu n={} as={} gpu={:#x} mode={} override={} execute={} "
                 "operands_read={} initial={:#x}/{:#x} current={:#x}/{:#x}",
                 trace, memory_manager.GetID(), condition_address,
                 static_cast<u32>(regs.render_enable.mode),
                 static_cast<u32>(regs.render_enable_override), execute_on, operands_read,
                 cmp.initial_sequence, cmp.initial_mode, cmp.current_sequence, cmp.current_mode);
    }
}

void Maxwell3D::ProcessCounterReset() {
    const VideoCommon::Nx2Profile::FineScope profile_scope{
        VideoCommon::Nx2Profile::Zone::CounterReset};
    const auto query_type = [clear_report = regs.clear_report_value]() {
        switch (clear_report) {
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::ZPassPixelCount:
            return VideoCommon::QueryType::ZPassPixelCount64;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::StreamingPrimitivesSucceeded:
            return VideoCommon::QueryType::StreamingPrimitivesSucceeded;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::PrimitivesGenerated:
            return VideoCommon::QueryType::PrimitivesGenerated;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::VtgPrimitivesOut:
            return VideoCommon::QueryType::VtgPrimitivesOut;
        default:
            LOG_DEBUG(HW_GPU, "Unimplemented counter reset={}", clear_report);
            return VideoCommon::QueryType::Payload;
        }
    }();
    rasterizer->ResetCounter(query_type);
}

void Maxwell3D::ProcessSyncPoint() {
    if (GetNativeMacroEngine().native_turing) {
        LOG_DEBUG(HW_GPU, "Native 3D syncpoint increment: id={}, raw={:#x}", regs.sync_info.sync_point.Value(), regs.reg_array[MAXWELL3D_REG_INDEX(sync_info)]);
    }
    const u32 sync_point = regs.sync_info.sync_point.Value();
    [[maybe_unused]] const u32 cache_flush = regs.sync_info.clean_l2.Value();
    rasterizer->SignalSyncPoint(sync_point);
}

void Maxwell3D::ProcessCBBind(size_t stage_index) {
    const VideoCommon::Nx2Profile::FineScope profile_scope{VideoCommon::Nx2Profile::Zone::CbBind};
    // Bind the buffer currently in CB_ADDRESS to the specified index in the desired shader
    // stage.
    const auto& bind_data = regs.bind_groups[stage_index];
    static std::array<std::array<bool, 32>, 5> seen{};
    if (!seen[stage_index][bind_data.shader_slot]) {
        seen[stage_index][bind_data.shader_slot] = true;
        LOG_INFO(HW_GPU, "Constant buffer bind: stage {} slot {} address {:#x} size {:#x}", stage_index,
                 bind_data.shader_slot.Value(), regs.const_buffer.Address(), regs.const_buffer.size);
    }
    if (bind_data.shader_slot >= Regs::MaxConstBuffers) {
        return; // slot field is 5 bits; the state array holds 18
    }
    auto& buffer = state.shader_stages[stage_index].const_buffers[bind_data.shader_slot];
    if (VideoCommon::Nx2FastDrawCount::g_on) [[unlikely]] {
        if (buffer.enabled != (bind_data.valid.Value() != 0) ||
            buffer.address != regs.const_buffer.Address() ||
            buffer.size != regs.const_buffer.size) {
            VideoCommon::Nx2FastDrawCount::NoteBind(stage_index, bind_data.shader_slot);
        }
    }
    buffer.enabled = bind_data.valid.Value() != 0;
    buffer.address = regs.const_buffer.Address();
    buffer.size = regs.const_buffer.size;
    if (VideoCommon::MemoryStageLink::BackServed()) [[unlikely]] {
        return; // FRITCHII_MEMSTAGE_THREAD=2: the memory stage bound it on the buffer cache
    }

    const bool is_enabled = bind_data.valid.Value() != 0;
    if (!is_enabled) {
        rasterizer->DisableGraphicsUniformBuffer(stage_index, bind_data.shader_slot);
        return;
    }
    const GPUVAddr gpu_addr = regs.const_buffer.Address();
    const u32 size = regs.const_buffer.size;
    rasterizer->BindGraphicsUniformBuffer(stage_index, bind_data.shader_slot, gpu_addr, size);
}

void Maxwell3D::ProcessCBMultiData(const u32* start_base, u32 amount) {
    const Common::WriteRing::SiteScope write_site{"3D constant buffer data", false};
    const VideoCommon::Nx2Profile::FineScope profile_scope{VideoCommon::Nx2Profile::Zone::CbUpload};
    // Write the input value to the current const buffer at the current position.
    const GPUVAddr buffer_address = regs.const_buffer.Address();
    ASSERT(buffer_address != 0);

    // Don't allow writing past the end of the buffer.
    const size_t copy_size = amount * sizeof(u32);
    const size_t room = regs.const_buffer.offset < regs.const_buffer.size
                            ? regs.const_buffer.size - regs.const_buffer.offset
                            : 0;
    const size_t write_size = std::min(copy_size, room / sizeof(u32) * sizeof(u32));
    if (VideoCommon::MemoryStageLink::BackServed()) [[unlikely]] {
        // FRITCHII_MEMSTAGE_THREAD=2: the memory stage wrote the words and collected their range
        // (the back's half: the offset).
        regs.const_buffer.offset += static_cast<u32>(copy_size);
        return;
    }
    // EDEN_NX2_MEMSTAGE_STATS: constant buffer words, and whether they land on texture pages.
    if (VideoCommon::MemStageStats::Enabled() && write_size != 0) [[unlikely]] {
        VideoCommon::MemStageStats::StreamWrite(
            VideoCommon::MemStageStats::Write::CbWords,
            memory_manager.GpuToCpuAddress(buffer_address + regs.const_buffer.offset).value_or(0),
            write_size);
    }
    // EXP107: as in CallConstBufferWords.
    const ConstantWrite compared =
        write_size != 0
            ? CompareConstantData(buffer_address + regs.const_buffer.offset, start_base,
                                  write_size)
            : ConstantWrite::Changed;
    if (VideoCommon::Nx2FastDrawCount::g_on && write_size != 0) [[unlikely]] {
        const GPUVAddr address{buffer_address + regs.const_buffer.offset};
        NoteFastDrawCountData(address, address + write_size, compared != ConstantWrite::Changed);
    }
    if (write_size != 0 && compared != ConstantWrite::Skip &&
        !WriteBoundConstantBuffer(regs.const_buffer.offset, start_base,
                                  static_cast<u32>(write_size))) {
        const GPUVAddr address{buffer_address + regs.const_buffer.offset};
        memory_manager.WriteBlockCached(address, start_base, write_size);
    }
    if (write_size != copy_size) {
        static std::atomic<bool> warned{};
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            LOG_ERROR(HW_GPU,
                      "Dropping constant buffer words past the end: offset={:#x}, size={:#x}, "
                      "bytes={:#x}, written={:#x}",
                      regs.const_buffer.offset, regs.const_buffer.size, copy_size, write_size);
        }
    }

    // Increment the current buffer position.
    regs.const_buffer.offset += static_cast<u32>(copy_size);
}

void Maxwell3D::ProcessCBData(u32 value) {
    ProcessCBMultiData(&value, 1);
}

bool Maxwell3D::WriteBoundConstantBuffer(u32 offset, const void* data, u32 bytes) {
    namespace CbPath = VideoCommon::Nx2CbPath;
    const int mode = CbPath::Mode();
    if ((mode & CbPath::DirectBit) == 0) {
        return false;
    }
    auto& counters = CbPath::g_counters;
    const GPUVAddr address = regs.const_buffer.Address();
    const u32 size = regs.const_buffer.size;
    const u64 map_serial = memory_manager.MapSerial();
    BoundConstantBuffer& bound = bound_constant_buffer;
    // One translation per selection (fable52_answer_next70 3.2): the same bytes reach the same
    // device memory and the same ranges the same invalidation, translated when written instead
    // of when flushed; map_serial covers a remap between writes.
    if (bound.address != address || bound.size != size || bound.map_serial != map_serial) {
        const Tegra::MemoryManager::MapRead read = memory_manager.BeginMapRead();
        bound = {address, size, read.serial, 0};
        if (address != 0 && size != 0 && memory_manager.IsContinuousRange(address, size)) {
            bound.device = memory_manager.GpuToCpuAddress(address).value_or(0);
        }
        // A page table change overlapped the translation: used for this write only (the
        // counter never reaches ~0, so the next write translates again).
        if (!memory_manager.MapReadStable(read)) {
            bound.map_serial = ~u64{0};
        }
    }
    if (bound.device == 0) {
        CbPath::Count(counters.fallback_device0);
        return false;
    }
    const GPUVAddr begin = address + offset;
    // Words on Switch 2 sparse memory go the general path (Nx2BackSparseWrite).
    if (begin + bytes > Tegra::g_nx2_sparse_low.load(std::memory_order_relaxed) &&
        begin < Tegra::g_nx2_sparse_high.load(std::memory_order_relaxed)) {
        CbPath::Count(counters.fallback_sparse);
        return false;
    }
    const bool verify = (mode & CbPath::VerifyBit) != 0;
    memory_manager.WriteConstantBufferData(bound.device + offset, begin, data, bytes, verify);
    CbPath::Count(counters.direct_writes);
    if (verify) {
        // The bytes read back through the general path's translation.
        std::array<u8, 256> read;
        bool differs = false;
        for (u32 done = 0; done < bytes;) {
            const u32 amount = (std::min<u32>)(bytes - done, static_cast<u32>(read.size()));
            memory_manager.ReadBlockUnsafe(begin + done, read.data(), amount);
            differs |= std::memcmp(read.data(), static_cast<const u8*>(data) + done, amount) != 0;
            done += amount;
        }
        CbPath::Count(counters.verify_direct_checked);
        if (differs) {
            CbPath::Count(counters.verify_direct_mismatch);
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_ERROR(HW_GPU,
                          "EXP88 verify: constant buffer data at {:#x}+{:#x} (device {:#x}) "
                          "reads back different",
                          begin, bytes, bound.device + offset);
            }
        }
    }
    return true;
}

void Maxwell3D::ProcessInlineIndexMultiData(u32 method, const u32* start_base, u32 amount) {
    if (amount == 0) {
        return;
    }
    const u32 argument = ProcessShadowRam(method, start_base[amount - 1]);
    ProcessDirtyRegisters(method, argument);
    draw_manager.SetInlineIndexBuffer(*this, method, start_base, amount);
}

Texture::TICEntry Maxwell3D::GetTICEntry(u32 tic_index) const {
    const GPUVAddr tic_address_gpu{regs.tex_header.Address() +
                                   tic_index * sizeof(Texture::TICEntry)};
    Texture::TICEntry tic_entry;
    memory_manager.ReadBlockUnsafe(tic_address_gpu, &tic_entry, sizeof(Texture::TICEntry));
    return tic_entry;
}

Texture::TSCEntry Maxwell3D::GetTSCEntry(u32 tsc_index) const {
    const GPUVAddr tsc_address_gpu{regs.tex_sampler.Address() +
                                   tsc_index * sizeof(Texture::TSCEntry)};
    Texture::TSCEntry tsc_entry;
    memory_manager.ReadBlockUnsafe(tsc_address_gpu, &tsc_entry, sizeof(Texture::TSCEntry));
    return tsc_entry;
}

u32 Maxwell3D::GetRegisterValue(u32 method) const {
    ASSERT(method < Regs::NUM_REGS && "Invalid Maxwell3D register");
    return regs.reg_array[method];
}

void Maxwell3D::SetHLEReplacementAttributeType(u32 bank, u32 offset,
                                               HLEReplacementAttributeType name) {
    const u64 key = (static_cast<u64>(bank) << 32) | offset;
    replace_table.emplace(key, name);
}

// Split GPU thread with macros on its front half (gpu_split.h, EDEN_NX2_SPLIT=2). The front keeps
// what macros read: the registers as the back thread will hold them when it reaches the same
// point of the method stream. That is every write the front sends, with the shadow RAM applied,
// plus the few changes the back makes on its own (constant buffer offset, shadow_scratch[0] on
// falcon[4], the macro bind pointer); the front loads and binds the macro code itself.
struct Maxwell3D::FrontState {
    std::array<ControlWrite, 16> control_writes{};
    u32 control_count{};
    std::array<u32, Regs::NUM_REGS> regs{};
    std::array<u32, Regs::NUM_REGS> shadow{};
    Regs::ShadowRamControl control{};
    u32 executing_macro{};
    std::vector<u32> parameters;
    /// Which FrontRunMacro body runs (Fable's feature byte, fable_answer_regression_2 3.1): 0 =
    /// v5.7's (FrontRunMacroPlain); bit 0 EXP53, 1 EDEN_NX2_OOT_MACROS, 2 EDEN_NX2_FOLD_INSTANCE,
    /// 3 EDEN_NX2_DRAW_PACKET = FrontRunMacroFeatures. Refreshed when
    /// Nx2Debug::g_switch_generation moved since switches_seen, and every 1024 calls.
    u8 macro_features{};
    bool macro_features_logged{};
    u32 switches_seen{~u32{0}};
    u32 macro_calls{};
    /// EXP53: per macro, 0 unknown, 1 a render-enable predicate, 2 something else.
    std::vector<u8> predicate_macros;
    u64 predicate_generation{~u64{0}};
    /// Per macro slot (FrontRunMacro): the code generation and start it was identified for
    /// and its OoT replacement (macro_hle_oot.h, by code hash; None for other code). Used by
    /// EDEN_NX2_OOT_MACROS and EDEN_NX2_FOLD_INSTANCE.
    static constexpr size_t MacroSlots = 0x80;
    std::array<u64, MacroSlots> oot_generation{};
    std::array<u32, MacroSlots> oot_start{};
    std::array<bool, MacroSlots> oot_known{};
    std::array<OotMacros::Id, MacroSlots> oot_id{};
    /// EDEN_NX2_OOT_MACROS statistics, logged every ~10 s while the switch is on ("OoT macros").
    struct OotStats {
        u64 native;      ///< Calls a handler ran (mode 1)
        u64 declined;    ///< Handler calls that returned false: run through the JIT
        u64 unmatched;   ///< Calls of macros without a handler (JIT)
        u64 verified;    ///< Calls compared (modes 2 and 3)
        u64 differences; ///< Of those, with a difference
        u64 unverifiable; ///< Calls the recording context could not reproduce (data RAM DMA)
        u64 folds;       ///< EDEN_NX2_FOLD_INSTANCE: first instances folded
    };
    OotStats oot_stats{};
    u64 oot_ticks{};
    std::chrono::steady_clock::time_point oot_report{};
    /// Calls per replacement (modes 2 and 3 pick the calls to compare from it).
    std::array<u64, 32> oot_calls{};
    std::array<u32, 32> oot_mismatch_logs{};
    /// Verify buffers: the JIT's emitted (method, value) pairs, the handler's, and the data RAM
    /// before the call (the handler runs on this copy after the JIT).
    std::vector<u32> oot_jit_writes;
    std::vector<u32> oot_handler_writes;
    std::array<u32, OotMacros::DataRamWords> oot_dram{};
    /// EDEN_NX2_DRAW_PACKET: this macro call's draw writes go in the packet (set per call in
    /// FrontRunMacro), the mode (2 = the back verifies), and the held writes (method | argument
    /// << 32), the values the front's registers took, and the entries' combined effect.
    bool packet_on{};
    int packet_mode{};
    static constexpr size_t MaxPacketWrites = 64;
    std::vector<u64> packet;
    std::vector<u32> packet_values;
    VideoCommon::GpuSplit::Effect packet_effect{VideoCommon::GpuSplit::Effect::None};
    u64 packets{};       ///< Packets sent (the "OoT macros" line)
    u64 packet_writes{}; ///< Writes they carried
    Core::System* system{};
    std::vector<u32> dma_words; ///< FrontDmaReadEarly's words
    u64 calls{};
    /// A method the draw manager handles was sent since the back last ran DrawDeferred: only
    /// then can it have a deferred instanced draw to issue at the end of a macro.
    bool draw_pending{true};
    /// The back holds state writes in its method sink, which its next executable method consumes
    /// (issuing a deferred draw a changed one ends). Front() consumes it before copying the state.
    /// Instance folding folds only while it is empty.
    bool sink_pending{};
    /// List draws (nx2_list_draw.h): by start in the macro code of
    /// list_draw_generation, whether the macro is the array draw FrontTakeListDraw sends as an
    /// address.
    std::unordered_map<u32, bool> list_draw_macros;
    u64 list_draw_generation{~u64{0}};
    /// List guard (nx2_list_dispatch.h): the registers the last list draw left with their old values
    /// (bit 0: vertex_buffer.count, bit 1: vertex_buffer.first), on the front and the back
    /// alike, until they are written again; the list its words are in; whether an index count
    /// was written since the last draw.end (the draw then uses neither register).
    u32 list_draw_stale{};
    Nx2ListDispatch::ListReader list_draw_list;
    bool list_draw_indexed{};
    bool in_list_draw{};
    /// By start in the macro code of data_ram_generation (an index per instruction), what the
    /// macro's code can do (AnalyzeEffects): bit 0 use the data RAM, bit 1 read
    /// vertex_buffer.first or count; 0xff not analyzed yet.
    std::vector<u8> data_ram_macros;
    u64 data_ram_generation{~u64{0}};
    /// Set by FrontRunMacro for one call of FrontRunMacroFeatures, whose code can read one of
    /// the two stale registers: check each of its register reads (our macro bodies; the plain
    /// one stays v5.7's).
    bool guard_list_reads{};
    /// Set by FrontRunMacroFeatures for one call of the 0xe2c occlusion macro (hash
    /// 0x3b26ab7eb3fc9863, OotMacros::Id::E2C), handler or JIT: its MME DMA read is counted in
    /// the "Front 0xe2c misses" line (FrontDmaRead).
    bool e2c_call{};
    bool skip_same{}; ///< SkipSameValueWrites() && FrontSkipSame(), read once
    /// The inline upload being sent, when its words land linearly (EDEN_NX2_DMA_WAIT_WRITERS).
    bool upload_known{};
    /// The back's draw manager is in an instance loop (DrawMode::Instance): draw.end does nothing.
    bool instance_loop{};
    /// EDEN_NX2_EXP2, read in Front(): same-value index_buffer.first writes are not sent.
    int same_experiment{};
    int early_same{}; ///< EXP14, read once per front call like same_experiment
    /// EDEN_NX2_EXP22 (the original front state, draw batches and same-value draw
    /// registers), read once per front call like early_same.
    int exp22{};
    /// StateDelta blocks would skip the back's write history and first-write trace
    /// (EDEN_NX2_3D_HISTORY, EDEN_NX2_3D_METHODS): off with them.
    bool delta_allowed{};
    /// EXP22 bit 2: the last complete instance-loop group (draw.begin Subsequent and the loop
    /// writes after it) held back, how many times in a row it came, and the next group while it
    /// is captured. Held only until the next ring entry, publish or drain (FrontFlushPending).
    struct BatchWrite {
        u32 method;
        u32 argument;
        bool is_last;
        bool from_macro;
    };
    static constexpr u32 MaxBatchWrites = 8;
    std::array<BatchWrite, MaxBatchWrites> batch{};
    u32 batch_size{};
    u32 batch_repeats{};
    std::array<BatchWrite, MaxBatchWrites> capture{};
    u32 capture_size{};
    /// An inline or small index draw (which binds an index buffer not taken from the registers)
    /// was sent since the last index_buffer.count write. Starts set: nothing is known.
    bool other_index_draw{true};
    /// EDEN_NX2_EXP2 counters, logged every ~5 s: writes the experiment dropped, writes the
    /// existing same-value filters dropped, and writes sent.
    u64 exp_skipped{};
    u64 exp_same{};
    u64 exp_sent{};
    u64 exp_ticks{};
    std::chrono::steady_clock::time_point exp_report{};
    u64 upload_device{};
    u32 upload_size{};
    u32 upload_offset{};
    /// EXP13=2-4: the upload's write-stamp tag (Nx2PatchQmdCopy looks past its own stamps).
    u64 upload_tag{};
    /// Macro DMA read snapshots (EDEN_NX2_DMA_SNAPSHOT, from a ported dma-snapshot fix): device
    /// pages a drained read flushed whole, and the split position right after. Cleared past 8192
    /// pages.
    struct DmaSnapshot {
        u64 position;
        /// EXP10: the noted compute and copy ranges and recorded dispatch writes already found
        /// to miss the page (GpuSplit::WritersMayWriteSince, DispatchesMayWriteSince).
        u64 writer_ranges;
        u64 dispatch_writes;
        /// EXP10 +8: the noted ranges and dispatch writes since that do overlap the page; a
        /// read of bytes none of them overlaps still skips the drain.
        VideoCommon::GpuSplit::ListedWrites listed{};
    };
    std::unordered_map<u64, DmaSnapshot> dma_snapshots;
};

namespace {

// EDEN_NX2_OOT_MACROS=1: the context Ocarina of Time's native macro handlers (macro_hle_oot.h,
// ported, its maxwell_3d.cpp OotContext) run in on the
// split front. Emit is FrontRunMacro's own emit, the one the JIT calls, so every front transform
// of a macro write (EXP53's predicate, EDEN_NX2_FOLD_INSTANCE, EDEN_NX2_MACRO_TRACE, the MME DMA
// reads) applies to a handler's writes exactly as to the JIT's.
template <typename EmitFn>
struct OotFrontContext {
    std::vector<u32>& parameters;
    u32* dram;
    std::span<const u32> regs;
    EmitFn& emit;
    const char* error{};

    [[nodiscard]] u32 Count() const {
        return static_cast<u32>(parameters.size());
    }
    [[nodiscard]] u32 Param(u32 index) const {
        return parameters[index];
    }
    [[nodiscard]] u32* Dram() {
        return dram;
    }
    [[nodiscard]] u32 Read(u32 method) const {
        return method < regs.size() ? regs[method] : 0U;
    }
    [[nodiscard]] bool OcclusionHle() const {
        return false;
    }
    void FoldFirstInstance() {
        // FrontRunMacro's emit folds at the loop's first draw.end (the same point: after the
        // loop's first draw.begin, checked against the front's copy of it).
    }
    void Emit(u32 method, u32 value) {
        if (!emit(method, value)) [[unlikely]] {
            Fail("method past the macro range");
        }
    }
    void EmitRepeated(const std::array<u32, 2>* writes, u32 count, u32 times) {
        for (u32 time = 0; time < times; ++time) {
            for (u32 i = 0; i < count; ++i) {
                Emit(writes[i][0], writes[i][1]);
            }
        }
    }
    void Fail(const char* text) {
        if (error == nullptr) {
            error = text;
        }
    }
};

// EDEN_NX2_OOT_MACROS=2/3 (verify): a handler run after the JIT ran the same call, without
// effects: its writes are recorded, it works on a copy of the data RAM from before the call, and
// the FIFO is the one the JIT left (the MME DMA reads appended), grown to the size it had at each
// point of the call as the handler's own MME_DMA_READ_FIFOED writes reach it.
struct OotRecordContext {
    std::span<const u32> parameters;
    u32 count;
    u32* dram;
    std::span<const u32> regs;
    std::vector<u32>& writes;
    const char* error{};
    bool unverifiable{};

    [[nodiscard]] u32 Count() const {
        return count;
    }
    [[nodiscard]] u32 Param(u32 index) const {
        return index < parameters.size() ? parameters[index] : 0U;
    }
    [[nodiscard]] u32* Dram() {
        return dram;
    }
    [[nodiscard]] u32 Read(u32 method) const {
        return method < regs.size() ? regs[method] : 0U;
    }
    [[nodiscard]] bool OcclusionHle() const {
        return false;
    }
    void FoldFirstInstance() {}
    void Emit(u32 method, u32 value) {
        writes.push_back(method);
        writes.push_back(value);
        if (method >= OotMacros::FirstMacroMethod) [[unlikely]] {
            Fail("method past the macro range");
        } else if (method == 0x158) {
            // MME_DMA_READ_FIFOED: FrontDmaRead appended up to 0x1000 words (EXP53: `value`).
            const u64 grown = u64{count} + std::min<u32>(value, 0x1000);
            count = static_cast<u32>(std::min<u64>(grown, parameters.size()));
        } else if (method == 0x157) {
            // MME_DMA_READ into the data RAM: the copy doesn't get the words.
            unverifiable = true;
        }
    }
    void EmitRepeated(const std::array<u32, 2>* list, u32 list_count, u32 times) {
        for (u32 time = 0; time < times; ++time) {
            for (u32 i = 0; i < list_count; ++i) {
                Emit(list[i][0], list[i][1]);
            }
        }
    }
    void Fail(const char* text) {
        if (error == nullptr) {
            error = text;
        }
    }
};

enum class OotVerdict { Same, Different, Declined, Unverifiable };

// EDEN_NX2_DRAW_PACKET: the writes of OoT's draw macros (macro_hle_oot.h E04, E52, EA0) that go
// in the packet: state and draw registers that write no memory and that FrontWrite notes nothing
// for after sending. The constant buffer words of 0xea0 (0x8e4, 0x8e5) are not among them.
static_assert(MAXWELL3D_REG_INDEX(draw.end) == 0x585 && MAXWELL3D_REG_INDEX(draw.begin) == 0x586);
static_assert(MAXWELL3D_REG_INDEX(const_buffer.offset) == 0x8e3 &&
              MAXWELL3D_REG_INDEX(const_buffer.buffer) == 0x8e4);
[[nodiscard]] bool IsDrawPacketMethod(u32 method) {
    switch (method) {
    case 0x446: // base vertex
    case 0x50d: // base vertex copy
    case 0x50e: // base instance
    case 0x585: // draw.end
    case 0x586: // draw.begin
    case 0x5f2: // index_buffer address high, low and format (0xe04)
    case 0x5f3:
    case 0x5f6:
    case 0x5f7: // index_buffer.first
    case 0x5f8: // index_buffer.count
    case 0x8e3: // const_buffer.offset
        return true;
    default:
        return false;
    }
}

[[nodiscard]] bool IsDrawMacro(OotMacros::Id id) {
    return id == OotMacros::Id::E04 || id == OotMacros::Id::E52 || id == OotMacros::Id::EA0;
}

// Runs replacement `id` with an OotRecordContext for the call the JIT just ran and compares the
// writes, the data RAM and the failure with the JIT's. Logs the first differences of each
// replacement (up to `logs` 4 times).
OotVerdict VerifyOotCall(OotMacros::Id id, u32 method, u32 start, std::span<const u32> parameters,
                         size_t pre_count, std::span<const u32> regs, std::span<u32> dram_copy,
                         std::span<const u32> dram_after, std::span<const u32> jit_writes,
                         bool jit_ok, std::vector<u32>& handler_writes, u32& logs) {
    handler_writes.clear();
    OotRecordContext context{parameters, static_cast<u32>(std::min(pre_count, parameters.size())),
                             dram_copy.data(), regs, handler_writes};
    if (!OotMacros::Run(id, context)) {
        return OotVerdict::Declined;
    }
    if (context.unverifiable) {
        return OotVerdict::Unverifiable;
    }
    const bool handler_ok = context.error == nullptr;
    const bool same_writes = std::ranges::equal(jit_writes, handler_writes);
    const bool same_dram = std::ranges::equal(dram_after, dram_copy);
    if (same_writes && same_dram && handler_ok == jit_ok) [[likely]] {
        return OotVerdict::Same;
    }
    if (logs >= 4) {
        return OotVerdict::Different;
    }
    ++logs;
    std::string params;
    for (size_t i = 0; i < std::min<size_t>(pre_count, 16); ++i) {
        params += fmt::format(" {:x}", parameters[i]);
    }
    std::string text = fmt::format(
        "OoT macro verify: {} (method {:#x} start {:#x}) differs from the JIT; parameters{}{}; "
        "writes JIT {} handler {}",
        OotMacros::Name(id), method, start, params, pre_count > 16 ? " ..." : "",
        jit_writes.size() / 2, handler_writes.size() / 2);
    if (!same_writes) {
        size_t i = 0;
        while (i + 1 < jit_writes.size() && i + 1 < handler_writes.size() &&
               jit_writes[i] == handler_writes[i] && jit_writes[i + 1] == handler_writes[i + 1]) {
            i += 2;
        }
        const auto pair = [](std::span<const u32> list, size_t at) {
            return at + 1 < list.size() ? fmt::format("{:x}={:x}", list[at], list[at + 1])
                                        : std::string{"(end)"};
        };
        text += fmt::format(", first difference at write {}: JIT {} handler {}", i / 2,
                            pair(jit_writes, i), pair(handler_writes, i));
    }
    if (!same_dram) {
        size_t differing = 0;
        size_t first = dram_after.size();
        for (size_t i = 0; i < dram_after.size(); ++i) {
            if (dram_after[i] != dram_copy[i]) {
                first = std::min(first, i);
                ++differing;
            }
        }
        text += fmt::format(", data RAM: {} words differ, first [{:#x}] JIT {:x} handler {:x}",
                            differing, first, dram_after[first], dram_copy[first]);
    }
    if (handler_ok != jit_ok) {
        text += fmt::format(", JIT {} handler {}", jit_ok ? "ok" : "failed",
                            handler_ok ? "ok" : context.error);
    }
    LOG_WARNING(HW_GPU, "{}", text);
    return OotVerdict::Different;
}

// Each macro slot's identification, logged once per method, start and code hash (at most 160
// lines a session): "OoT macro slot". EDEN_NX2_OOT_MACROS_DUMP=1 also logs the code.
void LogOotSlot(u32 method, u32 start, u64 hash, std::span<const u32> code,
                const OotMacros::Macro* found) {
    static std::mutex mutex;
    static std::unordered_set<u64> logged;
    static const bool dump = std::getenv("EDEN_NX2_OOT_MACROS_DUMP") != nullptr;
    std::scoped_lock lock{mutex};
    if (logged.size() >= 160 ||
        !logged.insert(hash ^ (u64{method} << 48) ^ (u64{start} << 24)).second) {
        return;
    }
    LOG_INFO(HW_GPU, "OoT macro slot: method {:#x} start {:#x} instructions {} hash {:016x}: {}",
             method, start, OotMacros::MacroLength(code, start), hash,
             found != nullptr ? found->name : "no handler");
    if (dump) {
        OotMacros::DumpCode(method, start, hash, code);
    }
}

// EDEN_NX2_SPLIT_VERIFY=1: every 4096 front macro calls, wait for the back and compare the
// front's registers with the back's (logs and resynchronizes differences).
bool FrontVerifyEnabled() {
    static const bool verify = std::getenv("EDEN_NX2_SPLIT_VERIFY") != nullptr;
    return verify;
}

// EDEN_NX2_MACRO_TRACE=<method>: the first 8 calls of that macro (with a parameter other than 0),
// parameters and writes, counted across both FrontRunMacro bodies.
u32 MacroTraced() {
    static const u32 traced = [] {
        const char* value = std::getenv("EDEN_NX2_MACRO_TRACE");
        return value != nullptr ? static_cast<u32>(std::strtoul(value, nullptr, 16)) : 0U;
    }();
    return traced;
}
u32 g_macro_traced_calls{};

// Methods whose execution on the back writes guest memory (inline uploads, reports).
bool WritesGuestMemory(u32 method) {
    return method == MAXWELL3D_REG_INDEX(launch_dma) ||
           method == MAXWELL3D_REG_INDEX(inline_data) ||
           method == MAXWELL3D_REG_INDEX(report_semaphore.query);
}

// Executable methods ProcessMethodCall handles itself; the rest go to the draw manager.
bool FrontHandledOutsideDrawManager(u32 method) {
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method >= cb_data && method < cb_data + 16) {
        return true;
    }
    switch (method) {
    case Maxwell3D::RootTables::SelectMethod:
    case Maxwell3D::RootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true;
    default:
        return false;
    }
}

// What an executable method may write on the back, besides a deferred draw it may issue
// (GpuSplit::Effect). Constant buffer data and reports write ranges the front notes.
VideoCommon::GpuSplit::Effect FrontMethodEffect(u32 method) {
    using Effect = VideoCommon::GpuSplit::Effect;
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method >= cb_data && method < cb_data + 16) {
        return Effect::None;
    }
    switch (method) {
    case Maxwell3D::RootTables::SelectMethod:
    case Maxwell3D::RootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
        return Effect::None;
    default:
        return FrontHandledOutsideDrawManager(method) ? Effect::Unknown : Effect::Gpu;
    }
}

// Where the next constant buffer data word goes (ProcessCBMultiData), from the front's registers.
GPUVAddr ConstBufferWriteAddress(std::span<const u32> regs) {
    return ((GPUVAddr{regs[MAXWELL3D_REG_INDEX(const_buffer.address_high)]} << 32) |
            regs[MAXWELL3D_REG_INDEX(const_buffer.address_low)]) +
           regs[MAXWELL3D_REG_INDEX(const_buffer.offset)];
}

// A report and what it may write: payload and timestamp.
GPUVAddr ReportAddress(std::span<const u32> regs) {
    return (GPUVAddr{regs[MAXWELL3D_REG_INDEX(report_semaphore.address_high)]} << 32) |
           regs[MAXWELL3D_REG_INDEX(report_semaphore.address_low)];
}
constexpr u64 ReportBytes = 16;

// EDEN_NX2_SPLIT_SKIP=0: send state writes that keep the register's value too.
bool FrontSkipSame() {
    static const bool skip = [] {
        const char* value = std::getenv("EDEN_NX2_SPLIT_SKIP");
        return value == nullptr || value[0] != '0';
    }();
    return skip;
}

} // Anonymous namespace

Maxwell3D::FrontState& Maxwell3D::Front(Core::System& system, VideoCommon::GpuSplit& split) {
    if (!front) [[unlikely]] {
        // Copy the back's state and take the macros over while it is idle (state writes it
        // still holds in its method sink first).
        split.Drain(VideoCommon::DrainSite::FrontStart);
        ConsumeSink(system);
        front = std::make_unique<FrontState>();
        std::memcpy(front->regs.data(), regs.reg_array.data(), sizeof(front->regs));
        std::memcpy(front->shadow.data(), shadow_state.reg_array.data(), sizeof(front->shadow));
        front->control = shadow_state.shadow_ram_control;
        front->skip_same = SkipSameValueWrites() && FrontSkipSame();
        front->delta_allowed = !record_write_history && !g_trace_first_method_writes;
        front->executing_macro = executing_macro;
        front->system = &system;
        front->parameters = std::move(macro_params);
        macro_params.clear();
        executing_macro = 0;
        front_macros = true;
    }
    front->same_experiment = VideoCore::Nx2Debug::Experiment(2);
    front->early_same = VideoCore::Nx2Debug::Experiment(14);
    front->exp22 = VideoCore::Nx2Debug::Experiment(22);
    return *front;
}

namespace {

// Indexed draws that bind an index buffer other than the registers' (inline indices, small index
// draws): the back sets and consumes Dirty::IndexBuffer for them itself.
bool IsOtherIndexDraw(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        return true;
    default:
        return false;
    }
}

} // Anonymous namespace

void Maxwell3D::FrontTrackIndexDraws(FrontState& fs, u32 method) {
    if (method == MAXWELL3D_REG_INDEX(index_buffer.count)) {
        fs.other_index_draw = false;
    } else if (IsOtherIndexDraw(method)) {
        fs.other_index_draw = true;
    }
}

bool Maxwell3D::FrontSkipsIndexFirst(FrontState& fs, u32 method, u32 value, bool shadow_kept) {
    // EDEN_NX2_EXP2: index_buffer.first written with its value. On the back it only stores the
    // register and sets Dirty::IndexBuffer (an instance loop method: no deferred draw flush, and
    // the draw manager reads the register when it draws). The flag matters only when the index
    // buffer binding is stale for the next register-indexed draw; that draw needs an
    // index_buffer.count write (draw_indexed), which always sets the flag, unless the count was
    // written before an inline or small index draw that rebound and consumed the flag (then the
    // write is sent).
    if (method != MAXWELL3D_REG_INDEX(index_buffer.first) || !fs.skip_same || !shadow_kept ||
        fs.regs[method] != value || fs.other_index_draw) {
        return false;
    }
    ++fs.exp_skipped;
    return true;
}

void Maxwell3D::FrontReportSameWrites(FrontState& fs) {
    if ((++fs.exp_ticks & 0xffff) != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (fs.exp_report == std::chrono::steady_clock::time_point{}) {
        fs.exp_report = now;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - fs.exp_report).count();
    if (seconds < 5.0) {
        return;
    }
    LOG_INFO(HW_GPU,
             "EDEN_NX2_EXP2: 3D writes per second: {:.0f} sent, {:.0f} same-value index_buffer.first "
             "not sent, {:.0f} other same-value writes not sent",
             fs.exp_sent / seconds, fs.exp_skipped / seconds, fs.exp_same / seconds);
    fs.exp_skipped = fs.exp_same = fs.exp_sent = 0;
    fs.exp_report = now;
}

void Maxwell3D::FrontWrite(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                           u32 argument, bool is_last_call, bool from_macro) {
    namespace Profile = VideoCommon::Nx2Profile;
    if (fs.list_draw_stale != 0) [[unlikely]] {
        FrontNoteListDrawWrite(split, fs, method);
    }
    // As ProcessShadowRam does on the back.
    u32 value = argument;
    bool shadow_kept = true;
    if (fs.control == Regs::ShadowRamControl::Track ||
        fs.control == Regs::ShadowRamControl::TrackWithFilter) {
        shadow_kept = fs.shadow[method] == argument;
        fs.shadow[method] = argument;
    } else if (fs.control == Regs::ShadowRamControl::Replay) {
        value = fs.shadow[method];
    }
    // EXP14: an unchanged state register (6.6M of Ocarina's 11.4M macro writes a second) is
    // dropped before the checks below, as the same-value skip further down would.
    if (!execution_mask[method] && fs.regs[method] == value && shadow_kept && fs.skip_same &&
        fs.same_experiment == 0 && fs.early_same != 0) [[likely]] {
        if (Profile::Enabled()) [[unlikely]] {
            Profile::Add(Profile::Counter::MethodWrites, 1);
            Profile::Add(Profile::Counter::SameValueWrites, 1);
        }
        return;
    }
    if (Profile::Enabled()) [[unlikely]] {
        Profile::Add(Profile::Counter::MethodWrites, 1);
    }
    // EXP22 bit 1 (the original's EDEN_NX2_FRONT_STATE): a write to a state register that
    // writes no memory, which the rest of this function would only compare, store and send as a
    // Direct MacroWrite (a macro's, EDEN_NX2_MACRO_SINK off) or a PairWrite (EDEN_NX2_RING_PAIRS:
    // the command list's, and the macros' with MACRO_SINK), goes in the open StateDelta block at
    // that same place, with the effect that path notes. Sunk Method entries (MACRO_SINK without
    // RING_PAIRS) apply later than a block would and stay as they are.
    if ((fs.exp22 & 1) != 0 && !execution_mask[method] && !memory_write_mask[method] &&
        fs.delta_allowed && fs.same_experiment == 0) {
        const bool direct = from_macro && !VideoCommon::GpuSplit::MacroSink();
        if (direct || VideoCommon::GpuSplit::RingPairs()) {
            if (fs.regs[method] == value && shadow_kept && fs.skip_same) {
                if (Profile::Enabled()) [[unlikely]] {
                    Profile::Add(Profile::Counter::SameValueWrites, 1);
                }
                return;
            }
            fs.regs[method] = value;
            using Effect = VideoCommon::GpuSplit::Effect;
            split.DeltaWrite(this, method, argument,
                             direct && fs.draw_pending ? Effect::Gpu : Effect::None);
            fs.sink_pending = false; // ApplyFrontDelta consumes the sink first
            if (from_macro && !direct) {
                split.CountBatched(VideoCommon::GpuSplit::Batched::MacroSunk);
            }
            return;
        }
    }
    if (method == MAXWELL3D_REG_INDEX(shadow_ram_control) && FrontVerifyEnabled()) [[unlikely]] {
        fs.control_writes[fs.control_count++ % fs.control_writes.size()] = {
            argument, static_cast<u32>(fs.control), from_macro ? 2U : 1U};
    }
    // Ported: instance loop writes the back would ignore.
    if (execution_mask[method] && FrontSkipsLoopWrite(fs, method, value, shadow_kept)) {
        if (Profile::Enabled()) [[unlikely]] {
            Profile::Add(Profile::Counter::SameValueWrites, 1);
        }
        if (fs.same_experiment != 0) [[unlikely]] {
            ++fs.exp_same;
            FrontReportSameWrites(fs);
        }
        return;
    }
    // EXP2, and EXP22 bit 4 (the original's EDEN_NX2_SKIP_SAME_DRAWREG: its same-value vertex_buffer
    // first and count are the loop writes skipped above).
    if ((fs.same_experiment != 0 || (fs.exp22 & 4) != 0) && execution_mask[method] &&
        FrontSkipsIndexFirst(fs, method, value, shadow_kept)) {
        if (Profile::Enabled()) [[unlikely]] {
            Profile::Add(Profile::Counter::SameValueWrites, 1);
        }
        if (fs.same_experiment != 0) [[unlikely]] {
            FrontReportSameWrites(fs);
        }
        split.CountBatched(VideoCommon::GpuSplit::Batched::SameIndexFirst);
        return;
    }
    // Instance folding (EDEN_NX2_FOLD_INSTANCES): a later instance of an
    // instance loop. The register keeps its value (instance id Subsequent, the same topology)
    // and nothing waits in the back's sink, so all the write does there is count one more
    // instance (DrawManager::DrawBegin): the count is sent once, before the front's next entry
    // (GpuSplit::FoldInstance, held as the split's pending flush).
    if (method == MAXWELL3D_REG_INDEX(draw.begin) && fs.instance_loop &&
        ((value >> 26) & 3) == 1 && fs.skip_same && !fs.sink_pending && shadow_kept &&
        fs.regs[method] == value && FoldInstancesEnabled()) {
        fs.draw_pending = true;
        split.FoldInstance(&FlushFoldedInstances, this);
        return;
    }
    // What a constant buffer upload word or a report writes, for the front's macro memory reads.
    const auto [write_begin, write_end] = FrontWriteRange(fs, method, 1);
    using Effect = VideoCommon::GpuSplit::Effect;
    // What the entry may write on the back (FrontDmaRead): any method may issue a deferred
    // instanced draw first.
    Effect effect = fs.draw_pending ? Effect::Gpu : Effect::None;
    std::optional<GPUVAddr> cb_write;
    if (!execution_mask[method]) {
        // Nothing would change on the back: CallMethod and ConsumeSinkImpl skip same-value state
        // writes, and the shadow RAM (tracked by default) holds the value already.
        if (fs.regs[method] == value && shadow_kept && fs.skip_same) {
            if (Profile::Enabled()) [[unlikely]] {
                Profile::Add(Profile::Counter::SameValueWrites, 1);
            }
            if (fs.same_experiment != 0) [[unlikely]] {
                ++fs.exp_same;
                FrontReportSameWrites(fs);
            }
            return;
        }
        fs.regs[method] = value;
    } else {
        fs.regs[method] = value;
        FrontTrackIndexDraws(fs, method);
        effect = VideoCommon::SplitWriteTracker::Combine(effect, FrontMethodEffect(method));
        constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
        switch (method) {
        case MAXWELL3D_REG_INDEX(shadow_ram_control):
            // The back's mode is shadow_state.shadow_ram_control: the shadow copy of this very
            // register, which a later Replay write reads back.
            fs.control = static_cast<Regs::ShadowRamControl>(argument);
            fs.shadow[method] = argument;
            break;
        case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
            GetNativeMacroEngine().ClearCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)]);
            break;
        case MAXWELL3D_REG_INDEX(load_mme.instruction):
            GetNativeMacroEngine().AddCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)], value);
            break;
        case MAXWELL3D_REG_INDEX(load_mme.start_address): {
            const auto positions = GetNativeMacroPositions();
            u32& pointer = fs.regs[MAXWELL3D_REG_INDEX(load_mme.start_address_ptr)];
            if (pointer < positions.size()) {
                positions[pointer] = value;
            }
            ++pointer;
            break;
        }
        case MAXWELL3D_REG_INDEX(falcon[4]):
            fs.regs[MAXWELL3D_REG_INDEX(shadow_scratch)] = 1;
            break;
        default:
            if (method >= cb_data && method < cb_data + 16) {
                cb_write = ConstBufferWriteAddress(fs.regs);
                fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)] += sizeof(u32);
            } else if (!FrontHandledOutsideDrawManager(method)) {
                fs.draw_pending = true;
            }
            break;
        }
    }
    if (fs.same_experiment != 0) [[unlikely]] {
        ++fs.exp_sent;
        FrontReportSameWrites(fs);
    }
    // EXP22 bit 2: instance loop methods, which nothing after the send below concerns.
    if ((fs.exp22 & 2) != 0 && execution_mask[method] &&
        FrontBatchWrite(split, fs, method, argument, value, is_last_call, from_macro)) {
        fs.sink_pending = false; // Held executable writes: each consumes the sink when sent
        return;
    }
    // FRITCHII_CB_WORDS_BLOCK (live, default 0): a constant buffer data word goes into the open
    // CbWords block as the very Method word its own entry would carry (Direct and last from a
    // macro, as MacroWrite; the command list's last-call bit, as Method), committed and noted at
    // its own position (GpuSplit::CbWord); the notes below are made as for its entry. 2 also
    // sends the address noted for a block's first word, which the back compares.
    int cb_block = 0;
    if (cb_write) {
        cb_block = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_cb_words_block,
                                                    "FRITCHII_CB_WORDS_BLOCK");
    }
    if (cb_block != 0) {
        split.CbWord(this, method, argument, from_macro || is_last_call, from_macro, effect,
                     cb_block == 2 ? &*cb_write : nullptr);
        fs.sink_pending = false; // The back's block consumes the sink first, as each entry did
    } else if (from_macro && (execution_mask[method] || !VideoCommon::GpuSplit::MacroSink())) {
        if (fs.packet_on && IsDrawPacketMethod(method)) [[unlikely]] {
            // EDEN_NX2_DRAW_PACKET: the same Direct write, held in the draw packet (none of
            // these methods writes memory or is noted below).
            FrontCapturePacket(split, fs, method, argument, value, static_cast<u8>(effect));
        } else {
            split.MacroWrite(this, method, argument, effect);
        }
        fs.sink_pending = false; // Direct writes (and the packet) consume the sink first
    } else {
        // EDEN_NX2_MACRO_SINK: a macro's state write goes to the back's method sink as the
        // command list's own do; the back applies the sink in order (shadow RAM and same-value
        // skip as CallMethod) before the next executable or Direct write and in
        // FinishFrontMacro, so the order against executable writes and deferred draws is kept.
        // EDEN_NX2_RING_PAIRS: the same writes as pairs, applied the same way (ApplyPairs) when
        // the back reaches them, which is never later than the sink would.
        if (!execution_mask[method] && VideoCommon::GpuSplit::RingPairs()) {
            split.PairWrite(this, method, argument);
            fs.sink_pending = false; // The back's Pairs entry consumes the sink first
        } else {
            split.Method(this, method, argument, is_last_call, nullptr, effect);
            fs.sink_pending = !execution_mask[method];
        }
        if (from_macro) {
            split.CountBatched(VideoCommon::GpuSplit::Batched::MacroSunk);
        }
    }
    if (cb_write) {
        split.NoteWrite(memory_manager, *cb_write, sizeof(u32));
    } else if (method == MAXWELL3D_REG_INDEX(report_semaphore.query)) {
        split.NoteWrite(memory_manager, ReportAddress(fs.regs), ReportBytes);
    }
    if (write_end != 0) {
        constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
        const bool upload = method >= cb_data && method < cb_data + 16;
        split.RangeLastWrite(write_begin, write_end, upload ? &value : nullptr,
                             memory_manager.GetID() + 1, &memory_manager);
        if (method == MAXWELL3D_REG_INDEX(report_semaphore.query)) {
            FrontNoteRelease(split, fs, value);
        }
    }
    if (WritesGuestMemory(method)) {
        FrontNoteMemoryWrite(split, fs, method, &value, 1);
    }
}

void Maxwell3D::FrontNoteRelease(VideoCommon::GpuSplit& split, const FrontState& fs,
                                 u32 query) const {
    // A semaphore release of the payload register (no reduction) writes a value the front knows.
    Regs::ReportSemaphore report{};
    report.query.raw = query;
    if (report.query.operation != Regs::ReportSemaphore::Operation::Release ||
        report.query.report != Regs::ReportSemaphore::Report::Payload ||
        report.query.reduction_enable != 0) {
        return;
    }
    const GPUVAddr address =
        (GPUVAddr{fs.regs[MAXWELL3D_REG_INDEX(report_semaphore.address_high)]} << 32) |
        fs.regs[MAXWELL3D_REG_INDEX(report_semaphore.address_low)];
    if (const auto device = memory_manager.GpuToCpuAddress(address)) {
        const u32 payload = fs.regs[MAXWELL3D_REG_INDEX(report_semaphore.payload)];
        split.NoteRelease(*device, address, payload,
                          report.query.short_query != 0
                              ? VideoCommon::GpuSplit::ReleaseKind::Payload3DShort
                              : VideoCommon::GpuSplit::ReleaseKind::Payload3DLong);
        if (VideoCommon::QmdPrediction::MirrorMode()) {
            // EXP13=2: recorded so the reduction bound sees it queued or run but not landed.
            namespace Prediction = VideoCommon::QmdPrediction;
            Prediction::Note(*device,
                             {.writer = Prediction::Writer::Report,
                              .kind = Prediction::Kind::Plain,
                              .payload = payload,
                              .op = 0,
                              .is_signed = false},
                             memory_manager.Read<u32>(address), split.SentPosition(), 0);
        }
    }
}

std::pair<GPUVAddr, GPUVAddr> Maxwell3D::FrontWriteRange(const FrontState& fs, u32 method,
                                                         u32 amount) const {
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method >= cb_data && method < cb_data + 16) {
        // Uploaded words go to the buffer's address plus the running offset.
        const GPUVAddr address =
            (GPUVAddr{fs.regs[MAXWELL3D_REG_INDEX(const_buffer.address_high)]} << 32) |
            fs.regs[MAXWELL3D_REG_INDEX(const_buffer.address_low)];
        const GPUVAddr begin = address + fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)];
        return {begin, begin + GPUVAddr{amount} * sizeof(u32)};
    }
    if (method == MAXWELL3D_REG_INDEX(report_semaphore.query) && amount == 1) {
        // A report writes its payload or a 16-byte counter and timestamp at its address.
        const GPUVAddr begin =
            (GPUVAddr{fs.regs[MAXWELL3D_REG_INDEX(report_semaphore.address_high)]} << 32) |
            fs.regs[MAXWELL3D_REG_INDEX(report_semaphore.address_low)];
        return {begin, begin + 16};
    }
    return {0, 0};
}

void Maxwell3D::FrontCallMethod(Core::System& system, VideoCommon::GpuSplit& split, u32 method,
                                u32 argument, bool is_last_call) {
    auto& fs = Front(system, split);
    if (method >= MacroRegistersStart) {
        if (fs.executing_macro == 0) {
            fs.executing_macro = method;
        }
        fs.parameters.push_back(argument);
        if (is_last_call) {
            FrontRunMacro(system, split, fs);
        }
        return;
    }
    FrontWrite(split, fs, method, argument, is_last_call, false);
}

void Maxwell3D::FrontCallMultiMethod(Core::System& system, VideoCommon::GpuSplit& split,
                                     u32 method, const u32* base_start, u32 amount,
                                     u32 methods_pending) {
    auto& fs = Front(system, split);
    if (method >= MacroRegistersStart) {
        if (fs.executing_macro == 0) {
            fs.executing_macro = method;
        }
        fs.parameters.insert(fs.parameters.end(), base_start, base_start + amount);
        if (amount == methods_pending) {
            FrontRunMacro(system, split, fs);
        }
        return;
    }
    // List guard: each word as FrontWrite notes it (the PR's front notes
    // single writes only; a draw.end here would draw from the stale registers).
    for (u32 i = 0; i < amount && fs.list_draw_stale != 0; ++i) {
        FrontNoteListDrawWrite(split, fs, method);
    }
    // The instance loop state the back ends in (Replay: it acts on the shadow copy). The back
    // runs draw.begin once per word, so each word can switch the loop; other methods leave it
    // as their last word does.
    const bool replay = fs.control == Regs::ShadowRamControl::Replay;
    if (method == MAXWELL3D_REG_INDEX(draw.begin)) {
        for (u32 i = 0; i < amount; ++i) {
            FrontTrackInstanceLoop(fs, method, replay ? fs.shadow[method] : base_start[i]);
        }
    } else if (amount != 0) {
        FrontTrackInstanceLoop(fs, method, replay ? fs.shadow[method] : base_start[amount - 1]);
    }
    if (amount != 0) {
        FrontTrackIndexDraws(fs, method);
    }
    if (fs.same_experiment != 0) [[unlikely]] {
        fs.exp_sent += amount;
        FrontReportSameWrites(fs);
    }
    // What CallMultiMethod does to the registers on the back, then one entry for all the words.
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    const auto [write_begin, write_end] = FrontWriteRange(fs, method, amount);
    const auto tracked = [&fs] {
        return fs.control == Regs::ShadowRamControl::Track ||
               fs.control == Regs::ShadowRamControl::TrackWithFilter;
    };
    using Effect = VideoCommon::GpuSplit::Effect;
    Effect effect = fs.draw_pending ? Effect::Gpu : Effect::None;
    std::optional<GPUVAddr> cb_write;
    if (method >= cb_data && method < cb_data + 16) {
        cb_write = ConstBufferWriteAddress(fs.regs);
        fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)] += amount * sizeof(u32);
    } else if (method == MAXWELL3D_REG_INDEX(inline_data)) {
        // Upload data only.
        effect = Effect::Unknown;
    } else if ((method == MAXWELL3D_REG_INDEX(draw_inline_index) ||
                method == MAXWELL3D_REG_INDEX(inline_index_2x16.even) ||
                method == MAXWELL3D_REG_INDEX(inline_index_4x8.index0)) &&
               fs.control != Regs::ShadowRamControl::Replay) {
        if (amount != 0) {
            const u32 last = base_start[amount - 1];
            if (tracked()) {
                fs.shadow[method] = last;
            }
            fs.regs[method] = last;
        }
        fs.draw_pending = true;
        effect = VideoCommon::SplitWriteTracker::Combine(effect, Effect::Gpu);
    } else {
        // CallMethod for each word, macro uploads among them (load_mme.instruction). A word may
        // change the shadow RAM mode for the next (a run of SET_SHADOW_RAM_CONTROL writes).
        for (u32 i = 0; i < amount; ++i) {
            const u32 argument = base_start[i];
            u32 value = argument;
            if (method == MAXWELL3D_REG_INDEX(shadow_ram_control) && FrontVerifyEnabled()) [[unlikely]] {
                fs.control_writes[fs.control_count++ % fs.control_writes.size()] = {
                    argument, static_cast<u32>(fs.control), 3U};
            }
            if (tracked()) {
                fs.shadow[method] = argument;
            } else if (fs.control == Regs::ShadowRamControl::Replay) {
                value = fs.shadow[method];
            }
            fs.regs[method] = value;
            if (!execution_mask[method]) {
                continue;
            }
            effect = VideoCommon::SplitWriteTracker::Combine(effect, FrontMethodEffect(method));
            switch (method) {
            case MAXWELL3D_REG_INDEX(shadow_ram_control):
                fs.control = static_cast<Regs::ShadowRamControl>(argument);
                fs.shadow[method] = argument; // The same word on the back (see FrontWrite)
                break;
            case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
                GetNativeMacroEngine().ClearCode(value);
                break;
            case MAXWELL3D_REG_INDEX(load_mme.instruction):
                GetNativeMacroEngine().AddCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)], value);
                break;
            case MAXWELL3D_REG_INDEX(load_mme.start_address): {
                const auto positions = GetNativeMacroPositions();
                u32& pointer = fs.regs[MAXWELL3D_REG_INDEX(load_mme.start_address_ptr)];
                if (pointer < positions.size()) {
                    positions[pointer] = value;
                }
                ++pointer;
                break;
            }
            case MAXWELL3D_REG_INDEX(falcon[4]):
                fs.regs[MAXWELL3D_REG_INDEX(shadow_scratch)] = 1;
                break;
            default:
                if (!FrontHandledOutsideDrawManager(method)) {
                    fs.draw_pending = true;
                }
                break;
            }
        }
    }
    split.MultiMethod(this, method, base_start, amount, methods_pending, nullptr, effect);
    fs.sink_pending = false; // The back consumes the sink before a run
    if (write_end != 0) {
        // Uploaded words, unless the shadow RAM replays other values for them on the back.
        const bool upload = method >= cb_data && method < cb_data + 16 &&
                            fs.control != Regs::ShadowRamControl::Replay;
        split.RangeLastWrite(write_begin, write_end, upload ? base_start : nullptr,
                             memory_manager.GetID() + 1, &memory_manager);
        if (method == MAXWELL3D_REG_INDEX(report_semaphore.query) && amount == 1) {
            FrontNoteRelease(split, fs, base_start[0]);
        }
    }
    if (cb_write) {
        split.NoteWrite(memory_manager, *cb_write, u64{amount} * sizeof(u32));
    } else if (method == MAXWELL3D_REG_INDEX(report_semaphore.query) && amount != 0) {
        split.NoteWrite(memory_manager, ReportAddress(fs.regs), ReportBytes);
    }
    if (WritesGuestMemory(method) && amount != 0) {
        FrontNoteMemoryWrite(split, fs, method, base_start, amount);
    }
}

void Maxwell3D::FrontRunMacro(Core::System& system, VideoCommon::GpuSplit& split,
                              FrontState& fs) {
    // Fable's feature byte (fable_answer_regression_2 3.1): with every macro switch off, v5.7's
    // body and emit, without the per-call switch reads, slot lookup, predicate test, fold and
    // packet setup and report tick the switches added (FrontRunMacro itself 4 -> 17 ns a call,
    // the emit 1.4 -> 3.7 ns a write in the 5.7 vs HEAD sampler pair). The features body reads
    // the switches per call as before, so turning one off is exact at once; turning one on
    // bumps g_switch_generation, seen at the next call.
    if (fs.switches_seen !=
            VideoCore::Nx2Debug::g_switch_generation.load(std::memory_order_relaxed) ||
        (++fs.macro_calls & 1023) == 0) [[unlikely]] {
        FrontRefreshSwitches(fs);
    }
    // List guard (nx2_list_dispatch.h), ahead of both bodies so the plain
    // one stays v5.7's: a data RAM that holds a list dispatch's marker (the compute engine shares
    // it), and the two registers a list draw left with their old values.
    if (grid_marker.In(native_data_ram) || fs.list_draw_stale != 0) [[unlikely]] {
        if (FrontListGuard(split, fs)) {
            // The call may read a stale register: the features body checks each read.
            fs.guard_list_reads = true;
            FrontRunMacroFeatures(system, split, fs);
            fs.guard_list_reads = false;
            return;
        }
    }
    if (fs.macro_features == 0) [[likely]] {
        FrontRunMacroPlain(system, split, fs);
        return;
    }
    FrontRunMacroFeatures(system, split, fs);
}

void Maxwell3D::FrontRefreshSwitches(FrontState& fs) {
    namespace Debug = VideoCore::Nx2Debug;
    // The generation first: a switch stored after this load bumps it again, so the next call
    // refreshes once more (the SwitchValue first reads below do that once too).
    const u32 generation = Debug::g_switch_generation.load(std::memory_order_acquire);
    u8 features = 0;
    if (Debug::Experiment(53) != 0) {
        features |= 1;
    }
    if (Debug::SwitchValue(Debug::g_oot_macros, "EDEN_NX2_OOT_MACROS", 1) != 0) {
        features |= 2;
    }
    if (Debug::SwitchValue(Debug::g_fold_instance, "EDEN_NX2_FOLD_INSTANCE") != 0) {
        features |= 4;
    }
    if (Debug::SwitchValue(Debug::g_draw_packet, "EDEN_NX2_DRAW_PACKET") != 0) {
        features |= 8;
    }
    if (!fs.macro_features_logged || features != fs.macro_features) {
        fs.macro_features_logged = true;
        LOG_INFO(HW_GPU,
                 "Front macro switches: features={:#x} (EXP53, OOT_MACROS, FOLD_INSTANCE, "
                 "DRAW_PACKET bits), generation {}: {} body",
                 features, generation, features == 0 ? "plain (v5.7)" : "features");
    }
    fs.macro_features = features;
    fs.switches_seen = generation;
}

void Maxwell3D::FrontRunMacroPlain(Core::System& system, VideoCommon::GpuSplit& split,
                                   FrontState& fs) {
    // v5.7's (52317bf0e5) FrontRunMacro, verbatim but for the shared trace counter.
    const u32 method = fs.executing_macro;
    fs.executing_macro = 0;
    auto& engine = GetNativeMacroEngine();
    const auto positions = GetNativeMacroPositions();
    const u32 start = positions[((method - MacroRegistersStart) >> 1) % positions.size()];
    VideoCommon::Nx2Profile::CountMethod(method, false);
    // EDEN_NX2_MACRO_TRACE=<method>: the first calls of that macro, parameters and writes.
    const u32 traced = MacroTraced();
    const bool trace = traced != 0 && method == traced &&
                       std::ranges::any_of(fs.parameters, [](u32 p) { return p != 0; }) &&
                       g_macro_traced_calls++ < 8;
    std::string trace_text;
    if (trace) [[unlikely]] {
        trace_text = fmt::format("3D macro {:#x} start {:#x} params", traced, start);
        for (const u32 p : fs.parameters) {
            trace_text += fmt::format(" {:x}", p);
        }
        trace_text += " writes";
    }
    {
        VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Macro};
        engine.native_interpreter.active_dram = &native_data_ram;
        const auto parameter_count = fs.parameters.size();
        const auto read = [&fs](u32 address) {
            return address < Regs::NUM_REGS ? fs.regs[address] : 0U;
        };
        const auto emit = [&](u32 address, u32 value) {
            if (trace) [[unlikely]] {
                trace_text += fmt::format(" {:x}={:x}", address, value);
            }
            if (address >= MacroRegistersStart) {
                return false;
            }
            FrontWrite(split, fs, address, value, true, true);
            // MME_DMA_READ (into data RAM) and MME_DMA_READ_FIFOED (into the load FIFO).
            if (address == 0x157 || address == 0x158) {
                FrontDmaRead(split, fs, method, address, value);
            }
            return true;
        };
#ifdef ARCHITECTURE_x86_64
        if (!engine.native_jit) {
            engine.native_jit = std::make_shared<TuringMacroJit>();
        }
        engine.native_jit->Prepare(engine.native_code, engine.native_code_generation);
        const auto result = engine.native_jit->Execute(
            start, fs.parameters, std::span<u32>{native_data_ram}, read, emit);
#else
        const auto result =
            engine.native_interpreter.Execute(engine.native_code, start, fs.parameters, read, emit);
#endif
        if (!result.success) {
            engine.native_execution_failed = true;
            LOG_ERROR(HW_GPU, "Native MME (split front) failed: start={:#x}, consumed={}/{}, error={}",
                      start, result.consumed, parameter_count, result.error);
        }
    }
    if (trace) [[unlikely]] {
        LOG_INFO(HW_GPU, "{}", trace_text);
    }
    fs.parameters.clear();
    if (fs.draw_pending) {
        fs.draw_pending = false;
        split.Defer(StageCall::FinishMacro{this, &system}, VideoCommon::GpuSplit::Effect::Gpu,
                    VideoCommon::Nx2FastDrawCount::Neutral() | StageCall::FinishMacroFlags);
    }
    if (FrontVerifyEnabled() && (++fs.calls & 4095) == 0) [[unlikely]] {
        FrontVerify(split, fs);
    }
}

void Maxwell3D::FrontRunMacroFeatures(Core::System& system, VideoCommon::GpuSplit& split,
                                      FrontState& fs) {
    const u32 method = fs.executing_macro;
    fs.executing_macro = 0;
    auto& engine = GetNativeMacroEngine();
    const auto positions = GetNativeMacroPositions();
    const u32 start = positions[((method - MacroRegistersStart) >> 1) % positions.size()];
    VideoCommon::Nx2Profile::CountMethod(method, false);
    // EDEN_NX2_MACRO_TRACE=<method>: the first calls of that macro, parameters and writes.
    const u32 traced = MacroTraced();
    const bool trace = traced != 0 && method == traced &&
                       std::ranges::any_of(fs.parameters, [](u32 p) { return p != 0; }) &&
                       g_macro_traced_calls++ < 8;
    std::string trace_text;
    if (trace) [[unlikely]] {
        trace_text = fmt::format("3D macro {:#x} start {:#x} params", traced, start);
        for (const u32 p : fs.parameters) {
            trace_text += fmt::format(" {:x}", p);
        }
        trace_text += " writes";
    }
    // EDEN_NX2_EXP53 (live, 0 = off; ported): NVN's conditional
    // render macro reads the six Compare words at an address and sets render_enable.mode from
    // them, which drained the split on every call (our largest front wait, "macro dma read").
    // For such a macro the front skips the read, feeds six zero words and writes the address
    // with mode IfNotEqual: the back evaluates the same condition in order.
    const bool predicate = VideoCore::Nx2Debug::Experiment(53) != 0 &&
                           fs.parameters.size() == 3 && fs.parameters[0] == 0 &&
                           IsPredicateMacro(fs, method);
    // The slot's OoT replacement (macro_hle_oot.h, ported):
    // the hash of the code at the slot's start, found again when the code or the
    // start changed, as the original FrontRunMacro does. Each slot is logged once ("OoT macro
    // slot") so the hashes of the running game's macros can be checked against the table. Only
    // in this body: with no macro switch on, nothing needs the slots (FrontRunMacroPlain).
    OotMacros::Id oot = OotMacros::Id::None;
    if (const size_t slot = ((method - MacroRegistersStart) >> 1) % positions.size();
        slot < FrontState::MacroSlots) [[likely]] {
        if (!fs.oot_known[slot] || fs.oot_generation[slot] != engine.native_code_generation ||
            fs.oot_start[slot] != start) [[unlikely]] {
            fs.oot_known[slot] = true;
            fs.oot_generation[slot] = engine.native_code_generation;
            fs.oot_start[slot] = start;
            const std::span<const u32> code{engine.native_code};
            const u64 hash = OotMacros::HashMacro(code, start);
            const OotMacros::Macro* const found = OotMacros::Find(hash);
            fs.oot_id[slot] = found != nullptr ? found->id : OotMacros::Id::None;
            LogOotSlot(method, start, hash, code, found);
        }
        oot = fs.oot_id[slot];
    }
    // EDEN_NX2_OOT_MACROS (live, 0 = off; 1, the default since 2026-11-12): 1 runs the
    // replacement instead of the JIT, 2 and 3 verify it against the JIT (see nx2_debug_control.h
    // g_oot_macros).
    const int oot_mode = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_oot_macros,
                                                          "EDEN_NX2_OOT_MACROS", 1);
    const size_t oot_index = static_cast<size_t>(oot);
    bool oot_verify = false;
    if (oot != OotMacros::Id::None && oot_mode >= 2) [[unlikely]] {
        const u64 n = fs.oot_calls[oot_index]++;
        oot_verify = oot_mode >= 3 || n < 4096 || (n & 63) == 0;
    }
    // EDEN_NX2_FOLD_INSTANCE (live, 0 = off, the default; ported from its
    // macro_hle_oot.h DrawLoop and maxwell_3d.cpp OotContext::FoldFirstInstance):
    // for a call of an instanced draw macro (0xe52 or 0xea0, identified as above, the handlers'
    // own lookup) with 2 or more instances, the back's draw manager is told before the loop's
    // first draw.end that this instance is the first of a loop, so it is not drawn alone: the
    // loop's deferred draw (instance_count + 1 from the base instance) covers it. Without it
    // instance 0 is drawn twice. The same with the handlers on (they call the emit below).
    const bool fold_on = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_fold_instance,
                                                          "EDEN_NX2_FOLD_INSTANCE") != 0;
    bool fold_pending = fold_on &&
                        (oot == OotMacros::Id::E52 || oot == OotMacros::Id::EA0) &&
                        fs.parameters.size() >= 6 && fs.parameters[2] > 1 &&
                        fs.parameters[2] <= OotMacros::MaxInstances;
    // EDEN_NX2_DRAW_PACKET (live, 0 = off, the default): a draw macro's Direct draw and state
    // writes go as one ring entry (FrontCapturePacket). Not with the paths that keep their own
    // open blocks or held writes (EXP22, EXP2, EDEN_NX2_MACRO_SINK).
    fs.packet_mode = IsDrawMacro(oot) ? VideoCore::Nx2Debug::SwitchValue(
                                            VideoCore::Nx2Debug::g_draw_packet,
                                            "EDEN_NX2_DRAW_PACKET")
                                      : 0;
    fs.packet_on = fs.packet_mode != 0 && fs.exp22 == 0 && fs.same_experiment == 0 &&
                   !VideoCommon::GpuSplit::MacroSink();
    fs.e2c_call = oot == OotMacros::Id::E2C;
    if (fs.e2c_call) {
        // FRITCHII_DELAY_E2C_US (live, 0 = off): the delay test "W", every call (hits too),
        // before its read, whether the handler or the JIT runs it. Off: one load and a branch.
        VideoCommon::GpuSplit::DelayWorkIf(VideoCore::Nx2Debug::g_delay_e2c_us,
                                           VideoCommon::GpuSplit::DelayTest::E2c);
    }
    {
        VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Macro};
        engine.native_interpreter.active_dram = &native_data_ram;
        const auto parameter_count = fs.parameters.size();
        const auto read = [&](u32 address) {
            if (address >= Regs::NUM_REGS) {
                return 0U;
            }
            if (fs.guard_list_reads && fs.list_draw_stale != 0 &&
                (address == Nx2ListDraw::VertexFirst || address == Nx2ListDraw::VertexCount))
                [[unlikely]] {
                // List guard: a register the last list draw left
                // unwritten (FrontRunMacro set guard_list_reads only with the guard on).
                Nx2ListDispatch::Count(Nx2ListDispatch::g_guard.stale_reads);
                if (Nx2ListDispatch::GuardMode() == 2) {
                    FrontResolveListDraw(split, fs);
                }
            }
            return fs.regs[address];
        };
        const auto emit = [&](u32 address, u32 value) {
            if (oot_verify) [[unlikely]] {
                fs.oot_jit_writes.push_back(address);
                fs.oot_jit_writes.push_back(value);
            }
            if (trace) [[unlikely]] {
                trace_text += fmt::format(" {:x}={:x}", address, value);
            }
            if (address >= MacroRegistersStart) {
                return false;
            }
            if (predicate && address == MAXWELL3D_REG_INDEX(render_enable.mode)) {
                FrontWrite(split, fs, MAXWELL3D_REG_INDEX(render_enable.address_high),
                           fs.regs[0x154] & 0x1ffffff, true, true);
                FrontWrite(split, fs, MAXWELL3D_REG_INDEX(render_enable.address_low),
                           fs.regs[0x155], true, true);
                FrontWrite(split, fs, address,
                           static_cast<u32>(Regs::RenderEnable::Mode::IfNotEqual), true, true);
                return true;
            }
            if (fold_pending && address == MAXWELL3D_REG_INDEX(draw.end)) [[unlikely]] {
                // The loop's first draw.end, the macro's first. Only when the back's draw.begin
                // (the front's copy, after the shadow RAM) holds instance id First: the draw
                // manager then has no loop open and draws this one alone (DrawMode::General).
                fold_pending = false;
                if (((fs.regs[MAXWELL3D_REG_INDEX(draw.begin)] >> 26) & 3) == 0) {
                    split.Defer(StageCall::FoldFirstInstance{this},
                                VideoCommon::GpuSplit::Effect::None,
                                VideoCommon::Nx2FastDrawCount::Neutral() |
                                    StageCall::FoldFirstInstanceFlags);
                    ++fs.oot_stats.folds;
                    static u32 fold_logged{};
                    if (fold_logged++ < 4) {
                        LOG_INFO(HW_GPU, "Native macro {:#x} (start {:#x}, {}): instanced draw, "
                                 "first instance folded into the loop's draw "
                                 "(EDEN_NX2_FOLD_INSTANCE)",
                                 method, start, OotMacros::Name(oot));
                    }
                }
            }
            FrontWrite(split, fs, address, value, true, true);
            // MME_DMA_READ (into data RAM) and MME_DMA_READ_FIFOED (into the load FIFO).
            if (address == 0x157 || address == 0x158) {
                if (predicate && address == 0x158) {
                    // The back reads the words when it evaluates the condition: the macro only
                    // has to consume six.
                    fs.parameters.insert(fs.parameters.end(), value, 0U);
                    static std::atomic<u64> skipped{};
                    if (const u64 n = skipped.fetch_add(1, std::memory_order_relaxed);
                        n == 0 || (n + 1) % 100000 == 0) {
                        LOG_INFO(HW_GPU, "Conditional render macro (EXP53): {} reads left to the back",
                                 n + 1);
                    }
                } else {
                    FrontDmaRead(split, fs, method, address, value);
                }
            }
            return true;
        };
        bool handled = false;
        // Not while the list guard checks the call's register reads (the handlers read the
        // registers directly): the JIT runs it.
        if (oot_mode == 1 && oot != OotMacros::Id::None && !fs.guard_list_reads) {
            // The replacement instead of the JIT, through the same emit. It returns false
            // before any effect for a call it can't reproduce: that call runs through the JIT.
            static_assert(sizeof(native_data_ram) == OotMacros::DataRamWords * sizeof(u32));
            OotFrontContext<decltype(emit)> context{fs.parameters, native_data_ram.data(),
                                                    std::span<const u32>{fs.regs}, emit};
            handled = OotMacros::Run(oot, context);
            if (handled) {
                ++fs.oot_stats.native;
                if (context.error != nullptr) [[unlikely]] {
                    engine.native_execution_failed = true;
                    LOG_ERROR(HW_GPU, "Native MME (split front, OoT macro {:#x}) failed: {}",
                              method, context.error);
                }
            } else {
                ++fs.oot_stats.declined;
            }
        } else if (oot_mode != 0 && oot == OotMacros::Id::None) {
            ++fs.oot_stats.unmatched;
        }
        if (!handled) {
            if (oot_verify) [[unlikely]] {
                fs.oot_jit_writes.clear();
                fs.oot_dram = native_data_ram;
            }
#ifdef ARCHITECTURE_x86_64
            if (!engine.native_jit) {
                engine.native_jit = std::make_shared<TuringMacroJit>();
            }
            engine.native_jit->Prepare(engine.native_code, engine.native_code_generation);
            const auto result = engine.native_jit->Execute(
                start, fs.parameters, std::span<u32>{native_data_ram}, read, emit);
#else
            const auto result = engine.native_interpreter.Execute(engine.native_code, start,
                                                                  fs.parameters, read, emit);
#endif
            if (!result.success) {
                engine.native_execution_failed = true;
                LOG_ERROR(HW_GPU, "Native MME (split front) failed: start={:#x}, consumed={}/{}, error={}",
                          start, result.consumed, parameter_count, result.error);
            }
            if (oot_verify) [[unlikely]] {
                // The JIT's writes counted; now the replacement on copies, compared.
                switch (VerifyOotCall(oot, method, start, fs.parameters, parameter_count,
                                      fs.regs, fs.oot_dram, native_data_ram, fs.oot_jit_writes,
                                      result.success, fs.oot_handler_writes,
                                      fs.oot_mismatch_logs[oot_index])) {
                case OotVerdict::Same:
                    ++fs.oot_stats.verified;
                    break;
                case OotVerdict::Different:
                    ++fs.oot_stats.verified;
                    ++fs.oot_stats.differences;
                    break;
                case OotVerdict::Declined:
                    ++fs.oot_stats.declined;
                    break;
                case OotVerdict::Unverifiable:
                    ++fs.oot_stats.unverifiable;
                    break;
                }
            }
        }
    }
    fs.e2c_call = false;
    if (fs.packet_on) [[unlikely]] {
        // The call's last packet goes now (it is the pending flush when it holds writes).
        fs.packet_on = false;
        if (!fs.packet.empty()) {
            split.RunPendingFlush();
        }
    }
    if ((oot_mode != 0 || fold_on || fs.packet_mode != 0) && (++fs.oot_ticks & 0xffff) == 0)
        [[unlikely]] {
        const auto now = std::chrono::steady_clock::now();
        if (fs.oot_report == std::chrono::steady_clock::time_point{}) {
            fs.oot_report = now;
        } else if (const double seconds = std::chrono::duration<double>(now - fs.oot_report).count();
                   seconds >= 10.0) {
            const auto& s = fs.oot_stats;
            LOG_INFO(HW_GPU,
                     "OoT macros (EDEN_NX2_OOT_MACROS={}, FOLD_INSTANCE={}), last {:.0f} s: {} "
                     "native calls, {} declined to the JIT, {} calls of macros without a handler; "
                     "verify: {} compared, {} differences, {} not verifiable; {} first instances "
                     "folded; draw packets (EDEN_NX2_DRAW_PACKET) {} with {} writes",
                     oot_mode, fold_on ? 1 : 0, seconds, s.native, s.declined, s.unmatched,
                     s.verified, s.differences, s.unverifiable, s.folds, fs.packets,
                     fs.packet_writes);
            fs.oot_stats = {};
            fs.packets = 0;
            fs.packet_writes = 0;
            fs.oot_report = now;
        }
    }
    if (trace) [[unlikely]] {
        LOG_INFO(HW_GPU, "{}", trace_text);
    }
    fs.parameters.clear();
    if (fs.draw_pending) {
        fs.draw_pending = false;
        split.Defer(StageCall::FinishMacro{this, &system}, VideoCommon::GpuSplit::Effect::Gpu,
                    VideoCommon::Nx2FastDrawCount::Neutral() | StageCall::FinishMacroFlags);
    }
    if (FrontVerifyEnabled() && (++fs.calls & 4095) == 0) [[unlikely]] {
        FrontVerify(split, fs);
    }
}

// List draws (nx2_list_draw.h): whether the macro at `start` does nothing
// but begin a draw, write its first parameter to vertex_buffer.count and its second to
// vertex_buffer.first, and end the draw.
bool Maxwell3D::FrontIsListDrawMacro(FrontState& fs, const MacroEngine& engine, u32 start) {
    if (fs.list_draw_generation != engine.native_code_generation) {
        fs.list_draw_macros.clear();
        fs.list_draw_generation = engine.native_code_generation;
    }
    const auto [it, inserted] = fs.list_draw_macros.try_emplace(start, false);
    if (!inserted) {
        return it->second;
    }
    static_assert(Nx2ListDraw::DrawBegin == MAXWELL3D_REG_INDEX(draw.begin) &&
                  Nx2ListDraw::DrawEnd == MAXWELL3D_REG_INDEX(draw.end) &&
                  Nx2ListDraw::VertexFirst == MAXWELL3D_REG_INDEX(vertex_buffer.first) &&
                  Nx2ListDraw::VertexCount == MAXWELL3D_REG_INDEX(vertex_buffer.count));
    if (!Nx2ListDraw::IsArrayDrawMacro(std::span<const u32>{engine.native_code}, start)) {
        return false;
    }
    LOG_INFO(HW_GPU, "List draw: the macro at {:#x} is an array draw of (count, first)", start);
    it->second = true;
    return true;
}

u8 Maxwell3D::FrontListGuardEffects(FrontState& fs, const MacroEngine& engine, u32 start) {
    if (fs.data_ram_generation != engine.native_code_generation) {
        fs.data_ram_macros.assign(engine.native_code.size() / 3, u8{0xff});
        fs.data_ram_generation = engine.native_code_generation;
    }
    if (start >= fs.data_ram_macros.size()) {
        return u8{3}; // Nothing known: the guard assumes the worst
    }
    u8& known = fs.data_ram_macros[start];
    if (known == 0xff) {
        const auto effects = TuringMacroInterpreter::AnalyzeEffects(
            std::span<const u32>{engine.native_code}, start);
        const bool reads_list_draw =
            !effects.registers_read_known ||
            std::ranges::binary_search(effects.registers_read, Nx2ListDraw::VertexFirst) ||
            std::ranges::binary_search(effects.registers_read, Nx2ListDraw::VertexCount);
        known = static_cast<u8>((effects.data_ram ? 1 : 0) | (reads_list_draw ? 2 : 0));
    }
    return known;
}

bool Maxwell3D::FrontMacroUsesDataRam(FrontState& fs, const MacroEngine& engine, u32 start) {
    return (FrontListGuardEffects(fs, engine, start) & 1) != 0;
}

bool Maxwell3D::FrontListGuard(VideoCommon::GpuSplit& split, FrontState& fs) {
    const int guard = Nx2ListDispatch::GuardMode();
    auto& engine = GetNativeMacroEngine();
    const auto positions = GetNativeMacroPositions();
    if (guard == 0 || positions.empty()) {
        return false;
    }
    const u32 start =
        positions[((fs.executing_macro - MacroRegistersStart) >> 1) % positions.size()];
    // The data RAM holds the marker of a grid the front did not read (the compute engine shares
    // it). A 3D macro that uses the data RAM gets the real words.
    if (grid_marker.In(native_data_ram) && FrontMacroUsesDataRam(fs, engine, start)) {
        Nx2ListDispatch::Count(Nx2ListDispatch::g_guard.dram_macros);
        if (guard == 2) {
            Nx2ListDispatch::ResolveGrid(grid_marker, native_data_ram);
        }
    }
    // A macro whose code can read vertex_buffer.first or count while a list draw left them with
    // their old values: its reads are checked one by one (FrontRunMacroFeatures).
    const bool checked =
        fs.list_draw_stale != 0 && (FrontListGuardEffects(fs, engine, start) & 2) != 0;
    if (fs.list_draw_stale != 0 && Nx2ListDraw::Diagnostics()) [[unlikely]] {
        // EDEN_NX2_LIST_DIAG: how much of the macro traffic runs while the registers are stale,
        // and how much of it goes through the read-checking body.
        Nx2ListDispatch::Count(Nx2ListDispatch::g_guard.stale_calls);
        if (checked) {
            Nx2ListDispatch::Count(Nx2ListDispatch::g_guard.checked_calls);
        }
    }
    return checked;
}

void Maxwell3D::FrontNoteListDrawWrite(VideoCommon::GpuSplit& split, FrontState& fs, u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
        fs.list_draw_stale &= ~1U;
        break;
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
        fs.list_draw_stale &= ~2U;
        break;
    // The small index draws (index_buffer32_first and the like) draw at once, from their own
    // words: the draw.end after them is still an array draw.
    case MAXWELL3D_REG_INDEX(index_buffer.count):
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        fs.list_draw_indexed = true;
        break;
    case MAXWELL3D_REG_INDEX(draw.end):
        if (fs.in_list_draw) {
            break;
        }
        // An array draw takes its first vertex and its count from the two registers.
        if (!fs.list_draw_indexed) {
            if (const int guard = Nx2ListDispatch::GuardMode(); guard != 0) {
                Nx2ListDispatch::Count(Nx2ListDispatch::g_guard.stale_draws);
                if (guard == 2) {
                    FrontResolveListDraw(split, fs);
                }
            }
        }
        fs.list_draw_indexed = false;
        break;
    default:
        break;
    }
}

void Maxwell3D::FrontResolveListDraw(VideoCommon::GpuSplit& split, FrontState& fs) {
    const u32 stale = std::exchange(fs.list_draw_stale, 0);
    std::array<u32, 2> words{};
    if (!fs.list_draw_list.Read(words)) {
        Nx2ListDispatch::Count(Nx2ListDispatch::g_guard.failed);
        return;
    }
    // As the macro would have written them: the first vertex, then the count.
    if ((stale & 2) != 0) {
        FrontWrite(split, fs, MAXWELL3D_REG_INDEX(vertex_buffer.first), words[1], true, true);
    }
    if ((stale & 1) != 0) {
        FrontWrite(split, fs, MAXWELL3D_REG_INDEX(vertex_buffer.count), words[0], true, true);
    }
    Nx2ListDispatch::Count(Nx2ListDispatch::g_guard.draw_resolves);
}

bool Maxwell3D::FrontTakeListDraw(Core::System& system, VideoCommon::GpuSplit& split, u32 method,
                                  GPUVAddr address, const Nx2ListDispatch::ListReader& reader) {
    auto& fs = Front(system, split);
    // The registers must take the words themselves (no shadow RAM replay), and the two words
    // must be the whole call.
    if (fs.executing_macro != 0 || !fs.parameters.empty() ||
        fs.control == Regs::ShadowRamControl::Replay || rasterizer == nullptr ||
        !rasterizer->Nx2ListDrawSupported()) {
        return false;
    }
    auto& engine = GetNativeMacroEngine();
    const auto positions = GetNativeMacroPositions();
    if (!engine.native_turing || positions.empty()) {
        return false;
    }
    const u32 start = positions[((method - MacroRegistersStart) >> 1) % positions.size()];
    if (!FrontIsListDrawMacro(fs, engine, start)) {
        return false;
    }
    VideoCommon::Nx2Profile::CountMethod(method, false);
    // As the writes of the macro, in its order. vertex_buffer.first and count keep the values of
    // the last draw that was not one of these, here and on the back: only this draw reads the
    // words.
    fs.in_list_draw = true;
    FrontWrite(split, fs, MAXWELL3D_REG_INDEX(draw.begin), 0, true, true);
    // Writes no guest memory itself (the draw.end after it notes the draw's writes). The memory
    // stage stores the two fields in its copy of the draw state (StageCall::ListDraw).
    split.Defer(StageCall::ListDraw{this, address, reader.keep_id},
                VideoCommon::GpuSplit::Effect::None,
                VideoCommon::Nx2FastDrawCount::Neutral() | StageCall::ListDrawFlags);
    FrontWrite(split, fs, MAXWELL3D_REG_INDEX(draw.end), 0, true, true);
    fs.in_list_draw = false;
    // List guard (nx2_list_dispatch.h): the two registers keep their old values from here on.
    fs.list_draw_stale = 3;
    fs.list_draw_list = reader;
    fs.list_draw_indexed = false;
    if (fs.draw_pending) {
        fs.draw_pending = false;
        split.Defer(StageCall::FinishMacro{this, &system}, VideoCommon::GpuSplit::Effect::Gpu,
                    VideoCommon::Nx2FastDrawCount::Neutral() | StageCall::FinishMacroFlags);
    }
    return true;
}

bool Maxwell3D::FoldInstancesEnabled() {
    // EDEN_NX2_FOLD_INSTANCES=0 (the ported switch; not our EDEN_NX2_FOLD_INSTANCE, which folds
    // a loop's first instance into its deferred draw): every later instance is sent as its own
    // draw.begin write again.
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_FOLD_INSTANCES");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

void Maxwell3D::FlushFoldedInstances(void* engine, u32 count) {
    auto* const self = static_cast<Maxwell3D*>(engine);
    if (Nx2ListDraw::Diagnostics()) {
        // EDEN_NX2_LIST_DIAG: how many instances reach the back in how many calls.
        static u64 instances{};
        static u64 calls{};
        instances += count;
        ++calls;
        static auto window = std::chrono::steady_clock::now();
        if (const auto now = std::chrono::steady_clock::now();
            now - window >= std::chrono::seconds{2}) {
            window = now;
            LOG_INFO(HW_GPU, "Instance fold per 2 s: {} instances sent in {} calls", instances,
                     calls);
            instances = calls = 0;
        }
    }
    // As the draw.begin writes: the draw they defer may write anywhere a draw does.
    VideoCommon::GpuSplit::Get().Defer(
        StageCall::AddLoopInstances{self, count}, VideoCommon::GpuSplit::Effect::Gpu,
        VideoCommon::Nx2FastDrawCount::Neutral() | StageCall::AddLoopInstancesFlags);
}

bool Maxwell3D::IsPredicateMacro(FrontState& fs, u32 method) {
#ifdef ARCHITECTURE_x86_64
    auto& engine = GetNativeMacroEngine();
    const auto positions = GetNativeMacroPositions();
    if (positions.empty() || engine.native_execution_failed) {
        return false;
    }
    const size_t index = ((method - MacroRegistersStart) >> 1) % positions.size();
    if (fs.predicate_generation != engine.native_code_generation ||
        fs.predicate_macros.size() != positions.size()) {
        fs.predicate_generation = engine.native_code_generation;
        fs.predicate_macros.assign(positions.size(), 0);
    }
    if (fs.predicate_macros[index] != 0) {
        return fs.predicate_macros[index] == 1;
    }
    // Dry run against a copy of the data RAM, feeding test words when the macro asks for six. It
    // must write exactly the DMA address, the read of six words and render_enable.mode, and the
    // mode must be what IfNotEqual makes of the Compare words for every test vector.
    constexpr u32 address_high = 0x21, address_low = 0x13d60da0;
    constexpr std::array<std::array<u32, 6>, 8> vectors{{
        {0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 0}, {1, 0, 0, 0, 1, 0},
        {0, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1}, {5, 0, 0, 0, 0, 0}, {2, 3, 7, 9, 2, 3},
    }};
    if (!engine.native_jit) {
        engine.native_jit = std::make_shared<TuringMacroJit>();
    }
    engine.native_jit->Prepare(engine.native_code, engine.native_code_generation);
    bool matches = true;
    std::array<u32, vectors.size()> modes{};
    for (size_t i = 0; i < vectors.size() && matches; ++i) {
        std::vector<u32> parameters{0, address_high, address_low};
        std::vector<u32> scratch_ram(native_data_ram.begin(), native_data_ram.end());
        std::vector<std::pair<u32, u32>> writes;
        const auto result = engine.native_jit->Execute(
            positions[index], parameters, std::span<u32>{scratch_ram},
            [&fs](u32 address) { return address < Regs::NUM_REGS ? fs.regs[address] : 0U; },
            [&](u32 address, u32 value) {
                if (address >= MacroRegistersStart || writes.size() >= 16) {
                    return false;
                }
                writes.emplace_back(address, value);
                if (address == 0x158 && value == 6) {
                    parameters.insert(parameters.end(), vectors[i].begin(), vectors[i].end());
                }
                return true;
            });
        matches = result.success && writes.size() == 4 && writes[0].first == 0x154 &&
                  writes[0].second == address_high && writes[1].first == 0x155 &&
                  writes[1].second == address_low && writes[2] == std::pair<u32, u32>{0x158, 6} &&
                  writes[3].first == MAXWELL3D_REG_INDEX(render_enable.mode);
        if (matches) {
            const auto& w = vectors[i];
            modes[i] = writes[3].second;
            matches = modes[i] == ((w[0] != w[4] || w[1] != w[5]) ? 1U : 0U);
        }
    }
    fs.predicate_macros[index] = matches ? 1 : 2;
    LOG_INFO(HW_GPU, "Native macro {:#x}: {} (EXP53; modes for the test vectors: {})", method,
             matches ? "render-enable predicate, the back evaluates IfNotEqual"
                     : "not a render-enable predicate (or not IfNotEqual), runs as before",
             fmt::join(modes, " "));
    return matches;
#else
    static_cast<void>(fs);
    static_cast<void>(method);
    return false;
#endif
}

bool Maxwell3D::FrontMacroCountsDraw(Core::System& system, VideoCommon::GpuSplit& split,
                                     u32 method, u32 count) {
#ifdef ARCHITECTURE_x86_64
    FrontState& fs = Front(system, split);
    if (count != 2 || fs.executing_macro != 0 || !fs.parameters.empty()) {
        return false;
    }
    auto& engine = GetNativeMacroEngine();
    if (engine.native_execution_failed) {
        return false;
    }
    const auto positions = GetNativeMacroPositions();
    const u32 start = positions[((method - MacroRegistersStart) >> 1) % positions.size()];
    // Cached per macro start and code generation: the dry runs only repeat when the code changes.
    struct Verdict {
        u32 start;
        u64 generation;
        bool counts;
    };
    static std::array<Verdict, 8> verdicts{};
    for (const Verdict& verdict : verdicts) {
        if (verdict.start == start && verdict.generation == engine.native_code_generation + 1) {
            return verdict.counts;
        }
    }
    if (!engine.native_jit) {
        engine.native_jit = std::make_shared<TuringMacroJit>();
    }
    engine.native_jit->Prepare(engine.native_code, engine.native_code_generation);
    const auto run = [&](u32 first, u32 second, std::vector<std::pair<u32, u32>>& writes,
                         std::array<u32, 0xc00>& dram) {
        dram = native_data_ram;
        std::vector<u32> parameters{first, second};
        bool reads_memory = false;
        const auto read = [&fs](u32 address) {
            return address < Regs::NUM_REGS ? fs.regs[address] : 0U;
        };
        const auto emit = [&](u32 address, u32 value) {
            if (address >= MacroRegistersStart) {
                return false;
            }
            reads_memory |= address == 0x157 || address == 0x158;
            writes.emplace_back(address, value);
            return true;
        };
        const auto result =
            engine.native_jit->Execute(start, parameters, std::span<u32>{dram}, read, emit);
        return result.success && !reads_memory && parameters.size() == 2;
    };
    constexpr u32 vertex_count = MAXWELL3D_REG_INDEX(vertex_buffer.count);
    constexpr u32 first_vertex = MAXWELL3D_REG_INDEX(vertex_buffer.first);
    std::vector<std::pair<u32, u32>> a, b;
    std::array<u32, 0xc00> dram_a{}, dram_b{};
    bool counts = run(0x13572468u, 0x0BADF00Du, a, dram_a) && run(0x2468ACE0u, 0x1337C0DEu, b, dram_b) &&
                  a.size() == b.size() && dram_a == dram_b;
    bool copied = false;
    for (size_t i = 0; counts && i < a.size(); ++i) {
        if (a[i].first != b[i].first) {
            counts = false;
        } else if (a[i].second != b[i].second) {
            // The first parameter, unchanged, into the vertex count; the second into the first
            // vertex (0 in the GPU's words: the back draws from vertex 0 and checks in debug).
            const bool count_write = a[i].first == vertex_count && a[i].second == 0x13572468u &&
                                     b[i].second == 0x2468ACE0u;
            const bool first_write = a[i].first == first_vertex && a[i].second == 0x0BADF00Du &&
                                     b[i].second == 0x1337C0DEu;
            counts = count_write || first_write;
            copied |= count_write;
        }
    }
    counts = counts && copied;
    static u32 next_verdict{};
    verdicts[next_verdict++ % verdicts.size()] = {start, engine.native_code_generation + 1, counts};
    LOG_INFO(HW_GPU, "GPU-counted draws: 3D macro {:#x} (start {:#x}) {}", method, start,
             counts ? "copies its count into the draw" : "uses its parameters otherwise: read as before");
    return counts;
#else
    return false;
#endif
}

// Not inlined on GCC: GCC 13 -O3 unswitched a per-word loop on this
// switch there and dropped the macro upload cases from it (an empty macro RAM on the front).
#if defined(__GNUC__) && !defined(__clang__)
[[gnu::noinline]]
#endif
void Maxwell3D::FrontTrackInstanceLoop(FrontState& fs, u32 method, u32 value) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.begin):
        // Instance id, bits 27:26: First leaves an instance loop, Subsequent enters one.
        if (((value >> 26) & 3) == 0) {
            fs.instance_loop = false;
        } else if (((value >> 26) & 3) == 1) {
            fs.instance_loop = true;
        }
        break;
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        fs.instance_loop = false; // DrawMode::InlineIndex: draw.end draws
        break;
    default:
        break;
    }
}

bool Maxwell3D::FrontSkipsLoopWrite(FrontState& fs, u32 method, u32 value, bool shadow_kept) {
    FrontTrackInstanceLoop(fs, method, value);
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
        break;
    default:
        return false;
    }
    if (!fs.skip_same || !shadow_kept || fs.regs[method] != value) {
        return false;
    }
    // Instance loop methods flush no deferred draw, and the draw manager ignores these: DrawEnd
    // draws nothing in an instance loop, and it only reads the vertex buffer first and count
    // (index_buffer writes also mark the index buffer dirty, so they are always sent).
    return method != MAXWELL3D_REG_INDEX(draw.end) || fs.instance_loop;
}

void Maxwell3D::FrontNoteMemoryWrite(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                                     const u32* words, u32 count) {
    split.NoteMemoryWriter();
    const auto unknown = [&split](const char* what) { split.NoteMemoryWrite(0, ~u64{0}, what); };
    if (method == MAXWELL3D_REG_INDEX(report_semaphore.query)) {
        constexpr u32 base = MAXWELL3D_REG_INDEX(report_semaphore);
        const GPUVAddr address = (GPUVAddr{fs.regs[base]} << 32) | fs.regs[base + 1];
        Regs::ReportSemaphore query_regs{};
        query_regs.query.raw = words[count - 1];
        const auto device = memory_manager.GpuToCpuAddress(address);
        if (!device) {
            return; // CounterReport writes nothing at an unmapped address.
        }
        // A short payload release is a fence: written when the back runs it (fast fences).
        if (query_regs.query.operation == Regs::ReportSemaphore::Operation::Release &&
            query_regs.query.short_query != 0 &&
            query_regs.query.report == Regs::ReportSemaphore::Report::Payload) {
            const u32 payload = fs.regs[base + 2];
            split.NoteMemoryWriteData(*device, &payload, 1, VideoCommon::NotedPayloadRelease);
        } else {
            // Counters and timed releases are written when the host query completes, not when
            // the back runs them: reads of these bytes wait until the back ran the report and
            // the host query completed. (Which of the two, for the "Front 0xe2c misses" line.)
            split.NoteAsyncWrite(*device, 16);
            split.NoteMemoryWrite(*device, 16,
                                  query_regs.query.report ==
                                          Regs::ReportSemaphore::Report::Payload
                                      ? VideoCommon::NotedPayloadReport
                                      : VideoCommon::NotedCounterReport);
        }
        return;
    }
    constexpr u32 base = MAXWELL3D_REG_INDEX(upload);
    if (method == MAXWELL3D_REG_INDEX(launch_dma)) {
        const u64 line_length = fs.regs[base];
        const u64 lines = (std::max)(fs.regs[base + 1], 1U);
        const u64 pitch = (std::max<u64>)(fs.regs[base + 4], line_length);
        const GPUVAddr address = (GPUVAddr{fs.regs[base + 2]} << 32) | fs.regs[base + 3];
        const u64 size = pitch * (lines - 1) + line_length;
        const auto device = memory_manager.GpuToCpuAddress(address);
        // LAUNCH_DMA: bit 0 pitch (linear) layout, bits 5:4 completion (2 releases a semaphore).
        const u32 launch = words[count - 1];
        const bool linear = (launch & 1) != 0;
        if (((launch >> 4) & 3) == 2) {
            unknown("upload release");
        }
        fs.upload_known = device && linear && lines == 1 && (*device & 3) == 0 &&
                          memory_manager.GpuToCpuAddress(address + size - 1) ==
                              *device + size - 1;
        fs.upload_device = device.value_or(0);
        fs.upload_size = static_cast<u32>((std::min<u64>)(size, ~0U));
        fs.upload_offset = 0;
        fs.upload_tag = split.NewWriteStampTag();
        if (!fs.upload_known) {
            if (!device || size == 0 || size > (u64{64} << 20)) {
                return unknown("upload unmapped or huge");
            }
            split.NoteMemoryWrite(*device, size, "upload (not linear)");
        }
        // Its bytes are noted (above, or by its inline_data words); an upload release is an
        // unknown write of its own.
        split.RangedMethodWrite();
        return;
    }
    // inline_data words of the current upload.
    if (!fs.upload_known) {
        const auto device = memory_manager.GpuToCpuAddress(
            (GPUVAddr{fs.regs[base + 2]} << 32) | fs.regs[base + 3]);
        if (!device) {
            return unknown("upload data unmapped");
        }
        split.NoteMemoryWrite(*device, fs.regs[base] * u64{(std::max)(fs.regs[base + 1], 1U)} +
                                           u64{fs.regs[base + 4]} * fs.regs[base + 1],
                              "upload data");
        split.RangedMethodWrite();
        return;
    }
    const u32 remaining_bytes =
        fs.upload_offset < fs.upload_size ? fs.upload_size - fs.upload_offset : 0;
    const u32 whole = (std::min<u32>)(count, remaining_bytes / 4);
    // Tagged, so the QMD copy's patch below looks past this upload's own stamps.
    split.SetWriteStampTag(fs.upload_tag);
    split.NoteMemoryWriteData(fs.upload_device + fs.upload_offset, words, whole);
    if (whole < count && remaining_bytes % 4 != 0) {
        split.NoteMemoryWrite(fs.upload_device + fs.upload_offset + whole * 4, remaining_bytes % 4);
    }
    split.SetWriteStampTag(0);
    if (VideoCommon::QmdPrediction::MirrorMode()) {
        // EXP13=2-4: a QMD patched by a 3D inline upload (6h): the copy takes the bytes too.
        // EDEN_NX2_QMD_3D_PATCH: this method's write mark is retired first (its bytes are noted
        // above, under the tag the patch looks past), so the holders' stamp check does not take
        // the upload itself for a later writer. The same call below then changes nothing.
        if (VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_qmd_3d_patch,
                                             "EDEN_NX2_QMD_3D_PATCH") != 0) {
            split.RangedMethodWrite();
        }
        const GPUVAddr destination = (GPUVAddr{fs.regs[base + 2]} << 32) | fs.regs[base + 3];
        Nx2PatchQmdCopy(split, memory_manager, destination + fs.upload_offset,
                        reinterpret_cast<const u8*>(words),
                        (std::min<u64>)(u64{count} * sizeof(u32), remaining_bytes),
                        split.SentPosition(), fs.upload_tag);
    }
    split.RangedMethodWrite();
    fs.upload_offset += (std::min<u32>)(count * 4, remaining_bytes);
}

namespace {

// "Front 0xe2c misses" (fable_front_drains 1.3 A, the counter first; FRITCHII_E2C_CENSUS, live,
// default 0 since 2026-11-14: about 20 ms/s of the front at 2,600 misses a second; only at misses):
// per MME DMA read of the 0xe2c occlusion macro that misses its page snapshot for a noted write
// (EDEN_NX2_DMA_SNAPSHOT's reason "noted write"), the newest noted write since the snapshot over
// bytes 0-8, 8-16 and 16-24 of the read (the macro compares the 64-bit words at 0 and 16; 8-16
// is the timestamp): what wrote it (a 16-byte payload report, whose payload the front had in its
// registers; a 4-byte payload release; a counter or sample count report; anything else: puller
// semaphores, uploads, unknown ranges), whether the front knows its words, whether the back had
// run it; how far behind the back was (ring words between `consumed` and the front) and whether
// the query cache listed a report on the bytes (the hit path's own test). Front thread only.
struct E2cMissStats {
    enum Kind : u32 { PayloadReport, PayloadRelease, CounterReport, Other, None, KindCount };
    static constexpr std::array<const char*, KindCount> kind_names{
        "payload report (16 bytes)", "payload release (4 bytes)", "counter or sample report",
        "other", "none"};
    /// [byte range][kind][words known][run by the back]
    std::array<std::array<std::array<std::array<u64, 2>, 2>, KindCount>, 3> notes{};
    u64 misses{};
    u64 behind_words{};
    u64 behind_max{};
    u64 listed{};
    u64 overflows{};
    std::chrono::steady_clock::time_point since{};
    std::chrono::steady_clock::time_point last{}; ///< The previous miss counted
};
E2cMissStats g_e2c_misses;

void NoteE2cMiss(VideoCommon::GpuSplit& split, u64 snapshot, u64 device, bool listed) {
    auto& s = g_e2c_misses;
    const auto now = std::chrono::steady_clock::now();
    // A miss 10 s after the previous one: the switch was off (or nothing missed), so the window
    // starts over here rather than reporting the old counts over the whole time.
    if (s.since != std::chrono::steady_clock::time_point{} &&
        now - s.last > std::chrono::seconds{10}) {
        s = {};
    }
    s.last = now;
    const u64 done = split.ConsumedNow();
    const u64 behind = split.SentPosition() - (std::min)(done, split.SentPosition());
    std::array<VideoCommon::GpuSplit::NotedOver, 3> over;
    bool overflow = false;
    split.NewestNotedWritesOver(snapshot, device, 8, over, overflow);
    ++s.misses;
    s.behind_words += behind;
    s.behind_max = (std::max)(s.behind_max, behind);
    s.listed += listed ? 1 : 0;
    s.overflows += overflow ? 1 : 0;
    for (size_t range = 0; range < over.size(); ++range) {
        const auto& note = over[range];
        const E2cMissStats::Kind kind =
            note.position == 0                             ? E2cMissStats::None
            : note.what == VideoCommon::NotedPayloadReport ? E2cMissStats::PayloadReport
            : note.what == VideoCommon::NotedPayloadRelease ? E2cMissStats::PayloadRelease
            : note.what == VideoCommon::NotedCounterReport ? E2cMissStats::CounterReport
                                                           : E2cMissStats::Other;
        const bool ran = note.position != 0 && note.position <= done;
        ++s.notes[range][kind][note.known ? 1 : 0][ran ? 1 : 0];
    }
    if (s.since == std::chrono::steady_clock::time_point{}) {
        s.since = now;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - s.since).count();
    if (seconds < 10.0) {
        return;
    }
    const auto rate = [seconds](u64 count) { return static_cast<double>(count) / seconds; };
    std::string ranges;
    constexpr std::array<const char*, 3> range_names{"0-8", "8-16", "16-24"};
    for (size_t range = 0; range < s.notes.size(); ++range) {
        ranges += fmt::format("; bytes {}:", range_names[range]);
        for (u32 kind = 0; kind < E2cMissStats::KindCount; ++kind) {
            const auto& n = s.notes[range][kind];
            if (n[0][0] + n[0][1] + n[1][0] + n[1][1] == 0) {
                continue;
            }
            ranges += fmt::format(" {} known {:.0f}/{:.0f} unknown {:.0f}/{:.0f},",
                                  E2cMissStats::kind_names[kind], rate(n[1][1]), rate(n[1][0]),
                                  rate(n[0][1]), rate(n[0][0]));
        }
        if (ranges.back() == ',') {
            ranges.pop_back();
        }
    }
    LOG_INFO(HW_GPU,
             "Front 0xe2c misses (noted write) {:.0f} s: {:.0f}/s, back behind {:.0f} ring words "
             "on average ({} at most), a listed report on the bytes {:.0f}/s, notes past the "
             "table {:.0f}/s; the newest note since the page snapshot per byte range, /s, value "
             "known at the front run/queued, unknown run/queued{}",
             seconds, rate(s.misses),
             s.misses != 0 ? static_cast<double>(s.behind_words) / static_cast<double>(s.misses)
                           : 0.0,
             s.behind_max, rate(s.listed), rate(s.overflows), ranges);
    const auto start = now;
    s = {};
    s.since = start;
    s.last = start;
}

// FRITCHII_FRONT_RAN_READS (live, default 0; the market's "Front 0xe2c misses" line: about 90%
// of the noted-write misses wait for writes the back already ran): a macro DMA read that misses
// its page snapshot only for a noted write (reason 3) reads guest memory at once, without the
// drain, when the drained read could not give other bytes:
//   (a) the back ran every write noted over the bytes since the snapshot (GpuSplit::
//       NotedWritesRan: none past ConsumedNow, the dropped ones included);
//   (e) the read is on one page (the snapshot, the GPU-written bit and (a) are per page);
//   (f) the page never held GPU-modified buffer data, or no indirect grid placeholder was ever
//       handed out: the grid's DMA copy inside a 3D inline upload is the one noted write whose
//       run leaves its bytes in the buffer cache only (marking the page GPU-written), which the
//       drained path's FlushRegion would bring back. Without placeholders every noted write
//       (3D reports and payload releases, puller semaphores, inline uploads, KeplerMemory data
//       with EXP120) writes guest memory itself, at its run or when its async report completes;
//   (c) no async report write to the bytes is still owed: every report issued to their 16-byte
//       slots completed (Nx2AsyncReportsSettled; counter reports, timed and fence releases, with
//       any accuracy or fence setting, EXP33 and EXP119's runs; issued on the back before it
//       passed the report's entry), and the newest async note over them has run (SEM_FAST's
//       rule). A report starting up to 15 bytes before the read is included;
//   (b) the query cache lists no report on the bytes (the hit path's own test, 12 bytes back).
// The other miss reasons (compute and copies, unranged writers, and with EXP10 the dispatch
// check and byte lists, which come after the noted-write test) are asked as for a hit and still
// drain. 2 = verify: the same decision, then the read drains anyway, and the bytes read at once
// are compared with the drained ones ("Front ran reads (must be 0)"), which are used.
// The page snapshot is left alone: (a)-(c) hold for the bytes read, not for the rest of the page,
// whose noted writes may still be queued or owe a report; a snapshot moved to `done` would let
// the hit path (which asks neither (a) nor (c)) serve those.
// The same switch answers EXP10's dispatch check past its scan limit (DispatchCheck::ScanLimit:
// more dispatch writes recorded since the page's last check than the scan compares, or the ring
// dropped them): the scan has made every dispatch sent since run by then, each recorded write
// stamped its 64 KB blocks before the record was counted, so no stamp at or after the snapshot
// on the page's block is the scan's own "miss" without its limit (GpuSplit::
// DispatchStampsMissSince); a block stamped since, or a write that stamped everything (an
// unbounded or unconfirmed dispatch, an image without a guest range), still drains. Front
// thread only.
struct RanReadStats {
    enum Outcome : u32 { Served, Queued, PastTable, TwoPages, GpuWritten, Async, Listed, Count };
    static constexpr std::array<const char*, Count> names{
        "served",          "a note still queued", "notes past the table", "two pages",
        "GPU-written page", "an async report owed", "a listed report"};
    std::array<u64, Count> outcomes{};
    u64 then_missed{}; ///< Spared the noted write, then drained for another writer
    /// The compute/copy misses (reason 1) of reads while the switch is on, by cause. The dispatch
    /// scan's limit (DispatchCheck::ScanLimit) is then answered by the dispatch stamps
    /// (GpuSplit::DispatchStampsMissSince): passed, or refused by a stamp on the page's block or
    /// by a write that stamped everything.
    enum Compute : u32 {
        WriterUnbounded,
        WriterOverlap,
        WriterScanLimit,
        WriterForgotten,
        DispatchQueued,
        DispatchOverlap,
        DispatchUnbounded,
        ScanPassed,
        ScanBlock,
        ScanWide,
        ByteOverlap,
        ComputeCount,
    };
    static constexpr std::array<const char*, ComputeCount> compute_names{
        "unbounded writer",          "writer range overlap",
        "writer ranges past the scan", "writer ranges left the ring",
        "dispatch queued",           "dispatch write overlap",
        "unbounded dispatch",        "dispatches past the scan, passed by the stamps",
        "dispatches past the scan, block stamped since", "dispatches past the scan, a wide write since",
        "listed byte overlap"};
    std::array<u64, ComputeCount> compute{};
    /// Reads served (with 2: that would be) by what was spared: the noted write, the dispatch
    /// scan's limit, both.
    std::array<u64, 3> served{};
    u64 compared{};
    u64 differed{};
    u64 session_differed{};
    std::chrono::steady_clock::time_point since{};
};
RanReadStats g_ran_reads;

RanReadStats::Outcome RanReadCheck(VideoCommon::GpuSplit& split,
                                   VideoCore::RasterizerInterface& rasterizer, u64 since,
                                   u64 device, u64 bytes, bool one_page) {
    if (!one_page || device < 16) {
        return RanReadStats::TwoPages;
    }
    // The back's position first: everything it ran before it is visible from here on.
    const u64 done = split.ConsumedNow();
    switch (split.NotedWritesRan(since, done, device, bytes)) {
    case VideoCommon::GpuSplit::NotedRun::Queued:
        return RanReadStats::Queued;
    case VideoCommon::GpuSplit::NotedRun::PastTable:
        return RanReadStats::PastTable;
    case VideoCommon::GpuSplit::NotedRun::Ran:
        break;
    }
    if (Tegra::Nx2IndirectGrid::AnyLive() &&
        (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed) ||
         Tegra::Nx2MayBeGpuWritten(device, bytes))) {
        return RanReadStats::GpuWritten;
    }
    // A 16-byte report written up to 15 bytes before the read overlaps it.
    if (split.HasAsyncWrite(device - 15, bytes + 15) > done ||
        !Tegra::Nx2AsyncReportsSettled(device - 15, bytes + 15)) {
        return RanReadStats::Async;
    }
    if (rasterizer.HasCachedReports(device - 12, bytes + 12)) {
        return RanReadStats::Listed;
    }
    return RanReadStats::Served;
}

void MaybeLogRanReads(int mode) {
    auto& s = g_ran_reads;
    const auto now = std::chrono::steady_clock::now();
    if (s.since == std::chrono::steady_clock::time_point{}) {
        s.since = now;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - s.since).count();
    if (seconds < 10.0) {
        return;
    }
    const auto rate = [seconds](u64 count) { return static_cast<double>(count) / seconds; };
    u64 checked = 0;
    std::string drained;
    for (u32 i = 0; i < RanReadStats::Count; ++i) {
        checked += s.outcomes[i];
        if (i != RanReadStats::Served) {
            drained += fmt::format("{} {} {:.0f}/s", drained.empty() ? "" : ",",
                                   RanReadStats::names[i], rate(s.outcomes[i]));
        }
    }
    std::string compute;
    for (u32 i = 0; i < RanReadStats::ComputeCount; ++i) {
        if (s.compute[i] != 0) {
            compute += fmt::format("{} {} {:.0f}/s", compute.empty() ? "" : ",",
                                   RanReadStats::compute_names[i], rate(s.compute[i]));
        }
    }
    LOG_INFO(HW_GPU,
             "Front ran reads (FRITCHII_FRONT_RAN_READS={}) {:.0f} s: noted-write misses checked "
             "{:.0f}/s, passed {:.0f}/s{} (of them drained for another writer after it "
             "{:.0f}/s); drained:{}; compute/copy checks:{}; reads served{}: {:.0f}/s spared a "
             "noted write, {:.0f}/s the dispatch scan's limit, {:.0f}/s both; verify: {} "
             "compared, {} differed (must be 0; {} this session)",
             mode, seconds, rate(checked), rate(s.outcomes[RanReadStats::Served]),
             mode == 2 ? " (mode 2: would be; drained and compared)" : "", rate(s.then_missed),
             drained, compute.empty() ? std::string{" none"} : compute,
             mode == 2 ? " (would be)" : "", rate(s.served[0]), rate(s.served[1]),
             rate(s.served[2]), s.compared, s.differed, s.session_differed);
    const u64 session = s.session_differed;
    s = {};
    s.since = now;
    s.session_differed = session;
}

} // Anonymous namespace

void Maxwell3D::FrontDmaRead(VideoCommon::GpuSplit& split, FrontState& fs, u32 macro,
                             u32 method, u32 count) {
    // The words may be written by methods still queued for the back (indirect draw arguments
    // from a culling dispatch, constant buffer and inline uploads, reports), and the caches that
    // may hold newer ones belong to it. Draws write memory the 3D engine reads only after a
    // host-level wait, so the work after the last one need not have run. EDEN_NX2_MACRO_DMA_DRAIN=1
    // waits for everything queued, as before. With EDEN_NX2_CMDPAGE_DIAG the read is repeated
    // after a full drain and the words that differ are counted.
    const GPUVAddr source = (GPUVAddr{fs.regs[0x154] & 0x1ffffff} << 32) | fs.regs[0x155];
    const u32 ram = fs.regs[0x156];
    const u32 words = std::min<u32>(count, 0x1000);
    const size_t read_bytes = size_t{words} * sizeof(u32);
    static const bool full_drain = std::getenv("EDEN_NX2_MACRO_DMA_DRAIN") != nullptr;
    // EDEN_NX2_DEPTH_CENSUS (fable52_answer_potroom 4b): the occlusion reads of macro 0xe2c.
    namespace ReportCensus = VideoCommon::Nx2ReportCensus;
    const bool census_read = macro == 0xe2c && ReportCensus::Enabled();
    const auto flush_if_dirty = [&] {
        if (memory_manager.IsMemoryDirty(source, read_bytes, VideoCommon::CacheType::BufferCache)) {
            split.Run(
                [memory = &memory_manager, source, read_bytes] {
                    memory->FlushRegion(source, read_bytes, VideoCommon::CacheType::BufferCache);
                },
                VideoCommon::DrainSite::MacroDmaFlush,
                VideoCommon::Nx2FastDrawCount::Neutral());
        }
    };
    // Words a constant buffer upload still queued for the back writes come from the upload
    // itself (9,600 waits a second in Ocarina's forest).
    if (!full_drain && !Tegra::Nx2CmdPageDiag::g_enabled &&
        !VideoCommon::QueryDma::TraceEnabled()) {
        thread_local std::vector<u32> forwarded;
        forwarded.resize(words);
        if (split.ForwardRead(memory_manager.GetID() + 1, source, words, forwarded.data())) {
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::DmaReadsForwarded, 1);
            if (census_read) [[unlikely]] {
                ReportCensus::NoteE2cRead(forwarded.data(), words,
                                          ReportCensus::E2cPath::Forwarded, false);
            }
            for (u32 i = 0; i < words; ++i) {
                if (method == 0x158) {
                    fs.parameters.push_back(forwarded[i]);
                } else if (ram + i < native_data_ram.size()) {
                    native_data_ram[ram + i] = forwarded[i];
                }
            }
            return;
        }
    }
    // EDEN_NX2_DMA_SNAPSHOT (a ported dma-snapshot fix): Ocarina reads an occlusion report for
    // each object (macro 0xe2c, MME_DMA_READ_FIFOED), ~35,000 times a second, and its pages are
    // GPU-written, so each read drained the back: the front and the back took turns. A drained
    // read now flushes its whole device pages and remembers the split position; later reads of
    // those pages skip the drain while nothing sent since may write them
    // (GpuSplit::MayBeWrittenSince) and the query cache lists no report there. 10 also drains,
    // reads again and logs differences.
    const int snapshot_mode =
        VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_dma_snapshot, "EDEN_NX2_DMA_SNAPSHOT",
                                         9);
    // Waits until pending reports in the range have landed, as the drained path below.
    const auto flush_queries = [&](GPUVAddr begin, size_t bytes) {
        split.Run(
            [this, begin, bytes] {
                VideoCommon::QueryDma::FlushPending(memory_manager, *rasterizer, begin, bytes);
            },
            VideoCommon::DrainSite::MacroDmaFlush,
            VideoCommon::Nx2FastDrawCount::Neutral());
    };
    if (snapshot_mode >= 9 && !full_drain && !Tegra::Nx2CmdPageDiag::g_enabled &&
        method == 0x158 && words != 0) {
        const auto device = memory_manager.GpuToCpuAddress(source);
        if (device &&
            memory_manager.GpuToCpuAddress(source + read_bytes - 1) == *device + read_bytes - 1) {
            static u64 hits{};
            static u64 misses{};
            static u64 differences{};
            static u64 report_reads{};
            static u64 miss_report_flushes{};
            static std::array<u64, 4> miss_reasons{};
            // EXP10: compute and copies sent since count only where their writes may reach the
            // pages read: the ranges the fronts noted (GpuSplit::WritersMayWriteSince), the
            // writes the back recorded for the dispatches (DispatchesMayWriteSince, once it ran
            // them: 1 waits for that, 2 drains instead). Copies and other work whose ranges
            // nobody can tell still count for every read.
            const int exp10 = VideoCore::Nx2Debug::Experiment(10);
            const bool ranged_writers = exp10 != 0;
            // EXP10 +8 (fable_answer_long_drains R1): a page snapshot is current for every byte
            // no writer since touches. The page scans list the overlapping writes (8 a page) and
            // the read misses only when its own bytes overlap one of them.
            const bool byte_ranges = (exp10 & 8) != 0;
            // EXP122 (fable_answer_long_drains 8.2 a'): the pages with a snapshot keep the noted
            // ranges since it, so WritersMayWriteSince answers them without the ring scan.
            const int exp122 = VideoCore::Nx2Debug::Experiment(122);
            split.SetPageWatch(exp122);
            static u64 byte_skips{};
            static u64 byte_overlaps{};
            static u64 byte_differences{};
            static u64 byte_merges{};
            using WriterCheck = VideoCommon::GpuSplit::WriterCheck;
            static std::array<u64, static_cast<size_t>(WriterCheck::Count)> writer_checks{};
            bool byte_skip = false;
            const u64 read_end = *device + read_bytes;
            using DispatchCheck = VideoCommon::GpuSplit::DispatchCheck;
            static u64 writer_unbounded{};
            static u64 writer_overlaps{};
            static std::array<u64, static_cast<size_t>(DispatchCheck::Count)> dispatch_checks{};
            static u64 dispatch_waits{};
            static u64 range_skips{};
            static u64 range_differences{};
            bool range_skip = false;
            // EXP120: hits that EXP120 = 0 would have drained (an unranged writer note it spared
            // since a page's snapshot), with their own verify count.
            static u64 unranged_skips{};
            static u64 unranged_differences{};
            bool unranged_skip = false;
            const u64 first_page = *device >> 12;
            const u64 last_page = (*device + read_bytes - 1) >> 12;
            // FRITCHII_FRONT_RAN_READS (live, 0 = off; RanReadCheck above): a noted write since
            // the snapshot whose every note over the bytes already ran and landed is no miss.
            // The checks after it (EXP10's dispatches and byte lists) are asked as for a hit.
            const int ran_mode = VideoCore::Nx2Debug::SwitchValue(
                VideoCore::Nx2Debug::g_front_ran_reads, "FRITCHII_FRONT_RAN_READS");
            bool ran_passed = false;
            // The dispatch scan's limit answered by the dispatch stamps (below).
            bool scan_passed = false;
            const auto noted_write = [&](u64 since) {
                if (!split.NotedWriteSince(since, *device, read_bytes)) {
                    return false;
                }
                if (ran_mode == 0) {
                    return true;
                }
                const auto outcome = RanReadCheck(split, *rasterizer, since, *device, read_bytes,
                                                  first_page == last_page);
                ++g_ran_reads.outcomes[outcome];
                if (outcome != RanReadStats::Served) {
                    return true;
                }
                ran_passed = true;
                return false;
            };
            bool hit = true;
            bool census_pending = false;
            for (u64 page = first_page; page <= last_page && hit; ++page) {
                const auto it = fs.dma_snapshots.find(page);
                if (it == fs.dma_snapshots.end()) {
                    hit = false;
                    ++miss_reasons[0];
                    break;
                }
                // Compute/copy sent since, unranged writer sent since, noted write.
                auto& snapshot = it->second;
                const u64 since = snapshot.position;
                const bool writer_since = split.LastWriterPosition() > since;
                auto* const listed = byte_ranges ? &snapshot.listed : nullptr;
                const u32 merges_before = snapshot.listed.merges;
                WriterCheck writer_check = WriterCheck::Miss;
                u32 reason = 4;
                if (!ranged_writers) {
                    reason = writer_since                             ? 1
                             : split.LastUnrangedPosition() > since ? 2
                             : noted_write(since)                   ? 3
                                                                    : 4;
                } else if (split.WritersMayWriteSince(since, page << 12, 0x1000,
                                                      &snapshot.writer_ranges, listed,
                                                      &writer_check)) {
                    // Whole pages: the ranges found to miss one need no second look.
                    reason = 1;
                    ++(split.LastUnboundedWriterPosition() > since ? writer_unbounded
                                                                   : writer_overlaps);
                    ++writer_checks[static_cast<size_t>(writer_check)];
                    if (ran_mode != 0) {
                        ++g_ran_reads.compute[writer_check == WriterCheck::Unbounded
                                                  ? RanReadStats::WriterUnbounded
                                              : writer_check == WriterCheck::ScanLimit
                                                  ? RanReadStats::WriterScanLimit
                                              : writer_check == WriterCheck::Forgotten
                                                  ? RanReadStats::WriterForgotten
                                                  : RanReadStats::WriterOverlap];
                    }
                } else if (split.LastUnrangedPosition() > since) {
                    reason = 2;
                } else if (noted_write(since)) {
                    reason = 3;
                } else {
                    // Last: the only check that may wait.
                    bool waited = false;
                    DispatchCheck check = split.DispatchesMayWriteSince(
                        since, page << 12, 0x1000, (exp10 & 3) != 2, snapshot.dispatch_writes,
                        waited, listed);
                    dispatch_waits += waited ? 1 : 0;
                    // FRITCHII_FRONT_RAN_READS: past the scan's limit every dispatch sent since
                    // has run and recorded its writes (the scan waits for that first, or answers
                    // Queued); the dispatch stamps answer what the scan without its limit would.
                    if (check == DispatchCheck::ScanLimit && ran_mode != 0) {
                        u64 count = 0;
                        bool wide = false;
                        if (split.DispatchStampsMissSince(since, page << 12, 0x1000, count,
                                                          wide)) {
                            // As a scan miss: the records below `count` miss the page, and
                            // there was nothing to list.
                            check = DispatchCheck::Miss;
                            snapshot.dispatch_writes = count;
                            scan_passed = true;
                            ++g_ran_reads.compute[RanReadStats::ScanPassed];
                        } else {
                            ++g_ran_reads.compute[wide ? RanReadStats::ScanWide
                                                       : RanReadStats::ScanBlock];
                        }
                    } else if (check != DispatchCheck::Miss && ran_mode != 0) {
                        ++g_ran_reads.compute[check == DispatchCheck::Queued ? RanReadStats::DispatchQueued
                                              : check == DispatchCheck::Overlap
                                                  ? RanReadStats::DispatchOverlap
                                                  : RanReadStats::DispatchUnbounded];
                    }
                    if (check != DispatchCheck::Miss) {
                        reason = 1;
                        ++dispatch_checks[static_cast<size_t>(check)];
                    } else if (listed != nullptr && listed->count != 0) {
                        // The listed writes are all since this page's snapshot.
                        if (listed->Overlaps(page << 12, *device, read_end)) {
                            reason = 1;
                            ++byte_overlaps;
                            if (ran_mode != 0) {
                                ++g_ran_reads.compute[RanReadStats::ByteOverlap];
                            }
                        } else {
                            byte_skip = true;
                        }
                    } else {
                        range_skip |= writer_since;
                    }
                }
                byte_merges += snapshot.listed.merges - merges_before;
                if (reason == 3 && fs.e2c_call && words == 6 &&
                    VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_e2c_census,
                                                     "FRITCHII_E2C_CENSUS") != 0) [[unlikely]] {
                    // "Front 0xe2c misses" (FRITCHII_E2C_CENSUS, live, default 0: nothing here):
                    // what the noted write over the read's bytes is.
                    NoteE2cMiss(split, since, *device,
                                rasterizer->HasCachedReports(*device - 12,
                                                             u64{read_bytes} + 12));
                }
                if (reason != 4) {
                    hit = false;
                    ++miss_reasons[reason];
                } else if (split.LastUnrangedPositionWithoutExp120() > since) {
                    unranged_skip = true;
                }
            }
            if (hit && range_skip) {
                ++range_skips;
            }
            if (hit && byte_skip) {
                ++byte_skips;
            }
            if (hit && unranged_skip) {
                ++unranged_skips;
            }
            // FRITCHII_FRONT_RAN_READS: spared the noted write, then drained for another writer;
            // with 2 a read it would serve drains all the same, the bytes read now kept to compare.
            thread_local std::vector<u32> ran_words;
            bool ran_verify = false;
            if (ran_passed && !hit) {
                ++g_ran_reads.then_missed;
            }
            const size_t spared = ran_passed && scan_passed ? 2 : ran_passed ? 0 : 1;
            if ((ran_passed || scan_passed) && hit) {
                if (ran_mode == 2) {
                    // Counted here (would be served); 1 counts them where they are served.
                    ++g_ran_reads.served[spared];
                    ran_words.resize(words);
                    memory_manager.ReadBlockUnsafe(source, ran_words.data(), read_bytes);
                    ran_verify = true;
                    hit = false;
                    ++miss_reasons[ran_passed ? 3 : 1];
                }
            }
            if (ran_mode != 0) {
                MaybeLogRanReads(ran_mode);
            }
            // A report's value is up to 8 bytes, listed by its first 4: look 12 bytes back. A
            // listed report may not be in guest memory yet: take the drained path.
            if (!hit || !rasterizer->HasCachedReports(*device - 12, u64{read_bytes} + 12)) {
                thread_local std::vector<u32> snap_words;
                snap_words.resize(words);
                if (hit) {
                    ++hits;
                    if (ran_passed || scan_passed) {
                        ++g_ran_reads.served[spared];
                    }
                    memory_manager.ReadBlockUnsafe(source, snap_words.data(), read_bytes);
                    if (snapshot_mode == 10) {
                        split.Drain(VideoCommon::DrainSite::MacroDmaRead);
                        flush_if_dirty();
                        flush_queries(source, read_bytes);
                        std::vector<u32> exact(words);
                        memory_manager.ReadBlockUnsafe(source, exact.data(), read_bytes);
                        if (exact != snap_words) {
                            range_differences += range_skip ? 1 : 0;
                            byte_differences += byte_skip ? 1 : 0;
                            unranged_differences += unranged_skip ? 1 : 0;
                            if (differences++ < 32) {
                                LOG_WARNING(HW_GPU,
                                            "DMA snapshot check: macro {:#x} read of {} words at "
                                            "{:#x} differs from the drained read ({:#x} vs "
                                            "{:#x}){}{}{}",
                                            macro, words, source, snap_words[0], exact[0],
                                            range_skip ? " (skipped by EXP10 ranges)" : "",
                                            byte_skip ? " (skipped by EXP10 byte ranges)" : "",
                                            unranged_skip ? " (skipped by EXP120)" : "");
                            }
                            snap_words = std::move(exact);
                        }
                    }
                } else {
                    ++misses;
                    if (census_read) [[unlikely]] {
                        census_pending =
                            rasterizer->HasCachedReports(*device - 12, u64{read_bytes} + 12);
                    }
                    // The GPU data of the whole pages (a flush of only the bytes read left stale
                    // neighbours), the pending reports of the bytes read.
                    const GPUVAddr flush_begin = source & ~GPUVAddr{0xfff};
                    const GPUVAddr flush_end = (source + read_bytes + 0xfff) & ~GPUVAddr{0xfff};
                    const size_t flush_bytes = static_cast<size_t>(flush_end - flush_begin);
                    // FRITCHII_DELAY_MACRO_DRAIN_US (live, 0 = off): the delay test of this drain.
                    using DelayTest = VideoCommon::GpuSplit::DelayTest;
                    const u32 delay_us = split.DelayBeforeWait(DelayTest::MacroDrain);
                    split.Drain(VideoCommon::DrainSite::MacroDmaRead);
                    VideoCommon::GpuSplit::DelayAfterWait(DelayTest::MacroDrain, delay_us);
                    if (memory_manager.IsMemoryDirty(flush_begin, flush_bytes,
                                                     VideoCommon::CacheType::BufferCache)) {
                        split.Run(
                            [memory = &memory_manager, flush_begin, flush_bytes] {
                                memory->FlushRegion(flush_begin, flush_bytes,
                                                    VideoCommon::CacheType::BufferCache);
                            },
                            VideoCommon::DrainSite::MacroDmaFlush,
                            VideoCommon::Nx2FastDrawCount::Neutral());
                    }
                    // EDEN_NX2_EXP73 (live, 0 = off; fable52_answer_bosswave_2 3b): only when
                    // reports are listed over the bytes read (the hit path's own test, 12 bytes
                    // back): otherwise FlushPending finds nothing and split.Run is an empty round
                    // trip to the back, ~2,850 a second in the Stalchild fight.
                    if (VideoCore::Nx2Debug::Experiment(73) == 0 ||
                        rasterizer->HasCachedReports(*device - 12, u64{read_bytes} + 12)) {
                        ++miss_report_flushes;
                        flush_queries(source, read_bytes);
                    }
                    if (fs.dma_snapshots.size() > 8192) {
                        fs.dma_snapshots.clear();
                    }
                    for (u64 page = first_page; page <= last_page; ++page) {
                        fs.dma_snapshots[page] = {split.SentPosition(), split.WriterRangeCount(),
                                                  split.DispatchWriteCount(), {}};
                        split.WatchPage(page);
                    }
                    memory_manager.ReadBlockUnsafe(source, snap_words.data(), read_bytes);
                    if (ran_verify) {
                        // FRITCHII_FRONT_RAN_READS=2: the drained words are used.
                        ++g_ran_reads.compared;
                        if (ran_words != snap_words) {
                            ++g_ran_reads.differed;
                            if (g_ran_reads.session_differed++ < 16) {
                                u32 index = 0;
                                while (index + 1 < words && ran_words[index] == snap_words[index]) {
                                    ++index;
                                }
                                LOG_WARNING(HW_GPU,
                                            "Front ran reads (must be 0): macro {:#x} read of {} "
                                            "words at {:#x} (device {:#x}) differs from the "
                                            "drained read: word {} {:#x}, drained {:#x}",
                                            macro, words, source, *device, index,
                                            ran_words[index], snap_words[index]);
                            }
                        }
                    }
                }
                static auto last_log = std::chrono::steady_clock::now();
                if (const auto now = std::chrono::steady_clock::now();
                    now - last_log >= std::chrono::seconds{5}) {
                    last_log = now;
                    LOG_INFO(HW_GPU,
                             "DMA snapshot: {} reads skipped the drain, {} drained, {} differed, "
                             "{} reads of listed reports (drained path); misses: {} new page, {} "
                             "compute/copy, {} unranged, {} noted write; {} misses flushed reports "
                             "(EXP73={})",
                             hits, misses, differences, report_reads, miss_reasons[0],
                             miss_reasons[1], miss_reasons[2], miss_reasons[3], miss_report_flushes,
                             VideoCore::Nx2Debug::Experiment(73));
                    {
                        // fable_answer_long_drains 8.4: who notes the unranged writers, the
                        // KeplerMemory and Fermi2D methods (single method / non-incrementing run).
                        using Kind = VideoCommon::GpuSplit::EngineWriteKind;
                        using Note = VideoCommon::GpuSplit::EngineWriteNote;
                        const auto& notes = split.GetEngineWriteNotes();
                        const auto kind = [&notes](Kind k) {
                            const auto at = [&](size_t call, Note n) {
                                return notes[call][static_cast<size_t>(k)][static_cast<size_t>(n)];
                            };
                            return fmt::format(
                                "{}/{} unranged, {}/{} ranged, {}/{} not noted",
                                at(0, Note::Unranged), at(1, Note::Unranged), at(0, Note::Ranged),
                                at(1, Note::Ranged), at(0, Note::Nothing), at(1, Note::Nothing));
                        };
                        LOG_INFO(HW_GPU,
                                 "DMA snapshot unranged notes (EXP120={}; single/run): "
                                 "KeplerMemory data {}; KeplerMemory registers {}; Fermi2D blits "
                                 "{} (left unranged by EXP120=2: {}); Fermi2D registers {}; {} "
                                 "reads skipped by EXP120 ({} differed)",
                                 VideoCore::Nx2Debug::Experiment(120), kind(Kind::I2mData),
                                 kind(Kind::I2mRegister), kind(Kind::BlitLaunch),
                                 Tegra::FritchiiBlitMissText(), kind(Kind::BlitRegister),
                                 unranged_skips, unranged_differences);
                    }
                    if (ranged_writers) {
                        using Unbounded = VideoCommon::GpuSplit::DispatchUnbounded;
                        const auto& sent = split.UnboundedWriters();
                        const auto dispatches = split.GetDispatchStats();
                        const auto checks = [](DispatchCheck check) {
                            return dispatch_checks[static_cast<size_t>(check)];
                        };
                        const auto unbounded = [&dispatches](Unbounded reason) {
                            return dispatches.unbounded[static_cast<size_t>(reason)];
                        };
                        u64 dispatch_drains = 0;
                        for (const u64 count : dispatch_checks) {
                            dispatch_drains += count;
                        }
                        u64 dispatches_unbounded = 0;
                        for (const u64 count : dispatches.unbounded) {
                            dispatches_unbounded += count;
                        }
                        using Release = VideoCommon::GpuSplit::ReleaseNote;
                        const auto& releases = split.ReleaseNotes();
                        const auto watch = split.GetPageWatchStats();
                        const auto release = [&releases](Release note) {
                            return releases[static_cast<size_t>(note)];
                        };
                        LOG_INFO(HW_GPU,
                                 "DMA snapshot EXP10: compute/copy: {} drained ({} unbounded "
                                 "writer, {} overlapping range, {} on listed bytes; dispatches: "
                                 "{} queued, {} overlapping, {} unbounded, {} past the scan), {} "
                                 "skipped by ranges ({} differed), {} by byte ranges ({} "
                                 "differed), {} waited for dispatches only; dispatch launches "
                                 "sent: {}, dispatches run: {} bounded, {} unbounded ({} "
                                 "unconfirmed, {} replaced, {} image); unbounded writers sent: "
                                 "{} upload, {} release, {} compute, {} copy, {} other; compute "
                                 "releases: {} ranged, {} unmapped (write nothing), {} write "
                                 "nothing, unbounded: {} pending map, {} sparse, {} unmapped "
                                 "reduction, {} split; page scan misses: {} overlap (no list), "
                                 "{} past the scan, {} left the ring; {} byte list merges; {} "
                                 "ranges noted, {} ranges compared by the scans; page watch "
                                 "(EXP122={}): {} answered by the page watch ({} may write), {} "
                                 "watched pages, {} byte-list overflows, {} verified, {} verify "
                                 "mismatches; left to the ring: {} page not watched, {} watched "
                                 "after the snapshot, {} over two pages; {} pages replaced",
                                 writer_unbounded + writer_overlaps + byte_overlaps +
                                     dispatch_drains,
                                 writer_unbounded, writer_overlaps, byte_overlaps,
                                 checks(DispatchCheck::Queued),
                                 checks(DispatchCheck::Overlap), checks(DispatchCheck::Unbounded),
                                 checks(DispatchCheck::ScanLimit), range_skips, range_differences,
                                 byte_skips, byte_differences, dispatch_waits, dispatches.sent,
                                 dispatches.bounded,
                                 dispatches_unbounded, unbounded(Unbounded::Unconfirmed),
                                 unbounded(Unbounded::Replaced), unbounded(Unbounded::Image),
                                 sent[0], sent[1], sent[2], sent[3], sent[4],
                                 release(Release::Ranged), release(Release::Unmapped),
                                 release(Release::NoWrite), release(Release::PendingMap),
                                 release(Release::Sparse), release(Release::LateUnmapped),
                                 release(Release::Split),
                                 writer_checks[static_cast<size_t>(WriterCheck::Overlap)],
                                 writer_checks[static_cast<size_t>(WriterCheck::ScanLimit)],
                                 writer_checks[static_cast<size_t>(WriterCheck::Forgotten)],
                                 byte_merges, split.NotedWriterRanges(),
                                 VideoCommon::g_writer_ranges_compared.load(
                                     std::memory_order_relaxed),
                                 exp122, watch.answered, watch.answered_write, watch.watched,
                                 watch.overflows, watch.verified, watch.mismatches,
                                 watch.unwatched_absent, watch.unwatched_newer,
                                 watch.unwatched_spans, watch.replaced);
                    }
                }
                if (census_read) [[unlikely]] {
                    ReportCensus::NoteE2cRead(snap_words.data(), words,
                                              hit ? ReportCensus::E2cPath::Snapshot
                                                  : ReportCensus::E2cPath::SnapshotMiss,
                                              census_pending);
                }
                for (u32 i = 0; i < words; ++i) {
                    fs.parameters.push_back(snap_words[i]);
                }
                return;
            }
            ++report_reads;
        }
    }
    VideoCommon::Nx2Profile::CountDmaRead(macro);
    if (full_drain) {
        split.Drain(VideoCommon::DrainSite::MacroDmaRead);
    } else {
        split.WaitForMemoryWriters(VideoCommon::DrainSite::MacroDmaRead, source,
                                   source + read_bytes);
    }
    if (VideoCommon::Nx2Profile::Enabled()) {
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::DmaReadBytes, read_bytes);
    }
    // EDEN_NX2_QUERY_DMA_TRACE: reads of reports whose host query result is still pending.
    if (VideoCommon::QueryDma::TraceEnabled()) {
        split.Run(
            [this, macro, method, source, read_bytes] {
                if (VideoCommon::QueryDma::HasPendingQueries(memory_manager, *rasterizer, source,
                                                             read_bytes)) {
                    VideoCommon::QueryDma::Trace(memory_manager, *rasterizer, macro, method,
                                                 source, read_bytes);
                }
            },
            VideoCommon::DrainSite::MacroDmaFlush,
            VideoCommon::Nx2FastDrawCount::Neutral());
    }
    flush_if_dirty();
    std::vector<u32> diag_before;
    if (Tegra::Nx2CmdPageDiag::g_enabled) {
        diag_before.resize(words);
        memory_manager.ReadBlockUnsafe(source, diag_before.data(), read_bytes);
        split.Drain(VideoCommon::DrainSite::MacroDmaCheck);
        flush_if_dirty();
    }
    // Write out the reports held in the query cache, or the read returns stale occlusion counts.
    bool census_pending = false;
    split.Run(
        [this, source, read_bytes, census_read, &census_pending] {
            if (census_read) [[unlikely]] {
                census_pending = VideoCommon::QueryDma::HasPendingQueries(
                    memory_manager, *rasterizer, source, read_bytes);
            }
            VideoCommon::QueryDma::FlushPending(memory_manager, *rasterizer, source, read_bytes);
        },
        VideoCommon::DrainSite::MacroDmaFlush,
        VideoCommon::Nx2FastDrawCount::Neutral());
    thread_local std::vector<u32> census_words;
    if (census_read) [[unlikely]] {
        census_words.clear();
    }
    u32 changed{};
    for (u32 i = 0; i < words; ++i) {
        const u32 word = memory_manager.Read<u32>(source + u64{i} * sizeof(u32));
        if (census_read) [[unlikely]] {
            census_words.push_back(word);
        }
        if (method == 0x158) {
            fs.parameters.push_back(word);
        } else if (ram + i < native_data_ram.size()) {
            native_data_ram[ram + i] = word;
        }
        if (!diag_before.empty() && diag_before[i] != word) {
            ++changed;
        }
    }
    if (census_read) [[unlikely]] {
        ReportCensus::NoteE2cRead(census_words.data(), census_words.size(),
                                  ReportCensus::E2cPath::Drained, census_pending);
    }
    if (Tegra::Nx2CmdPageDiag::g_enabled) {
        bool gpu_written = false;
        for (GPUVAddr page = source & ~GPUVAddr{0xfff}; page < source + read_bytes; page += 0x1000) {
            const auto device = memory_manager.GpuToCpuAddress(page);
            gpu_written |= device && Tegra::Nx2IsGpuWritten(*device);
        }
        Tegra::Nx2CmdPageDiag::RecordDmaRead(source, words, changed, gpu_written);
    }
}

bool Maxwell3D::FrontDmaReadEarly(VideoCommon::GpuSplit& split, FrontState& fs, u32 macro,
                                  u32 method, GPUVAddr source, u32 ram, u32 words) {
    namespace Profile = VideoCommon::Nx2Profile;
    const u64 bytes = u64{words} * sizeof(u32);
    const bool sparse = source < Tegra::g_nx2_sparse_high.load(std::memory_order_relaxed) &&
                        Tegra::g_nx2_sparse_low.load(std::memory_order_relaxed) < source + bytes;
    if (words == 0 || sparse) {
        return false;
    }
    const u64 consumed = split.SettleWrites();
    std::array<u64, 5> pages;
    size_t page_count = 0;
    u64 target = 0;
    u64 write_target = 0;
    for (GPUVAddr page = source & ~GPUVAddr{0xfff}; page < source + bytes; page += 0x1000) {
        const std::optional<DAddr> device = memory_manager.GpuToCpuAddress(page);
        if (page_count == pages.size() || !device || Tegra::Nx2IsGpuWritten(*device)) {
            return false;
        }
        const u64 device_page = *device & ~u64{0xfff};
        pages[page_count++] = device_page;
        target = (std::max)(target, split.ReadTarget(device_page, device_page + 0x1000));
        write_target = (std::max)(write_target,
                                  split.Writes().LastWriteTo(device_page, device_page + 0x1000));
    }
    const auto wait_reason = target == split.Writes().LastUnknown()
                                 ? Profile::Counter::DmaReadsWaitUnknown
                             : target == write_target ? Profile::Counter::DmaReadsWaitWrite
                                                      : Profile::Counter::DmaReadsWaitGpu;
    split.WaitUntil(target, VideoCommon::DrainSite::MacroDmaWrites);
    for (size_t i = 0; i < page_count; ++i) {
        if (Tegra::Nx2IsGpuWritten(pages[i])) {
            return false;
        }
    }
    if (Profile::Enabled()) {
        Profile::Add(Profile::Counter::DmaReadBytes, bytes);
        Profile::Add(Profile::Counter::DmaReadsEarly, 1);
        if (target > consumed) {
            Profile::Add(wait_reason, 1);
        }
    }
    auto& read = fs.dma_words;
    read.resize(words);
    for (u32 i = 0; i < words; ++i) {
        read[i] = memory_manager.Read<u32>(source + u64{i} * sizeof(u32));
    }
    if (perf_verify) {
        // The old path: drain, flush, read again. Its words are the ones used.
        split.Drain(VideoCommon::DrainSite::MacroDmaRead);
        if (memory_manager.IsMemoryDirty(source, bytes, VideoCommon::CacheType::BufferCache)) {
            split.Run(
                [memory = &memory_manager, source, bytes] {
                    memory->FlushRegion(source, bytes, VideoCommon::CacheType::BufferCache);
                },
                VideoCommon::DrainSite::MacroDmaFlush,
                VideoCommon::Nx2FastDrawCount::Neutral());
        }
        static u32 logged{};
        for (u32 i = 0; i < words; ++i) {
            const u32 drained = memory_manager.Read<u32>(source + u64{i} * sizeof(u32));
            if (drained != read[i] && logged < 64) {
                ++logged;
                LOG_ERROR(HW_GPU,
                          "Nx2Perf split: macro {:#x} DMA read of {} words at {:#x}: word {} is "
                          "{:#x} without the drain, {:#x} after it",
                          macro, words, source, i, read[i], drained);
            }
            read[i] = drained;
        }
    }
    for (u32 i = 0; i < words; ++i) {
        if (method == 0x158) {
            fs.parameters.push_back(read[i]);
        } else if (ram + i < native_data_ram.size()) {
            native_data_ram[ram + i] = read[i];
        }
    }
    return true;
}

void Maxwell3D::FrontVerify(VideoCommon::GpuSplit& split, FrontState& fs) {
    split.Drain(VideoCommon::DrainSite::FrontVerify);
    static u64 checks{};
    static u64 differences{};
    ++checks;
    // State writes the back still holds in its method sink count as done.
    std::array<u32, Regs::NUM_REGS> expected;
    std::memcpy(expected.data(), regs.reg_array.data(), sizeof(expected));
    for (const auto& [method, value] : method_sink) {
        expected[method] = shadow_state.shadow_ram_control == Regs::ShadowRamControl::Replay
                               ? shadow_state.reg_array[method]
                               : value;
    }
    // EDEN_NX2_MACRO_SINK sinks macro writes too: in a tracking mode their shadow RAM copies are
    // written only when the sink is applied, so they count as done for the shadow check as well
    // (else the check would "resynchronize" the front's shadow to older values).
    std::array<u32, Regs::NUM_REGS> expected_shadow;
    std::memcpy(expected_shadow.data(), shadow_state.reg_array.data(), sizeof(expected_shadow));
    if (VideoCommon::GpuSplit::MacroSink() &&
        (shadow_state.shadow_ram_control == Regs::ShadowRamControl::Track ||
         shadow_state.shadow_ram_control == Regs::ShadowRamControl::TrackWithFilter)) {
        for (const auto& [method, value] : method_sink) {
            expected_shadow[method] = value;
        }
    }
    static u64 control_differences{};
    if ((fs.control != shadow_state.shadow_ram_control ||
         fs.shadow[MAXWELL3D_REG_INDEX(shadow_ram_control)] != expected_shadow[MAXWELL3D_REG_INDEX(shadow_ram_control)] ||
         fs.regs[MAXWELL3D_REG_INDEX(shadow_ram_control)] != expected[MAXWELL3D_REG_INDEX(shadow_ram_control)]) &&
        control_differences < 4) {
        LOG_WARNING(HW_GPU, "Split shadow RAM control writes, front ({} total) / back ({} total):",
                    fs.control_count, back_control_count);
        for (u32 i = 0; i < 16; ++i) {
            const auto& f = fs.control_writes[(fs.control_count + i) % 16];
            const auto& b = back_control_writes[(back_control_count + i) % 16];
            LOG_WARNING(HW_GPU, "  front {:#x} (mode {}, path {}) | back {:#x} (mode {}, last {})",
                        f.argument, f.before, f.path, b.argument, b.before, b.path);
        }
    }
    for (u32 i = 0; i < Regs::NUM_REGS; ++i) {
        if (fs.regs[i] != expected[i]) {
            if (differences++ < 64) {
                LOG_WARNING(HW_GPU, "Split front register {:#x}: front {:#x}, back {:#x}", i,
                            fs.regs[i], expected[i]);
            }
            fs.regs[i] = expected[i];
        }
    }
    if (fs.control != shadow_state.shadow_ram_control) {
        if (control_differences++ < 16) {
            LOG_WARNING(HW_GPU, "Split front shadow RAM control {} vs back {}",
                        static_cast<u32>(fs.control),
                        static_cast<u32>(shadow_state.shadow_ram_control));
        }
        fs.control = shadow_state.shadow_ram_control;
    }
    for (u32 i = 0; i < Regs::NUM_REGS; ++i) {
        if (fs.shadow[i] != expected_shadow[i]) {
            if (control_differences++ < 16) {
                LOG_WARNING(HW_GPU, "Split front shadow register {:#x}: front {:#x}, back {:#x}", i,
                            fs.shadow[i], expected_shadow[i]);
            }
            fs.shadow[i] = expected_shadow[i];
        }
    }
    if ((checks & 63) == 1) {
        LOG_INFO(HW_GPU, "Split front register checks: {} done, {} differences", checks, differences);
    }
}

void Maxwell3D::FinishFrontMacro(Core::System& system) {
    const VideoCommon::Nx2Profile::FineScope profile_scope{VideoCommon::Nx2Profile::Zone::MacroEnd};
    // What ProcessMacro and CallMacroMethod do around the macro.
    ConsumeSink(system);
    draw_manager.DrawDeferred(*this);
}

namespace {

// EXP22 bit 2: the instance loop writes that may follow a draw.begin (Subsequent) in a draw
// batch. Written again with the same values they store what the registers hold and raise dirty
// flags that are raised (no deferred draw goes before an instance loop method), set draw_indexed
// again (index_buffer.count) or draw nothing (draw.end in DrawMode::Instance).
bool IsBatchMiddle(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(index_buffer.count):
    case MAXWELL3D_REG_INDEX(draw.end):
        return true;
    default:
        return false;
    }
}

// A held instance loop write (EXP22), sent as the entry it would have been. Draw manager
// methods: FrontMethodEffect's Gpu.
template <typename Write>
void SendHeldWrite(VideoCommon::GpuSplit& split, EngineInterface* engine, const Write& write) {
    constexpr auto effect = VideoCommon::GpuSplit::Effect::Gpu;
    if (write.from_macro) {
        split.MacroWrite(engine, write.method, write.argument, effect);
    } else {
        split.Method(engine, write.method, write.argument, write.is_last, nullptr, effect);
    }
}

// ApplyFrontDelta past a deferred draw, with the same-value skip and a shadow RAM that tracks
// (Track) or passes through: each write's register and its two dirty flags, raised together at
// the end (nothing reads them in between), as ProcessDirtyRegisters raises them one by one.
template <bool Track>
void ApplyDeltaWrites(Maxwell3D& maxwell3d, const u64* writes, u32 count) {
    static_assert(std::tuple_size<Maxwell3D::DirtyState::Tables>::value == 2);
    const auto& table0 = maxwell3d.dirty.tables[0];
    const auto& table1 = maxwell3d.dirty.tables[1];
    u32* const reg_array = maxwell3d.regs.reg_array.data();
    u32* const shadow_array = maxwell3d.shadow_state.reg_array.data();
    Maxwell3D::DirtyState::Flags raised;
    for (u32 i = 0; i < count; ++i) {
        const u32 method = static_cast<u32>(writes[i]) & 0xffff;
        const u32 value = static_cast<u32>(writes[i] >> 32);
        if constexpr (Track) {
            shadow_array[method] = value;
        }
        if (reg_array[method] == value) {
            continue;
        }
        if (VideoCommon::Nx2FastDrawCount::g_on) [[unlikely]] {
            maxwell3d.NoteFastDrawCountWrite(method, value);
        }
        reg_array[method] = value;
        raised[table0[method]] = true;
        raised[table1[method]] = true;
        // One follower, step 2+: as ProcessDirtyRegisters notes it.
        if (VideoCommon::OneFollower::g_track_writes.load(std::memory_order_relaxed)) [[unlikely]] {
            VideoCommon::OneFollower::NoteRegWrite(method);
        }
    }
    maxwell3d.dirty.flags |= raised;
}

// EXP22 bit 8 (back thread): every 65536th check, how many were made and how many differed.
struct Exp22Checks {
    u64 deltas;
    u64 delta_differences;
    u64 batches;
    u64 batch_differences;
};
Exp22Checks g_exp22_checks{};

void ReportExp22Checks() {
    const Exp22Checks& c = g_exp22_checks;
    if (((c.deltas + c.batches) & 0xffff) == 1) {
        LOG_INFO(HW_GPU,
                 "EXP22 check: {} state deltas checked, {} differed; {} draw batches checked, "
                 "{} differed",
                 c.deltas, c.delta_differences, c.batches, c.batch_differences);
    }
}

} // Anonymous namespace

void Maxwell3D::ApplyFrontDelta(Core::System& system, const u64* writes, u32 count) {
    // The Direct MacroWrite entries or the Pairs block the block stands for, each write as
    // CallMethod and ApplyPairs apply a state register write: the sink first, then the shadow
    // RAM, the same-value skip and ProcessDirtyRegisters. The shadow RAM mode can't change within
    // a block (its register is executable, and any other entry ends the block).
    ConsumeSink(system);
    if (VideoCommon::Nx2Profile::Enabled()) [[unlikely]] {
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::MethodWrites, count);
    }
    if (VideoCommon::FritchiiDrawKinds::g_on) [[unlikely]] {
        VideoCommon::FritchiiDrawKinds::NoteWrites(count);
    }
    const auto control = shadow_state.shadow_ram_control;
    const bool track = control == Regs::ShadowRamControl::Track ||
                       control == Regs::ShadowRamControl::TrackWithFilter;
    const bool fast = SkipSameValueWrites() &&
                      (track || control == Regs::ShadowRamControl::Passthrough);
    // One by one while a deferred instanced draw waits (the first changed write issues it), and
    // with a replaying shadow RAM or without the same-value skip.
    u32 i = 0;
    for (; i < count && (!fast || draw_manager.HasDeferredDraw()); ++i) {
        const u32 method = static_cast<u32>(writes[i]) & 0xffff;
        const u32 value = ProcessShadowRam(method, static_cast<u32>(writes[i] >> 32));
        if (!SkipSameValueWrites() || regs.reg_array[method] != value) {
            ProcessDirtyRegisters(method, value);
        }
    }
    if (i == count) {
        return;
    }
    if ((VideoCore::Nx2Debug::Experiment(22) & 8) != 0) [[unlikely]] {
        CheckFrontDelta(writes + i, count - i);
    } else if (track) {
        ApplyDeltaWrites<true>(*this, writes + i, count - i);
    } else {
        ApplyDeltaWrites<false>(*this, writes + i, count - i);
    }
}

void Maxwell3D::CheckFrontDelta(const u64* writes, u32 count) {
    // ApplyDeltaWrites, undone, then the single writes as CallMethodFast applies them (no
    // deferred draw waits here); the two results must match, and the second one stays.
    struct Saved {
        u32 reg;
        u32 shadow;
        u32 fast_reg;
        u32 fast_shadow;
    };
    thread_local std::vector<Saved> saved;
    saved.resize(count);
    const auto method_of = [writes](u32 k) { return static_cast<u32>(writes[k]) & 0xffff; };
    for (u32 k = 0; k < count; ++k) {
        saved[k].reg = regs.reg_array[method_of(k)];
        saved[k].shadow = shadow_state.reg_array[method_of(k)];
    }
    const DirtyState::Flags flags = dirty.flags;
    const auto control = shadow_state.shadow_ram_control;
    if (control == Regs::ShadowRamControl::Track ||
        control == Regs::ShadowRamControl::TrackWithFilter) {
        ApplyDeltaWrites<true>(*this, writes, count);
    } else {
        ApplyDeltaWrites<false>(*this, writes, count);
    }
    const DirtyState::Flags fast_flags = dirty.flags;
    for (u32 k = 0; k < count; ++k) {
        saved[k].fast_reg = regs.reg_array[method_of(k)];
        saved[k].fast_shadow = shadow_state.reg_array[method_of(k)];
    }
    // Newest first: a register written twice gets its value from before the block back.
    for (u32 k = count; k-- > 0;) {
        regs.reg_array[method_of(k)] = saved[k].reg;
        shadow_state.reg_array[method_of(k)] = saved[k].shadow;
    }
    dirty.flags = flags;
    for (u32 k = 0; k < count; ++k) {
        const u32 value = ProcessShadowRam(method_of(k), static_cast<u32>(writes[k] >> 32));
        if (!SkipSameValueWrites() || regs.reg_array[method_of(k)] != value) {
            ProcessDirtyRegisters(method_of(k), value);
        }
    }
    u32 first = count;
    for (u32 k = 0; k < count && first == count; ++k) {
        if (regs.reg_array[method_of(k)] != saved[k].fast_reg ||
            shadow_state.reg_array[method_of(k)] != saved[k].fast_shadow) {
            first = k;
        }
    }
    const bool differs = first != count || fast_flags != dirty.flags;
    ++g_exp22_checks.deltas;
    if (differs && g_exp22_checks.delta_differences++ < 32) {
        LOG_ERROR(HW_GPU,
                  "EXP22 check: state delta of {} writes differs: register {:#x}, dirty flags {}",
                  count, first != count ? method_of(first) : 0U,
                  fast_flags != dirty.flags ? "differ" : "match");
    }
    ReportExp22Checks();
}

void Maxwell3D::RunDrawBatch(Core::System& system, const u64* writes, u32 count, u32 repeats) {
    // The first group as its single entries ran: the sink, then CallMethod (is_last_call only
    // matters to inline data).
    const auto run_group = [&] {
        for (u32 i = 0; i < count; ++i) {
            ConsumeSink(system);
            CallMethod(system, static_cast<u32>(writes[i]) & 0xffff,
                       static_cast<u32>(writes[i] >> 32), true);
        }
    };
    run_group();
    if (repeats <= 1) {
        return;
    }
    auto& state = draw_manager.draw_state;
    if ((VideoCore::Nx2Debug::Experiment(22) & 8) == 0) [[likely]] {
        // The repeats write what the registers and shadow RAM hold, raise flags that are raised,
        // set draw_indexed again and draw nothing (IsBatchMiddle): each counts an instance in
        // its draw.begin (DrawManager::DrawBegin, Subsequent).
        state.instance_count += repeats - 1;
        return;
    }
    // EXP22 bit 8: the repeats as their single entries, against that.
    constexpr u32 MaxWrites = FrontState::MaxBatchWrites;
    const u32 checked = (std::min)(count, MaxWrites);
    const u32 expected = state.instance_count + (repeats - 1);
    const DirtyState::Flags flags = dirty.flags;
    const auto mode = state.draw_mode;
    const bool indexed = state.draw_indexed;
    const auto topology = state.topology;
    std::array<u32, MaxWrites> group_regs{};
    std::array<u32, MaxWrites> group_shadow{};
    for (u32 i = 0; i < checked; ++i) {
        const u32 method = static_cast<u32>(writes[i]) & 0xffff;
        group_regs[i] = regs.reg_array[method];
        group_shadow[i] = shadow_state.reg_array[method];
    }
    for (u32 repeat = 1; repeat < repeats; ++repeat) {
        run_group();
    }
    bool differs = state.instance_count != expected || dirty.flags != flags ||
                   state.draw_mode != mode || state.draw_indexed != indexed ||
                   state.topology != topology;
    for (u32 i = 0; i < checked; ++i) {
        const u32 method = static_cast<u32>(writes[i]) & 0xffff;
        differs |= regs.reg_array[method] != group_regs[i] ||
                   shadow_state.reg_array[method] != group_shadow[i];
    }
    ++g_exp22_checks.batches;
    if (differs && g_exp22_checks.batch_differences++ < 32) {
        LOG_ERROR(HW_GPU,
                  "EXP22 check: draw batch of {} writes x {} differs: {} instances, expected {}",
                  count, repeats, state.instance_count, expected);
    }
    ReportExp22Checks();
}

bool Maxwell3D::FrontBatchWrite(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                                u32 argument, u32 value, bool is_last_call, bool from_macro) {
    const FrontState::BatchWrite write{method, argument, is_last_call, from_macro};
    if (method == MAXWELL3D_REG_INDEX(draw.begin)) {
        // Not with a replaying shadow RAM: the back's instance id is then the argument's.
        constexpr u32 subsequent = static_cast<u32>(Regs::Draw::InstanceId::Subsequent);
        if (fs.control == Regs::ShadowRamControl::Replay || ((value >> 26) & 3) != subsequent) {
            return false;
        }
        if (fs.capture_size != 0) {
            FrontEndCapture(split, fs);
        }
        split.SetPendingFlush(&FrontFlushPending, this);
        fs.capture[0] = write;
        fs.capture_size = 1;
        return true;
    }
    if (fs.capture_size == 0 || fs.capture_size == FrontState::MaxBatchWrites ||
        !IsBatchMiddle(method)) {
        // Sent as before, after what is held (GpuSplit::SelectEngine sends that first).
        return false;
    }
    fs.capture[fs.capture_size++] = write;
    return true;
}

bool Maxwell3D::FrontCaptureRepeats(const FrontState& fs) {
    if (fs.batch_size != fs.capture_size) {
        return false;
    }
    for (u32 i = 0; i < fs.capture_size; ++i) {
        if (fs.batch[i].method != fs.capture[i].method ||
            fs.batch[i].argument != fs.capture[i].argument) {
            return false;
        }
    }
    return true;
}

void Maxwell3D::FrontEndCapture(VideoCommon::GpuSplit& split, FrontState& fs) {
    // The captured writes are one more of the held batch, or the next batch to hold.
    if (FrontCaptureRepeats(fs)) {
        ++fs.batch_repeats;
        fs.capture_size = 0;
        return;
    }
    split.DropPendingFlush(this); // The held batch goes now, by itself
    FrontEmitBatch(split, fs);
    fs.batch = fs.capture;
    fs.batch_size = fs.capture_size;
    fs.batch_repeats = 1;
    fs.capture_size = 0;
    split.SetPendingFlush(&FrontFlushPending, this);
}

void Maxwell3D::FrontEmitBatch(VideoCommon::GpuSplit& split, FrontState& fs) {
    const u32 size = fs.batch_size;
    const u32 repeats = fs.batch_repeats;
    fs.batch_size = 0;
    fs.batch_repeats = 0;
    if (size == 0) {
        return;
    }
    if (repeats > 1) {
        std::array<u64, FrontState::MaxBatchWrites> words;
        for (u32 i = 0; i < size; ++i) {
            words[i] = u64{fs.batch[i].method} | (u64{fs.batch[i].argument} << 32);
        }
        split.DrawBatch(this, words.data(), size, repeats, VideoCommon::GpuSplit::Effect::Gpu);
        return;
    }
    for (u32 i = 0; i < size; ++i) {
        SendHeldWrite(split, this, fs.batch[i]);
    }
}

void Maxwell3D::FrontEmitCapture(VideoCommon::GpuSplit& split, FrontState& fs) {
    const u32 size = fs.capture_size;
    fs.capture_size = 0;
    for (u32 i = 0; i < size; ++i) {
        SendHeldWrite(split, this, fs.capture[i]);
    }
}

void Maxwell3D::FrontFlushPending(void* maxwell3d) {
    // GpuSplit cleared the hook before the call: what is sent here does not come back to it.
    auto* const self = static_cast<Maxwell3D*>(maxwell3d);
    FrontState& fs = *self->front;
    auto& split = VideoCommon::GpuSplit::Get();
    if (fs.capture_size != 0 && FrontCaptureRepeats(fs)) {
        ++fs.batch_repeats;
        fs.capture_size = 0;
    }
    self->FrontEmitBatch(split, fs);
    self->FrontEmitCapture(split, fs);
}

void Maxwell3D::FrontCapturePacket(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                                   u32 argument, u32 value, u8 effect) {
    // EDEN_NX2_DRAW_PACKET (Fable's back-thread plan, Build C step 1). The packet is held as
    // GpuSplit's pending flush, so anything else the front sends, publishes or drains for sends
    // it first and the method order stays as with single entries.
    if (fs.packet.size() >= FrontState::MaxPacketWrites) {
        split.RunPendingFlush();
    }
    if (fs.packet.empty()) {
        split.SetPendingFlush(&Maxwell3D::FrontFlushPacket, this);
        fs.packet_effect = VideoCommon::GpuSplit::Effect::None;
    }
    fs.packet.push_back(u64{method} | (u64{argument} << 32));
    fs.packet_values.push_back(value);
    fs.packet_effect = VideoCommon::SplitWriteTracker::Combine(
        fs.packet_effect, static_cast<VideoCommon::GpuSplit::Effect>(effect));
}

void Maxwell3D::FrontFlushPacket(void* maxwell3d) {
    // GpuSplit cleared the hook before the call: the Reserve below does not come back here.
    auto* const self = static_cast<Maxwell3D*>(maxwell3d);
    FrontState& fs = *self->front;
    if (fs.packet.empty()) {
        return;
    }
    auto& split = VideoCommon::GpuSplit::Get();
    // The writes as MacroWrite sends each one (Direct, is_last): the engine is part of the
    // payload, so no engine entry is needed; the back's engine selection stays as it is.
    const size_t count = fs.packet.size();
    // EDEN_NX2_DRAW_PACKET=2: also each written register once, with the value the front's copy
    // took for it (its last write in the packet), for the back to compare after applying.
    std::array<u64, FrontState::MaxPacketWrites> checks;
    size_t check_count = 0;
    if (fs.packet_mode >= 2) [[unlikely]] {
        for (size_t i = 0; i < count; ++i) {
            const u32 method = static_cast<u32>(fs.packet[i] & 0xffff);
            const bool later = std::any_of(fs.packet.begin() + i + 1, fs.packet.end(),
                                           [method](u64 write) {
                                               return static_cast<u32>(write & 0xffff) == method;
                                           });
            if (!later) {
                checks[check_count++] = u64{method} | (u64{fs.packet_values[i]} << 32);
            }
        }
    }
    split.CallWithPayload(&Maxwell3D::ApplyDrawPacket, 3 + count + check_count, fs.packet_effect,
                          [&](u64* payload) {
                              payload[0] = reinterpret_cast<u64>(self);
                              payload[1] = reinterpret_cast<u64>(fs.system);
                              payload[2] = u64{count} | (u64{check_count} << 32);
                              std::memcpy(payload + 3, fs.packet.data(), count * sizeof(u64));
                              std::memcpy(payload + 3 + count, checks.data(),
                                          check_count * sizeof(u64));
                          },
                          // The memory stage reads these words (StageTag::Writes).
                          VideoCommon::StageTagBits(VideoCommon::StageTag::Writes));
    ++fs.packets;
    fs.packet_writes += count;
    fs.packet.clear();
    fs.packet_values.clear();
}

void Maxwell3D::ApplyDrawPacket(const u64* payload) {
    // EDEN_NX2_DRAW_PACKET on the back: what GpuSplit::Execute does for each of the packet's
    // Direct Method entries (ConsumeSink, then CallMethod), in one call. Nothing between the
    // writes can fill the method sink, so it is consumed once.
    auto* const self = reinterpret_cast<Maxwell3D*>(payload[0]);
    auto& system = *reinterpret_cast<Core::System*>(payload[1]);
    const u32 count = static_cast<u32>(payload[2]);
    const u32 checks = static_cast<u32>(payload[2] >> 32);
    const u64* const writes = payload + 3;
    {
        // GpuSplit::Execute: the fast path leaves the timing to CallMethod's own sample, the
        // other path times the entry.
        std::optional<VideoCommon::Nx2Profile::Scope> scope;
        if (!self->fast_maxwell3d) {
            scope.emplace(VideoCommon::Nx2Profile::Zone::Method);
        }
        self->ConsumeSink(system);
        for (u32 i = 0; i < count; ++i) {
            self->CallMethod(system, static_cast<u32>(writes[i] & 0xffff),
                             static_cast<u32>(writes[i] >> 32), true);
        }
    }
    if (checks == 0) [[likely]] {
        return;
    }
    // EDEN_NX2_DRAW_PACKET=2: the registers must hold what the front's copy took.
    static std::atomic<u64> checked{};
    static std::atomic<u64> differing{};
    static std::atomic<u32> logged{};
    const u64* const check = writes + count;
    for (u32 i = 0; i < checks; ++i) {
        const u32 method = static_cast<u32>(check[i] & 0xffff);
        const u32 expected = static_cast<u32>(check[i] >> 32);
        if (self->regs.reg_array[method] != expected) [[unlikely]] {
            differing.fetch_add(1, std::memory_order_relaxed);
            if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_WARNING(HW_GPU,
                            "Draw packet verify: register {:#x} is {:#x} on the back, the front "
                            "wrote {:#x} ({} writes in the packet)",
                            method, self->regs.reg_array[method], expected, count);
            }
        }
    }
    if (const u64 n = checked.fetch_add(1, std::memory_order_relaxed) + 1;
        n == 1 || n % 200000 == 0) {
        LOG_INFO(HW_GPU, "Draw packet verify: {} packets checked, {} register differences", n,
                 differing.load(std::memory_order_relaxed));
    }
}

} // namespace Tegra::Engines
