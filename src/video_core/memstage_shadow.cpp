// SPDX-License-Identifier: GPL-3.0-or-later

// Memory stage piece 5b, "the stage's inputs, on one thread" (fable_memstage_step6_spec.md 6.2),
// FRITCHII_MEMSTAGE_SHADOW=1 (2: the whole register file in compare b).
//
// What the future memory stage thread (GPUMem) will do with each ring entry, done here on the back
// thread, just before the back executes the entry (GpuSplit::BackLoop), producing nothing:
// - classify the entry as pass, work or level (spec section 3);
// - keep its own copy of each Maxwell3D's registers, shadow RAM, constant buffer bindings, root
//   tables, render-enable state and draw manager (MemStageShadowEngine), from pass and work
//   entries only;
// - run the draw manager's own code on that copy (DrawManager is a template on the engine; the
//   back's and the stage's instantiation are the same lines), which finds the stage's draws;
// - collect the ranges its constant buffer words write (computed, not written).
// A level entry is not looked at: the copies go stale, and at the next entry that needs one it is
// copied again from the back (the resync of spec 1.5), with the back's waiting method sink applied
// to it, which is what the back will apply before its next executable method.
//
// The back then issues its draws. At each one (DrawManager::ProcessDraw, before the render-enable
// test) the shadow's next draw is taken and compared:
// a  it was issued at the same entry (earlier is allowed only for a deferred draw that a write in
//    the back's method sink flushes: the back flushes it when it consumes the sink); a back draw
//    with no shadow draw at a non-level entry, or a shadow draw the back never issued, counts;
// b  the shadow's state at its draw against the back's at its own: the registers CacheInputs reads
//    and the draw's registers (all of them with =2), the constant buffer bindings, the root tables,
//    the draw manager's state, execute_on, the channel and the engine; and every dirty flag the
//    shadow raised since its last draw must be raised on the back (the back has more: the state
//    tracker raises them on every new command buffer, spec 7.4). The first 16 differences of the
//    first mismatching draws are logged, named by register;
// c  at the pipeline lookup (RasterizerVulkan::PrepareDraw): the facts table, keyed by the
//    stage's program key (the program registers, which is what the stage can see), against the
//    pipeline's facts digest; conflicts are counted per program set (the shader hashes);
// d  at FlushCaching: the device ranges the shadow's constant buffer words cover against the ones
//    the back hands InnerInvalidation. Every shadow range must be there; with no level entry since
//    the last flush, every back range must be a shadow range too.
// Every count of a to d must read 0.
//
// The counters (one line every 10 s, plus the level kinds): entries a second by kind, level runs
// a second, level entries a second by kind and channel (top 12: the 3D method, the engine, the
// closure's address in the executable, the constant buffer write that hit a watched page), draws by
// the kinds of spec 3.4, facts misses and watched-page hits. A third line counts what runs on
// the back inside a draw, between its FillDraw and its record (inside Configure, e.g. from a
// texture read): RasterizerVulkan::FlushRegion by the cache halves that ran (texture, buffer,
// query) and the collected game writes applied (OnCacheInvalidations / OnCacheInvalidation). The
// buffer half there must stay 0 in play, or such draws become level entries in step 6.
//
// What it does not cover (yet): the query and shader code pages of the watched-page filter (only
// image pages and the texture/sampler tables are tested).
//
// Piece 5c (fable_memstage_6b_plan.md sections 2 to 4 and 6.1): the classification of 6b and
// "follow".
// - Four classes. Pass and work as before; **hard** is a level entry that needs the cache on the
//   back (or does something the stage cannot know); **neutral** is another engine's plain register
//   write, which touches nothing the stage reads. Only hard entries start a run. A run starts at
//   a hard entry and ends after the last hard entry reached through pass and neutral entries only:
//   a work entry, or any entry that issues a draw, ends it.
// - Plan section 2: clear_report_value, fragment_barrier and tiled_cache_barrier are pass (the
//   store); report_semaphore.query is pass unless it is a fence report (Release with a short
//   query: hard); render_enable.mode is work with no packet when its result is fixed by the
//   registers (an override, or mode False / True: the stage sets execute_on), else hard (and
//   execute_on is taken from the back after it). Other engines: plain writes neutral, their
//   execution_mask methods hard. Closures: StageTag::Cache is hard with no copy after; ListDraw is
//   a draw loop tag (work).
// - Follow (plan 3.2): the copy is no longer thrown away at every level entry. Hard entries are
//   applied to it too (the store; falcon[4]'s scratch word; constant buffer words as before), so
//   it stays current. Only an entry that cannot be followed marks the copies stale (an untagged
//   closure, an unknown tag, a 3D macro run on the back, methods before any engine entry); the
//   next use takes the full copy from the back as in 5b (Resync), counted by cause. After a hard
//   render_enable.mode the stage takes execute_on from the back, and after a draw the back issued
//   inside a hard entry (or a GPU-counted draw whose count the rasterizer wrote into the draw
//   state) the draw manager's scalar draw state; both at the next entry, where the back has run
//   exactly what the stage has applied.
// - The cache's flags (plan 4.2): compare b looks at flags 15 to 48 only (VertexBuffers,
//   VertexBuffer0..31, IndexBuffer; the others belong to the texture cache and the state tracker).
//   The copy keeps its flags as the cache's own state: its stores raise them, a full copy takes
//   the back's, and what the back's fill clears (the flags before the fill and not after it) is
//   cleared on the copy too, except a flag the stage raised again after its draw. The cache's own
//   marks (a deleted buffer's scrub, a repack change) reach the copy through CacheInputs' mark
//   hook. At the back's draw the flags must be equal both ways when the state tracker has not
//   raised them since the back's previous draw (g_flag_invalidations), and the copy's a subset of
//   the back's otherwise. Mismatches are counted per flag.
// - SHADOW=3 (plan 6.1 step B): the real fill reads the copy. At a draw the back issues outside a
//   hard entry, whose shadow draw matched and whose copy has not changed since (a generation
//   count), the rasterizer points the buffer cache's inputs at the copy (CacheInputs holding
//   pointers, plan 5.4). Otherwise (indirect, a hard entry, a stale or moved-on copy) it reads
//   the engine. The cache's flags are then the copy's outside runs: at a run's start the back's
//   flags 15-48 are set to the copy's (the lend), and inside a run the copy takes the back's at
//   every entry (the return). Needs full vertex binds (the state tracker's raises on the back are
//   dropped at the lend): while FRITCHII_MEMSTAGE_FULL_BINDS is off it runs as 2 (counted). The
//   flag part of compare b is skipped at 3 (the copy's fills clear the copy's flags, not the
//   back's).
//
// Counters for 5c (the "Memory stage 5c" lines): runs a second by first hard kind and channel
// (top 12) and entries per run; served draws between two runs (histogram); full copies a second
// and bytes by cause; render_enable.mode by override and mode, and the hard ones; fence reports;
// other engines' hard methods; b mismatches by flag; SHADOW=3's fills from the copy and the
// reasons for the others; the OoT trackers' record against today's query. Then the gate line of
// plan 6.1: runs above 15,000 a second (at the market), full copies above 2,000 a second, or a
// render_enable.mode the registers do not decide, each fails it.
//
// Piece 6b (FRITCHII_MEMSTAGE_THREAD=2, memory_stage_link.cpp): the same code runs on GPUMem, on
// each ring entry before the back (StageProcess), and its copy is the one the fills read. The
// state below is then GPUMem's: the back's hooks (Before, BackDraw's compares, BackFill, ...) are
// off (SetRingBack), and the compares give way to the ring's own checks. What changes on the stage:
// - constant buffer words of a served entry are written to game memory here (WriteBlockStage, the
//   bytes the back's cached write writes) and their device ranges collected (StageRanges): the
//   next fill applies their buffer half, a lend hands them to the back. In a run, or after the
//   entry turned hard, they only advance the offset (the back writes them);
// - bind_groups binds and the Channel closure act on the buffer cache through the ring backend
//   (the back skips them in served mode);
// - a draw the copy's draw manager issues with execute_on, outside a run and before the entry
//   turned hard, is filled at once (the ring backend's fill_draw, from the copy) and its packet
//   goes to ring B with the draw's sequence number (every draw counts, on both threads, from the
//   ring's start). A draw the stage cannot serve makes its entry hard: indirect draws, list draws
//   (served with the window open and FRITCHII_MEMSTAGE_LIST_DRAWS on, unless the back reads their
//   words back), GPU-counted draws, draws while the back asks for level draws (diagnostics),
//   facts misses and texel buffers (fill_draw says so);
// - StageLend hands the cache's flags 15-48 to the back's engines, StageReturn takes them back,
//   with execute_on (5c's take), the scalar draw state when the back drew in the run, and full
//   copies of the copies an unfollowed entry left stale. The back is parked at both moments.

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include <boost/container/small_vector.hpp>

#include "common/fritchii_write_ring.h"
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/buffer_cache/cache_inputs.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/kepler_memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/memory_stage_link.h"
#include "video_core/memstage_shadow.h"
#include "video_core/memstage_shadow_engine.h"
#include "video_core/nx2.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_indirect_grid.h"
#include "video_core/nx2_profile.h"
#include "video_core/page_filter.h"
#include "video_core/query_cache/types.h"

namespace VideoCommon {

/// Friend of GpuSplit and Maxwell3D: the ring's layout and the back's engine settings.
struct MemStageShadowAccess {
    using Maxwell3D = Tegra::Engines::Maxwell3D;
    using Kind = GpuSplit::Kind;
    static constexpr u64 IsLast = GpuSplit::IsLast;
    static constexpr u64 HasSegment = GpuSplit::HasSegment;
    static constexpr u64 Direct = GpuSplit::Direct;
    static constexpr u64 Sink = GpuSplit::Sink;
    static constexpr u64 CbVerify = GpuSplit::CbVerify;
    static constexpr size_t RingWords = GpuSplit::RingWords;
    static constexpr size_t RingMask = GpuSplit::RingMask;

    static const u64* Ring() {
        return GpuSplit::Get().ring.get();
    }
    /// How far the front has published (GPUMem may read the entries before it).
    static u64 Published() {
        return GpuSplit::Get().published.load(std::memory_order_acquire);
    }
    static bool SkipSame(const Maxwell3D& maxwell3d) {
        return maxwell3d.skip_same_value_writes;
    }
    static bool FrontMacros(const Maxwell3D& maxwell3d) {
        return maxwell3d.front_macros;
    }
    static Tegra::Engines::EngineInterface* BackEngine() {
        return GpuSplit::Get().back_engine;
    }
};

namespace MemStageShadow {
namespace {

using Access = MemStageShadowAccess;
using Kind = Access::Kind;
using Maxwell3D = Tegra::Engines::Maxwell3D;
using Regs = Maxwell3D::Regs;
using Engine = MemStageShadowEngine;
using Flags = Maxwell3D::DirtyState::Flags;

constexpr u32 MacroRegistersStart = 0xE00;
constexpr u32 CbData = MAXWELL3D_REG_INDEX(const_buffer.buffer);

thread_local bool t_on_back{};
const PageFilter* g_watch_filter{};
int g_mode{};

/// Level is "hard" in 5c's words (plan section 2): the entry needs the cache on the back.
enum class Class : u8 { Pass, Work, Level, Neutral };

/// What a level entry is, for the by-kind counter.
enum class LevelType : u8 {
    Method3D,   ///< id: the method
    Macro3D,    ///< id: the macro method
    Packet3D,   ///< a tagged draw packet holding a level method; id: the method
    Batch3D,    ///< a draw batch holding a level method; id: the method
    Engine,     ///< (5b) methods on another engine; id: EngineType
    NoEngine,   ///< methods before any engine entry
    Call,       ///< an untagged closure; id: its address in the executable
    CbWatched,  ///< constant buffer words onto a watched page
    CbMode,     ///< constant buffer words with EXP107 (the stage implements the default mode)
    UnknownTag, ///< a tag the shadow does not know
    // 5c
    FenceReport,  ///< report_semaphore.query, a fence (Release, short query)
    Condition,    ///< render_enable.mode the registers do not decide; id: override << 8 | mode
    EngineMethod, ///< another engine's executable method; id: EngineType << 16 | method
    CacheCall,    ///< a closure tagged StageTag::Cache; id: its address in the executable
    // 6b
    LevelDraw,    ///< a draw the stage cannot serve (THREAD = 2); id: the kind
    // 7
    CbGpuWritten, ///< constant buffer words onto GPU-written buffer data, the window open
};
enum class EngineType : u32 { KeplerCompute, KeplerMemory, MaxwellDMA, Fermi2D, Other };

/// The cache's flags (plan 4.2): VertexBuffers, VertexBuffer0..31, IndexBuffer.
constexpr u32 CacheFlagFirst = VideoCommon::Dirty::VertexBuffers;
constexpr u32 CacheFlagLast = VideoCommon::Dirty::IndexBuffer;
constexpr u32 CacheFlagCount = CacheFlagLast - CacheFlagFirst + 1;
static_assert(CacheFlagCount == 34);

const Maxwell3D::DirtyState::Flags& CacheFlagMask() {
    static const Maxwell3D::DirtyState::Flags mask = [] {
        Maxwell3D::DirtyState::Flags bits;
        for (u32 flag = CacheFlagFirst; flag <= CacheFlagLast; ++flag) {
            bits[flag] = true;
        }
        return bits;
    }();
    return mask;
}

u64 LevelKey(LevelType type, s32 channel, u64 id) {
    return (u64{static_cast<u8>(type)} << 56) | (u64{static_cast<u8>(channel)} << 48) |
           (id & 0xffff'ffff'ffffULL);
}

const char* MethodName(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(wait_for_idle):
        return "wait_for_idle";
    case MAXWELL3D_REG_INDEX(falcon[4]):
        return "falcon[4]";
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
        return "report_semaphore.query";
    case MAXWELL3D_REG_INDEX(render_enable.mode):
        return "render_enable.mode";
    case MAXWELL3D_REG_INDEX(clear_report_value):
        return "clear_report_value";
    case MAXWELL3D_REG_INDEX(sync_info):
        return "sync_info";
    case MAXWELL3D_REG_INDEX(launch_dma):
        return "launch_dma";
    case MAXWELL3D_REG_INDEX(inline_data):
        return "inline_data";
    case MAXWELL3D_REG_INDEX(fragment_barrier):
        return "fragment_barrier";
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
        return "invalidate_texture_data_cache";
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return "tiled_cache_barrier";
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
        return "load_mme.instruction_ptr";
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
        return "load_mme.instruction";
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
        return "load_mme.start_address";
    default:
        return "";
    }
}

const char* EngineName(u64 type) {
    switch (static_cast<EngineType>(type)) {
    case EngineType::KeplerCompute:
        return "KeplerCompute";
    case EngineType::KeplerMemory:
        return "KeplerMemory";
    case EngineType::MaxwellDMA:
        return "MaxwellDMA";
    case EngineType::Fermi2D:
        return "Fermi2D";
    default:
        return "other engine";
    }
}

std::string LevelName(u64 key) {
    const auto type = static_cast<LevelType>(key >> 56);
    const s32 channel = static_cast<s8>((key >> 48) & 0xff);
    const u64 id = key & 0xffff'ffff'ffffULL;
    std::string what;
    switch (type) {
    case LevelType::Method3D:
        what = fmt::format("3D {:#x} {}", id, MethodName(static_cast<u32>(id)));
        break;
    case LevelType::Macro3D:
        what = fmt::format("3D macro {:#x} (on the back)", id);
        break;
    case LevelType::Packet3D:
        what = fmt::format("draw packet with 3D {:#x} {}", id, MethodName(static_cast<u32>(id)));
        break;
    case LevelType::Batch3D:
        what = fmt::format("draw batch with 3D {:#x} {}", id, MethodName(static_cast<u32>(id)));
        break;
    case LevelType::Engine:
        what = fmt::format("engine {}", EngineName(id));
        break;
    case LevelType::NoEngine:
        what = "methods before an engine entry";
        break;
    case LevelType::Call:
        what = fmt::format("call +{:#x}", id);
        break;
    case LevelType::CbWatched:
        what = "constant buffer words on a watched page";
        break;
    case LevelType::CbMode:
        what = "constant buffer words, EXP107 on";
        break;
    case LevelType::UnknownTag:
        what = fmt::format("call with stage tag {}", id);
        break;
    case LevelType::FenceReport:
        what = "3D 0x6c3 report_semaphore.query, fence";
        break;
    case LevelType::Condition:
        what = fmt::format("3D 0x556 render_enable.mode {} (override {})", id & 0xff, id >> 8);
        break;
    case LevelType::EngineMethod:
        what = fmt::format("{} method {:#x}", EngineName(id >> 16), id & 0xffff);
        break;
    case LevelType::CacheCall:
        what = fmt::format("cache call +{:#x}", id);
        break;
    case LevelType::LevelDraw: {
        // MemStageShadowEngine::StageDraw's ids (RingLevel).
        static constexpr std::array<const char*, 11> names{
            "indirect", "GPU count", "texel buffers", "forced",
            "fill refused", "no room", "late", "other",
            "list with the window closed or the switch off", "list read back",
            "list fill refused"};
        what = fmt::format("level draw ({})", id < names.size() ? names[id] : "?");
        break;
    }
    case LevelType::CbGpuWritten:
        what = "constant buffer words onto GPU-written data (the window open)";
        break;
    }
    return fmt::format("{} ch{}", what, channel);
}

/// A device range a constant buffer write covers (spec 3.2's "collect the range").
struct Range {
    DAddr begin;
    u64 size;
    const Tegra::MemoryManager* memory;
    u64 position; ///< Of the entry that wrote it
};

/// The compared state: segments of words in a fixed order (Visit), named for the diff.
enum class Segment : u8 {
    Registers, ///< reg_base: the register of the first word
    ShadowRam,
    ConstBuffers, ///< 4 words per binding (address low, high, size, enabled), stage * 18 + slot
    RootTables,
    RootTableState,
    DrawManagerState,
    Indirect,
    Execute,
    Channel,
    Engine,
};

struct SegmentInfo {
    const char* name;
    Segment segment;
    u32 reg_base;
    u32 words;
};

#define FRITCHII_SHADOW_REGS(field)                                                                \
    SegmentInfo {                                                                                  \
        #field, Segment::Registers, static_cast<u32>(MAXWELL3D_REG_INDEX(field)),                  \
            static_cast<u32>(sizeof(Regs::field) / sizeof(u32))                                    \
    }

constexpr u32 DrawStateWords = 26;
constexpr u32 IndirectWords = 12;
constexpr u32 CbufWords = Regs::MaxShaderStage * Regs::MaxConstBuffers * 4;

/// The registers of compare b with FRITCHII_MEMSTAGE_SHADOW=1: the ones CacheInputs exposes
/// (vertex streams, limits, attribute formats for the repack layout, native stream sizes,
/// transform feedback) and the ones a draw and its facts key read.
const std::vector<SegmentInfo>& Segments(bool full) {
    static const std::vector<SegmentInfo> narrow = [] {
        std::vector<SegmentInfo> list{
            FRITCHII_SHADOW_REGS(vertex_streams),
            FRITCHII_SHADOW_REGS(vertex_stream_limits),
            FRITCHII_SHADOW_REGS(vertex_attrib_format),
            SegmentInfo{"native stream sizes", Segment::Registers, 0x600 / 4, 64},
            FRITCHII_SHADOW_REGS(index_buffer),
            FRITCHII_SHADOW_REGS(vertex_buffer),
            FRITCHII_SHADOW_REGS(transform_feedback_enabled),
            FRITCHII_SHADOW_REGS(transform_feedback),
            FRITCHII_SHADOW_REGS(pipelines),
            FRITCHII_SHADOW_REGS(program_region),
            FRITCHII_SHADOW_REGS(const_buffer),
            FRITCHII_SHADOW_REGS(bind_groups),
            FRITCHII_SHADOW_REGS(global_base_vertex_index),
            FRITCHII_SHADOW_REGS(global_base_instance_index),
            FRITCHII_SHADOW_REGS(draw),
            FRITCHII_SHADOW_REGS(render_enable),
            FRITCHII_SHADOW_REGS(render_enable_override),
            FRITCHII_SHADOW_REGS(primitive_topology_control),
            FRITCHII_SHADOW_REGS(topology_override),
            FRITCHII_SHADOW_REGS(tex_header),
            FRITCHII_SHADOW_REGS(tex_sampler),
            FRITCHII_SHADOW_REGS(sampler_binding),
        };
        return list;
    }();
    static const std::vector<SegmentInfo> whole{
        SegmentInfo{"regs", Segment::Registers, 0, Regs::NUM_REGS},
        SegmentInfo{"shadow RAM", Segment::ShadowRam, 0, Regs::NUM_REGS},
    };
    static const std::vector<SegmentInfo> rest{
        SegmentInfo{"constant buffer bindings", Segment::ConstBuffers, 0, CbufWords},
        SegmentInfo{"root tables", Segment::RootTables, 0, 8 * 64},
        SegmentInfo{"root table select", Segment::RootTableState, 0, 3},
        SegmentInfo{"draw state", Segment::DrawManagerState, 0, DrawStateWords},
        SegmentInfo{"indirect state", Segment::Indirect, 0, IndirectWords},
        SegmentInfo{"execute_on", Segment::Execute, 0, 1},
        SegmentInfo{"channel", Segment::Channel, 0, 1},
        SegmentInfo{"engine", Segment::Engine, 0, 2},
    };
    static const std::vector<SegmentInfo> narrow_all = [] {
        std::vector<SegmentInfo> list = narrow;
        list.insert(list.end(), rest.begin(), rest.end());
        return list;
    }();
    static const std::vector<SegmentInfo> full_all = [] {
        std::vector<SegmentInfo> list = whole;
        list.insert(list.end(), rest.begin(), rest.end());
        return list;
    }();
    return full ? full_all : narrow_all;
}

#undef FRITCHII_SHADOW_REGS

u64 HashBytes(const void* data, size_t size, u64 hash = 0xcbf29ce484222325ULL) {
    const auto* bytes = static_cast<const u8*>(data);
    for (size_t i = 0; i < size; ++i) {
        hash = (hash ^ bytes[i]) * 0x100000001b3ULL;
    }
    return hash;
}

/// Calls `emit(info, words)` for each segment of Segments(full), with the segment's words from
/// `e` (Maxwell3D or the shadow's copy: the same names). The back and the shadow call it alike.
template <typename E, typename Emit>
void Visit(const E& e, bool full, s32 channel, const Maxwell3D* engine, Emit&& emit) {
    std::array<u32, 8 * 64> scratch{};
    for (const SegmentInfo& info : Segments(full)) {
        switch (info.segment) {
        case Segment::Registers:
            emit(info, &e.regs.reg_array[info.reg_base]);
            break;
        case Segment::ShadowRam:
            emit(info, &e.shadow_state.reg_array[0]);
            break;
        case Segment::ConstBuffers: {
            std::array<u32, CbufWords> words{};
            size_t w = 0;
            for (const auto& stage : e.state.shader_stages) {
                for (const auto& buffer : stage.const_buffers) {
                    words[w++] = static_cast<u32>(buffer.address);
                    words[w++] = static_cast<u32>(buffer.address >> 32);
                    words[w++] = buffer.size;
                    words[w++] = buffer.enabled ? 1 : 0;
                }
            }
            emit(info, words.data());
            break;
        }
        case Segment::RootTables:
            for (size_t t = 0; t < 8; ++t) {
                std::memcpy(&scratch[t * 64], e.root_tables.tables[t].data(), 64 * sizeof(u32));
            }
            emit(info, scratch.data());
            break;
        case Segment::RootTableState: {
            const std::array<u32, 3> words{e.root_tables.selected, e.root_tables.offset,
                                           e.root_tables.loaded ? 1U : 0U};
            emit(info, words.data());
            break;
        }
        case Segment::DrawManagerState: {
            const auto& dm = e.draw_manager;
            const auto& s = dm.draw_state;
            std::array<u32, DrawStateWords> words{};
            static_assert(sizeof(s.vertex_buffer) <= 4 * sizeof(u32));
            static_assert(sizeof(s.index_buffer) <= 10 * sizeof(u32));
            words[0] = static_cast<u32>(s.topology);
            words[1] = static_cast<u32>(s.draw_mode);
            words[2] = s.draw_indexed ? 1 : 0;
            words[3] = s.draw_element_count;
            words[4] = s.base_index;
            words[5] = s.base_instance;
            words[6] = s.instance_count;
            words[7] = s.drawn_instances;
            std::memcpy(&words[8], &s.vertex_buffer, sizeof(s.vertex_buffer));
            std::memcpy(&words[12], &s.index_buffer, sizeof(s.index_buffer));
            const u64 inline_hash =
                HashBytes(s.inline_index_draw_indexes.data(), s.inline_index_draw_indexes.size());
            words[22] = static_cast<u32>(inline_hash) ^ static_cast<u32>(inline_hash >> 32);
            words[23] = static_cast<u32>(s.inline_index_draw_indexes.size());
            words[24] = static_cast<u32>(s.list_draw_address ^ (s.list_draw_address >> 32));
            words[25] = (dm.fold_first_instance ? 1U : 0U) | (dm.first_folded ? 2U : 0U);
            emit(info, words.data());
            break;
        }
        case Segment::Indirect: {
            const auto& in = e.draw_manager.indirect_state;
            const std::array<u32, IndirectWords> words{
                in.is_byte_count ? 1U : 0U,
                in.is_indexed ? 1U : 0U,
                in.include_count ? 1U : 0U,
                static_cast<u32>(in.count_start_address),
                static_cast<u32>(in.count_start_address >> 32),
                static_cast<u32>(in.indirect_start_address),
                static_cast<u32>(in.indirect_start_address >> 32),
                static_cast<u32>(in.buffer_size),
                static_cast<u32>(in.max_draw_counts),
                static_cast<u32>(in.stride),
                in.list_words ? 1U : 0U,
                static_cast<u32>(in.list_keep_id),
            };
            emit(info, words.data());
            break;
        }
        case Segment::Execute: {
            const u32 word = e.ShouldExecute() ? 1 : 0;
            emit(info, &word);
            break;
        }
        case Segment::Channel: {
            const u32 word = static_cast<u32>(channel);
            emit(info, &word);
            break;
        }
        case Segment::Engine: {
            const u64 pointer = reinterpret_cast<u64>(engine);
            const std::array<u32, 2> words{static_cast<u32>(pointer),
                                           static_cast<u32>(pointer >> 32)};
            emit(info, words.data());
            break;
        }
        }
    }
}

/// The name of word `index` of segment `info`, for the diff.
std::string WordName(const SegmentInfo& info, u32 index) {
    switch (info.segment) {
    case Segment::Registers:
        return fmt::format("{}+{} (register {:#x})", info.name, index, info.reg_base + index);
    case Segment::ShadowRam:
        return fmt::format("shadow RAM register {:#x}", index);
    case Segment::ConstBuffers: {
        static constexpr const char* fields[4]{"address low", "address high", "size", "enabled"};
        const u32 binding = index / 4;
        return fmt::format("constant buffer stage {} slot {} {}", binding / Regs::MaxConstBuffers,
                           binding % Regs::MaxConstBuffers, fields[index % 4]);
    }
    case Segment::RootTables:
        return fmt::format("root table {} word {}", index / 64, index % 64);
    case Segment::DrawManagerState: {
        static constexpr const char* fields[]{
            "topology",        "draw_mode",       "draw_indexed",  "draw_element_count",
            "base_index",      "base_instance",   "instance_count", "drawn_instances"};
        if (index < 8) {
            return fmt::format("draw state {}", fields[index]);
        }
        if (index < 12) {
            return fmt::format("draw state vertex_buffer+{}", index - 8);
        }
        if (index < 22) {
            return fmt::format("draw state index_buffer+{}", index - 12);
        }
        static constexpr const char* tail[]{"inline index hash", "inline index bytes",
                                            "list draw address", "fold flags"};
        return fmt::format("draw state {}", tail[index - 22]);
    }
    default:
        return fmt::format("{}+{}", info.name, index);
    }
}

/// The stage's program key for the facts table: what selects the programs, as the stage sees it
/// (the program region and every pipeline slot's registers, native addresses included).
/// Step 7: an xxh64-style hash of 8-byte words (four lanes, xxh64's round, merge and
/// avalanche). The byte loop was 392 dependent multiplies, about 0.4 us for every served draw on
/// the back (its facts publish) and again for every fill on GPUMem; both sides use this one
/// function, so their keys stay equal. The round's rotate brings high bits down, so differences
/// in the top bits of words do not cancel (a plain xor-multiply lane lost them).
// ProgramWordCount (memstage_shadow.h): the six pipeline slots (48 words), then the region.
template <typename E>
void ProgramWords(const E& e, std::array<u64, ProgramWordCount>& words) {
    static_assert(sizeof(e.regs.pipelines) == 48 * sizeof(u64));
    static_assert(sizeof(e.regs.program_region) == sizeof(u64));
    std::memcpy(words.data(), &e.regs.pipelines, sizeof(e.regs.pipelines));
    std::memcpy(words.data() + 48, &e.regs.program_region, sizeof(u64));
}

constexpr u64 XxhPrime1 = 0x9E3779B185EBCA87ULL;
constexpr u64 XxhPrime2 = 0xC2B2AE3D27D4EB4FULL;
constexpr u64 XxhPrime3 = 0x165667B19E3779F9ULL;
constexpr u64 XxhPrime4 = 0x85EBCA77C2B2AE63ULL;

constexpr u64 XxhRound(u64 acc, u64 input) {
    acc += input * XxhPrime2;
    acc = std::rotl(acc, 31);
    return acc * XxhPrime1;
}

constexpr u64 XxhMerge(u64 acc, u64 value) {
    acc ^= XxhRound(0, value);
    return acc * XxhPrime1 + XxhPrime4;
}

u64 HashProgramWords(const std::array<u64, ProgramWordCount>& words) {
    std::array<u64, 4> lanes{XxhPrime1 + XxhPrime2, XxhPrime2, 0, 0 - XxhPrime1};
    for (size_t i = 0; i < 48; i += 4) {
        for (size_t lane = 0; lane < 4; ++lane) {
            lanes[lane] = XxhRound(lanes[lane], words[i + lane]);
        }
    }
    u64 hash = std::rotl(lanes[0], 1) + std::rotl(lanes[1], 7) + std::rotl(lanes[2], 12) +
               std::rotl(lanes[3], 18);
    for (const u64 lane : lanes) {
        hash = XxhMerge(hash, lane);
    }
    hash += ProgramWordCount * sizeof(u64);
    // The last word (the program region), as xxh64's 8-byte tail; then its avalanche.
    hash ^= XxhRound(0, words[48]);
    hash = std::rotl(hash, 27) * XxhPrime1 + XxhPrime4;
    hash ^= hash >> 33;
    hash *= XxhPrime2;
    hash ^= hash >> 29;
    hash *= XxhPrime3;
    hash ^= hash >> 32;
    return hash;
}

template <typename E>
u64 ProgramKey(const E& e) {
    std::array<u64, ProgramWordCount> words;
    ProgramWords(e, words);
    return HashProgramWords(words);
}

/// One draw the shadow issued, waiting for the back's.
struct Record {
    u64 position{};
    bool sunk{};      ///< flushed by a write the back keeps in its method sink
    bool indirect{};  ///< ProcessDrawIndirect
    bool executes{};  ///< execute_on at the draw
    Maxwell3D* engine{};
    s32 channel{};
    u64 program_key{};
    Flags dirty{};        ///< The copy's flags at the draw
    Flags raised_before{}; ///< Raised on the copy since its previous draw
    u64 generation{};     ///< The copy's generation at the draw (SHADOW=3)
    bool hard{};          ///< Issued inside a hard entry
    std::vector<u32> words;
    std::vector<Range> ranges; ///< Written since the shadow's previous draw
};

/// One Maxwell3D's copy.
struct Copy {
    Maxwell3D* back{};
    std::unique_ptr<Engine> e;
    Tegra::MemoryManager* memory{};
    const Maxwell3D::DirtyState::Tables* tables{};
    bool skip_same{};
    bool front_macros{};
    bool stale{true};
    /// Flags raised on the copy since its last draw (StageDraw hands them to the pending records).
    Flags raised{};
    /// Bumped at every change to what the fill reads (SHADOW=3: a draw whose copy moved on since
    /// the shadow's draw reads the engine instead).
    u64 generation{};
    /// Step 7 fps fix 4: ResolveStage of the last 4 KiB GPU page a constant buffer write went to,
    /// at the page's start, and the memory map counter it was looked up under (CbTarget).
    u64 cb_page{~u64{0}};
    u64 cb_map_serial{};
    Tegra::MemoryManager::StageTarget cb_target{};
};

/// Histogram buckets of served draws between two runs: 0, 1, 2-3, 4-7, 8-15, 16+.
constexpr size_t GapBuckets = 6;
size_t GapBucket(u64 draws) {
    if (draws == 0) {
        return 0;
    }
    const size_t bit = static_cast<size_t>(std::bit_width(draws)); // 1 -> 1, 2-3 -> 2, ...
    return (std::min)(bit, GapBuckets - 1);
}

/// SHADOW=3: why a draw's fill read the engine and not the copy.
enum class FillFrom : u8 {
    Copy,
    NoRecord,   ///< the back's draw matched no shadow draw
    Hard,       ///< issued inside a hard entry (the back fills in a run)
    Indirect,   ///< an indirect draw (level in 6b)
    Stale,      ///< the copy is stale (after an entry that cannot be followed)
    MovedOn,    ///< the copy changed after the shadow's draw (later writes of the same entry)
    Other,      ///< another engine or channel than the shadow's draw
    NoFullBinds, ///< FRITCHII_MEMSTAGE_FULL_BINDS off
    Count,
};

/// Counters for one 10 s window.
struct Counters {
    u64 pass{}, work{}, level{};
    u64 level_runs{};
    u64 resyncs{};
    u64 resync_bytes{};
    // Draws by kind (spec 3.4) and how they were compared.
    u64 back_draws{}, compared{}, served{}, render_off{}, indirect{}, list{}, gpu_count{};
    u64 texel{}, diagnostic{}, host_conditional{}, level_draws{}, sunk{};
    // Compare a.
    u64 a_missing{}, a_extra{}, a_position{};
    // Compare b.
    u64 b_mismatch{}, b_dirty{};
    // Compare c.
    u64 facts_checked{}, facts_misses{}, key_changes{}, c_conflicts{}, facts_unchecked{};
    // Compare d.
    u64 flushes{}, flushes_strict{}, d_missing{}, d_extra{};
    // Join 2.
    u64 watched_hits{};
    // Inside a draw (between its fill and its record) on the back: FlushRegion calls by the caches
    // whose half ran, and collected game writes applied (fable_memstage_readlog1.md).
    u64 draw_flushes{}, draw_flush_texture{}, draw_flush_buffer{}, draw_flush_query{};
    u64 draw_gathers{}, draw_gather_ranges{}, draw_single_invalidations{};
    std::unordered_map<u64, u64> level_kinds;
    // 5c.
    u64 neutral{};
    u64 run_entries{};     ///< Entries of the runs that ended in this window
    u64 runs_ended{};
    std::array<u64, GapBuckets> gaps{}; ///< Served draws between two runs
    u64 served_draws{};    ///< Shadow draws outside hard entries with execute_on
    std::unordered_map<u64, u64> run_kinds;   ///< Runs by their first hard entry's kind
    std::unordered_map<u64, u64> copy_causes; ///< Full copies by the entry that made them stale
    std::unordered_map<u32, u64> conditions;  ///< render_enable.mode: override << 8 | mode
    u64 condition_hard{};
    u64 fence_reports{}, reports{};
    u64 macros_on_back{};
    u64 execute_takes{}, draw_state_takes{}, draw_state_written{};
    std::array<u64, CacheFlagCount> flag_missing{}; ///< Raised on the copy, not on the back
    std::array<u64, CacheFlagCount> flag_extra{};   ///< On the back, not the copy (no invalidation)
    u64 flag_compares{}, flag_compares_strict{}, flag_settles{};
    std::array<u64, static_cast<size_t>(FillFrom::Count)> fill_from{};
    u64 lends{}, returns{};
    // The OoT trackers' record (plan 5.1) against today's query.
    u64 tracker_compared{}, tracker_mismatches{}, tracker_without{};
};

struct State {
    u64 cursor{};
    u64 current{};             ///< The entry the back executes now
    bool current_level{};      ///< ... and whether the shadow classified it level
    bool in_level_run{};
    bool level_since_flush{};
    bool processing_sunk{};    ///< Applying writes the back keeps in its sink
    u64 processing_position{}; ///< The entry being applied
    Tegra::Engines::EngineInterface* engine{};
    Copy* engine_copy{}; ///< When `engine` is a Maxwell3D
    EngineType engine_type{EngineType::Other}; ///< Otherwise
    s32 channel{};
    s32 back_channel{};
    bool host_condition{};
    bool channel_stale{};
    std::deque<Copy> copies; ///< A deque: pointers to its elements stay valid
    std::deque<Record> records;
    std::vector<Record> spare;
    bool in_draw_window{};      ///< Between the back's FillDraw and its record
    bool has_current_draw{};    ///< The back's draw matched a shadow draw (for c) ...
    u64 current_program_key{};  ///< ... with the shadow's program key
    std::vector<Range> ranges_since_draw;
    std::vector<Range> due; ///< Ranges of draws the back issued, not flushed yet
    std::unordered_map<u64, u64> program_sets; ///< Program key -> shader set key
    std::unordered_map<u64, u64> facts;        ///< Shader set key -> facts digest
    // 5c: the entry being processed.
    bool entry_hard{};        ///< A hard entry (plan section 2)
    u64 entry_key{};          ///< ... its first hard kind
    bool entry_unfollowed{};  ///< It cannot be followed: the copies go stale
    LevelType entry_type{LevelType::Method3D}; ///< How a hard 3D method in it is named
    u32 entry_draws{};        ///< Shadow draws it issued
    u32 entry_served{};       ///< ... with execute_on
    u32 entry_launches{};     ///< Compute launches in it (0xaf, 0xb0, 0x107; StageResult)
    u64 stale_cause{};        ///< The entry that last made the copies stale (full copy counter)
    // Runs (plan 3.2).
    bool in_run{};
    u64 run_entries{};        ///< Up to its last hard entry
    u64 run_trailing{};       ///< Pass and neutral entries after its last hard entry
    u64 served_since_run{};
    // Taken from the back at the next entry (plan 3.3).
    Copy* take_execute{};
    Copy* take_draw_state{};
    // The back's draw being prepared: its record's flags, for the fill's clears (plan 4.2).
    Copy* match_copy{};       ///< The copy of the back's current draw's record, or null
    u64 match_generation{};
    bool match_hard{};
    bool match_indirect{};
    s32 match_channel{};
    bool fill_pending{};      ///< Between BackFill and the settle
    Flags fill_pre{};         ///< The back's cache flags at BackFill (and marks made since)
    Maxwell3D* fill_engine{};
    bool fill_from_copy{};
    u64 invalidations_seen{}; ///< g_flag_invalidations at the back's previous draw
    bool full_binds{};        ///< SHADOW=3: FRITCHII_MEMSTAGE_FULL_BINDS at the last check
    // 6b (THREAD = 2): this state is GPUMem's.
    bool ring{};              ///< StageProcess runs it on GPUMem
    bool ring_in_run{};       ///< The entry is followed inside a level run
    u64 ring_seq{};           ///< Draws the copies issued since the ring started
    u32 ring_packets{};       ///< Packets the entry sent
    bool ring_ran_on_back{};  ///< The back ran its gather or semaphore release (StageResult)
    bool ring_publish_now{};  ///< GPUMem passed it (FRITCHII_MEMSTAGE_PASS; StageResult)
    // fable_stage_levelpoints 6.1: the mirror of the back's query_cache.HasPendingSync() at
    // GPUMem's place in the stream. A sync may be pending unless `sync_unknown` is false and no
    // report went by since the last sync point. The alternative mirror (P0's counter only) is
    // also cleared by a lone sync point that runs (the back runs its wait for idle at the
    // entry), and a run's return leaves it.
    bool sync_unknown{true};
    u32 sync_reports{};
    bool alt_unknown{true};
    u32 alt_reports{};
    /// The entry is a lone 0x3dd or 0x44: what its end does to the mirror.
    enum class EntrySync : u8 { None, Cleared, Ran } entry_sync{EntrySync::None};
    /// Dry (the switch off): the entry would have passed; the run it starts is not one P1 has.
    bool entry_dry_pass{};
    bool run_from_dry_pass{};
    // fable_stage_p3 4.1, for its dry counters (section 6): the device pages (4 KiB) of the
    // reports GPUMem saw since the last sync point, and whether others may be pending. Kept as
    // P3 would have it (an entry P3 would pass does not count as hard or as a run).
    bool p3_unknown{true};
    boost::container::small_vector<u64, 16> p3_pages;
    enum class P3Effect : u8 { None, Cleared, Kept } entry_p3{P3Effect::None};
    bool run_from_p3_pass{};
    /// The last report's GPU page, its device page (~0: unmapped), the address space and the
    /// map counter it was looked up under.
    u64 report_gpu_page{~u64{0}};
    u64 report_device_page{};
    const Tegra::MemoryManager* report_memory{};
    u64 report_map_serial{};
    Class last_class{Class::Pass};
    std::vector<std::pair<u64, std::size_t>> ring_ranges; ///< Collected, not applied yet
    /// Step 7: the flags a deferred lend hands over, per engine (BackTakeLend).
    std::vector<std::pair<Maxwell3D*, Flags>> lend_flags;
    Counters counters;
    Counters totals;
    std::chrono::steady_clock::time_point window_start{};
    u64 entries_since_check{};
    u32 b_logged{}, a_logged{}, c_logged{}, d_logged{}, t_logged{};
#ifdef _WIN32
    u64 module_base{};
#endif
};

State& S() {
    static State state;
    return state;
}

/// The entry being processed is hard (a level entry); the first kind names it.
void Hard(State& s, LevelType type, u64 id) {
    if (!s.entry_hard) {
        s.entry_hard = true;
        s.entry_key = LevelKey(type, s.channel, id);
    }
}

/// The entry cannot be followed: hard, and the copies go stale after it (plan 3.2).
void Unfollow(State& s, LevelType type, u64 id) {
    Hard(s, type, id);
    if (!s.entry_unfollowed) {
        s.entry_unfollowed = true;
        s.stale_cause = LevelKey(type, s.channel, id);
    }
}

/// The draw manager's scalar draw state, the inline index bytes left out (plan 3.3).
void CopyScalarDrawState(Maxwell3D::DrawManager::State& to,
                         const Maxwell3D::DrawManager::State& from) {
    to.topology = from.topology;
    to.draw_mode = from.draw_mode;
    to.draw_indexed = from.draw_indexed;
    to.draw_element_count = from.draw_element_count;
    to.base_index = from.base_index;
    to.vertex_buffer = from.vertex_buffer;
    to.index_buffer = from.index_buffer;
    to.base_instance = from.base_instance;
    to.instance_count = from.instance_count;
    to.drawn_instances = from.drawn_instances;
    to.list_draw_address = from.list_draw_address;
    to.list_keep_id = from.list_keep_id;
}

Copy* FindCopy(State& s, Maxwell3D* back) {
    for (Copy& copy : s.copies) {
        if (copy.back == back) {
            return &copy;
        }
    }
    Copy copy;
    copy.back = back;
    copy.e = std::make_unique<Engine>();
    copy.e->back = back;
    copy.memory = &back->GetMemoryManager();
    copy.tables = &back->dirty.tables;
    copy.skip_same = Access::SkipSame(*back);
    copy.front_macros = Access::FrontMacros(*back);
    s.copies.push_back(std::move(copy));
    return &s.copies.back();
}

EngineType TypeOf(Tegra::Engines::EngineInterface* engine) {
    if (dynamic_cast<Tegra::Engines::KeplerCompute*>(engine) != nullptr) {
        return EngineType::KeplerCompute;
    }
    if (dynamic_cast<Tegra::Engines::KeplerMemory*>(engine) != nullptr) {
        return EngineType::KeplerMemory;
    }
    if (dynamic_cast<Tegra::Engines::MaxwellDMA*>(engine) != nullptr) {
        return EngineType::MaxwellDMA;
    }
    if (dynamic_cast<Tegra::Engines::Fermi2D*>(engine) != nullptr) {
        return EngineType::Fermi2D;
    }
    return EngineType::Other;
}

// --- The stage's method handlers on a copy (spec 3.2), as Maxwell3D's own paths apply them ---

/// Maxwell3D::ProcessShadowRam.
u32 ShadowRam(Engine& e, u32 method, u32 argument) {
    const auto control = e.shadow_state.shadow_ram_control;
    if (control == Regs::ShadowRamControl::Track ||
        control == Regs::ShadowRamControl::TrackWithFilter) {
        return e.shadow_state.reg_array[method] = argument;
    }
    if (control == Regs::ShadowRamControl::Replay) {
        return e.shadow_state.reg_array[method];
    }
    return argument;
}

/// Maxwell3D::ProcessDirtyRegisters: the deferred draw goes first, through the draw manager's
/// own code, then the store and the flags.
void DirtyStore(Copy& c, u32 method, u32 value) {
    Engine& e = *c.e;
    if (e.draw_manager.HasDeferredDraw() &&
        !Maxwell3D::DrawManager::IsInstanceLoopMethod(method)) [[unlikely]] {
        e.draw_manager.DrawDeferred(e, true);
    }
    ++c.generation;
    e.regs.reg_array[method] = value;
    for (const auto& table : *c.tables) {
        e.dirty.flags[table[method]] = true;
        c.raised[table[method]] = true;
    }
}

/// Maxwell3D::WriteStateRegister (and ConsumeSinkImpl, ApplyPairs, ApplyFrontDelta per write).
void StateWrite(Copy& c, u32 method, u32 argument) {
    const u32 value = ShadowRam(*c.e, method, argument);
    if (!c.skip_same || c.e->regs.reg_array[method] != value) {
        DirtyStore(c, method, value);
    }
}

/// FRITCHII_MEMSTAGE_FAST_LOOP (GPUMem with the window open): a Pairs or StateDelta block's
/// writes, ending in exactly the state StateWrite leaves them in one by one, in one loop:
/// - while a deferred draw waits, write by write through StateWrite: the first stored write that
///   is not an instance loop method draws it (reading the registers and flags as they are then).
///   That draw leaves none waiting (DrawDeferred zeroes the count and the folded instance), and a
///   state write cannot start one (only executable methods reach the draw manager), so after it
///   no write of the block issues a draw, and nothing reads the copy until the block's end;
/// - then the shadow RAM's mode read once (StateWrite reads it per write: only a tracked write
///   to shadow_ram_control's own word changes it, and the loop reads it again after one), the
///   same-value skip as StateWrite's, each stored register's two dirty flags gathered and raised
///   on the copy and in `raised` at the end, and the generation advanced by the stores (one per
///   store, as DirtyStore).
void PairsFast(Copy& c, const u64* writes, u32 count) {
    constexpr u32 ShadowRamControlMethod = MAXWELL3D_REG_INDEX(shadow_ram_control);
    static_assert(std::tuple_size_v<Maxwell3D::DirtyState::Tables> == 2);
    Engine& e = *c.e;
    u32 i = 0;
    for (; i < count && e.draw_manager.HasDeferredDraw(); ++i) {
        StateWrite(c, static_cast<u32>(writes[i] & 0xffff), static_cast<u32>(writes[i] >> 32));
    }
    if (i == count) {
        return;
    }
    const auto& table0 = (*c.tables)[0];
    const auto& table1 = (*c.tables)[1];
    u32* const reg_array = e.regs.reg_array.data();
    u32* const shadow_array = e.shadow_state.reg_array.data();
    const bool skip_same = c.skip_same;
    Flags raised;
    u64 stores = 0;
    while (i < count) {
        // shadow_state.shadow_ram_control, read as the word the loop writes (no aliasing doubt).
        const auto control =
            static_cast<Regs::ShadowRamControl>(shadow_array[ShadowRamControlMethod]);
        const bool track = control == Regs::ShadowRamControl::Track ||
                           control == Regs::ShadowRamControl::TrackWithFilter;
        const bool replay = control == Regs::ShadowRamControl::Replay;
        for (; i < count; ++i) {
            const u32 method = static_cast<u32>(writes[i] & 0xffff);
            u32 value = static_cast<u32>(writes[i] >> 32);
            if (track) {
                shadow_array[method] = value;
            } else if (replay) {
                value = shadow_array[method];
            }
            if (!skip_same || reg_array[method] != value) {
                reg_array[method] = value;
                raised[table0[method]] = true;
                raised[table1[method]] = true;
                ++stores;
            }
            if (track && method == ShadowRamControlMethod) [[unlikely]] {
                ++i;
                break; // the mode is the word just tracked: read it again
            }
        }
    }
    c.generation += stores;
    e.dirty.flags |= raised;
    c.raised |= raised;
}

bool CbMode() {
    return VideoCore::Nx2Debug::Experiment(107) != 0;
}

using StageTarget = Tegra::MemoryManager::StageTarget;

/// Step 7 fps fix 4: whether a constant buffer write lies within one 4 KiB GPU page (every single
/// word does, and most runs of words). Then its address is looked up once (ResolveStage) and
/// Watched, Collect, the own-write ring and the write take that lookup, where each translated it
/// again (the write by its page walk).
bool OnePage(GPUVAddr address, u64 size) {
    constexpr u64 Page = 0x1000;
    return (address & (Page - 1)) + size <= Page;
}

/// ResolveStage(address) for a write within one 4 KiB page, through the copy's last page: kept
/// while the page and the memory map counter (MemoryManager::MapSerial, EXP88's key on the back)
/// stay the same. A map or unmap bumps the counter before its first table write and after its
/// last (MemoryManager::MapChange), so a word after a change started misses and looks the page
/// up again, and a lookup that overlapped a change is not kept (BeginMapRead, MapReadStable).
/// Within a 4 KiB page the lookup only shifts by the offset (the same table entries, the same
/// device page).
StageTarget CbTarget(Copy& c, GPUVAddr address) {
    constexpr u64 Page = 0x1000;
    const u64 page = address & ~(Page - 1);
    StageTarget target;
    if (page == c.cb_page && c.memory->MapSerial() == c.cb_map_serial) [[likely]] {
        target = c.cb_target;
    } else {
        // A sequence lock's read: kept only if no page table change ran when the lookup started
        // and none started before it ended (a change can mark a page mapped before it writes
        // the page's address). Otherwise used for this word only.
        const Tegra::MemoryManager::MapRead read = c.memory->BeginMapRead();
        target = c.memory->ResolveStage(page);
        if (c.memory->MapReadStable(read)) {
            c.cb_target = target;
            c.cb_page = page;
            c.cb_map_serial = read.serial;
        }
    }
    const u64 offset = address - page;
    target.write_device += offset;
    if (target.device) {
        *target.device += offset;
    }
    return target;
}

/// The one piece Collect's loop makes of a write within one 4 KiB page: the 32-byte rounding
/// stays in the page (its end is 32-byte aligned), and the rounded start is on the same device
/// page as `address`, so its device address is the lookup's minus the rounding.
std::optional<std::pair<u64, std::size_t>> OnePiece(const StageTarget& target, GPUVAddr address,
                                                    u64 size) {
    constexpr u64 Atom = 32;
    if (!target.device) {
        return std::nullopt;
    }
    const GPUVAddr begin = address & ~(Atom - 1);
    const GPUVAddr end = (address + size + Atom - 1) & ~(Atom - 1);
    return std::pair<u64, std::size_t>{*target.device - (address - begin),
                                       static_cast<std::size_t>(end - begin)};
}

/// Whether [address, address + size) is on a page the back's caches watch (join 2): image
/// pages (the texture cache's filter) or the texture header and sampler pools. `target`: the
/// address already looked up (ResolveStage), or null.
bool Watched(const Copy& c, GPUVAddr address, u64 size, const StageTarget* target) {
    const Regs& regs = c.e->regs;
    constexpr u64 HeaderSize = 0x20;
    const GPUVAddr tic = regs.tex_header.Address();
    const u64 tic_size = (u64{regs.tex_header.limit} + 1) * HeaderSize;
    const GPUVAddr tsc = regs.tex_sampler.Address();
    const u64 tsc_size = (u64{regs.sampler_binding == Regs::SamplerBinding::ViaHeaderBinding
                                  ? regs.tex_header.limit
                                  : regs.tex_sampler.limit} +
                          1) *
                         HeaderSize;
    const GPUVAddr end = address + size;
    if ((address < tic + tic_size && tic < end) || (address < tsc + tsc_size && tsc < end)) {
        return true;
    }
    if (g_watch_filter == nullptr) {
        return false;
    }
    const std::optional<DAddr> device =
        target != nullptr ? target->device : c.memory->GpuToCpuAddress(address);
    return device && g_watch_filter->Any(*device, size);
}

/// The device ranges of a constant buffer write at [address, address + size), rounded as the
/// back's invalidation rounds (32 bytes), split where the GPU pages map. `target`: as Watched's.
void Collect(State& s, const Copy& c, GPUVAddr address, u64 size, const StageTarget* target) {
    if (target != nullptr) {
        if (const auto piece = OnePiece(*target, address, size)) {
            s.ranges_since_draw.push_back({piece->first, piece->second, c.memory,
                                           s.processing_position});
        }
        return;
    }
    constexpr u64 Atom = 32;
    constexpr u64 Page = 0x1000;
    GPUVAddr begin = address & ~(Atom - 1);
    const GPUVAddr end = (address + size + Atom - 1) & ~(Atom - 1);
    while (begin < end) {
        const GPUVAddr piece_end = (std::min)(end, (begin + Page) & ~(Page - 1));
        if (const std::optional<DAddr> device = c.memory->GpuToCpuAddress(begin)) {
            s.ranges_since_draw.push_back({*device, piece_end - begin, c.memory,
                                           s.processing_position});
        }
        begin = piece_end;
    }
}

/// 6b: the bytes Maxwell3D::ProcessCBMultiData's write puts in game memory, written by the stage
/// (words on Switch 2 sparse memory one at a time, as WriteWordsCached), and their device ranges
/// collected for the buffer half (the back's accumulator is not used).
/// `target`: as Watched's.
void StageWriteWords(State& s, Copy& c, GPUVAddr address, const u32* data, u64 size,
                     const StageTarget* target) {
    const Common::WriteRing::SiteScope write_site{"3D constant buffer data (memory stage)", false};
    MemoryStageLink& link = MemoryStageLink::Get();
    const bool ahead = MemoryStageLink::Ahead() != 0;
    // The write's device ranges, rounded as the back's invalidation rounds (32 bytes), split where
    // the GPU pages map: the buffer half the next fill (or a lend) applies.
    boost::container::small_vector<std::pair<u64, std::size_t>, 4> pieces;
    if (target != nullptr) {
        if (const auto piece = OnePiece(*target, address, size)) {
            pieces.push_back(*piece);
        }
    } else {
        constexpr u64 Atom = 32;
        constexpr u64 Page = 0x1000;
        GPUVAddr begin = address & ~(Atom - 1);
        const GPUVAddr end = (address + size + Atom - 1) & ~(Atom - 1);
        while (begin < end) {
            const GPUVAddr piece_end = (std::min)(end, (begin + Page) & ~(Page - 1));
            if (const std::optional<DAddr> device = c.memory->GpuToCpuAddress(begin)) {
                pieces.emplace_back(*device, static_cast<std::size_t>(piece_end - begin));
            }
            begin = piece_end;
        }
    }
    // Step 7, the window open (Fable's review, A): words onto GPU-written buffer data. A served
    // download the back makes before this entry would copy the GPU's older bytes over them, and
    // the claim would then upload those. Coming level and claiming at once is not enough: the
    // packets of this entry's earlier draws are still in ring B (a writing draw into these pages,
    // a join's copy) and the claim's download would be recorded before them. So the entry turns
    // hard here, before the write: the back runs it in a level run and writes the words with its
    // own code, in stream order (re-review N1). The words written before in this entry are the
    // same bytes; the back writes them again.
    const auto& backend = link.Backend();
    if (ahead && !pieces.empty() && backend.ranges_gpu_written != nullptr &&
        backend.ranges_gpu_written(backend.context, pieces.data(), pieces.size())) [[unlikely]] {
        link.NoteWriteLevel();
        Hard(s, LevelType::CbGpuWritten, 0);
        return;
    }
    // The stage's own ring keeps these writes with their entry while the window's verify runs
    // (the read compare leaves them out, the image check looks for them); the emulator's write
    // ring would only wrap on them over a lead of a frame, so they stay out of it then.
    const bool kept = MemoryStageLink::StageWritesKept();
    struct Quiet {
        bool was;
        explicit Quiet(bool on) : was{Common::WriteRing::t_quiet} {
            Common::WriteRing::t_quiet = was || on;
        }
        ~Quiet() {
            Common::WriteRing::t_quiet = was;
        }
    } quiet{kept};
    if (kept && target != nullptr) {
        if (target->device) {
            link.NoteStageWrite(*target->device, size, s.processing_position);
        }
    } else if (kept) {
        constexpr u64 Page = 0x1000;
        for (GPUVAddr piece = address; piece < address + size;) {
            const GPUVAddr piece_end = (std::min)(address + size, (piece + Page) & ~(Page - 1));
            if (const std::optional<DAddr> device = c.memory->GpuToCpuAddress(piece)) {
                link.NoteStageWrite(*device, piece_end - piece, s.processing_position);
            }
            piece = piece_end;
        }
    }
    if (address + size > Tegra::g_nx2_sparse_low.load(std::memory_order_relaxed) &&
        address < Tegra::g_nx2_sparse_high.load(std::memory_order_relaxed)) {
        for (u64 i = 0; i < size / sizeof(u32); ++i) {
            c.memory->WriteBlockStage(address + i * sizeof(u32), data + i, sizeof(u32));
        }
    } else if (target != nullptr) {
        c.memory->WriteBlockStageAt(*target, address, data, size);
    } else {
        c.memory->WriteBlockStage(address, data, size);
    }
    s.ring_ranges.insert(s.ring_ranges.end(), pieces.begin(), pieces.end());
}

/// Maxwell3D::ProcessCBMultiData: computes the write and its range, writes nothing. With EXP107
/// (a mode the stage does not implement: hard) only the offset advances, as in a run (plan 3.2).
/// 6b: a served entry's words are written here (StageWriteWords).
void ConstBufferWords(State& s, Copy& c, u32 amount, const u32* data) {
    // Step 7 fps fix 4 (profiling): GPUMem's constant buffer path, a fine zone inside its entries'.
    const Nx2Profile::Scope zone{Nx2Profile::Zone::StageCbWords, s.ring && Nx2Profile::Fine(),
                                 false};
    Regs& regs = c.e->regs;
    const GPUVAddr address = regs.const_buffer.Address();
    const u64 copy_size = u64{amount} * sizeof(u32);
    const u64 room = regs.const_buffer.offset < regs.const_buffer.size
                         ? regs.const_buffer.size - regs.const_buffer.offset
                         : 0;
    const u64 write_size = (std::min)(copy_size, room / sizeof(u32) * sizeof(u32));
    ++c.generation;
    if (write_size != 0 && !CbMode()) {
        const GPUVAddr begin = address + regs.const_buffer.offset;
        // Step 7 fps fix 4: one lookup for the whole path when the write lies within one page.
        std::optional<StageTarget> one_page;
        if (OnePage(begin, write_size)) [[likely]] {
            one_page = CbTarget(c, begin);
        }
        const StageTarget* const target = one_page ? &*one_page : nullptr;
        if (Watched(c, begin, write_size, target)) [[unlikely]] {
            // Join 2: the stage leaves this entry to the back (hard). It follows it: the
            // offset below; the back writes the words.
            ++s.counters.watched_hits;
            Hard(s, LevelType::CbWatched, 0);
        }
        if (!s.ring) {
            Collect(s, c, begin, write_size, target);
        } else if (!s.ring_in_run && !s.entry_hard) {
            StageWriteWords(s, c, begin, data, write_size, target);
        }
    }
    regs.const_buffer.offset += static_cast<u32>(copy_size);
}

/// Maxwell3D::ProcessCBBind: the binding the cache gets (Bind/DisableGraphicsUniformBuffer).
void ConstBufferBind(Copy& c, size_t stage) {
    Engine& e = *c.e;
    ++c.generation;
    const auto& bind = e.regs.bind_groups[stage];
    if (bind.shader_slot >= Regs::MaxConstBuffers) {
        return;
    }
    auto& buffer = e.state.shader_stages[stage].const_buffers[bind.shader_slot];
    buffer.enabled = bind.valid.Value() != 0;
    buffer.address = e.regs.const_buffer.Address();
    buffer.size = e.regs.const_buffer.size;
    // 6b: the buffer cache's half of Maxwell3D::ProcessCBBind (the back skips it when served).
    State& s = S();
    if (s.ring && !s.ring_in_run && !s.entry_hard) {
        const auto& backend = MemoryStageLink::Get().Backend();
        if (backend.bind_uniform != nullptr) {
            backend.bind_uniform(backend.context, stage, bind.shader_slot, buffer.address,
                                 buffer.size, buffer.enabled);
        }
    }
}

/// How an executable 3D method classifies before its handler (plan section 2; the handler can
/// still make it hard: a fence report, a render_enable.mode the registers do not decide, constant
/// buffer words onto a watched page).
enum class MethodKind : u8 { Work, Pass, Hard };

/// FRITCHII_MEMSTAGE_GATHER: a gather (invalidate_texture_data_cache) GPUMem may have the back run
/// at its entry (StageGather decides there, at the method). Anything else keeps it hard.
bool GatherCandidate(const State& s) {
    return s.ring && !s.ring_in_run && !s.entry_hard && MemoryStageLink::GatherServed();
}

/// FRITCHII_MEMSTAGE_PASS bit 1: a gather or wait for idle GPUMem may pass (StagePass decides
/// there, at the method). Anything else keeps it hard.
bool PassCandidate(const State& s) {
    return s.ring && !s.ring_in_run && !s.entry_hard && MemoryStageLink::PassServed(1);
}

MethodKind MethodKindOf(const Copy& c, u32 method) {
    switch (method) {
    // Hard: they need the back's caches (the gather, the wait, the upload engine's writes),
    // or (falcon[4]) were level in the first version. Followed: the store, and falcon[4]'s
    // scratch word.
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
        return MethodKind::Hard;
    case MAXWELL3D_REG_INDEX(wait_for_idle):
        return PassCandidate(S()) ? MethodKind::Work : MethodKind::Hard;
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
        return GatherCandidate(S()) || PassCandidate(S()) ? MethodKind::Work : MethodKind::Hard;
    // Pass (plan 2.2, 2.4, 2.5): the query streamer's reset, the scheduler's barrier, an empty
    // handler, and a report that is not a fence (the handler decides that). The store only.
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
        return MethodKind::Pass;
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
        return c.front_macros ? MethodKind::Work : MethodKind::Hard;
    default:
        return method - CbData < 16 && CbMode() ? MethodKind::Hard : MethodKind::Work;
    }
}

/// render_enable.mode after its store (plan 2.3): execute_on from the registers when they decide
/// it (an override, or mode False / True: the back's AccelerateConditionalRendering then only ends
/// host conditional rendering and returns false). Otherwise hard, and execute_on is taken from the
/// back at the next entry.
void Condition(State& s, Copy& c) {
    using RenderEnable = Regs::RenderEnable;
    Engine& e = *c.e;
    ++c.generation;
    const auto override_value = e.regs.render_enable_override;
    const auto mode = e.regs.render_enable.mode;
    const u32 key = (static_cast<u32>(override_value) << 8) | (static_cast<u32>(mode) & 0xff);
    ++s.counters.conditions[key];
    switch (override_value) {
    case RenderEnable::Override::AlwaysRender:
        e.execute_on = true;
        return;
    case RenderEnable::Override::NeverRender:
        e.execute_on = false;
        return;
    default:
        break;
    }
    if (override_value == RenderEnable::Override::UseRenderEnable) {
        if (mode == RenderEnable::Mode::True) {
            e.execute_on = true;
            return;
        }
        if (mode == RenderEnable::Mode::False) {
            e.execute_on = false;
            return;
        }
    }
    ++s.counters.condition_hard;
    Hard(s, LevelType::Condition, key);
    s.take_execute = &c;
}

void StageGather(State& s, Copy& c, u32 method);

/// The ring backend's pass_ready (read tracking, rule C) at GPUMem's position.
MemoryStageLink::PassReady PassReadyNow() {
    const auto& backend = MemoryStageLink::Get().Backend();
    return backend.pass_ready != nullptr
               ? static_cast<MemoryStageLink::PassReady>(backend.pass_ready(backend.context))
               : MemoryStageLink::PassReady::Tracking;
}

/// P0 (fable_stage_levelpoints 5.2): a fence entry outside a run, and whether rule C would hold
/// there (`kind` 0: the puller's release closure, 1: a 3D fence report). Returns the answer.
MemoryStageLink::PassReady NoteFenceEntry(u32 kind) {
    const MemoryStageLink::PassReady ready = PassReadyNow();
    MemoryStageLink::Get().NoteFenceEntry(kind, ready);
    return ready;
}

// --- fable_stage_p3 section 6: P3's dry counters, and its mirror of the report pages (4.1) ---

constexpr u64 ReportPage = 0x1000;
constexpr u64 NoPage = ~u64{0};

/// The device page of a report at `address` (NoPage: unmapped), the last lookup kept under the
/// address space's map counter (reports come in rows on one page).
u64 ReportDevicePage(State& s, Tegra::MemoryManager& memory, GPUVAddr address) {
    const u64 gpu_page = address & ~(ReportPage - 1);
    if (gpu_page == s.report_gpu_page && &memory == s.report_memory &&
        memory.MapSerial() == s.report_map_serial) [[likely]] {
        return s.report_device_page;
    }
    const auto read = memory.BeginMapRead();
    const std::optional<DAddr> device = memory.GpuToCpuAddress(gpu_page);
    const u64 page = device ? (*device & ~(ReportPage - 1)) : NoPage;
    if (memory.MapReadStable(read)) {
        s.report_gpu_page = gpu_page;
        s.report_device_page = page;
        s.report_memory = &memory;
        s.report_map_serial = read.serial;
    }
    return page;
}

/// A report P3 would leave pending (4.1): its page joins the mirror. Across a page end: unknown.
void P3AddReport(State& s, Tegra::MemoryManager& memory, GPUVAddr address) {
    if ((address & (ReportPage - 1)) > ReportPage - 8) {
        s.p3_unknown = true;
        return;
    }
    const u64 page = ReportDevicePage(s, memory, address);
    if (page == NoPage) {
        return; // the back's CounterReport returns before WriteCounter
    }
    if (std::find(s.p3_pages.begin(), s.p3_pages.end(), page) != s.p3_pages.end()) {
        return;
    }
    if (s.p3_pages.size() >= 256) {
        s.p3_unknown = true;
        return;
    }
    s.p3_pages.push_back(page);
}

/// P3's targets (4.2), dry: the mirror's pages and `own_page` as maximal runs of adjacent pages,
/// each tested as the sync would look it up.
MemoryStageLink::P3Outcome P3Targets(State& s, u64 own_page) {
    using Outcome = MemoryStageLink::P3Outcome;
    MemoryStageLink& link = MemoryStageLink::Get();
    boost::container::small_vector<u64, 17> pages(s.p3_pages.begin(), s.p3_pages.end());
    if (own_page != NoPage) {
        pages.push_back(own_page);
    }
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    link.NoteP3Pages(pages.size());
    boost::container::small_vector<std::pair<u64, u64>, 8> runs;
    for (const u64 page : pages) {
        if (!runs.empty() && runs.back().first + runs.back().second == page) {
            runs.back().second += ReportPage;
            continue;
        }
        if (runs.size() == 8) {
            return Outcome::TooManyRuns;
        }
        runs.emplace_back(page, ReportPage);
    }
    if (runs.empty()) {
        return Outcome::Passed;
    }
    const auto& backend = link.Backend();
    if (backend.sync_probe == nullptr) {
        return Outcome::NoBuffer;
    }
    return static_cast<Outcome>(backend.sync_probe(backend.context, runs.data(), runs.size()));
}

/// P3's decision (4.3), dry, past the structural tests: a sync point (`fence_entry` false) or a
/// fence-flagged release or report with its own report's page.
MemoryStageLink::P3Outcome P3Decide(State& s, MemoryStageLink::PassReady ready, bool fence_entry,
                                    u64 own_page) {
    using Outcome = MemoryStageLink::P3Outcome;
    using Ready = MemoryStageLink::PassReady;
    if (fence_entry && (VideoCore::Nx2Debug::Experiment(28) & 1) != 0) {
        return Outcome::Trim;
    }
    if (ready == Ready::Tracking) {
        return Outcome::Tracking;
    }
    if (ready == Ready::Flushes) {
        return Outcome::Flushes;
    }
    if (!fence_entry && !s.p3_unknown && s.p3_pages.empty()) {
        return Outcome::Passed; // P1's pass: nothing can be pending
    }
    if (s.p3_unknown) {
        return Outcome::ReportsUnknown;
    }
    if (Settings::IsGPUFenceBehaviorAccurate()) {
        return Outcome::Accurate;
    }
    return P3Targets(s, own_page);
}

/// Counts P3's decision of a kind; one it would pass changes the mirror as P3 would.
void P3Note(State& s, MemoryStageLink::P3Kind kind, MemoryStageLink::P3Outcome what) {
    using Outcome = MemoryStageLink::P3Outcome;
    MemoryStageLink::Get().NoteP3(kind, what);
    if (what == Outcome::Passed) {
        s.entry_p3 = State::P3Effect::Cleared;
    } else if (what == Outcome::PassedUnmapped || what == Outcome::PassedNoFence) {
        s.entry_p3 = State::P3Effect::Kept;
    }
}

/// fable_stage_p3 6 at a release closure GPUMem reaches outside a run (`words`: its payload,
/// StageReleaseWord's layout).
void P3Release(State& s, const u64* words, MemoryStageLink::PassReady ready) {
    using Kind3 = MemoryStageLink::P3Kind;
    using Outcome = MemoryStageLink::P3Outcome;
    const GPUVAddr address = words[static_cast<size_t>(StageReleaseWord::Address)];
    const u32 properties =
        static_cast<u32>(words[static_cast<size_t>(StageReleaseWord::PayloadProperties)] >> 32);
    auto* const memory = reinterpret_cast<Tegra::MemoryManager*>(
        words[static_cast<size_t>(StageReleaseWord::Memory)]);
    const u64 page = memory != nullptr ? ReportDevicePage(s, *memory, address) : NoPage;
    if (page == NoPage) {
        return P3Note(s, Kind3::Release, Outcome::PassedUnmapped);
    }
    constexpr u32 IsAFence = static_cast<u32>(VideoCommon::QueryPropertiesFlags::IsAFence);
    if ((properties & IsAFence) == 0) {
        P3AddReport(s, *memory, address);
        return P3Note(s, Kind3::Release, Outcome::PassedNoFence);
    }
    P3Note(s, Kind3::Release, P3Decide(s, ready, true, page));
}

/// FRITCHII_MEMSTAGE_PASS (fable_stage_levelpoints 6.2), at a gather (0x3dd) or a wait for idle
/// (0x44) of the copy, before StageGather: P1's tests, in its order, each way out counted. Off
/// (dry), the decision is counted and the entry goes on as before (hard: a run, or GATHER's
/// request); the mirror then takes it as P1 would (EntrySync, run_from_dry_pass).
///
/// On (bit 1, the window open), a passed entry: GPUMem has done what the entry does to the buffer
/// cache, at its position, and goes on; the back runs the entry later, served, with its own code,
/// and finds nothing to do in the buffer cache there (levelpoints section 4, rules C, G, Q):
/// - rule Q: no query sync can be pending (the mirror), so the wait for idle's NotifyWFI does
///   nothing (the back checks: PassQuerySync);
/// - rule C: no uncommitted flush (read under the lock when the note is set), so the flush
///   grouping (SignalOrdering) has nothing to group (the back checks: PassFlushes);
/// - rule G, 0x3dd: GPUMem takes the game's collected writes and applies their buffer half at once
///   (the fable amendment: WriteMemory only marks on pages without GPU-written data, no command);
///   the record carries them to the back for the texture and pipeline caches. A write on such a
///   page: nothing applied, the record whole, the entry a level run (its gather applies them).
/// Every passed entry pushes a record (fable_stage_p3 1): the back asks for it at the entry.
enum class PassResult : u8 {
    Today,  ///< not passed: today's path (StageGather, or hard)
    Passed, ///< passed: work, no run
    Hard,   ///< the take found GPU-written data: hard, the record whole (no StageGather)
};
PassResult StagePass(State& s, Copy& c, u32 method) {
    using Outcome = MemoryStageLink::PassOutcome;
    using Ready = MemoryStageLink::PassReady;
    using Take = MemoryStageLink::PassTake;
    static_cast<void>(c);
    MemoryStageLink& link = MemoryStageLink::Get();
    const bool gather = method == MAXWELL3D_REG_INDEX(invalidate_texture_data_cache);
    const auto which =
        gather ? MemoryStageLink::PassMethod::Gather : MemoryStageLink::PassMethod::WaitForIdle;
    if (!MemoryStageLink::StageWindowOpen()) {
        link.NotePass(which, Outcome::Closed);
        return PassResult::Today;
    }
    const bool dry = !MemoryStageLink::PassServed(1);
    const auto kind3 = gather ? MemoryStageLink::P3Kind::Gather : MemoryStageLink::P3Kind::WaitForIdle;
    // Step 1: outside a run, the entry not hard.
    if (s.ring_in_run) {
        link.NotePass(which, Outcome::InRun);
        link.NoteP3(kind3, MemoryStageLink::P3Outcome::NotLone);
        return PassResult::Today;
    }
    // Dry, the method made the entry hard itself (MethodKindOf; a lone method has nothing
    // before it). Otherwise MethodKindOf said work: hard here means hard before it.
    if (s.entry_hard && (!dry || s.entry_key != LevelKey(s.entry_type, s.channel, method))) {
        link.NotePass(which, Outcome::Late);
        link.NoteP3(kind3, MemoryStageLink::P3Outcome::NotLone);
        return PassResult::Today;
    }
    // Step 2: a lone method entry, no draw before it in it (as StageGather).
    const u64 head = Access::Ring()[s.processing_position & Access::RingMask];
    if (static_cast<Kind>(head & 0xf) != Kind::Method || s.entry_draws != 0) {
        link.NotePass(which, Outcome::NotFirst);
        link.NoteP3(kind3, MemoryStageLink::P3Outcome::NotLone);
        return PassResult::Today;
    }
    // A lone sync point: whatever becomes of it, the back runs its wait for idle at this entry.
    s.entry_sync = State::EntrySync::Ran;
    const auto& backend = link.Backend();
    // Step 4's answer (read tracking, rule C), asked once for P1 and P3's dry counters.
    const Ready ready = PassReadyNow();
    P3Note(s, kind3, P3Decide(s, ready, false, NoPage)); // fable_stage_p3 6 (dry)
    // Step 3: no query sync can be pending (rule Q, the mirror).
    const bool clear = !s.sync_unknown && s.sync_reports == 0;
    const bool alt_clear = !s.alt_unknown && s.alt_reports == 0;
    if (!clear && !alt_clear) {
        link.NotePass(which, Outcome::SyncPending);
        return PassResult::Today;
    }
    // Step 4: no read tracking, no uncommitted flush (rule C).
    if (!clear) {
        link.NotePass(which, Outcome::SyncPending, ready == Ready::Ready);
        return PassResult::Today;
    }
    if (ready == Ready::Tracking) {
        link.NotePass(which, Outcome::Tracking);
        return PassResult::Today;
    }
    if (ready == Ready::Flushes) {
        link.NotePass(which, Outcome::Flushes);
        return PassResult::Today;
    }
    if (dry) {
        link.NotePass(which, Outcome::Passed);
        s.entry_sync = State::EntrySync::Cleared;
        s.entry_dry_pass = true;
        return PassResult::Today;
    }
    // Step 5: room for the record (every passed entry has one).
    if (!link.PassRoom() || (gather && backend.pass_gather == nullptr)) {
        link.NotePass(which, Outcome::NoRoom);
        return PassResult::Today;
    }
    MemoryStageLink::PassRecord& record = link.PassBegin(s.processing_position);
    if (gather) {
        // Steps 6 and 7: the take, the page-aligned GPU-written test and the buffer half, at
        // once (the backend, under the buffer cache's lock).
        Take took;
        try {
            took = static_cast<Take>(backend.pass_gather(backend.context, record.ranges));
        } catch (...) {
            // The ring ends (RingFailed); the taken writes must not be lost: the record goes
            // whole, so the back's run of this entry applies them all (the marks again, which
            // changes nothing where they were applied already).
            record.whole = true;
            link.PassPush();
            throw;
        }
        if (took == Take::GpuWritten) {
            // WriteMemory would download or upload in place there, which must be recorded where
            // the back is: the entry runs as a level run, and its gather applies the ranges
            // taken here whole (BackApplyPasses) before it takes more.
            record.whole = true;
            link.PassPush();
            link.NotePass(which, Outcome::GpuWritten);
            Hard(s, s.entry_type, method);
            return PassResult::Hard;
        }
        if (took == Take::Recorded) [[unlikely]] {
            MemoryStageLink::Note(MemoryStageLink::Check::PassCommand,
                                  "StagePass (a passed gather's buffer half)");
        }
    }
    link.PassPush();
    link.NotePass(which, Outcome::Passed);
    s.entry_sync = State::EntrySync::Cleared;
    s.entry_p3 = State::P3Effect::Cleared; // its wait for idle syncs every streamer
    s.ring_publish_now = true;
    return PassResult::Passed;
}

/// Maxwell3D::ProcessMethodCall's effects on the copy, for every executable method that is not a
/// macro: the stage's handler (spec 3.2), or for a hard method what follow applies (plan 3.2).
void MethodCall(State& s, Copy& c, u32 method, u32 value, u32 nonshadow) {
    Engine& e = *c.e;
    if (method - CbData < 16) {
        return ConstBufferWords(s, c, 1, &value);
    }
    ++c.generation;
    switch (method) {
    case MAXWELL3D_REG_INDEX(falcon[4]):
        // Maxwell3D::ProcessFirmwareCall4.
        e.regs.shadow_scratch[0] = 1;
        return;
    case MAXWELL3D_REG_INDEX(report_semaphore.query): {
        // Maxwell3D::ProcessQueryGet: a fence report signals a fence, whose manager commits the
        // cache's async flushes (plan 2.4): hard. Any other report is the store only.
        ++s.counters.reports;
        const auto& query = e.regs.report_semaphore.query;
        using Operation = Regs::ReportSemaphore::Operation;
        if (query.operation.Value() == Operation::Release && query.short_query != 0) {
            ++s.counters.fence_reports;
            if (s.ring && MemoryStageLink::StageWindowOpen()) {
                using Outcome3 = MemoryStageLink::P3Outcome;
                const auto kind3 = MemoryStageLink::P3Kind::FenceReport;
                const u64 head = Access::Ring()[s.processing_position & Access::RingMask];
                // P0: rule C's rate at the fence reports outside a run.
                const auto ready = s.ring_in_run ? MemoryStageLink::PassReady::Tracking
                                                 : NoteFenceEntry(1);
                if (s.ring_in_run || s.entry_hard || s.entry_draws != 0 ||
                    static_cast<Kind>(head & 0xf) != Kind::Method) {
                    // fable_stage_p3 4.3: not lone (in a packet or batch, a run): hard.
                    MemoryStageLink::Get().NoteP3(kind3, Outcome3::NotLone);
                } else {
                    // fable_stage_p3 6 (dry).
                    const GPUVAddr address = e.regs.report_semaphore.Address();
                    const u64 page = ReportDevicePage(s, *c.memory, address);
                    P3Note(s, kind3,
                           page == NoPage ? Outcome3::PassedUnmapped
                                          : P3Decide(s, ready, true, page));
                }
            }
            Hard(s, LevelType::FenceReport, method);
        } else if (query.operation.Value() == Operation::Release ||
                   query.operation.Value() == Operation::ReportOnly) {
            // Maxwell3D::ProcessQueryGet calls rasterizer->Query: a sync may be pending now
            // (the mirror, fable_stage_levelpoints 6.1; P3's pages, fable_stage_p3 4.1).
            ++s.sync_reports;
            ++s.alt_reports;
            if (s.ring) {
                P3AddReport(s, *c.memory, e.regs.report_semaphore.Address());
            }
        }
        return;
    }
    case MAXWELL3D_REG_INDEX(render_enable.mode):
        return Condition(s, c);
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
        if (s.ring) {
            // FRITCHII_MEMSTAGE_PASS first (dry: its counters only), then
            // FRITCHII_MEMSTAGE_GATHER and its counters. MethodKindOf made it work for either:
            // without GATHER's candidate it is hard here, as before the passes.
            if (StagePass(s, c, method) != PassResult::Today) {
                return;
            }
            if (!s.entry_hard && !GatherCandidate(s)) {
                Hard(s, s.entry_type, method);
            }
            StageGather(s, c, method);
        }
        return; // the store only
    case MAXWELL3D_REG_INDEX(wait_for_idle):
        if (s.ring && StagePass(s, c, method) == PassResult::Today && !s.entry_hard) {
            Hard(s, s.entry_type, method); // not passed: hard, as before the passes
        }
        return; // the store only
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return; // the store only
    case Maxwell3D::RootTables::SelectMethod:
        e.root_tables.selected = value & 7;
        e.root_tables.offset = (value >> 8) & 0xff;
        return;
    case Maxwell3D::RootTables::LoadMethod:
        if (e.root_tables.offset < 256 && e.root_tables.offset % 4 == 0) {
            e.root_tables.tables[e.root_tables.selected][e.root_tables.offset / 4] = value;
            e.root_tables.loaded = true;
        }
        e.root_tables.offset += 4;
        return;
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
        e.shadow_state.shadow_ram_control = static_cast<Regs::ShadowRamControl>(nonshadow);
        return;
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
        return; // front macros: the front owns the code
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
        ++e.regs.load_mme.start_address_ptr;
        return;
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
        return ConstBufferBind(c, 0);
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
        return ConstBufferBind(c, 1);
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
        return ConstBufferBind(c, 2);
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
        return ConstBufferBind(c, 3);
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return ConstBufferBind(c, 4);
    default:
        // Draws, clears, inline indexes, texture draws: the draw manager's own code.
        return e.draw_manager.ProcessMethodCall(e, method, value);
    }
}

/// Maxwell3D::CallMethod (CallMethodSlow's order), for every method: hard ones are followed. A
/// macro method (one the back would run) cannot be followed.
void CallMethod(State& s, Copy& c, u32 method, u32 argument) {
    if (method >= MacroRegistersStart) [[unlikely]] {
        ++s.counters.macros_on_back;
        return Unfollow(s, LevelType::Macro3D, method);
    }
    if (!c.back->execution_mask[method]) {
        return StateWrite(c, method, argument);
    }
    if (MethodKindOf(c, method) == MethodKind::Hard) {
        Hard(s, s.entry_type, method);
    }
    const u32 value = ShadowRam(*c.e, method, argument);
    DirtyStore(c, method, value);
    MethodCall(s, c, method, value, argument);
}

/// Maxwell3D::CallMultiMethod, for every method that is not a macro (hard ones are followed).
void CallMultiMethod(State& s, Copy& c, u32 method, const u32* words, u32 amount) {
    Engine& e = *c.e;
    if (e.draw_manager.HasDeferredDraw() &&
        !Maxwell3D::DrawManager::IsInstanceLoopMethod(method)) [[unlikely]] {
        e.draw_manager.DrawDeferred(e, true);
    }
    if (c.back->execution_mask[method] && MethodKindOf(c, method) == MethodKind::Hard) {
        Hard(s, s.entry_type, method);
    }
    if (method - CbData < 16) {
        return ConstBufferWords(s, c, amount, words);
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(inline_data):
        // The upload engine takes the words; no register is stored (Maxwell3D::CallMultiMethod).
        return;
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        if (e.shadow_state.shadow_ram_control != Regs::ShadowRamControl::Replay) {
            // Maxwell3D::ProcessInlineIndexMultiData.
            if (amount != 0) {
                const u32 value = ShadowRam(e, method, words[amount - 1]);
                DirtyStore(c, method, value);
                e.draw_manager.SetInlineIndexBuffer(e, method, words, amount);
            }
            return;
        }
        break;
    default:
        break;
    }
    for (u32 i = 0; i < amount; ++i) {
        CallMethod(s, c, method, words[i]);
    }
}

/// Takes the copy from the back (spec 1.5): the back has executed every entry before the one the
/// shadow is about to apply. Its waiting method sink is applied as ConsumeSinkImpl will.
void Resync(State& s, Copy& c) {
    Engine& e = *c.e;
    const Maxwell3D& back = *c.back;
    e.regs = back.regs;
    e.shadow_state = back.shadow_state;
    e.state = back.state;
    e.root_tables = back.root_tables;
    e.draw_manager = back.draw_manager;
    e.execute_on = back.ShouldExecute();
    // 5c (plan 4.2): the cache's flags are taken with the rest, not reset. A flag raised for a
    // vertex stream written before the entry that made the copy stale stays raised.
    e.dirty.flags = back.dirty.flags;
    c.raised.reset();
    ++c.generation;
    c.stale = false;
    ++s.counters.resyncs;
    s.counters.resync_bytes += 2 * sizeof(Regs) + sizeof(e.state) + sizeof(e.root_tables);
    ++s.counters.copy_causes[s.stale_cause];
    if (s.take_execute == &c) {
        s.take_execute = nullptr;
    }
    if (s.take_draw_state == &c) {
        s.take_draw_state = nullptr;
    }
    if (back.method_sink.empty()) {
        return;
    }
    const bool sunk = std::exchange(s.processing_sunk, true);
    for (const auto& [method, value] : back.method_sink) {
        if ((method & Tegra::Engines::EngineInterface::SplitPlainTag) != 0) {
            continue; // native compute only
        }
        StateWrite(c, method, value);
    }
    s.processing_sunk = sunk;
}

u64 Rva(u64 address) {
#ifdef _WIN32
    State& s = S();
    if (s.module_base == 0) {
        s.module_base = reinterpret_cast<u64>(GetModuleHandleW(nullptr));
    }
    return address - s.module_base;
#else
    return address;
#endif
}

/// The copy a 3D entry applies to: taken from the back again first when stale. Null when the
/// entry is one the back already ran (`executed`) on a stale copy: it is left out, the next
/// resync has it.
Copy* Use(State& s, Maxwell3D* engine, bool executed) {
    if (engine == nullptr) [[unlikely]] {
        return nullptr;
    }
    Copy* copy = engine == s.engine && s.engine_copy != nullptr ? s.engine_copy
                                                                 : FindCopy(s, engine);
    if (copy->stale) {
        if (executed) {
            return nullptr;
        }
        Resync(s, *copy);
    }
    return copy;
}

/// A tagged payload's engine word (StageCall, ApplyDrawPacket): a Maxwell3D by construction.
Maxwell3D* As3D(u64 pointer) {
    return reinterpret_cast<Maxwell3D*>(pointer);
}

/// SHADOW=3: whether the copy holds the cache's flags outside runs (needs full vertex binds).
bool OwnsFlags(const State& s) {
    return g_mode >= 3 && s.full_binds && !s.ring;
}

/// Sets `to`'s cache flags (15 to 48) to `from`'s.
void TakeCacheFlags(Flags& to, const Flags& from) {
    const Flags& mask = CacheFlagMask();
    to = (to & ~mask) | (from & mask);
}

/// SHADOW=3, a run starts (plan 3.3): the back holds the cache for it, with the copy's flags.
void Lend(State& s) {
    for (Copy& copy : s.copies) {
        if (!copy.stale) {
            TakeCacheFlags(copy.back->dirty.flags, copy.e->dirty.flags);
        }
    }
    ++s.counters.lends;
}

/// SHADOW=3, inside a run: the copy takes the back's flags (the back is at the same entry).
void Return(State& s) {
    for (Copy& copy : s.copies) {
        if (!copy.stale) {
            TakeCacheFlags(copy.e->dirty.flags, copy.back->dirty.flags);
            ++copy.generation;
        }
    }
    ++s.counters.returns;
}

/// FRITCHII_MEMSTAGE_GATHER (GPUMem, at the gather's method; memory_stage_link.cpp's header has
/// the whole argument). Why a gather can't be GPUMem's own work: the method is the game's CPU
/// writes into the caches *and* a wait for idle, whose query result sync (NotifyWFI, SyncValues)
/// and flush grouping (SignalOrdering) use the buffer cache at the gather's place in the stream,
/// with the back's query state there. So GPUMem comes level at the entry instead of lending a run:
/// the back, parked there with every packet before it played, runs the method's whole handler as
/// a request (the cache's user while it serves), and its own run of the entry skips it. What stays
/// a run: a gather that is not a lone Direct method (something before it in the entry would run on
/// the back after it), one the back's pending deferred draw would come before, one right before a
/// hard closure (the run takes it in), no record room, the switch off or the window closed.
///
/// The collected constant buffer ranges GPUMem holds (StageRanges) stay for its next fill, as on
/// THREAD 0, where the memory manager's cached ranges wait for FlushCaching at the next draw (no
/// part of the gather, the wait for idle, the query sync or a fence calls it): the request runs
/// with them pending, as the back's own gather would. (A lent run applies them at its start,
/// earlier than THREAD 0.)
void StageGather(State& s, Copy& c, u32 method);

/// The entry after the one at `position` (`size` words) is a hard closure the front has published
/// (a Cache or untagged Call: a semaphore release, a fence), which a run started here would take
/// in (StageExtendsRun).
bool NextIsHardClosure(u64 position, size_t size) {
    const u64 next = position + size;
    if (next >= Access::Published()) {
        return false;
    }
    const u64 head = Access::Ring()[next & Access::RingMask];
    return static_cast<Kind>(head & 0xf) == Kind::Call &&
           (StageTagOf(head) == StageTag::Cache || StageTagOf(head) == StageTag::Level);
}

void StageGather(State& s, Copy& c, u32 method) {
    using Outcome = MemoryStageLink::GatherOutcome;
    MemoryStageLink& link = MemoryStageLink::Get();
    if (s.ring_in_run) {
        return link.NoteGather(Outcome::InRun);
    }
    if (s.entry_hard) {
        // MethodKindOf made it hard for this gather (the switch off or the window closed: it
        // starts the run, as before), or something before it in the entry did.
        const bool starts = !MemoryStageLink::GatherServed() &&
                            s.entry_key == LevelKey(s.entry_type, s.channel, method);
        return link.NoteGather(starts ? Outcome::Closed : Outcome::Late);
    }
    // A candidate (MethodKindOf said work): served here, or hard after all.
    const auto level = [&](Outcome why) {
        link.NoteGather(why);
        Hard(s, s.entry_type, method);
    };
    const u64 head = Access::Ring()[s.processing_position & Access::RingMask];
    if (static_cast<Kind>(head & 0xf) != Kind::Method || s.entry_draws != 0) {
        // The back would run what comes before it in the entry (a draw the store flushed, other
        // writes of the packet) after the request: not in stream order.
        return level(Outcome::NotFirst);
    }
    // A hard closure right after it (a Cache or untagged Call: a semaphore release, a fence) would
    // join the gather's run (StageExtendsRun) and start one of its own here: GPUMem would come
    // level and then lend anyway. Kept as one run, when the front has published that entry.
    if (NextIsHardClosure(s.processing_position, (head & Access::HasSegment) != 0 ? 2 : 1)) {
        return level(Outcome::NextHard);
    }
    const auto& backend = link.Backend();
    if (backend.gather == nullptr) {
        return level(Outcome::NoBackend);
    }
    if (!link.GatherRoom()) {
        return level(Outcome::NoRoom);
    }
    const auto start = std::chrono::steady_clock::now();
    if (!link.StageLevel(s.processing_position, MemoryStageLink::LevelReason::Gather)) {
        return level(Outcome::Stopping);
    }
    // Level: the back has run every entry before this one and is parked at it. The record first:
    // if the request throws (the ring then ends at this entry), the back's run of the entry
    // without the ring still skips the gather, so it never runs twice. The request reads and
    // writes the back's state on the back (its idle hook's flush may run there meanwhile): it
    // declines when the engine has a deferred draw, which this entry would issue first (its
    // method sink's writes and the gather's own store flush it), before the gather; it hands the
    // cache's flags over around the gather (BackGatherLend, BackGatherReturn). That decline is
    // not a mismatch: the shadow applies sunk writes at once and may have issued that draw at an
    // earlier entry, while the back issues it when it consumes its sink here.
    link.PushGather(s.processing_position);
    u64 draw_seq = 0;
    if (!backend.gather(backend.context, c.back, draw_seq)) {
        link.RetractGather();
        return level(Outcome::DeferredDraw);
    }
    link.FinishGather(draw_seq);
    s.ring_ran_on_back = true;
    // The sync mirror (fable_stage_levelpoints 6.1) against what the back's run found, then
    // cleared: the gather's wait for idle synced every streamer at this entry.
    link.NoteMirrorCheck(!s.sync_unknown && s.sync_reports == 0,
                         !s.alt_unknown && s.alt_reports == 0, link.LastGatherSynced());
    s.entry_sync = State::EntrySync::Cleared;
    s.entry_p3 = State::P3Effect::Cleared;
    link.NoteGather(Outcome::Served,
                    static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - start)
                                         .count()));
}

/// FRITCHII_MEMSTAGE_RELEASE (GPUMem, at a Call entry tagged StageCacheCall::SemaphoreRelease, the
/// whole entry): true when the back ran the closure at this entry on GPUMem's request, so the entry
/// is work and no run; false: the caller makes it hard, as before. As StageGather (the gathers'
/// comment has the argument): GPUMem comes level at the entry, the back runs the closure's thunk
/// as a request (the cache's only user, the same call its Execute makes, at the same place in the
/// stream: the query cache's report, the fence and its commits, the semaphore word), with the
/// cache's flags handed over around it, and GpuSplit::Execute skips the entry when it reaches it.
/// A Call entry is only its call on the back (no method sink, no deferred draw is issued there),
/// so nothing of the entry comes before it. Kept as a run: a hard closure right after it (the run
/// takes it in), no record room, the switch off or the window closed.
bool StageRelease(State& s, const u64* entry, size_t size) {
    using Outcome = MemoryStageLink::ReleaseOutcome;
    MemoryStageLink& link = MemoryStageLink::Get();
    if (s.ring_in_run) {
        link.NoteRelease(Outcome::InRun);
        if (MemoryStageLink::StageWindowOpen()) {
            link.NoteP3(MemoryStageLink::P3Kind::Release, MemoryStageLink::P3Outcome::NotLone);
        }
        return false;
    }
    if (MemoryStageLink::StageWindowOpen()) {
        // P0: rule C's rate at the release closures; fable_stage_p3 6 (dry), on the closure's
        // fixed payload (puller.cpp's ReleaseCall).
        const auto ready = NoteFenceEntry(0);
        if (size == 2 + 4) {
            P3Release(s, entry + 2, ready);
        }
    }
    if (!MemoryStageLink::ReleaseServed()) {
        link.NoteRelease(Outcome::Closed);
        return false;
    }
    if (NextIsHardClosure(s.processing_position, size)) {
        link.NoteRelease(Outcome::NextHard);
        return false;
    }
    if (!link.ReleaseRoom()) {
        link.NoteRelease(Outcome::NoRoom);
        return false;
    }
    const auto start = std::chrono::steady_clock::now();
    if (!link.StageLevel(s.processing_position, MemoryStageLink::LevelReason::Release)) {
        link.NoteRelease(Outcome::Stopping);
        return false;
    }
    // Level: the back is parked at this entry, which stays in ring A until the back passes it. The
    // record first, as StageGather's: if the closure throws in the request (the ring then ends at
    // this entry), the back's run of the entry without the ring still skips it, so the report,
    // the fence and the semaphore word are never made twice.
    link.PushRelease(s.processing_position);
    struct Run {
        const u64* entry;
        u64 draw_seq;
    } run{entry, 0};
    auto body = [&run] {
        BackGatherLend();
        const BackGatherFlags flags; // BackGatherReturn, also if the closure throws
        reinterpret_cast<void (*)(const u64*)>(run.entry[1])(run.entry + 2);
        run.draw_seq = MemoryStageLink::Get().BackDrawSeq();
    };
    link.ServeNow(body);
    link.FinishRelease(run.draw_seq);
    s.ring_ran_on_back = true;
    link.NoteRelease(Outcome::Served,
                     static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now() - start)
                                          .count()));
    return true;
}

/// Run bookkeeping for one entry (plan 3.2): a run starts at a hard entry and ends after the last
/// hard entry reached through pass and neutral entries; a work entry, or one that issued a draw,
/// ends it.
void Runs(State& s, Class cls) {
    Counters& c = s.counters;
    if (cls == Class::Level) {
        if (!s.in_run) {
            s.in_run = true;
            ++c.level_runs;
            ++c.run_kinds[s.entry_key];
            ++c.gaps[GapBucket(s.served_since_run)];
            s.served_since_run = 0;
            s.run_entries = 1;
            s.run_trailing = 0;
            if (OwnsFlags(s)) {
                Lend(s);
            }
        } else {
            s.run_entries += s.run_trailing + 1;
            s.run_trailing = 0;
        }
        return;
    }
    if (cls == Class::Work || s.entry_draws != 0) {
        if (s.in_run) {
            s.in_run = false;
            ++c.runs_ended;
            c.run_entries += s.run_entries;
        }
        s.served_since_run += s.entry_served;
        return;
    }
    if (s.in_run) {
        ++s.run_trailing;
    }
}

/// Classifies the entry at `position`, applies it to the copies (hard entries too: follow, plan
/// 3.2, unless it cannot be followed), and returns its size in words (for a wrap: the rest of the
/// ring).
size_t Process(State& s, u64 position, bool executed) {
    const u64* const ring = Access::Ring();
    const size_t index = position & Access::RingMask;
    const u64* const entry = &ring[index];
    const u64 head = entry[0];
    s.processing_position = position;
    s.processing_sunk = false;
    s.entry_hard = false;
    s.entry_unfollowed = false;
    s.entry_type = LevelType::Method3D;
    s.entry_draws = 0;
    s.entry_served = 0;
    s.entry_launches = 0;
    s.entry_sync = State::EntrySync::None;
    s.entry_dry_pass = false;
    s.entry_p3 = State::P3Effect::None;
    if (s.channel_stale && !executed) {
        // After an entry that cannot be followed, the channel as the back has it (an untagged
        // closure may bind one).
        s.channel = s.back_channel;
        s.channel_stale = false;
    }
    Class cls = Class::Pass;
    size_t size = 1;
    // Another engine's method: neutral unless executable (plan 2.6).
    const auto other_engine = [&](u32 method, bool executable) {
        if (executable) {
            Hard(s, LevelType::EngineMethod,
                 (u64{static_cast<u32>(s.engine_type)} << 16) | (method & 0xffff));
        } else {
            cls = Class::Neutral;
        }
    };
    // fable_memstage_compute_brief 3: the compute launches among another engine's methods (as
    // the compute front notes dispatch writers), for the runs by size (counting only).
    // With the GPU profile only (the line is logged then), or while FRITCHII_DELAY_RUN_US is set
    // (its kind tells runs by launches).
    const bool launches_counted =
        Nx2Profile::Enabled() ||
        VideoCore::Nx2Debug::g_delay_run_us.load(std::memory_order_relaxed) > 0;
    const auto count_launch = [&](u32 method, u32 argument) {
        if (launches_counted && s.engine_type == EngineType::KeplerCompute &&
            ((method == 0xaf && (argument & 2) != 0) ||
             (method == 0xb0 && (argument & 15) > 1) || method == 0x107)) {
            ++s.entry_launches;
        }
    };
    switch (static_cast<Kind>(head & 0xf)) {
    case Kind::Wrap:
        size = Access::RingWords - index;
        break;
    case Kind::Engine: {
        size = 2;
        auto* const engine = reinterpret_cast<Tegra::Engines::EngineInterface*>(entry[1]);
        s.engine = engine;
        s.engine_copy = nullptr;
        if (auto* const maxwell3d = dynamic_cast<Maxwell3D*>(engine)) {
            s.engine_copy = FindCopy(s, maxwell3d);
        } else {
            s.engine_type = TypeOf(engine);
            cls = Class::Neutral;
        }
        break;
    }
    case Kind::Method: {
        size = (head & Access::HasSegment) != 0 ? 2 : 1;
        const u32 method = static_cast<u32>((head >> 8) & 0xffff);
        const u32 argument = static_cast<u32>(head >> 32);
        if (s.engine == nullptr) {
            Unfollow(s, LevelType::NoEngine, 0);
            break;
        }
        if (s.engine_copy == nullptr) {
            // A Sink write is a plain store (GpuSplit::SinkMethod); Direct is called.
            other_engine(method, (head & Access::Direct) != 0 ||
                                     ((head & Access::Sink) == 0 &&
                                      s.engine->execution_mask[method]));
            count_launch(method, argument);
            break;
        }
        Copy& c0 = *s.engine_copy;
        const bool executable = c0.back->execution_mask[method];
        if ((head & Access::Sink) != 0) {
            // Not sent for the 3D engine (native compute only): the back stores it later.
            Unfollow(s, LevelType::Method3D, method);
            break;
        }
        if ((head & Access::Direct) != 0 || executable) {
            if (method >= MacroRegistersStart) {
                ++s.counters.macros_on_back;
                Unfollow(s, LevelType::Macro3D, method);
                break;
            }
            cls = executable && MethodKindOf(c0, method) == MethodKind::Work ? Class::Work
                                                                             : Class::Pass;
            if (Copy* const c = Use(s, c0.back, executed)) {
                const Nx2Profile::Scope zone{Nx2Profile::Zone::StageMethods,
                                             s.ring && Nx2Profile::Fine(), false};
                CallMethod(s, *c, method, argument);
            } else if (executable && MethodKindOf(c0, method) == MethodKind::Hard) {
                Hard(s, LevelType::Method3D, method);
            }
            break;
        }
        // Into the back's method sink: applied now, which is the same state at the next
        // executable method (spec 4.2); a deferred draw it flushes comes later on the back.
        if (Copy* const c = Use(s, c0.back, executed)) {
            const Nx2Profile::Scope zone{Nx2Profile::Zone::StageMethods,
                                         s.ring && Nx2Profile::Fine(), false};
            s.processing_sunk = true;
            StateWrite(*c, method, argument);
            s.processing_sunk = false;
        }
        break;
    }
    case Kind::Multi: {
        const u32 method = static_cast<u32>((head >> 8) & 0xffff);
        const u32 amount = static_cast<u32>(head >> 32);
        const size_t data = (head & Access::HasSegment) != 0 ? 3 : 2;
        size = data + (amount + 1) / 2;
        if (s.engine == nullptr) {
            Unfollow(s, LevelType::NoEngine, 0);
            break;
        }
        if (s.engine_copy == nullptr) {
            // Direct: a macro's words (GpuSplit::MacroWords, EXP124), called as a Direct Method.
            other_engine(method, (head & Access::Direct) != 0 || s.engine->execution_mask[method]);
            if (launches_counted && (method == 0xaf || method == 0xb0 || method == 0x107)) {
                const u32* const words = reinterpret_cast<const u32*>(entry + data);
                for (u32 i = 0; i < amount; ++i) {
                    count_launch(method, words[i]);
                }
            }
            break;
        }
        Copy& c0 = *s.engine_copy;
        if (method >= MacroRegistersStart) {
            ++s.counters.macros_on_back;
            Unfollow(s, LevelType::Macro3D, method);
            break;
        }
        const bool executable = c0.back->execution_mask[method];
        cls = executable && MethodKindOf(c0, method) == MethodKind::Work ? Class::Work
                                                                         : Class::Pass;
        if (Copy* const c = Use(s, c0.back, executed)) {
            const Nx2Profile::Scope zone{Nx2Profile::Zone::StageMethods,
                                         s.ring && Nx2Profile::Fine(), false};
            CallMultiMethod(s, *c, method, reinterpret_cast<const u32*>(entry + data), amount);
        } else if (executable && MethodKindOf(c0, method) == MethodKind::Hard) {
            Hard(s, LevelType::Method3D, method);
        }
        break;
    }
    case Kind::CbWords: {
        // FRITCHII_CB_WORDS_BLOCK (GpuSplit::CbWord): 3D constant buffer data words that were
        // Method entries in a row, each followed as Kind::Method follows such an entry
        // (CallMethod); the block is one entry, hard as a whole when any word makes it hard
        // (the back then writes every word itself, the ones written here again).
        const u32 count = static_cast<u32>(head >> 32);
        const size_t first = (head & Access::CbVerify) != 0 ? 2 : 1;
        size = first + count;
        const u64* const words = entry + first;
        if (s.engine == nullptr) {
            Unfollow(s, LevelType::NoEngine, 0);
            break;
        }
        if (s.engine_copy == nullptr) {
            // Only the 3D engine sends blocks; any other engine's executable method is level.
            other_engine(count != 0 ? static_cast<u32>((words[0] >> 8) & 0xffff) : CbData, true);
            break;
        }
        Copy& c0 = *s.engine_copy;
        // As Kind::Method for an executable 3D method (what every word is, GpuSplit checks).
        cls = MethodKindOf(c0, CbData) == MethodKind::Work ? Class::Work : Class::Pass;
        if (Copy* const c = Use(s, c0.back, executed)) {
            const Nx2Profile::Scope zone{Nx2Profile::Zone::StageMethods,
                                         s.ring && Nx2Profile::Fine(), false};
            for (u32 i = 0; i < count; ++i) {
                CallMethod(s, *c, static_cast<u32>((words[i] >> 8) & 0xffff),
                           static_cast<u32>(words[i] >> 32));
            }
        } else if (MethodKindOf(c0, CbData) == MethodKind::Hard) {
            Hard(s, LevelType::Method3D, CbData);
        }
        break;
    }
    case Kind::Call: {
        const size_t payload = static_cast<size_t>((head >> 8) & 0xffffff);
        size = 2 + payload;
        const u64* const words = entry + 2;
        switch (StageTagOf(head)) {
        case StageTag::Level:
            Unfollow(s, LevelType::Call, Rva(entry[1]));
            break;
        case StageTag::Pass:
            break;
        case StageTag::Cache:
            // FRITCHII_MEMSTAGE_RELEASE: the puller's semaphore release, run by the back at its
            // entry on GPUMem's request (no run), or as any cache call.
            if (s.ring && StageCacheCallOf(head) == StageCacheCall::SemaphoreRelease &&
                StageRelease(s, entry, size)) {
                cls = Class::Work;
                // The back ran its Query at the entry, not hard: a sync may be pending now (the
                // mirror, levelpoints 6.1's "a Cache call that is not a passed one"), and P3's
                // reports are unknown unless P3 would have passed it (fable_stage_p3 4.1).
                s.sync_unknown = true;
                s.alt_unknown = true;
                if (s.entry_p3 == State::P3Effect::None) {
                    s.p3_unknown = true;
                }
                break;
            }
            // Uses the cache on the back, writes no 3D state: hard, no copy after (plan 3.3).
            Hard(s, LevelType::CacheCall, Rva(entry[1]));
            break;
        case StageTag::Channel:
            cls = Class::Work;
            s.channel = static_cast<s32>(static_cast<s64>(words[0]));
            // 6b: the buffer cache's channel bind (the back skips it when served).
            if (s.ring && !s.ring_in_run && !s.entry_hard) {
                const auto& backend = MemoryStageLink::Get().Backend();
                if (backend.bind_channel != nullptr) {
                    backend.bind_channel(backend.context, s.channel);
                }
            }
            break;
        case StageTag::Writes: {
            Maxwell3D* const engine = As3D(words[0]);
            const u32 count = static_cast<u32>(words[2]);
            const u64* const writes = words + 3;
            cls = Class::Work;
            s.entry_type = LevelType::Packet3D;
            if (Copy* const c = Use(s, engine, executed)) {
                const Nx2Profile::Scope zone{Nx2Profile::Zone::StagePackets,
                                             s.ring && Nx2Profile::Fine(), false};
                for (u32 i = 0; i < count; ++i) {
                    CallMethod(s, *c, static_cast<u32>(writes[i] & 0xffff),
                               static_cast<u32>(writes[i] >> 32));
                }
            } else if (engine == nullptr) {
                Unfollow(s, LevelType::Packet3D, 0);
            }
            break;
        }
        case StageTag::FinishMacro: {
            cls = Class::Work;
            if (Copy* const c = Use(s, As3D(words[0]), executed)) {
                // ConsumeSink (the shadow applied its writes already), then DrawDeferred.
                ++c->generation;
                c->e->draw_manager.DrawDeferred(*c->e);
            }
            break;
        }
        case StageTag::DrawLoop: {
            cls = Class::Work;
            if (Copy* const c = Use(s, As3D(words[0]), executed)) {
                ++c->generation;
                // The kind is in header bits 40-47; AddLoopInstances' count is the low half of
                // payload word 1 (the struct's u32 at offset 8, little-endian); ListDraw's address
                // and keep id are words 1 and 2.
                switch (StageDrawLoopOf(head)) {
                case StageDrawLoop::FoldFirstInstance:
                    c->e->draw_manager.FoldFirstInstance();
                    break;
                case StageDrawLoop::AddLoopInstances:
                    c->e->draw_manager.AddLoopInstances(*c->e, static_cast<u32>(words[1]));
                    break;
                case StageDrawLoop::ListDraw:
                    c->e->draw_manager.draw_state.list_draw_address = words[1];
                    c->e->draw_manager.draw_state.list_keep_id = words[2];
                    break;
                default:
                    Unfollow(s, LevelType::UnknownTag, static_cast<u64>(StageTag::DrawLoop));
                    break;
                }
            }
            break;
        }
        default:
            Unfollow(s, LevelType::UnknownTag, static_cast<u64>(StageTagOf(head)));
            break;
        }
        break;
    }
    case Kind::Pairs:
    case Kind::StateDelta: {
        const u32 count = static_cast<u32>(head >> 32);
        size = 1 + size_t{count};
        if (s.engine == nullptr) {
            Unfollow(s, LevelType::NoEngine, 0);
            break;
        }
        if (s.engine_copy == nullptr) {
            cls = Class::Neutral; // plain stores (GpuSplit::PairWrite)
            break;
        }
        if (Copy* const c = Use(s, s.engine_copy->back, executed)) {
            const Nx2Profile::Scope zone{Nx2Profile::Zone::StagePairs,
                                         s.ring && Nx2Profile::Fine(), false};
            if (s.ring && MemoryStageLink::FastLoop()) {
                PairsFast(*c, entry + 1, count);
                break;
            }
            for (u32 i = 0; i < count; ++i) {
                StateWrite(*c, static_cast<u32>(entry[1 + i] & 0xffff),
                           static_cast<u32>(entry[1 + i] >> 32));
            }
        }
        break;
    }
    case Kind::DrawBatch: {
        const u32 count = static_cast<u32>((head >> 8) & 0xff);
        const u32 repeats = static_cast<u32>(head >> 32);
        size = 1 + size_t{count};
        if (s.engine_copy == nullptr) {
            if (s.engine == nullptr) {
                Unfollow(s, LevelType::NoEngine, 0);
            } else {
                Unfollow(s, LevelType::Engine, static_cast<u64>(s.engine_type));
            }
            break;
        }
        Copy& c0 = *s.engine_copy;
        cls = Class::Work;
        s.entry_type = LevelType::Batch3D;
        if (Copy* const c = Use(s, c0.back, executed)) {
            const Nx2Profile::Scope zone{Nx2Profile::Zone::StagePackets,
                                         s.ring && Nx2Profile::Fine(), false};
            // Maxwell3D::RunDrawBatch: the group once, then the repeats as instances.
            for (u32 i = 0; i < count; ++i) {
                CallMethod(s, *c, static_cast<u32>(entry[1 + i] & 0xffff),
                           static_cast<u32>(entry[1 + i] >> 32));
            }
            if (repeats > 1) {
                ++c->generation;
                c->e->draw_manager.draw_state.instance_count += repeats - 1;
            }
        }
        break;
    }
    }
    if (s.entry_hard) {
        cls = Class::Level;
    }
    s.last_class = cls;
    Runs(s, cls);
    switch (cls) {
    case Class::Pass:
        ++s.counters.pass;
        break;
    case Class::Work:
        ++s.counters.work;
        break;
    case Class::Neutral:
        ++s.counters.neutral;
        break;
    case Class::Level:
        ++s.counters.level;
        ++s.counters.level_kinds[s.entry_key];
        s.level_since_flush = true;
        if (!executed) {
            s.current_level = true;
        }
        break;
    }
    if (s.entry_unfollowed) {
        s.channel_stale = true;
        for (Copy& copy : s.copies) {
            copy.stale = true;
        }
    }
    if (s.ring) {
        // The sync mirror (fable_stage_levelpoints 6.1): a pending sync is only ever added by
        // rasterizer->Query, which a served entry reaches only through the 3D report method
        // (counted in MethodCall); every other caller is a hard or unfollowed entry.
        switch (s.entry_sync) {
        case State::EntrySync::Cleared:
            // Passed, or run by the back at its entry: its wait for idle syncs every streamer.
            s.sync_unknown = false;
            s.sync_reports = 0;
            s.alt_unknown = false;
            s.alt_reports = 0;
            break;
        case State::EntrySync::Ran:
            s.sync_unknown = true;
            s.alt_unknown = false;
            s.alt_reports = 0;
            break;
        case State::EntrySync::None:
            if (s.entry_hard || s.entry_unfollowed) {
                s.sync_unknown = true;
                s.alt_unknown = true;
            }
            break;
        }
        // P3's mirror (fable_stage_p3 4.1) as P3 would keep it: an entry it would pass clears it
        // (a sync point or a fence) or leaves it (a release with no fence, already added).
        switch (s.entry_p3) {
        case State::P3Effect::Cleared:
            s.p3_pages.clear();
            s.p3_unknown = false;
            break;
        case State::P3Effect::Kept:
            break;
        case State::P3Effect::None:
            if (s.entry_hard || s.entry_unfollowed) {
                s.p3_unknown = true;
            }
            break;
        }
    }
    return size;
}

Record TakeSpare(State& s) {
    if (s.spare.empty()) {
        return Record{};
    }
    Record record = std::move(s.spare.back());
    s.spare.pop_back();
    record.words.clear();
    record.ranges.clear();
    return record;
}

void Recycle(State& s, Record&& record) {
    if (s.spare.size() < 64) {
        s.spare.push_back(std::move(record));
    }
}

/// A shadow draw the back never issued: compare a. Its ranges were written all the same.
void DropUnmatched(State& s, Record&& record, const char* why) {
    ++s.counters.a_extra;
    if (s.a_logged < 16) {
        ++s.a_logged;
        LOG_WARNING(HW_GPU,
                    "Memory stage shadow (a): the shadow drew at entry {:#x} ({}), the back did "
                    "not {}",
                    record.position, record.sunk ? "flushed by a sunk write" : "direct", why);
    }
    s.due.insert(s.due.end(), record.ranges.begin(), record.ranges.end());
    Recycle(s, std::move(record));
}

/// After the back's fill of a draw that matched a shadow draw (plan 4.2): what the fill cleared
/// on the back's cache flags is cleared on the copy and on the pending shadow draws of the same
/// copy, except a flag the stage raised again after the matched draw. Not at SHADOW=3 while the
/// copy holds the flags (its own fills clear them; the back's are a superset).
void Settle(State& s) {
    if (!s.fill_pending) {
        return;
    }
    s.fill_pending = false;
    Copy* const copy = s.match_copy;
    if (s.fill_from_copy || OwnsFlags(s) || copy == nullptr || copy->stale ||
        copy->back != s.fill_engine) {
        return;
    }
    const Flags cleared = s.fill_pre & ~(s.fill_engine->dirty.flags & CacheFlagMask());
    ++s.counters.flag_settles;
    if (cleared.none()) {
        return;
    }
    Flags since{};
    for (Record& record : s.records) {
        if (record.engine != copy->back) {
            continue;
        }
        since |= record.raised_before;
        record.dirty &= ~(cleared & ~since);
    }
    since |= copy->raised;
    copy->e->dirty.flags &= ~(cleared & ~since);
}

/// CacheInputs' mark hook (the scrub of a deleted buffer's bindings, a repack change): the marks
/// the cache makes on an engine's flags reach the copy too, as they will reach the stage's copy
/// in 6b. During a back fill they count as flags held before it (Settle).
void OnCacheMark(const Maxwell3D* engine, bool index_buffer, u32 vertex_mask) {
    if (!g_on.load(std::memory_order_relaxed) ||
        !(t_on_back || VideoCommon::MemoryStageLink::OnStageThread())) {
        return;
    }
    State& s = S();
    Flags bits{};
    if (index_buffer) {
        bits[VideoCommon::Dirty::IndexBuffer] = true;
    } else {
        bits[VideoCommon::Dirty::VertexBuffers] = true;
        for (u32 binding = 0; binding < 32; ++binding) {
            if ((vertex_mask >> binding) & 1) {
                bits[VideoCommon::Dirty::VertexBuffer0 + binding] = true;
            }
        }
    }
    for (Copy& copy : s.copies) {
        if (copy.back == engine) {
            copy.e->dirty.flags |= bits;
            break;
        }
    }
    if (s.fill_pending && s.fill_engine == engine) {
        s.fill_pre |= bits;
    }
}

/// "a: n/s; b: m/s" for the top `count` of a map by value, names from `name`.
template <typename Map, typename Name>
std::string Top(const Map& map, size_t count, double seconds, Name&& name) {
    std::vector<std::pair<typename Map::key_type, u64>> list(map.begin(), map.end());
    std::sort(list.begin(), list.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string text;
    for (size_t i = 0; i < list.size() && i < count; ++i) {
        text += fmt::format("{}{} {:.1f}/s", i == 0 ? "" : "; ", name(list[i].first),
                            static_cast<double>(list[i].second) / seconds);
    }
    return text.empty() ? std::string{"none"} : text;
}

/// The 5c lines (plan 6.1 step A item 7) and the gate.
void Report5c(State& s, double seconds) {
    Counters& c = s.counters;
    Counters& t = s.totals;
    const auto rate = [seconds](u64 value) { return static_cast<double>(value) / seconds; };
    t.condition_hard += c.condition_hard;
    t.macros_on_back += c.macros_on_back;
    t.tracker_mismatches += c.tracker_mismatches;
    t.flag_compares += c.flag_compares;
    const u64 copy_bytes_each =
        2 * sizeof(Regs) + sizeof(Maxwell3D::State) + sizeof(Maxwell3D::RootTables);
    LOG_INFO(HW_GPU,
             "Memory stage 5c runs: {:.0f}/s ({:.1f} entries each), neutral entries/s {:.0f}; "
             "served draws between two runs (0, 1, 2-3, 4-7, 8-15, 16+): {} {} {} {} {} {}; full "
             "copies/s {:.0f} ({:.1f} MB/s); takes/s execute_on {:.1f}, draw state {:.1f} (GPU "
             "count written {:.1f}); 3D macros on the back/s {:.1f}; fence reports/s {:.1f} of "
             "{:.0f} reports",
             rate(c.level_runs),
             c.runs_ended != 0 ? static_cast<double>(c.run_entries) / c.runs_ended : 0.0,
             rate(c.neutral), c.gaps[0], c.gaps[1], c.gaps[2], c.gaps[3], c.gaps[4], c.gaps[5],
             rate(c.resyncs), rate(c.resyncs * copy_bytes_each) / 1e6, rate(c.execute_takes),
             rate(c.draw_state_takes), rate(c.draw_state_written), rate(c.macros_on_back),
             rate(c.fence_reports), rate(c.reports));
    LOG_INFO(HW_GPU, "Memory stage 5c runs by first hard kind (top 12): {}",
             Top(c.run_kinds, 12, seconds, [](u64 key) { return LevelName(key); }));
    LOG_INFO(HW_GPU, "Memory stage 5c full copies by cause (top 8): {}",
             Top(c.copy_causes, 8, seconds, [](u64 key) {
                 return key == 0 ? std::string{"start"} : LevelName(key);
             }));
    LOG_INFO(HW_GPU,
             "Memory stage 5c render_enable.mode by override and mode: {}; not decided by the "
             "registers {} this window, {} this session (must be 0)",
             Top(c.conditions, 8, seconds,
                 [](u32 key) {
                     static constexpr const char* overrides[]{"use", "always", "never"};
                     static constexpr const char* modes[]{"False", "True", "Conditional",
                                                          "IfEqual", "IfNotEqual"};
                     const u32 o = key >> 8;
                     const u32 m = key & 0xff;
                     return fmt::format("{}/{}", o < 3 ? overrides[o] : "?",
                                        m < 5 ? modes[m] : "?");
                 }),
             c.condition_hard, t.condition_hard);
    std::string flags;
    for (u32 i = 0; i < CacheFlagCount; ++i) {
        if (c.flag_missing[i] != 0 || c.flag_extra[i] != 0) {
            flags += fmt::format("{}flag {}: copy only {}, back only {}", flags.empty() ? "" : "; ",
                                 CacheFlagFirst + i, c.flag_missing[i], c.flag_extra[i]);
        }
    }
    std::string fills;
    if (g_mode >= 3) {
        static constexpr const char* names[]{"copy",   "no record", "hard entry", "indirect",
                                             "stale",  "moved on",  "other",      "full binds off"};
        for (size_t i = 0; i < c.fill_from.size(); ++i) {
            fills += fmt::format("{}{} {:.0f}", i == 0 ? "" : ", ", names[i], rate(c.fill_from[i]));
        }
        fills = fmt::format("; SHADOW=3 fills/s from: {}; lends/s {:.0f}, returns/s {:.0f}", fills,
                            rate(c.lends), rate(c.returns));
    }
    LOG_INFO(HW_GPU,
             "Memory stage 5c flags 15-48 (compare b): {} draws compared ({} strict), {} settles; "
             "mismatches by flag: {}{}; OoT tracker record vs today's query: {} compared, {} "
             "mismatches (session {}, must be 0), {} draws without a record",
             c.flag_compares, c.flag_compares_strict, c.flag_settles,
             flags.empty() ? "none" : flags, fills, c.tracker_compared, c.tracker_mismatches,
             t.tracker_mismatches, c.tracker_without);
    // The stop thresholds of plan 6.1 (they are the market's: read this line in its window).
    constexpr double RunLimit = 15000.0;
    constexpr double CopyLimit = 2000.0;
    std::string why;
    if (rate(c.level_runs) > RunLimit) {
        why += fmt::format("; runs/s {:.0f} > {:.0f}", rate(c.level_runs), RunLimit);
    }
    if (rate(c.resyncs) > CopyLimit) {
        why += fmt::format("; full copies/s {:.0f} > {:.0f}", rate(c.resyncs), CopyLimit);
    }
    if (t.condition_hard != 0) {
        why += fmt::format("; render_enable.mode other than False/True in play ({} this session)",
                           t.condition_hard);
    }
    LOG_INFO(HW_GPU,
             "Memory stage 5c gate: {} (runs/s {:.0f}, limit {:.0f}; full copies/s {:.0f}, limit "
             "{:.0f}; undecided render_enable.mode this session {}, limit 0){}",
             why.empty() ? "pass" : "fail", rate(c.level_runs), RunLimit, rate(c.resyncs),
             CopyLimit, t.condition_hard, why);
}

void Report(State& s, std::chrono::steady_clock::time_point now) {
    const double seconds =
        std::chrono::duration<double>(now - s.window_start).count();
    s.window_start = now;
    if (seconds <= 0.0) {
        return;
    }
    Counters& c = s.counters;
    const auto rate = [seconds](u64 value) { return static_cast<double>(value) / seconds; };
    LOG_INFO(HW_GPU,
             "Memory stage shadow (FRITCHII_MEMSTAGE_SHADOW={}, {:.1f} s): entries/s pass {:.0f} "
             "work {:.0f} level {:.0f}; level runs/s {:.0f}, resyncs/s {:.0f} ({:.1f} MB/s); "
             "draws/s {:.0f}: compared {:.0f} (served {:.0f}, render-enable off {:.0f}, sunk "
             "{:.0f}), in level entries {:.0f}; draw kinds of spec 3.4/s: indirect {:.0f}, list "
             "{:.0f}, GPU count {:.0f}, texel buffers {:.0f}, diagnostics {:.0f}, host "
             "conditional {:.0f}; facts misses/s {:.1f}, watched-page hits {}",
             g_mode, seconds, rate(c.pass), rate(c.work), rate(c.level), rate(c.level_runs),
             rate(c.resyncs), rate(c.resync_bytes) / 1e6, rate(c.back_draws), rate(c.compared),
             rate(c.served), rate(c.render_off), rate(c.sunk), rate(c.level_draws),
             rate(c.indirect), rate(c.list), rate(c.gpu_count), rate(c.texel),
             rate(c.diagnostic), rate(c.host_conditional), rate(c.facts_misses),
             c.watched_hits);
    Counters& t = s.totals;
    t.a_missing += c.a_missing;
    t.a_extra += c.a_extra;
    t.a_position += c.a_position;
    t.b_mismatch += c.b_mismatch;
    t.b_dirty += c.b_dirty;
    t.c_conflicts += c.c_conflicts;
    t.key_changes += c.key_changes;
    t.d_missing += c.d_missing;
    t.d_extra += c.d_extra;
    t.watched_hits += c.watched_hits;
    LOG_INFO(HW_GPU,
             "Memory stage shadow compare (must be 0; this window / session): a missing {} / {}, "
             "extra {} / {}, position {} / {}; b {} / {} (dirty flags {} / {}); c {} / {} "
             "({} facts checked, {} program keys changed, {} pipelines without a compared draw); "
             "d missing {} / {}, extra {} / {} ({} flushes, {} strict)",
             c.a_missing, t.a_missing, c.a_extra, t.a_extra, c.a_position, t.a_position,
             c.b_mismatch, t.b_mismatch, c.b_dirty, t.b_dirty, c.c_conflicts, t.c_conflicts,
             c.facts_checked, c.key_changes, c.facts_unchecked, c.d_missing, t.d_missing,
             c.d_extra, t.d_extra, c.flushes, c.flushes_strict);
    t.draw_flush_buffer += c.draw_flush_buffer;
    LOG_INFO(HW_GPU,
             "Memory stage shadow inside draws (between fill and record, this window): FlushRegion "
             "{} (texture half {}, buffer half {}, query half {}; buffer half this session {}, "
             "must stay 0 in play), collected writes applied {} batches of {} ranges, {} single "
             "invalidations",
             c.draw_flushes, c.draw_flush_texture, c.draw_flush_buffer, c.draw_flush_query,
             t.draw_flush_buffer, c.draw_gathers, c.draw_gather_ranges,
             c.draw_single_invalidations);
    std::vector<std::pair<u64, u64>> kinds(c.level_kinds.begin(), c.level_kinds.end());
    std::sort(kinds.begin(), kinds.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string top;
    for (size_t i = 0; i < kinds.size() && i < 12; ++i) {
        top += fmt::format("{}{}) {} {:.1f}/s", i == 0 ? "" : "; ", i + 1,
                           LevelName(kinds[i].first), rate(kinds[i].second));
    }
    LOG_INFO(HW_GPU, "Memory stage shadow level entries by kind and channel (top 12 of {}): {}",
             kinds.size(), top.empty() ? "none" : top);
    Report5c(s, seconds);
    std::unordered_map<u64, u64> keep;
    keep.swap(c.level_kinds);
    keep.clear();
    c = Counters{};
    c.level_kinds.swap(keep);
}

/// Normalised device ranges: sorted, touching ones merged.
std::vector<std::pair<DAddr, DAddr>> Normalise(std::vector<std::pair<DAddr, DAddr>> list) {
    std::sort(list.begin(), list.end());
    std::vector<std::pair<DAddr, DAddr>> merged;
    for (const auto& [begin, end] : list) {
        if (!merged.empty() && begin <= merged.back().second) {
            merged.back().second = (std::max)(merged.back().second, end);
        } else {
            merged.emplace_back(begin, end);
        }
    }
    return merged;
}

/// The first part of `a` that `b` does not cover, or {0, 0}.
std::pair<DAddr, DAddr> FirstUncovered(const std::vector<std::pair<DAddr, DAddr>>& a,
                                       const std::vector<std::pair<DAddr, DAddr>>& b) {
    size_t j = 0;
    for (const auto& [begin, end] : a) {
        DAddr at = begin;
        while (at < end) {
            while (j < b.size() && b[j].second <= at) {
                ++j;
            }
            if (j == b.size() || b[j].first > at) {
                return {at, j == b.size() ? end : (std::min)(end, b[j].first)};
            }
            at = b[j].second;
        }
    }
    return {0, 0};
}

/// 6b: what the stage cannot serve, as a LevelDraw id and the link's level draw kind. The list
/// draw ids (8 and up) name run starters only; the link counts list draws on their own
/// (MemoryStageLink::ListDraw).
enum class RingLevel : u32 {
    Indirect = 0,
    GpuCount = 1,
    Forced = 3,
    Refused = 4,
    Late = 6,
    List = 8,         ///< a list draw, the window closed or FRITCHII_MEMSTAGE_LIST_DRAWS off
    ListReadBack = 9, ///< a list draw the back reads back on the CPU
    ListRefused = 10, ///< a list draw the fill refused (facts miss, texel buffers, no room)
};

/// The copy's state in compare b's words (the narrow set, channel 0), hashed: the verify of
/// compare b with two threads. The back hashes the same words of its engine at the take.
template <typename E>
u64 StateHash(const E& e, const Maxwell3D* engine) {
    u64 hash = 0xcbf29ce484222325ULL;
    Visit(e, false, 0, engine, [&](const SegmentInfo& info, const u32* words) {
        hash = HashBytes(words, size_t{info.words} * sizeof(u32), hash);
    });
    return hash;
}

} // Anonymous namespace

void SetRingBack(bool on) {
    // On the back thread: its hooks are off while GPUMem keeps the state (THREAD = 2), and on
    // again with a fresh state when the ring ends (SHADOW >= 1), or the shadow is off again.
    if (on) {
        t_on_back = false;
        g_on.store(true, std::memory_order_relaxed);
        return;
    }
    g_on.store(g_mode != 0, std::memory_order_relaxed);
    if (g_mode != 0) {
        Start();
    } else {
        t_on_back = false;
    }
}

void StageStart(s32 channel) {
    State& s = S();
    s = State{};
    s.window_start = std::chrono::steady_clock::now();
    s.ring = true;
    s.channel = channel;
    s.back_channel = channel;
    s.full_binds = true;
    CacheInputs::g_mark_hook = &OnCacheMark;
    // The engine the back runs now: entries before the next engine entry are its.
    if (Tegra::Engines::EngineInterface* const engine = Access::BackEngine()) {
        s.engine = engine;
        if (auto* const maxwell3d = dynamic_cast<Maxwell3D*>(engine)) {
            s.engine_copy = FindCopy(s, maxwell3d);
        } else {
            s.engine_type = TypeOf(engine);
        }
    }
}

StageResult StageProcess(u64 position, bool in_run) {
    State& s = S();
    s.ring_in_run = in_run;
    s.ring_packets = 0;
    s.ring_ran_on_back = false;
    s.ring_publish_now = false;
    s.current = position;
    const size_t size = Process(s, position, false);
    s.cursor = position + size;
    if (!in_run) {
        s.run_from_dry_pass = s.entry_dry_pass;
        s.run_from_p3_pass = s.entry_p3 != State::P3Effect::None;
    }
    return StageResult{
        .size = size,
        .cls = static_cast<u8>(s.last_class),
        .hard = s.entry_hard,
        .packets = s.ring_packets,
        .launches = s.entry_launches,
        .channel = s.channel,
        .ran_on_back = s.ring_ran_on_back,
        .publish_now = s.ring_publish_now,
    };
}

bool StageExtendsRun(u64 position) {
    State& s = S();
    const u64* const entry = &Access::Ring()[position & Access::RingMask];
    const u64 head = entry[0];
    switch (static_cast<Kind>(head & 0xf)) {
    case Kind::Wrap:
        return true;
    case Kind::Engine:
        return dynamic_cast<Maxwell3D*>(
                   reinterpret_cast<Tegra::Engines::EngineInterface*>(entry[1])) == nullptr;
    case Kind::Call:
        switch (StageTagOf(head)) {
        case StageTag::Level:
        case StageTag::Cache:
        case StageTag::Pass:
            return true;
        case StageTag::Channel:
        case StageTag::Writes:
        case StageTag::FinishMacro:
        case StageTag::DrawLoop:
            return false;
        default:
            return true; // an unknown tag: hard, cannot be followed
        }
    case Kind::Method:
    case Kind::Multi:
    case Kind::Pairs:
    case Kind::CbWords:
        return s.engine != nullptr && s.engine_copy == nullptr;
    default:
        return false;
    }
}

void StageLend() {
    State& s = S();
    for (Copy& copy : s.copies) {
        if (!copy.stale) {
            TakeCacheFlags(copy.back->dirty.flags, copy.e->dirty.flags);
        }
    }
    ++s.counters.lends;
}

void StageLendDeferred() {
    // GPUMem: what StageLend would set, kept for the back. Nothing changes the copies' flags
    // until the run returns (the run's entries GPUMem follows meanwhile are other engines' and
    // closures), and the back reads this only after the run is published.
    State& s = S();
    s.lend_flags.clear();
    for (Copy& copy : s.copies) {
        if (!copy.stale) {
            s.lend_flags.emplace_back(copy.back, copy.e->dirty.flags & CacheFlagMask());
        }
    }
    ++s.counters.lends;
}

void BackTakeLend() {
    // The back, at the run's first entry: S() is GPUMem's state, which is parked on this run.
    for (const auto& [engine, flags] : S().lend_flags) {
        TakeCacheFlags(engine->dirty.flags, flags);
    }
}

void BackGatherLend() {
    // The back, in GPUMem's gather request: S() is GPUMem's, which waits for the answer. As
    // StageLend (level mode's lend), without its count.
    for (Copy& copy : S().copies) {
        if (!copy.stale) {
            TakeCacheFlags(copy.back->dirty.flags, copy.e->dirty.flags);
        }
    }
}

void BackGatherReturn() {
    // As StageReturn's take of the flags: the gather may have marked bindings dirty (a buffer the
    // query sync's lookups deleted).
    for (Copy& copy : S().copies) {
        if (!copy.stale) {
            TakeCacheFlags(copy.e->dirty.flags, copy.back->dirty.flags);
            ++copy.generation;
        }
    }
}

bool StageMayCopy(u64 position) {
    // As Process and Use: the copy an entry applies to, when it is missing or stale, is taken
    // from the back (Resync).
    State& s = S();
    const u64* const entry = &Access::Ring()[position & Access::RingMask];
    const u64 head = entry[0];
    const auto needs = [&s](Maxwell3D* engine) {
        if (engine == nullptr) {
            return false;
        }
        for (const Copy& copy : s.copies) {
            if (copy.back == engine) {
                return copy.stale;
            }
        }
        return true; // a new copy starts stale
    };
    switch (static_cast<Kind>(head & 0xf)) {
    case Kind::Method:
    case Kind::Multi:
    case Kind::Pairs:
    case Kind::StateDelta:
    case Kind::DrawBatch:
    case Kind::CbWords:
        return s.engine_copy != nullptr && s.engine_copy->stale;
    case Kind::Call:
        switch (StageTagOf(head)) {
        case StageTag::Writes:
        case StageTag::FinishMacro:
        case StageTag::DrawLoop:
            return needs(As3D(entry[2]));
        default:
            return false;
        }
    default:
        return false;
    }
}

u64 StageReturn(const void* drew_engine, const void* state_engine, s32 back_channel) {
    State& s = S();
    // The sync mirror (6.1): after every run, a sync may be pending. Not after a run P1 would
    // not have (dry: its first entry would have passed; its other entries set it themselves).
    if (!std::exchange(s.run_from_dry_pass, false)) {
        s.sync_unknown = true;
    }
    if (!std::exchange(s.run_from_p3_pass, false)) {
        s.p3_unknown = true; // P3's mirror (fable_stage_p3 4.1), as P3 would keep it
    }
    for (Copy& copy : s.copies) {
        if (!copy.stale) {
            TakeCacheFlags(copy.e->dirty.flags, copy.back->dirty.flags);
            ++copy.generation;
        }
    }
    ++s.counters.returns;
    if (Copy* const copy = std::exchange(s.take_execute, nullptr); copy != nullptr && !copy->stale) {
        copy->e->execute_on = copy->back->ShouldExecute();
        ++copy->generation;
        ++s.counters.execute_takes;
    }
    // Plan 3.3: after a run in which the back issued a draw, or wrote a GPU count into its draw
    // state, the draw manager's scalar state.
    Copy* draw_copy = nullptr;
    if (drew_engine != nullptr) {
        draw_copy = FindCopy(s, static_cast<Maxwell3D*>(const_cast<void*>(drew_engine)));
    } else if (state_engine != nullptr) {
        draw_copy = s.engine_copy;
    }
    if (draw_copy != nullptr && !draw_copy->stale) {
        CopyScalarDrawState(draw_copy->e->draw_manager.draw_state,
                            draw_copy->back->draw_manager.draw_state);
        ++draw_copy->generation;
        ++s.counters.draw_state_takes;
    }
    s.take_draw_state = nullptr;
    s.back_channel = back_channel;
    if (s.channel_stale) {
        s.channel = back_channel;
        s.channel_stale = false;
    }
    // The full copies an unfollowed entry asks for, now that the back is parked at the run's
    // end (spec 1.5).
    u64 copies = 0;
    for (Copy& copy : s.copies) {
        if (copy.stale) {
            Resync(s, copy);
            ++copies;
        }
    }
    return copies;
}

void StageTick() {
    State& s = S();
    const auto now = std::chrono::steady_clock::now();
    if (now - s.window_start >= std::chrono::seconds{10}) {
        Report(s, now);
    }
}

std::vector<std::pair<u64, std::size_t>>& StageRanges() {
    return S().ring_ranges;
}

u64 StageDrawSeq(u64 back_seq) {
    return std::exchange(S().ring_seq, back_seq);
}

u64 BackStateHash(const Maxwell3D& maxwell3d) {
    return StateHash(maxwell3d, &maxwell3d);
}

void BackProgramId(const Maxwell3D& maxwell3d, ProgramId& out) {
    ProgramWords(maxwell3d, out.words);
    out.key = HashProgramWords(out.words);
}

void StageProgramId(const MemStageShadowEngine& copy, ProgramId& out) {
    ProgramWords(copy, out.words);
    out.key = HashProgramWords(out.words);
}

void ReadSwitch() {
    const char* const value = std::getenv("FRITCHII_MEMSTAGE_SHADOW");
    g_mode = value != nullptr ? std::atoi(value) : 0;
    g_on.store(g_mode != 0, std::memory_order_relaxed);
    if (g_mode >= 3) {
        LOG_INFO(HW_GPU,
                 "Memory stage shadow (FRITCHII_MEMSTAGE_SHADOW={}): the stage's inputs run on the "
                 "back and are compared at every draw, and the buffer cache's fills read the "
                 "stage's copy outside runs (needs FRITCHII_MEMSTAGE_FULL_BINDS=1; runs as 2 "
                 "while it is off)",
                 g_mode);
    } else if (g_mode != 0) {
        LOG_INFO(HW_GPU,
                 "Memory stage shadow (FRITCHII_MEMSTAGE_SHADOW={}): the stage's inputs run on the "
                 "back and are compared at every draw; nothing they compute is used",
                 g_mode);
    }
}

int Mode() {
    return g_mode;
}

void Start() {
    State& s = S();
    s = State{};
    s.window_start = std::chrono::steady_clock::now();
    t_on_back = true;
    CacheInputs::g_mark_hook = &OnCacheMark;
}

void Before(u64 position) {
    State& s = S();
    // The back finished the entry before: its draw's fill, if any, is done.
    Settle(s);
    s.match_copy = nullptr;
    s.full_binds = VideoCommon::MemoryStageLink::FullVertexBinds();
    if (position < s.cursor) [[unlikely]] {
        // The ring started over (not expected while the back runs): start from the back's state.
        s.cursor = position;
        for (Copy& copy : s.copies) {
            copy.stale = true;
        }
    }
    // Entries the back ran with the one before (its run of constant buffer words) come first.
    while (s.cursor < position) {
        const size_t size = Process(s, s.cursor, true);
        s.cursor += size != 0 ? size : Access::RingWords - (s.cursor & Access::RingMask);
    }
    // The copy and the back are at the same entry here: what plan 3.3 takes from the back after a
    // hard entry is taken now.
    if (Copy* const copy = std::exchange(s.take_execute, nullptr); copy != nullptr && !copy->stale) {
        copy->e->execute_on = copy->back->ShouldExecute();
        ++copy->generation;
        ++s.counters.execute_takes;
    }
    if (Copy* const copy = std::exchange(s.take_draw_state, nullptr);
        copy != nullptr && !copy->stale) {
        CopyScalarDrawState(copy->e->draw_manager.draw_state, copy->back->draw_manager.draw_state);
        ++copy->generation;
        ++s.counters.draw_state_takes;
    }
    if (s.in_run && OwnsFlags(s)) {
        Return(s);
    }
    s.current = position;
    s.current_level = false;
    s.in_draw_window = false;
    s.has_current_draw = false;
    // Shadow draws of earlier entries the back did not issue (a sunk write's deferred draw may
    // still come: the back flushes it when it consumes its sink).
    while (!s.records.empty() && s.records.front().position < position &&
           !s.records.front().sunk) {
        Record record = std::move(s.records.front());
        s.records.pop_front();
        DropUnmatched(s, std::move(record), "by the next entry");
    }
    const size_t size = Process(s, position, false);
    s.cursor = position + (size != 0 ? size : Access::RingWords - (position & Access::RingMask));
    if (++s.entries_since_check >= 4096) {
        s.entries_since_check = 0;
        const auto now = std::chrono::steady_clock::now();
        if (now - s.window_start >= std::chrono::seconds{10}) {
            Report(s, now);
        }
    }
}

void BackDraw(Maxwell3D& maxwell3d, bool indexed, u32 instances, bool indirect) {
    if (MemoryStageLink::RingActive() && GpuSplit::OnBackThread()) [[unlikely]] {
        // 6b: every draw counts (the packets' sequence numbers).
        MemoryStageLink::Get().NoteBackDraw(&maxwell3d);
        return;
    }
    if (!t_on_back) {
        return;
    }
    State& s = S();
    Counters& c = s.counters;
    ++c.back_draws;
    // An earlier draw of this entry: its fill is done (or it had none).
    Settle(s);
    s.match_copy = nullptr;
    // Plan 3.3: after a draw the back issues in a hard entry (a level draw in 6b), and after an
    // indirect one, the copy takes the draw manager's scalar state at the next entry.
    if (s.current_level || indirect) {
        s.take_draw_state = FindCopy(s, &maxwell3d);
    }
    const u64 invalidations = g_flag_invalidations.load(std::memory_order_relaxed);
    const bool flags_strict = invalidations == s.invalidations_seen;
    s.invalidations_seen = invalidations;
    const auto& in = maxwell3d.draw_manager.indirect_state;
    if (indirect) {
        if (in.list_words) {
            ++c.list;
        } else if (in.include_count) {
            ++c.gpu_count;
        } else {
            ++c.indirect;
        }
    }
    if (s.host_condition) {
        ++c.host_conditional;
    }
    if (s.records.empty() || s.records.front().position > s.current) {
        if (s.current_level) {
            ++c.level_draws;
            return;
        }
        ++c.a_missing;
        if (s.a_logged < 16) {
            ++s.a_logged;
            LOG_WARNING(HW_GPU,
                        "Memory stage shadow (a): the back drew at entry {:#x} (indexed {}, "
                        "{} instances, indirect {}), the shadow did not",
                        s.current, indexed, instances, indirect);
        }
        return;
    }
    Record record = std::move(s.records.front());
    s.records.pop_front();
    ++c.compared;
    if (record.sunk) {
        ++c.sunk;
    } else if (record.position != s.current) {
        ++c.a_position;
        if (s.a_logged < 16) {
            ++s.a_logged;
            LOG_WARNING(HW_GPU,
                        "Memory stage shadow (a): the shadow drew at entry {:#x}, the back at "
                        "{:#x}",
                        record.position, s.current);
        }
    }
    if (record.executes) {
        ++c.served;
    } else {
        ++c.render_off;
    }
    // Compare b.
    const bool full = g_mode >= 2;
    size_t at = 0;
    u32 differences = 0;
    std::string diff;
    Visit(maxwell3d, full, s.back_channel, &maxwell3d,
          [&](const SegmentInfo& info, const u32* words) {
              for (u32 i = 0; i < info.words; ++i, ++at) {
                  if (at >= record.words.size() || record.words[at] == words[i]) [[likely]] {
                      continue;
                  }
                  if (differences++ < 16) {
                      diff += fmt::format("; {}: back {:#x}, shadow {:#x}", WordName(info, i),
                                          words[i], record.words[at]);
                  }
              }
          });
    // The cache's flags (plan 4.2): equal both ways, or the copy's a subset of the back's when
    // the state tracker raised them since the back's previous draw. Not at SHADOW=3 while the
    // copy holds them.
    if (!OwnsFlags(s)) {
        const Flags& mask = CacheFlagMask();
        const Flags shadow_flags = record.dirty & mask;
        const Flags back_flags = maxwell3d.dirty.flags & mask;
        const Flags missing_flags = shadow_flags & ~back_flags;
        const Flags extra_flags = flags_strict ? back_flags & ~shadow_flags : Flags{};
        ++c.flag_compares;
        c.flag_compares_strict += flags_strict ? 1 : 0;
        if (missing_flags.any() || extra_flags.any()) {
            ++c.b_dirty;
            for (u32 flag = CacheFlagFirst; flag <= CacheFlagLast; ++flag) {
                if (missing_flags[flag]) {
                    ++c.flag_missing[flag - CacheFlagFirst];
                    if (differences++ < 16) {
                        diff += fmt::format("; dirty flag {} raised on the shadow, not on the back",
                                            flag);
                    }
                }
                if (extra_flags[flag]) {
                    ++c.flag_extra[flag - CacheFlagFirst];
                    if (differences++ < 16) {
                        diff += fmt::format("; dirty flag {} raised on the back, not on the shadow",
                                            flag);
                    }
                }
            }
        }
    }
    // For the fill (Settle, BackFill).
    s.match_copy = FindCopy(s, record.engine != nullptr ? record.engine : &maxwell3d);
    s.match_generation = record.generation;
    s.match_hard = record.hard;
    s.match_indirect = record.indirect;
    s.match_channel = record.channel;
    if (differences != 0) {
        ++c.b_mismatch;
        if (s.b_logged < 16) {
            ++s.b_logged;
            LOG_WARNING(HW_GPU,
                        "Memory stage shadow (b): draw at entry {:#x} (shadow entry {:#x}{}), {} "
                        "differences{}",
                        s.current, record.position, record.sunk ? ", sunk" : "", differences,
                        diff);
        }
    }
    s.due.insert(s.due.end(), record.ranges.begin(), record.ranges.end());
    s.has_current_draw = true;
    s.current_program_key = record.program_key;
    Recycle(s, std::move(record));
}

void BackChannel(s32 bind_id) {
    if (!t_on_back) {
        return;
    }
    S().back_channel = bind_id;
}

void BackCondition(bool host) {
    if (!t_on_back) {
        return;
    }
    S().host_condition = host;
}

void BackFlush(const Tegra::MemoryManager& memory,
               std::span<const std::pair<DAddr, std::size_t>> ranges) {
    if (!t_on_back) {
        return;
    }
    State& s = S();
    Counters& c = s.counters;
    // What the back must hand InnerInvalidation now: the ranges of the draws it issued, and those
    // written by entries before the current one that no draw took yet.
    std::vector<std::pair<DAddr, DAddr>> expected;
    const auto take = [&](std::vector<Range>& list, u64 before) {
        auto keep = list.begin();
        for (auto it = list.begin(); it != list.end(); ++it) {
            if (it->memory == &memory && it->position < before) {
                expected.emplace_back(it->begin, it->begin + it->size);
            } else {
                *keep++ = *it;
            }
        }
        list.erase(keep, list.end());
    };
    take(s.due, ~u64{0});
    take(s.ranges_since_draw, s.current);
    const bool strict = !s.level_since_flush;
    s.level_since_flush = false;
    if (expected.empty() && ranges.empty()) {
        return;
    }
    ++c.flushes;
    std::vector<std::pair<DAddr, DAddr>> back;
    back.reserve(ranges.size());
    for (const auto& [begin, size] : ranges) {
        back.emplace_back(begin, begin + size);
    }
    const auto shadow_set = Normalise(std::move(expected));
    const auto back_set = Normalise(std::move(back));
    const auto missing = FirstUncovered(shadow_set, back_set);
    if (missing.second != 0) {
        ++c.d_missing;
        if (s.d_logged < 16) {
            ++s.d_logged;
            LOG_WARNING(HW_GPU,
                        "Memory stage shadow (d): at entry {:#x} the shadow's constant buffer "
                        "words cover device {:#x}-{:#x}, the back's invalidation does not ({} "
                        "shadow ranges, {} back ranges)",
                        s.current, missing.first, missing.second, shadow_set.size(),
                        back_set.size());
        }
    }
    if (strict) {
        ++c.flushes_strict;
        const auto extra = FirstUncovered(back_set, shadow_set);
        if (extra.second != 0) {
            ++c.d_extra;
            if (s.d_logged < 16) {
                ++s.d_logged;
                LOG_WARNING(HW_GPU,
                            "Memory stage shadow (d): at entry {:#x} the back invalidates device "
                            "{:#x}-{:#x}, which no shadow write covers (no level entry since the "
                            "last flush; {} shadow ranges, {} back ranges)",
                            s.current, extra.first, extra.second, shadow_set.size(),
                            back_set.size());
            }
        }
    }
}

void BackPipeline(const std::array<u64, 6>& hashes, u64 facts_digest, u32 kinds) {
    if (!t_on_back) {
        return;
    }
    State& s = S();
    Counters& c = s.counters;
    if ((kinds & PipelineTexel) != 0) {
        ++c.texel;
    }
    if ((kinds & PipelineDiagnostic) != 0) {
        ++c.diagnostic;
    }
    if (!s.has_current_draw) {
        ++c.facts_unchecked;
        return;
    }
    ++c.facts_checked;
    const u64 shader_key = HashBytes(hashes.data(), sizeof(hashes));
    const auto [program, new_program] =
        s.program_sets.try_emplace(s.current_program_key, shader_key);
    if (new_program) {
        ++c.facts_misses;
    } else if (program->second != shader_key) {
        // The same program registers with other code: the stage's entry is dropped (spec 4.4).
        ++c.facts_misses;
        ++c.key_changes;
        program->second = shader_key;
    }
    const auto [facts, new_facts] = s.facts.try_emplace(shader_key, facts_digest);
    if (!new_facts && facts->second != facts_digest) {
        ++c.c_conflicts;
        if (s.c_logged < 16) {
            ++s.c_logged;
            LOG_WARNING(HW_GPU,
                        "Memory stage shadow (c): program set vs {:#x} fs {:#x} has facts {:#x}, "
                        "the table has {:#x} (the facts key is too small)",
                        hashes[1], hashes[5], facts_digest, facts->second);
        }
    }
    s.has_current_draw = false;
}

void BackDrawWindow(bool open) {
    if (!t_on_back) {
        return;
    }
    State& s = S();
    s.in_draw_window = open;
    if (!open) {
        // The fill and Configure are done.
        Settle(s);
    }
}

MemStageShadowEngine* BackFill(Maxwell3D& maxwell3d, bool indirect) {
    if (!t_on_back) {
        return nullptr;
    }
    State& s = S();
    Settle(s);
    s.fill_pending = true;
    s.fill_engine = &maxwell3d;
    s.fill_pre = maxwell3d.dirty.flags & CacheFlagMask();
    s.fill_from_copy = false;
    if (g_mode < 3) {
        return nullptr;
    }
    Copy* const copy = s.match_copy;
    FillFrom from = FillFrom::Copy;
    if (!s.full_binds) {
        from = FillFrom::NoFullBinds;
    } else if (copy == nullptr) {
        from = FillFrom::NoRecord;
    } else if (indirect || s.match_indirect) {
        from = FillFrom::Indirect;
    } else if (s.match_hard || s.current_level) {
        from = FillFrom::Hard;
    } else if (copy->back != &maxwell3d || s.match_channel != s.back_channel) {
        from = FillFrom::Other;
    } else if (copy->stale) {
        from = FillFrom::Stale;
    } else if (copy->generation != s.match_generation) {
        from = FillFrom::MovedOn;
    }
    ++s.counters.fill_from[static_cast<size_t>(from)];
    if (from != FillFrom::Copy) {
        return nullptr;
    }
    s.fill_from_copy = true;
    return copy->e.get();
}

void BackDrawStateWritten() {
    if (MemoryStageLink::RingActive() && GpuSplit::OnBackThread()) [[unlikely]] {
        MemoryStageLink::Get().NoteBackDrawState(reinterpret_cast<const void*>(1));
        return;
    }
    if (!t_on_back) {
        return;
    }
    State& s = S();
    ++s.counters.draw_state_written;
    if (s.engine_copy != nullptr) {
        s.take_draw_state = s.engine_copy;
    }
}

void BackTrackerRecord(bool had_record, u32 mismatches) {
    if (!t_on_back) {
        return;
    }
    State& s = S();
    Counters& c = s.counters;
    if (!had_record) {
        ++c.tracker_without;
        return;
    }
    ++c.tracker_compared;
    if (mismatches != 0) {
        ++c.tracker_mismatches;
        if (s.t_logged < 16) {
            ++s.t_logged;
            LOG_WARNING(HW_GPU,
                        "Memory stage shadow (OoT trackers): the draw packet's record of uniform "
                        "slots 3, 4, 6 differs from today's query at entry {:#x} (slot mask {:#x})",
                        s.current, mismatches);
        }
    }
}

void BackFlushRegion(bool texture, bool buffer, bool query) {
    if (!t_on_back) {
        return;
    }
    State& s = S();
    if (!s.in_draw_window) {
        return;
    }
    Counters& c = s.counters;
    ++c.draw_flushes;
    c.draw_flush_texture += texture ? 1 : 0;
    c.draw_flush_buffer += buffer ? 1 : 0;
    c.draw_flush_query += query ? 1 : 0;
}

void BackCacheInvalidation(bool batch, u64 ranges) {
    if (!t_on_back) {
        return;
    }
    State& s = S();
    if (!s.in_draw_window) {
        return;
    }
    Counters& c = s.counters;
    if (batch) {
        ++c.draw_gathers;
        c.draw_gather_ranges += ranges;
    } else {
        ++c.draw_single_invalidations;
    }
}

void SetWatchFilter(const PageFilter* filter) {
    g_watch_filter = filter;
}

} // namespace MemStageShadow

void MemStageShadowEngine::StageDraw(bool indexed, u32 instance_count, bool indirect) {
    namespace Shadow = MemStageShadow;
    Shadow::State& s = Shadow::S();
    if (s.ring) {
        // 6b (memstage_shadow.cpp's header): serve the draw, or make its entry hard.
        // Profiling: the draw's own share of GPUMem's entries (its fill is a zone of its own).
        const Nx2Profile::Scope zone{Nx2Profile::Zone::StageDrawSetup, Nx2Profile::Fine(), false};
        MemoryStageLink& link = MemoryStageLink::Get();
        const u64 seq = ++s.ring_seq;
        ++s.entry_draws;
        if (!execute_on) {
            return; // the back skips it too (DrawManager's render-enable test), and asks for nothing
        }
        if (s.ring_in_run) {
            return; // the back fills it in the run
        }
        if (s.entry_hard) {
            link.NoteLevelDraw(static_cast<u32>(Shadow::RingLevel::Late));
            return;
        }
        const auto level = [&](Shadow::RingLevel kind) {
            link.NoteLevelDraw(static_cast<u32>(kind));
            Shadow::Hard(s, Shadow::LevelType::LevelDraw, static_cast<u64>(kind));
        };
        const auto& backend = link.Backend();
        // List draws (FRITCHII_MEMSTAGE_LIST_DRAWS, the window open): DrawEnd's list branch, a
        // non-indexed draw from the two words at indirect_state.indirect_start_address. Served
        // when the back would draw them indirectly (not read them back): the fill binds the
        // words from the copy's indirect parameters and the packet carries their buffer
        // (vk_rasterizer.cpp, StageFillDraw). Every level one starts a run (entry_hard is false
        // here): counted by why.
        const bool list = indirect && draw_manager.indirect_state.list_words;
        const auto list_level = [&](MemoryStageLink::ListDraw why, Shadow::RingLevel kind) {
            link.NoteListDraw(why);
            Shadow::Hard(s, Shadow::LevelType::LevelDraw, static_cast<u64>(kind));
        };
        if (list) {
            if (!MemoryStageLink::ListDrawsServed()) {
                return list_level(MemoryStageLink::ListDraw::Closed, Shadow::RingLevel::List);
            }
            if (backend.list_reads_back == nullptr ||
                backend.list_reads_back(backend.context, *this)) {
                return list_level(MemoryStageLink::ListDraw::ReadBack,
                                  Shadow::RingLevel::ListReadBack);
            }
        } else if (indirect) {
            return level(Shadow::RingLevel::Indirect);
        }
        if (!indexed && Tegra::Nx2IndirectGrid::AnyLive() &&
            Tegra::Nx2IndirectGrid::LooksLikeDrawPlaceholder(
                draw_manager.draw_state.vertex_buffer.count)) {
            return level(Shadow::RingLevel::GpuCount);
        }
        if (Shadow::g_ring_level_draws.load(std::memory_order_relaxed)) {
            if (list) {
                link.NoteListDraw(MemoryStageLink::ListDraw::Forced);
            }
            return level(Shadow::RingLevel::Forced);
        }
        const u64 hash = MemoryStageLink::RingVerify() ? Shadow::StateHash(*this, back) : 0;
        // A list draw is drawn non-indexed (DrawListWords' PrepareDraw); DrawManager hands every
        // indirect draw over as indexed.
        if (backend.fill_draw == nullptr ||
            !backend.fill_draw(backend.context, *this, list ? false : indexed,
                               s.processing_position, seq, hash, s.ring_ranges)) {
            // fill_draw counted the kind (facts miss, texel buffers, no room).
            if (list) {
                return list_level(MemoryStageLink::ListDraw::Refused,
                                  Shadow::RingLevel::ListRefused);
            }
            Shadow::Hard(s, Shadow::LevelType::LevelDraw,
                         static_cast<u64>(Shadow::RingLevel::Refused));
            return;
        }
        if (list) {
            link.NoteListDraw(MemoryStageLink::ListDraw::Served);
        }
        ++s.entry_served;
        ++s.counters.served_draws;
        ++s.ring_packets;
        static_cast<void>(instance_count);
        return;
    }
    Shadow::Record record = Shadow::TakeSpare(s);
    record.position = s.processing_position;
    record.sunk = s.processing_sunk;
    record.indirect = indirect;
    record.executes = execute_on;
    record.engine = back;
    record.channel = s.channel;
    record.program_key = Shadow::ProgramKey(*this);
    // 5c: the flags are the cache's own state now (plan 4.2), not reset at a draw.
    record.dirty = dirty.flags;
    record.hard = s.entry_hard;
    if (Shadow::Copy* const copy = Shadow::FindCopy(s, back)) {
        record.raised_before = copy->raised;
        copy->raised.reset();
        record.generation = copy->generation;
    }
    ++s.entry_draws;
    if (execute_on && !indirect) {
        ++s.entry_served;
        ++s.counters.served_draws;
    }
    Shadow::Visit(*this, Shadow::g_mode >= 2, s.channel, back,
                  [&](const Shadow::SegmentInfo& info, const u32* words) {
                      record.words.insert(record.words.end(), words, words + info.words);
                  });
    record.ranges.swap(s.ranges_since_draw);
    s.ranges_since_draw.clear();
    static_cast<void>(indexed);
    static_cast<void>(instance_count);
    s.records.push_back(std::move(record));
}

void MemStageShadowEngine::NoteRaised(u32 flag) {
    namespace Shadow = MemStageShadow;
    if (Shadow::Copy* const copy = Shadow::FindCopy(Shadow::S(), back)) {
        copy->raised[flag] = true;
        ++copy->generation;
    }
}

void MemStageShadowEngine::NoteChanged() {
    namespace Shadow = MemStageShadow;
    if (Shadow::Copy* const copy = Shadow::FindCopy(Shadow::S(), back)) {
        ++copy->generation;
    }
}

} // namespace VideoCommon
