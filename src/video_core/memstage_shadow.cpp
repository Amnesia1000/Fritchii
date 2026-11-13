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
// What it does not cover (yet): compute and the other engines (level by design); 3D methods that
// are level in the first version (spec 3.2); render_enable.mode (level: the back asks the query
// cache and flushes, AccelerateConditionalRendering); the query and shader code pages of the
// watched-page filter (only image pages and the texture/sampler tables are tested).

#include <algorithm>
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

#include "common/logging.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/kepler_memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/memstage_shadow.h"
#include "video_core/memstage_shadow_engine.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/page_filter.h"

namespace VideoCommon {

/// Friend of GpuSplit and Maxwell3D: the ring's layout and the back's engine settings.
struct MemStageShadowAccess {
    using Maxwell3D = Tegra::Engines::Maxwell3D;
    using Kind = GpuSplit::Kind;
    static constexpr u64 IsLast = GpuSplit::IsLast;
    static constexpr u64 HasSegment = GpuSplit::HasSegment;
    static constexpr u64 Direct = GpuSplit::Direct;
    static constexpr u64 Sink = GpuSplit::Sink;
    static constexpr size_t RingWords = GpuSplit::RingWords;
    static constexpr size_t RingMask = GpuSplit::RingMask;

    static const u64* Ring() {
        return GpuSplit::Get().ring.get();
    }
    static bool SkipSame(const Maxwell3D& maxwell3d) {
        return maxwell3d.skip_same_value_writes;
    }
    static bool FrontMacros(const Maxwell3D& maxwell3d) {
        return maxwell3d.front_macros;
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

enum class Class : u8 { Pass, Work, Level };

/// What a level entry is, for the by-kind counter.
enum class LevelType : u8 {
    Method3D,   ///< id: the method
    Macro3D,    ///< id: the macro method
    Packet3D,   ///< a tagged draw packet holding a level method; id: the method
    Batch3D,    ///< a draw batch holding a level method; id: the method
    Engine,     ///< methods on another engine; id: EngineType
    NoEngine,   ///< methods before any engine entry
    Call,       ///< an untagged closure; id: its address in the executable
    CbWatched,  ///< constant buffer words onto a watched page
    CbMode,     ///< constant buffer words with EXP107 (the stage implements the default mode)
    UnknownTag, ///< a tag the shadow does not know
};
enum class EngineType : u32 { KeplerCompute, KeplerMemory, MaxwellDMA, Fermi2D, Other };

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
template <typename E>
u64 ProgramKey(const E& e) {
    u64 hash = HashBytes(&e.regs.program_region, sizeof(e.regs.program_region));
    return HashBytes(&e.regs.pipelines, sizeof(e.regs.pipelines), hash);
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
    Flags dirty{};
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
    Counters counters;
    Counters totals;
    std::chrono::steady_clock::time_point window_start{};
    u64 entries_since_check{};
    u32 b_logged{}, a_logged{}, c_logged{}, d_logged{};
#ifdef _WIN32
    u64 module_base{};
#endif
};

State& S() {
    static State state;
    return state;
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
    e.regs.reg_array[method] = value;
    for (const auto& table : *c.tables) {
        e.dirty.flags[table[method]] = true;
    }
}

/// Maxwell3D::WriteStateRegister (and ConsumeSinkImpl, ApplyPairs, ApplyFrontDelta per write).
void StateWrite(Copy& c, u32 method, u32 argument) {
    const u32 value = ShadowRam(*c.e, method, argument);
    if (!c.skip_same || c.e->regs.reg_array[method] != value) {
        DirtyStore(c, method, value);
    }
}

bool CbMode() {
    return VideoCore::Nx2Debug::Experiment(107) != 0;
}

/// Whether [address, address + size) is on a page the back's caches watch (join 2): image
/// pages (the texture cache's filter) or the texture header and sampler pools.
bool Watched(const Copy& c, GPUVAddr address, u64 size) {
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
    const std::optional<DAddr> device = c.memory->GpuToCpuAddress(address);
    return device && g_watch_filter->Any(*device, size);
}

/// The device ranges of a constant buffer write at [address, address + size), rounded as the
/// back's invalidation rounds (32 bytes), split where the GPU pages map.
void Collect(State& s, const Copy& c, GPUVAddr address, u64 size) {
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

/// Maxwell3D::ProcessCBMultiData: computes the write and its range, writes nothing.
void ConstBufferWords(State& s, Copy& c, u32 amount) {
    Regs& regs = c.e->regs;
    const GPUVAddr address = regs.const_buffer.Address();
    const u64 copy_size = u64{amount} * sizeof(u32);
    const u64 room = regs.const_buffer.offset < regs.const_buffer.size
                         ? regs.const_buffer.size - regs.const_buffer.offset
                         : 0;
    const u64 write_size = (std::min)(copy_size, room / sizeof(u32) * sizeof(u32));
    if (write_size != 0) {
        const GPUVAddr begin = address + regs.const_buffer.offset;
        if (Watched(c, begin, write_size)) [[unlikely]] {
            // Join 2: the stage would leave this entry to the back. Counted; the copy is
            // taken from the back again at the next entry, as after any level entry.
            ++s.counters.watched_hits;
            ++s.counters.level_kinds[LevelKey(LevelType::CbWatched, s.channel, 0)];
            c.stale = true;
        }
        Collect(s, c, begin, write_size);
    }
    regs.const_buffer.offset += static_cast<u32>(copy_size);
}

/// Maxwell3D::ProcessCBBind: the binding the cache gets (Bind/DisableGraphicsUniformBuffer).
void ConstBufferBind(Copy& c, size_t stage) {
    Engine& e = *c.e;
    const auto& bind = e.regs.bind_groups[stage];
    if (bind.shader_slot >= Regs::MaxConstBuffers) {
        return;
    }
    auto& buffer = e.state.shader_stages[stage].const_buffers[bind.shader_slot];
    buffer.enabled = bind.valid.Value() != 0;
    buffer.address = e.regs.const_buffer.Address();
    buffer.size = e.regs.const_buffer.size;
}

/// The executable 3D methods the first version leaves to the back (spec 3.2 and 8.1).
bool IsLevelMethod(const Copy& c, u32 method) {
    if (method >= MacroRegistersStart) {
        return true;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
    // The back asks the query cache (host conditional rendering) and flushes the invalidation
    // ranges first (RasterizerVulkan::AccelerateConditionalRendering): level in the first version.
    case MAXWELL3D_REG_INDEX(render_enable.mode):
        return true;
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
        return !c.front_macros;
    default:
        return method - CbData < 16 && CbMode();
    }
}

/// Maxwell3D::ProcessMethodCall for the methods that are not level.
void MethodCall(State& s, Copy& c, u32 method, u32 value, u32 nonshadow) {
    Engine& e = *c.e;
    if (method - CbData < 16) {
        return ConstBufferWords(s, c, 1);
    }
    switch (method) {
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

/// Maxwell3D::CallMethod for a method that is not level (CallMethodSlow's order).
void CallMethod(State& s, Copy& c, u32 method, u32 argument) {
    if (!c.back->execution_mask[method]) {
        return StateWrite(c, method, argument);
    }
    const u32 value = ShadowRam(*c.e, method, argument);
    DirtyStore(c, method, value);
    MethodCall(s, c, method, value, argument);
}

/// Maxwell3D::CallMultiMethod for a method that is not level.
void CallMultiMethod(State& s, Copy& c, u32 method, const u32* words, u32 amount) {
    Engine& e = *c.e;
    if (e.draw_manager.HasDeferredDraw() &&
        !Maxwell3D::DrawManager::IsInstanceLoopMethod(method)) [[unlikely]] {
        e.draw_manager.DrawDeferred(e, true);
    }
    if (method - CbData < 16) {
        return ConstBufferWords(s, c, amount);
    }
    switch (method) {
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
    e.dirty.flags.reset();
    c.stale = false;
    ++s.counters.resyncs;
    s.counters.resync_bytes += 2 * sizeof(Regs) + sizeof(e.state) + sizeof(e.root_tables);
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

void CountLevel(State& s, LevelType type, u64 id) {
    ++s.counters.level_kinds[LevelKey(type, s.channel, id)];
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

/// Classifies the entry at `position`, applies it to the copies unless it is level, and returns
/// its size in words (for a wrap: the rest of the ring).
size_t Process(State& s, u64 position, bool executed) {
    const u64* const ring = Access::Ring();
    const size_t index = position & Access::RingMask;
    const u64* const entry = &ring[index];
    const u64 head = entry[0];
    s.processing_position = position;
    s.processing_sunk = false;
    if (s.channel_stale && !executed) {
        // After a level run, the channel as the back has it (an untagged closure may bind one).
        s.channel = s.back_channel;
        s.channel_stale = false;
    }
    Class cls = Class::Pass;
    size_t size = 1;
    auto level = [&](LevelType type, u64 id) {
        cls = Class::Level;
        CountLevel(s, type, id);
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
            level(LevelType::Engine, static_cast<u64>(s.engine_type));
        }
        break;
    }
    case Kind::Method: {
        size = (head & Access::HasSegment) != 0 ? 2 : 1;
        const u32 method = static_cast<u32>((head >> 8) & 0xffff);
        const u32 argument = static_cast<u32>(head >> 32);
        if (s.engine == nullptr) {
            level(LevelType::NoEngine, 0);
            break;
        }
        if (s.engine_copy == nullptr) {
            level(LevelType::Engine, static_cast<u64>(s.engine_type));
            break;
        }
        Copy& c0 = *s.engine_copy;
        const bool executable = c0.back->execution_mask[method];
        if ((head & Access::Sink) != 0) {
            level(LevelType::Method3D, method);
            break;
        }
        if ((head & Access::Direct) != 0 || executable) {
            if (executable && IsLevelMethod(c0, method)) {
                level(method >= MacroRegistersStart ? LevelType::Macro3D : LevelType::Method3D,
                      method);
                break;
            }
            cls = executable ? Class::Work : Class::Pass;
            if (Copy* const c = Use(s, c0.back, executed)) {
                CallMethod(s, *c, method, argument);
            }
            break;
        }
        // Into the back's method sink: applied now, which is the same state at the next
        // executable method (spec 4.2); a deferred draw it flushes comes later on the back.
        if (Copy* const c = Use(s, c0.back, executed)) {
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
            level(LevelType::NoEngine, 0);
            break;
        }
        if (s.engine_copy == nullptr) {
            level(LevelType::Engine, static_cast<u64>(s.engine_type));
            break;
        }
        Copy& c0 = *s.engine_copy;
        if (method >= MacroRegistersStart) {
            level(LevelType::Macro3D, method);
            break;
        }
        const bool executable = c0.back->execution_mask[method];
        if (executable && IsLevelMethod(c0, method)) {
            level(LevelType::Method3D, method);
            break;
        }
        cls = executable ? Class::Work : Class::Pass;
        if (Copy* const c = Use(s, c0.back, executed)) {
            CallMultiMethod(s, *c, method, reinterpret_cast<const u32*>(entry + data), amount);
        }
        break;
    }
    case Kind::Call: {
        const size_t payload = static_cast<size_t>((head >> 8) & 0xffffff);
        size = 2 + payload;
        const u64* const words = entry + 2;
        switch (StageTagOf(head)) {
        case StageTag::Level:
            level(LevelType::Call, Rva(entry[1]));
            break;
        case StageTag::Pass:
            break;
        case StageTag::Channel:
            cls = Class::Work;
            s.channel = static_cast<s32>(static_cast<s64>(words[0]));
            break;
        case StageTag::Writes: {
            Maxwell3D* const engine = As3D(words[0]);
            const u32 count = static_cast<u32>(words[2]);
            const u64* const writes = words + 3;
            Copy* const c0 = engine != nullptr ? FindCopy(s, engine) : nullptr;
            if (c0 == nullptr) {
                level(LevelType::Packet3D, 0);
                break;
            }
            for (u32 i = 0; i < count; ++i) {
                const u32 method = static_cast<u32>(writes[i] & 0xffff);
                if (c0->back->execution_mask[method] && IsLevelMethod(*c0, method)) {
                    level(LevelType::Packet3D, method);
                    break;
                }
            }
            if (cls == Class::Level) {
                break;
            }
            cls = Class::Work;
            if (Copy* const c = Use(s, engine, executed)) {
                for (u32 i = 0; i < count; ++i) {
                    CallMethod(s, *c, static_cast<u32>(writes[i] & 0xffff),
                               static_cast<u32>(writes[i] >> 32));
                }
            }
            break;
        }
        case StageTag::FinishMacro: {
            cls = Class::Work;
            if (Copy* const c = Use(s, As3D(words[0]), executed)) {
                // ConsumeSink (the shadow applied its writes already), then DrawDeferred.
                c->e->draw_manager.DrawDeferred(*c->e);
            }
            break;
        }
        case StageTag::DrawLoop: {
            cls = Class::Work;
            if (Copy* const c = Use(s, As3D(words[0]), executed)) {
                if (static_cast<StageDrawLoop>(words[1]) == StageDrawLoop::FoldFirstInstance) {
                    c->e->draw_manager.FoldFirstInstance();
                } else {
                    c->e->draw_manager.AddLoopInstances(*c->e, static_cast<u32>(words[2]));
                }
            }
            break;
        }
        default:
            level(LevelType::UnknownTag, static_cast<u64>(StageTagOf(head)));
            break;
        }
        break;
    }
    case Kind::Pairs:
    case Kind::StateDelta: {
        const u32 count = static_cast<u32>(head >> 32);
        size = 1 + size_t{count};
        if (s.engine == nullptr) {
            level(LevelType::NoEngine, 0);
            break;
        }
        if (s.engine_copy == nullptr) {
            level(LevelType::Engine, static_cast<u64>(s.engine_type));
            break;
        }
        if (Copy* const c = Use(s, s.engine_copy->back, executed)) {
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
            level(s.engine == nullptr ? LevelType::NoEngine : LevelType::Engine,
                  s.engine == nullptr ? 0 : static_cast<u64>(s.engine_type));
            break;
        }
        Copy& c0 = *s.engine_copy;
        for (u32 i = 0; i < count; ++i) {
            const u32 method = static_cast<u32>(entry[1 + i] & 0xffff);
            if (c0.back->execution_mask[method] && IsLevelMethod(c0, method)) {
                level(LevelType::Batch3D, method);
                break;
            }
        }
        if (cls == Class::Level) {
            break;
        }
        cls = Class::Work;
        if (Copy* const c = Use(s, c0.back, executed)) {
            // Maxwell3D::RunDrawBatch: the group once, then the repeats as instances.
            for (u32 i = 0; i < count; ++i) {
                CallMethod(s, *c, static_cast<u32>(entry[1 + i] & 0xffff),
                           static_cast<u32>(entry[1 + i] >> 32));
            }
            if (repeats > 1) {
                c->e->draw_manager.draw_state.instance_count += repeats - 1;
            }
        }
        break;
    }
    }
    switch (cls) {
    case Class::Pass:
        ++s.counters.pass;
        s.in_level_run = false;
        break;
    case Class::Work:
        ++s.counters.work;
        s.in_level_run = false;
        break;
    case Class::Level:
        ++s.counters.level;
        if (!s.in_level_run) {
            s.in_level_run = true;
            ++s.counters.level_runs;
        }
        s.level_since_flush = true;
        s.channel_stale = true;
        for (Copy& copy : s.copies) {
            copy.stale = true;
        }
        if (!executed) {
            s.current_level = true;
        }
        break;
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

} // Anonymous namespace

void ReadSwitch() {
    const char* const value = std::getenv("FRITCHII_MEMSTAGE_SHADOW");
    g_mode = value != nullptr ? std::atoi(value) : 0;
    g_on = g_mode != 0;
    if (g_on) {
        LOG_INFO(HW_GPU,
                 "Memory stage shadow (FRITCHII_MEMSTAGE_SHADOW={}): the stage's inputs run on the "
                 "back and are compared at every draw; nothing they compute is used",
                 g_mode);
    }
}

void Start() {
    State& s = S();
    s = State{};
    s.window_start = std::chrono::steady_clock::now();
    t_on_back = true;
}

void Before(u64 position) {
    State& s = S();
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
    if (!t_on_back) {
        return;
    }
    State& s = S();
    Counters& c = s.counters;
    ++c.back_draws;
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
    const Flags missing_flags = record.dirty & ~maxwell3d.dirty.flags;
    if (missing_flags.any()) {
        ++c.b_dirty;
        for (size_t flag = 0; flag < missing_flags.size(); ++flag) {
            if (missing_flags[flag] && differences++ < 16) {
                diff += fmt::format("; dirty flag {} raised by the shadow, not on the back", flag);
            }
        }
    }
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
    S().in_draw_window = open;
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
    Shadow::Record record = Shadow::TakeSpare(s);
    record.position = s.processing_position;
    record.sunk = s.processing_sunk;
    record.indirect = indirect;
    record.executes = execute_on;
    record.engine = back;
    record.channel = s.channel;
    record.program_key = Shadow::ProgramKey(*this);
    record.dirty = dirty.flags;
    dirty.flags.reset();
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

} // namespace VideoCommon
