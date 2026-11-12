// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"

/// EDEN_NX2_FASTDRAW_COUNT=1 (read once, by the first Maxwell3D): counts the draws that could
/// take zortos293/Lichii#1161's delta ("fast") draw path, which keeps the previous draw's
/// pipeline, image views and samplers when only light state changed in between. Counting only:
/// no draw is resolved, recorded or skipped differently. Off, each site costs one test of g_on.
///
/// The rules are #1161's (nx2_fast_draw.h, Maxwell3D::IsFastDrawParameter and its lists,
/// RasterizerVulkan::TakeReusableDraw / DeltaStillHolds / NoteResolvedDraw), with its later
/// commits 792e12a47b (vertex_id_base, vertex formats, report operands and neutral methods are
/// light; the write list widens instead of ending reuse; handle words compared one by one),
/// e05a72847e (pipelines with computed handles take delta draws) and 7824123e58 (back macro
/// calls are neutral). Every break moves a serial, as there; it also ORs the break's group into a
/// mask, cleared at each draw, so a draw that state ended shows every group that ended it, not
/// only the last one (Fable plan, fable_answer_backthread_threads.md section 8).
namespace VideoCommon::Nx2FastDrawCount {

enum class Reason : u8 {
    Register,
    Method,
    InlineData,
    OtherEngine,
    Call,
    TickWork,
    Invalidation,
    Fence,
    IndirectReset,
    Frame,
    TextureHeaders,
    Map,
    HleMacro,
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Reason::Count)> reason_names{
    "register",       "method",        "inline data",   "other engine",        "deferred call",
    "tick work",      "invalidation",  "fence release", "indirect reset",      "frame",
    "texture header data", "memory map", "HLE macro",
};

/// The groups a draw's breaks are sorted into, a bit each. Register and method breaks go by
/// method (Maxwell3D::FastDrawCountGroup); with #1161's lists most of the first three are light
/// and stay 0: the mask shows what is left.
enum Group : u32 {
    VertexIdBase = 1U << 0,  ///< 0x446
    VertexLayout = 1U << 1,  ///< Vertex stream word 0 (enable, stride), attribute formats
    QueryMethods = 1U << 2,  ///< Counter reset, render enable, report operands, MME memory
    ListOverflow = 1U << 3,  ///< The 16-range write list was widened (not a break in #1161)
    RealState = 1U << 4,     ///< Any other 3D register or method
    OtherEngine = 1U << 5,
    Invalidation = 1U << 6,  ///< Texture cache writes and deletions, rescales, shader removals
    Fence = 1U << 7,
    TickWork = 1U << 8,
    Call = 1U << 9,          ///< Deferred calls on the split's back not marked neutral
    Frame = 1U << 10,
    Map = 1U << 11,          ///< GPU memory map changes, unmapped images
    Other = 1U << 12,        ///< Inline data, texture header data, indirect reset, HLE macros
};
inline constexpr u32 NumGroups = 13;
inline constexpr u32 BreakGroups = ((1U << NumGroups) - 1) & ~u32{ListOverflow};
inline constexpr std::array<const char*, NumGroups> group_names{
    "vertex_id_base", "vertex layout", "query methods", "write list overflow", "real state",
    "other engine",   "invalidation",  "fence",         "tick work",           "deferred call",
    "frame",          "memory map",    "other",
};

[[nodiscard]] constexpr u32 GroupOf(Reason reason) {
    switch (reason) {
    case Reason::OtherEngine:
        return OtherEngine;
    case Reason::Call:
        return Call;
    case Reason::TickWork:
        return TickWork;
    case Reason::Invalidation:
        return Invalidation;
    case Reason::Fence:
        return Fence;
    case Reason::Frame:
        return Frame;
    case Reason::Map:
        return Map;
    case Reason::Register:
    case Reason::Method:
        return RealState;
    default:
        return Other;
    }
}

/// EDEN_NX2_FASTDRAW_COUNT == "1". A plain bool: every site tests it and nothing else when off.
inline bool g_on{};

/// Reads the switch once (Maxwell3D's constructor: after main() applied nx2_switches.txt).
inline void Init() {
    static const bool once = [] {
        const char* const value = std::getenv("EDEN_NX2_FASTDRAW_COUNT");
        g_on = value != nullptr && value[0] == '1';
        if (g_on) {
            LOG_INFO(HW_GPU, "Delta draw count on (EDEN_NX2_FASTDRAW_COUNT=1): counting only, "
                             "zortos293/Lichii#1161 rules");
        }
        return true;
    }();
    (void)once;
}

[[nodiscard]] inline bool On() {
    return g_on;
}

/// A Call entry's flag on the split's ring (bit 7, free for Call entries): a deferred call that
/// can't change what a draw resolves to (#1161's GpuSplit::FastDrawNeutral). Set only when on.
inline constexpr u64 NeutralCallBit = u64{1} << 7;
[[nodiscard]] inline u64 Neutral() {
    return g_on ? NeutralCallBit : 0;
}

/// Whether vertex stream strides and enables are dynamic state (extended dynamic state and
/// dynamic vertex input): then they and the attribute formats are draw parameters. Set by the
/// pipeline cache.
inline bool g_vertex_streams_dynamic{};

inline std::atomic<u64> g_serial{1};
/// Groups broken since the last draw took its snapshot.
inline std::atomic<u32> g_pending{};
/// The last break's reason and, for Register and Method, its 3D method (bits 8 and up).
inline std::atomic<u32> g_last_break{};

/// A break, when on (callers test g_on). Breaks come from several threads.
inline void BreakOn(Reason reason, u32 group, u32 method = 0) {
    g_serial.fetch_add(1, std::memory_order_relaxed);
    g_last_break.store(static_cast<u32>(reason) | (method << 8), std::memory_order_relaxed);
    if ((g_pending.load(std::memory_order_relaxed) & group) != group) {
        g_pending.fetch_or(group, std::memory_order_relaxed);
    }
}

inline void Break(Reason reason) {
    if (g_on) [[unlikely]] {
        BreakOn(reason, GroupOf(reason));
    }
}

/// Light changes since the last draw that can reach texture handles: constant buffer binds and
/// inline constant buffer data by GPU address. Written and read on the thread that runs the 3D
/// engine and the rasterizer.
struct Delta {
    static constexpr size_t MaxWrites = 16;

    std::array<u32, 5> binds{}; ///< Constant buffer slots bound per stage
    std::array<u64, MaxWrites> write_begin{};
    std::array<u64, MaxWrites> write_end{};
    u32 num_writes{};
    bool widened{}; ///< The list was full and a range widened (#1161: no uniform keeps then)

    [[nodiscard]] bool Empty() const {
        return num_writes == 0 && (binds[0] | binds[1] | binds[2] | binds[3] | binds[4]) == 0;
    }

    [[nodiscard]] bool Reaches(u64 address, u64 size) const {
        for (u32 write = 0; write < num_writes; ++write) {
            if (write_begin[write] < address + size && address < write_end[write]) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] bool ReachesWords(u64 base, std::span<const std::pair<u32, u32>> words) const {
        for (u32 write = 0; write < num_writes; ++write) {
            for (const auto& [first, second] : words) {
                if (write_begin[write] < base + second && base + first < write_end[write]) {
                    return true;
                }
            }
        }
        return false;
    }
};

inline Delta g_delta;

/// #1161's AppendRange: extends a range the new one touches or overlaps, else appends it; a full
/// list widens the range that grows least (true then).
inline bool AppendRange(std::array<u64, Delta::MaxWrites>& begins,
                        std::array<u64, Delta::MaxWrites>& ends, u32& count, u64 begin, u64 end) {
    if (count != 0 && ends[count - 1] == begin) {
        ends[count - 1] = end;
        return false;
    }
    for (u32 i = 0; i < count; ++i) {
        if (begin <= ends[i] && begins[i] <= end) {
            begins[i] = (std::min)(begins[i], begin);
            ends[i] = (std::max)(ends[i], end);
            return false;
        }
    }
    if (count < Delta::MaxWrites) {
        begins[count] = begin;
        ends[count] = end;
        ++count;
        return false;
    }
    u32 best = 0;
    u64 best_growth = ~u64{0};
    for (u32 i = 0; i < count; ++i) {
        const u64 growth = ((std::max)(ends[i], end) - (std::min)(begins[i], begin)) -
                           (ends[i] - begins[i]);
        if (growth < best_growth) {
            best = i;
            best_growth = growth;
        }
    }
    begins[best] = (std::min)(begins[best], begin);
    ends[best] = (std::max)(ends[best], end);
    return true;
}

/// When on: a constant buffer bind that changed the slot.
inline void NoteBind(size_t stage, u32 slot) {
    if (stage < g_delta.binds.size()) {
        g_delta.binds[stage] |= 1U << slot;
    }
}

/// When on: inline constant buffer data written to GPU addresses [begin, end).
inline void NoteConstantData(u64 begin, u64 end) {
    Delta& delta = g_delta;
    delta.widened |= AppendRange(delta.write_begin, delta.write_end, delta.num_writes, begin, end);
}

/// What a pipeline's handles are read from (#1161's GraphicsPipeline handle masks), built on its
/// first counted draw.
struct PipelineInfo {
    static constexpr size_t NumStages = 5;
    static constexpr size_t NumSlots = 18;

    std::array<u32, NumStages> handle_cbufs{}; ///< Slots handles are read from, per stage
    /// The bytes [first, second) of a slot handles are read from...
    std::array<std::array<std::pair<u32, u32>, NumSlots>, NumStages> ranges{};
    /// ...and the same bytes word by word (sorted, disjoint), by (first, count) into words.
    std::vector<std::pair<u32, u32>> words;
    std::array<std::array<std::pair<u16, u16>, NumSlots>, NumStages> word_index{};
    bool writes_images{};
    bool mesh{};

    [[nodiscard]] std::span<const std::pair<u32, u32>> Words(size_t stage, u32 slot) const {
        const auto [first, count] = word_index[stage][slot];
        return std::span{words}.subspan(first, count);
    }
};

/// Why a draw would be resolved in full (#1161's FullDraw).
enum class Full : u8 {
    NoCandidate,
    Kind,
    StateChanged,
    Submitted,
    RenderPass,
    DeltaPipeline,
    VertexInput,
    HandleBind,
    HandleData,
    Count,
};

/// Why a draw can't be the previous draw of a delta draw (#1161's Ineligible).
enum class Ineligible : u8 {
    Indirect,
    Mesh,
    NotBuilt,
    WritesImages,
    TransformFeedback,
    InlineIndex,
    Quads,
    Repacked,
    NoRenderPass,
    PipelineNotBound,
    OffBackThread,
    Count,
};

/// The previous eligible draw (#1161's ReusableDraw).
struct PreviousDraw {
    const void* pipeline{};
    const void* maxwell3d{};
    u64 serial{};
    u64 map_serial{};
    u64 tick{};
    u64 render_pass{};
    u32 topology{};
    bool is_indexed{};
    bool valid{};
};

struct Stats {
    std::chrono::steady_clock::time_point window_start{};
    u64 draws{};     ///< PrepareDraw calls
    u64 not_drawn{}; ///< Returned before the decision (no pipeline, mesh draws skipped)
    u64 delta{};
    u64 other_pipeline{}; ///< Would-be delta draws whose full resolve picked another pipeline
    std::array<u64, static_cast<size_t>(Full::Count)> full{};
    std::array<u64, static_cast<size_t>(Ineligible::Count)> ineligible{};
    std::array<u64, static_cast<size_t>(Reason::Count)> breaks{}; ///< Last break, state changed
    std::vector<u32> break_methods = std::vector<u32>(0x1000);
    std::unordered_map<u32, u64> masks; ///< Break groups of the draws state changed
    std::array<u64, NumGroups> any{};
    std::array<u64, NumGroups> alone{};
};

/// The rasterizer's counting state (one per RasterizerVulkan, on its drawing thread).
struct Counter {
    PreviousDraw previous;
    // The draw being prepared, from Begin to Decide.
    PreviousDraw taken;
    Delta taken_delta;
    u64 take_serial{};
    u32 take_last_break{};
    bool vertex_input_dirty{};
    bool open{};
    // From Decide to Resolved.
    u64 draw_serial{};
    u64 draw_map_serial{};
    Stats stats;

    void CountStateChanged(u32 mask, u32 last_break) {
        ++stats.full[static_cast<size_t>(Full::StateChanged)];
        ++stats.breaks[(std::min)(last_break & 0xff, static_cast<u32>(Reason::Count) - 1)];
        if (const u32 method = last_break >> 8;
            method != 0 && method < stats.break_methods.size()) {
            ++stats.break_methods[method];
        }
        ++stats.masks[mask];
        const u32 breaks = mask & BreakGroups;
        for (u32 group = 0; group < NumGroups; ++group) {
            if ((mask >> group) & 1) {
                ++stats.any[group];
                if (breaks == (1U << group)) {
                    ++stats.alone[group];
                }
            }
        }
    }

    static std::string MaskName(u32 mask) {
        if (mask == 0) {
            return "none";
        }
        std::string name;
        for (u32 group = 0; group < NumGroups; ++group) {
            if ((mask >> group) & 1) {
                if (!name.empty()) {
                    name += '+';
                }
                name += group_names[group];
            }
        }
        return name;
    }

    /// Logs the window every 5 s (called once a frame).
    void MaybeLog() {
        const auto now = std::chrono::steady_clock::now();
        if (stats.window_start == std::chrono::steady_clock::time_point{}) {
            stats.window_start = now;
            return;
        }
        if (now - stats.window_start < std::chrono::seconds{5}) {
            return;
        }
        const double seconds = std::chrono::duration<double>(now - stats.window_start).count();
        const auto rate = [seconds](u64 count) { return static_cast<double>(count) / seconds; };
        const auto full = [&](Full reason) { return rate(stats.full[static_cast<size_t>(reason)]); };
        const u64 decided = stats.draws - stats.not_drawn;
        LOG_INFO(Render_Vulkan,
                 "Delta draw count (5 s, per second): {:.0f} draws, {:.0f} would-be delta "
                 "({:.1f}% of decided draws) | full: {:.0f} after an ineligible draw, {:.0f} "
                 "different kind, {:.0f} state changed, {:.0f} after a submit, {:.0f} render pass "
                 "or pipeline changed, {:.0f} pipeline takes no delta draws, {:.0f} vertex "
                 "attribute types changed, {:.0f} handle buffer bound, {:.0f} handle data "
                 "written, not counted: sampled image to prepare, computed handle changed | "
                 "{:.0f} not drawn (no pipeline or skipped) | {:.0f} would-be delta resolved to "
                 "another pipeline",
                 rate(stats.draws), rate(stats.delta),
                 decided == 0 ? 0.0
                              : 100.0 * static_cast<double>(stats.delta) /
                                    static_cast<double>(decided),
                 full(Full::NoCandidate), full(Full::Kind), full(Full::StateChanged),
                 full(Full::Submitted), full(Full::RenderPass), full(Full::DeltaPipeline),
                 full(Full::VertexInput), full(Full::HandleBind), full(Full::HandleData),
                 rate(stats.not_drawn), rate(stats.other_pipeline));
        static constexpr std::array<const char*, static_cast<size_t>(Ineligible::Count)>
            ineligible_names{"indirect",       "mesh",          "not built",
                             "writes images",  "transform feedback", "inline index",
                             "quads",          "repacked vertices",  "no render pass",
                             "pipeline not bound", "off the back thread"};
        std::string ineligible;
        for (size_t reason = 0; reason < stats.ineligible.size(); ++reason) {
            if (stats.ineligible[reason] != 0) {
                ineligible += fmt::format(" {} {:.0f},", ineligible_names[reason],
                                          rate(stats.ineligible[reason]));
            }
        }
        if (!ineligible.empty()) {
            ineligible.pop_back();
        }
        LOG_INFO(Render_Vulkan, "Delta draw count (5 s) ineligible draws per second:{}",
                 ineligible.empty() ? std::string{" none"} : ineligible);
        std::string breaks;
        for (size_t reason = 0; reason < stats.breaks.size(); ++reason) {
            if (stats.breaks[reason] != 0) {
                breaks += fmt::format(" {} {:.0f},", reason_names[reason],
                                      rate(stats.breaks[reason]));
            }
        }
        if (!breaks.empty()) {
            breaks.pop_back();
        }
        std::vector<std::pair<u32, u32>> methods;
        for (u32 method = 1; method < stats.break_methods.size(); ++method) {
            if (stats.break_methods[method] != 0) {
                methods.emplace_back(stats.break_methods[method], method);
            }
        }
        const size_t shown = (std::min)(methods.size(), size_t{12});
        std::partial_sort(methods.begin(), methods.begin() + shown, methods.end(),
                          std::greater{});
        std::string top;
        for (size_t i = 0; i < shown; ++i) {
            top += fmt::format(" {:#x}:{:.0f}", methods[i].second, rate(methods[i].first));
        }
        LOG_INFO(Render_Vulkan,
                 "Delta draw count (5 s) state changed per second, by last break:{} | top "
                 "methods by last break:{}",
                 breaks.empty() ? std::string{" none"} : breaks,
                 top.empty() ? std::string{" none"} : top);
        std::string any;
        std::string alone;
        for (u32 group = 0; group < NumGroups; ++group) {
            if (stats.any[group] != 0) {
                any += fmt::format(" {} {:.0f},", group_names[group], rate(stats.any[group]));
            }
            if (stats.alone[group] != 0) {
                alone += fmt::format(" {} {:.0f},", group_names[group], rate(stats.alone[group]));
            }
        }
        if (!any.empty()) {
            any.pop_back();
        }
        if (!alone.empty()) {
            alone.pop_back();
        }
        std::vector<std::pair<u64, u32>> masks;
        masks.reserve(stats.masks.size());
        for (const auto& [mask, count] : stats.masks) {
            masks.emplace_back(count, mask);
        }
        const size_t shown_masks = (std::min)(masks.size(), size_t{16});
        std::partial_sort(masks.begin(), masks.begin() + shown_masks, masks.end(),
                          std::greater{});
        std::string mask_list;
        for (size_t i = 0; i < shown_masks; ++i) {
            mask_list += fmt::format(" [{}] {:.0f},", MaskName(masks[i].second),
                                     rate(masks[i].first));
        }
        if (!mask_list.empty()) {
            mask_list.pop_back();
        }
        LOG_INFO(Render_Vulkan,
                 "Delta draw count (5 s) break groups of the state-changed draws per second, "
                 "group set:{} | group alone:{} | top masks:{}",
                 any.empty() ? std::string{" none"} : any,
                 alone.empty() ? std::string{" none"} : alone,
                 mask_list.empty() ? std::string{" none"} : mask_list);
        stats = Stats{};
        stats.window_start = now;
    }
};

} // namespace VideoCommon::Nx2FastDrawCount
