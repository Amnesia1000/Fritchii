// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iterator>
#include <string>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/nx2_debug_control.h"

/// FRITCHII_OCCL_FLAG (TEST; live between 0 and 2, but set it at boot: the device keeps the buffer
/// device address feature only when it is set then). Occlusion test draws that can show
/// "samples passed" without a host query: a draw that counts samples, writes no colour and
/// whose pixel program can change nothing (FRITCHII_OCCL_LEAN's Lean class) is drawn with a flag
/// pipeline, the same key with a host pixel program that, with early fragment tests, stores
/// 4096 into the 64-bit slot of the open host query (vulkan_occl_flag.frag).
/// 0 off. 2 verify: the host queries stay and decide everything; at each report the fencing
/// thread compares "the query counted samples" with "a flag was stored" for the reports whose
/// counted draws were all flag draws. The disagreements must be 0. One line every 10 s.
/// 1 on: a query window whose counted draws are all flag draws runs no host query; its slots
/// are flag slots, whose values the result copies (GPU) and reads (fencing thread) take from the
/// flag area. A window with other counted draws splits into flag and query slots (summed).
namespace VideoCommon::OcclFlag {

/// 3 (TEST): on without the colour flag variants (only colour-off draws take flag pipelines),
/// to measure what the colour variants add.
/// 4 (COST PROBE, wrong picture possible): as 3, and colour-off draws whose pixel program
/// discards take the host flag variant too (no discard, no query): what their queries cost.
/// 5 (TEST): as 3 (no colour variants), with the pre-pass for discarding draws (1 and 2 have it
/// too), to measure what the pre-pass adds.
enum Value : int {
    Off = 0,
    On = 1,
    Verify = 2,
    OnColourOff = 3,
    ProbeDiscards = 4,
    OnPrepassOnly = 5,
    /// 6 (COST PROBE, the picture can change): as 1, and colour Lean draws with subgroup
    /// operations take the colour flag variant too (early tests can change their results).
    ProbeSubgroup = 6,
    /// 8 (TEST): as 1 without the colour pre-pass, to measure what it adds.
    OnNoColourPrepass = 8,
};

[[nodiscard]] inline int RawMode() noexcept {
    return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_occl_flag,
                                            "FRITCHII_OCCL_FLAG");
}

[[nodiscard]] inline int Mode() noexcept {
    const int value = RawMode();
    return value == Verify                                                     ? Verify
           : (value == On || value == OnColourOff || value == ProbeDiscards ||
              value == OnPrepassOnly || value == ProbeSubgroup || value == OnNoColourPrepass)
               ? On
               : Off;
}

/// Counted colour-off draws whose program discards may take the flag pre-pass (1, 2, 5).
[[nodiscard]] inline bool PrepassAllowed() noexcept {
    const int value = RawMode();
    return value == On || value == Verify || value == OnPrepassOnly || value == ProbeSubgroup ||
           value == OnNoColourPrepass;
}

/// Colour-writing counted draws may take the colour flag variant (not with =3).
[[nodiscard]] inline bool ColourAllowed() noexcept {
    const int value = RawMode();
    return value == On || value == Verify || value == ProbeSubgroup ||
           value == OnNoColourPrepass;
}

/// Colour Lean draws the colour variant cannot take (subgroup or quad operations) may take the
/// colour pre-pass: the host flag program first with colour and depth writes off (1, 2).
[[nodiscard]] inline bool ColourPrepassAllowed() noexcept {
    const int value = RawMode();
    return value == On || value == Verify;
}

/// Read once at device creation: whether to keep the buffer device address feature.
[[nodiscard]] inline bool WantedAtBoot() noexcept {
    const char* value = std::getenv("FRITCHII_OCCL_FLAG");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

/// Census (with the flag lines): why a counted draw has no flag pipeline. Set by
/// PipelineCache::OcclFlagPipeline on the recording thread, read by the draw that follows.
enum class Census : u8 {
    Unknown,      ///< not decided for this draw (counted as blocking)
    Flag,         ///< drawn with a flag pipeline
    NotReady,     ///< Lean, its flag pipeline not built yet (becomes a flag draw)
    ColourLean,   ///< colour on, Lean pixel program, no colour variant (=3, or no int64)
    ColourQuad,   ///< colour on, Lean, but quad shuffles or FSWZADD (not taken: early tests)
    ColourSubgroup, ///< colour on, Lean, but subgroup votes, masks, shuffles or helper reads
    ColourA2C,    ///< colour on, Lean, alpha to coverage in the key
    ColourDiscards,
    ColourDiscardsNoWrites, ///< discards, otherwise Lean, no depth or stencil writes this draw
    ColourKeyState,
    ColourDepthOrMask,
    ColourStores,
    ColourOther,  ///< colour on: no pixel program, mesh, or other
    RegsRefused,  ///< alpha test or alpha to coverage in the registers
    OffDiscards,  ///< colour off, the pixel program discards
    OffDiscardsNoWrites, ///< as above, otherwise Lean, no depth or stencil writes this draw
    OffKeyState,  ///< colour off, alpha test or dual source in the key
    OffDepthOrMask,
    OffStores,
    OffOther,     ///< colour off: mesh or other
    Failed,
    Count,
};
inline constexpr const char* CensusNames[] = {
    "unknown", "flag", "not ready", "colour Lean", "colour Lean quad", "colour Lean subgroup",
    "colour Lean A2C", "colour discards", "colour discards no writes", "colour key state",
    "colour depth/mask", "colour stores", "colour other", "alpha test/coverage", "off discards",
    "off discards no writes", "off key state", "off depth/mask", "off stores", "off other",
    "failed"};
static_assert(std::size(CensusNames) == static_cast<size_t>(Census::Count));
inline thread_local Census t_census{};
/// Set by PipelineCache::OcclFlagPipeline when the host flag variant it returns is for a colour
/// pre-pass (read and cleared by the draw that follows).
inline thread_local bool t_colour_prepass{};

/// A draw of this census class keeps its window's host query even with flag stores added to
/// colour-writing Lean pixel programs.
[[nodiscard]] inline bool Blocks(Census c) noexcept {
    return c != Census::Flag && c != Census::NotReady && c != Census::ColourLean &&
           c != Census::ColourQuad && c != Census::ColourSubgroup && c != Census::ColourA2C;
}

struct Stats {
    std::atomic<u64> census[static_cast<size_t>(Census::Count)]{};
    std::atomic<u64> colour_lean_pipelines{}; ///< distinct pipelines of colour Lean draws
    /// Colour Lean draws refused for subgroup operations, by operation (a draw can count twice).
    std::atomic<u64> sg_vote{}, sg_mask{}, sg_shuffle{}, sg_invocation_id{}, sg_helper{};
    std::atomic<u64> qslot_injectable{}; ///< query slots whose draws were all flag or colour Lean
    std::atomic<u64> qslot_blocked{};    ///< query slots with a blocking draw
    std::atomic<u64> qslot_empty{};      ///< query slots with no counted draw marked
    std::atomic<u64> flag_draws{};     ///< counted draws drawn with a flag pipeline
    std::atomic<u64> other_counted{};  ///< counted draws drawn without one
    std::atomic<u64> not_lean{};       ///< colour off and counting, but not the Lean class
    std::atomic<u64> refused_regs{};   ///< colour on, alpha test or alpha to coverage
    std::atomic<u64> not_ready{};      ///< the flag pipeline is not made or built yet
    std::atomic<u64> made{};           ///< flag pipelines made
    std::atomic<u64> colour_made{};    ///< colour flag pipelines made (guest pixel program)
    std::atomic<u64> colour_flag_draws{}; ///< counted colour-writing draws with a flag pipeline
    std::atomic<u64> prepass_draws{};  ///< counted discarding draws drawn with the pre-pass
    std::atomic<u64> colour_prepass_draws{}; ///< counted colour draws drawn with the colour pre-pass
    std::atomic<u64> prepass_made{};   ///< pre-pass pipelines made
    std::atomic<u64> prepass_fallback{}; ///< pre-pass refused at the draw (indirect, delta, dump)
    std::atomic<u64> prepass_paused{}; ///< pre-passes drawn between two host queries
    std::atomic<u64> prepass_lost{};   ///< real draws the parent could not configure (must be 0)
    std::atomic<u64> prepass_skipped{}; ///< mode 1, a host query already open: real draw only
    std::atomic<u64> prepass_refused{}; ///< pre-pass variants whose descriptors differ
    std::atomic<u64> failed{};         ///< flag pipelines that failed to translate
    std::atomic<u64> no_slot{};        ///< flag draws with no open query (stored to the sink)
    std::atomic<u64> compared{};       ///< reports whose counted draws were all flag draws
    std::atomic<u64> agree{};
    std::atomic<u64> q0_f1{};          ///< the query counted 0, a flag was stored (must be 0)
    std::atomic<u64> q1_f0{};          ///< the query counted samples, no flag (must be 0)
    std::atomic<u64> mixed{};          ///< reports with flag and other counted draws
    std::atomic<u64> no_flag{};        ///< reports with no flag draw
    std::atomic<u64> reused{};         ///< reports whose banks were reserved again before the compare
    std::atomic<u64> flag_slots{};     ///< slots opened as flag slots (no host query; mode 1)
    std::atomic<u64> query_slots{};    ///< slots opened with a host query
    std::atomic<u64> disagree_total{}; ///< since boot (q0_f1 + q1_f0)
    std::atomic<s64> window_start{};
};
inline Stats g_stats;

inline void Add(std::atomic<u64>& counter, u64 amount = 1) noexcept {
    counter.fetch_add(amount, std::memory_order_relaxed);
}

inline void MaybeLog() {
    const s64 now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    s64 start = g_stats.window_start.load(std::memory_order_relaxed);
    if (start == 0) {
        g_stats.window_start.compare_exchange_strong(start, now, std::memory_order_relaxed);
        return;
    }
    if (now - start < 10'000'000'000LL ||
        !g_stats.window_start.compare_exchange_strong(start, now, std::memory_order_relaxed)) {
        return;
    }
    const double s = static_cast<double>(now - start) / 1e9;
    const auto take = [s](std::atomic<u64>& c) {
        return static_cast<double>(c.exchange(0, std::memory_order_relaxed)) / s;
    };
    auto& g = g_stats;
    const double flag_draws = take(g.flag_draws), other = take(g.other_counted),
                 not_lean = take(g.not_lean), refused = take(g.refused_regs),
                 not_ready = take(g.not_ready), made = take(g.made), failed = take(g.failed),
                 no_slot = take(g.no_slot), compared = take(g.compared), agree = take(g.agree),
                 q0_f1 = take(g.q0_f1), q1_f0 = take(g.q1_f0), mixed = take(g.mixed),
                 no_flag = take(g.no_flag), reused = take(g.reused),
                 flag_slots = take(g.flag_slots), query_slots = take(g.query_slots),
                 colour_flag_draws = take(g.colour_flag_draws), colour_made = take(g.colour_made),
                 prepass_draws = take(g.prepass_draws), prepass_made = take(g.prepass_made),
                 prepass_fallback = take(g.prepass_fallback),
                 prepass_paused = take(g.prepass_paused),
                 prepass_skipped = take(g.prepass_skipped),
                 colour_prepass_draws = take(g.colour_prepass_draws);
    LOG_INFO(Render_Vulkan,
             "Occlusion flags (FRITCHII_OCCL_FLAG={}) per second: counted draws {:.0f} with a "
             "flag pipeline, {:.0f} without (not Lean {:.0f}, regs refused {:.0f}, not ready "
             "{:.0f}); flag pipelines made {:.0f}, failed {:.0f}; flag draws with no open query "
             "{:.0f}; reports compared {:.0f}: agree {:.0f}, query 0 flag 1 {:.0f}, query 1 flag 0 "
             "{:.0f} (must be 0; {} since boot); mixed {:.0f}, no flag {:.0f}, bank reused {:.0f}; "
             "slots opened: {:.0f} flag (no host query), {:.0f} with a host query; colour flag "
             "draws {:.0f}, colour flag pipelines made {:.0f}; pre-pass draws {:.0f} ({:.0f} "
             "between host queries, {:.0f} skipped: a host query open), made {:.0f}, refused at "
             "the draw {:.0f}, refused for their descriptors {} since boot, real draws lost {} "
             "(must be 0); colour pre-pass draws {:.0f}",
             Mode(), flag_draws, other, not_lean, refused, not_ready, made, failed, no_slot,
             compared, agree, q0_f1, q1_f0, g.disagree_total.load(std::memory_order_relaxed),
             mixed, no_flag, reused, flag_slots, query_slots, colour_flag_draws, colour_made,
             prepass_draws, prepass_paused, prepass_skipped, prepass_made, prepass_fallback,
             g.prepass_refused.load(std::memory_order_relaxed),
             g.prepass_lost.load(std::memory_order_relaxed), colour_prepass_draws);
    std::string census;
    for (size_t i = 0; i < static_cast<size_t>(Census::Count); ++i) {
        census += fmt::format("{}{} {:.0f}", i == 0 ? "" : ", ", CensusNames[i], take(g.census[i]));
    }
    LOG_INFO(Render_Vulkan,
             "Occlusion flag census per second: counted draws by class: {}; colour Lean "
             "pipelines {} distinct in the window; colour Lean subgroup draws by operation: vote "
             "{:.0f}, mask {:.0f}, shuffle {:.0f}, invocation id {:.0f}, helper {:.0f}; query "
             "slots closed: {:.0f} all flag or colour Lean, "
             "{:.0f} blocked, {:.0f} no draw marked",
             census, g.colour_lean_pipelines.exchange(0, std::memory_order_relaxed),
             take(g.sg_vote), take(g.sg_mask), take(g.sg_shuffle), take(g.sg_invocation_id),
             take(g.sg_helper),
             take(g.qslot_injectable),
             take(g.qslot_blocked), take(g.qslot_empty));
}

} // namespace VideoCommon::OcclFlag
