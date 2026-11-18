// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>

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
/// 1 (flags in place of the queries) is not built yet.
namespace VideoCommon::OcclFlag {

enum Value : int { Off = 0, On = 1, Verify = 2 };

[[nodiscard]] inline int Mode() noexcept {
    const int value = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_occl_flag,
                                                       "FRITCHII_OCCL_FLAG");
    return value == Verify ? Verify : Off;
}

/// Read once at device creation: whether to keep the buffer device address feature.
[[nodiscard]] inline bool WantedAtBoot() noexcept {
    const char* value = std::getenv("FRITCHII_OCCL_FLAG");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

struct Stats {
    std::atomic<u64> flag_draws{};     ///< counted draws drawn with a flag pipeline
    std::atomic<u64> other_counted{};  ///< counted draws drawn without one
    std::atomic<u64> not_lean{};       ///< colour off and counting, but not the Lean class
    std::atomic<u64> refused_regs{};   ///< colour on, alpha test or alpha to coverage
    std::atomic<u64> not_ready{};      ///< the flag pipeline is not made or built yet
    std::atomic<u64> made{};           ///< flag pipelines made
    std::atomic<u64> failed{};         ///< flag pipelines that failed to translate
    std::atomic<u64> no_slot{};        ///< flag draws with no open query (stored to the sink)
    std::atomic<u64> compared{};       ///< reports whose counted draws were all flag draws
    std::atomic<u64> agree{};
    std::atomic<u64> q0_f1{};          ///< the query counted 0, a flag was stored (must be 0)
    std::atomic<u64> q1_f0{};          ///< the query counted samples, no flag (must be 0)
    std::atomic<u64> mixed{};          ///< reports with flag and other counted draws
    std::atomic<u64> no_flag{};        ///< reports with no flag draw
    std::atomic<u64> reused{};         ///< reports whose banks were reserved again before the compare
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
                 no_flag = take(g.no_flag), reused = take(g.reused);
    LOG_INFO(Render_Vulkan,
             "Occlusion flags (FRITCHII_OCCL_FLAG={}) per second: counted draws {:.0f} with a "
             "flag pipeline, {:.0f} without (not Lean {:.0f}, regs refused {:.0f}, not ready "
             "{:.0f}); flag pipelines made {:.0f}, failed {:.0f}; flag draws with no open query "
             "{:.0f}; reports compared {:.0f}: agree {:.0f}, query 0 flag 1 {:.0f}, query 1 flag 0 "
             "{:.0f} (must be 0; {} since boot); mixed {:.0f}, no flag {:.0f}, bank reused {:.0f}",
             Mode(), flag_draws, other, not_lean, refused, not_ready, made, failed, no_slot,
             compared, agree, q0_f1, q1_f0, g.disagree_total.load(std::memory_order_relaxed),
             mixed, no_flag, reused);
}

} // namespace VideoCommon::OcclFlag
