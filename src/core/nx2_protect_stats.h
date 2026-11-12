// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"

namespace Core::Nx2ProtectStats {

/// EDEN_NX2_PROTECT_STATS=1 (read once): counts the host page protection calls the device memory
/// manager makes when a page's cached count moves between 0 and 1 (protect = the page became
/// tracked, unprotect = it became untracked), how long they take, how many follow the previous
/// call of the same kind on the next page within one batch (one call could have done both), and
/// the pages that change most. Logged every 5 s ("Page protection (5 s"). Fable's next70 ask 3.
[[nodiscard]] inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_PROTECT_STATS");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

struct PageCount {
    u32 protects{};
    u32 unprotects{};
};

struct State {
    std::atomic<u64> calls[2]{};      ///< [0] unprotect, [1] protect
    std::atomic<u64> bytes[2]{};
    std::atomic<u64> ns[2]{};
    std::atomic<u64> mergeable[2]{};  ///< Calls that start where the previous same-kind call ended
    std::atomic<u64> batches{};
    std::mutex pages_mutex;
    std::unordered_map<u64, PageCount> pages; ///< Guest page number -> changes this window
    std::atomic<s64> next_report_ns{};
};

inline State& Get() {
    static State state;
    return state;
}

/// The previous call on this thread inside the current batch, for the mergeable count.
inline thread_local u64 t_last_end = ~u64{0};
inline thread_local bool t_last_cached = false;

inline void BeginBatch() {
    t_last_end = ~u64{0};
    Get().batches.fetch_add(1, std::memory_order_relaxed);
}

inline s64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline void Report(State& state, s64 now) {
    s64 next = state.next_report_ns.load(std::memory_order_relaxed);
    if (next == 0) {
        state.next_report_ns.compare_exchange_strong(next, now + 5'000'000'000LL);
        return;
    }
    if (now < next || !state.next_report_ns.compare_exchange_strong(next, now + 5'000'000'000LL)) {
        return;
    }
    const auto take = [](std::atomic<u64>& value) {
        return value.exchange(0, std::memory_order_relaxed);
    };
    const u64 calls_u = take(state.calls[0]);
    const u64 calls_p = take(state.calls[1]);
    const u64 bytes_u = take(state.bytes[0]);
    const u64 bytes_p = take(state.bytes[1]);
    const u64 ns_u = take(state.ns[0]);
    const u64 ns_p = take(state.ns[1]);
    const u64 merge_u = take(state.mergeable[0]);
    const u64 merge_p = take(state.mergeable[1]);
    const u64 batches = take(state.batches);
    std::vector<std::pair<u64, PageCount>> top;
    size_t distinct = 0;
    size_t cycled = 0;
    {
        std::scoped_lock lock{state.pages_mutex};
        distinct = state.pages.size();
        top.reserve(distinct);
        for (const auto& [page, count] : state.pages) {
            cycled += (count.protects > 0 && count.unprotects > 0) ? 1 : 0;
            top.emplace_back(page, count);
        }
        state.pages.clear();
    }
    const size_t keep = std::min<size_t>(16, top.size());
    std::partial_sort(top.begin(), top.begin() + keep, top.end(), [](const auto& a, const auto& b) {
        return a.second.protects + a.second.unprotects > b.second.protects + b.second.unprotects;
    });
    std::string pages_text;
    for (size_t i = 0; i < keep; ++i) {
        pages_text += fmt::format(" {:#x}:{}/{}", top[i].first << 12, top[i].second.protects,
                                  top[i].second.unprotects);
    }
    LOG_INFO(HW_GPU,
             "Page protection (5 s, per second): protect {:.0f} calls {:.0f} pages {:.2f} ms, "
             "unprotect {:.0f} calls {:.0f} pages {:.2f} ms; could merge with the previous call: "
             "protect {:.0f}, unprotect {:.0f}; batches {:.0f}; pages changed {} (both ways {})",
             calls_p / 5.0, bytes_p / 4096.0 / 5.0, ns_p / 5e6, calls_u / 5.0,
             bytes_u / 4096.0 / 5.0, ns_u / 5e6, merge_p / 5.0, merge_u / 5.0, batches / 5.0,
             distinct, cycled);
    LOG_INFO(HW_GPU, "Page protection top pages (address:protects/unprotects in 5 s):{}",
             pages_text);
}

/// Wraps one MarkRegionCaching call: `vaddr`/`bytes` are the guest range, `cached` true when
/// the pages became tracked (protected).
template <typename Func>
inline void Count(u64 vaddr, u64 bytes, bool cached, Func&& call) {
    State& state = Get();
    const s64 start = NowNs();
    call();
    const s64 end = NowNs();
    const size_t kind = cached ? 1 : 0;
    state.calls[kind].fetch_add(1, std::memory_order_relaxed);
    state.bytes[kind].fetch_add(bytes, std::memory_order_relaxed);
    state.ns[kind].fetch_add(static_cast<u64>(end - start), std::memory_order_relaxed);
    if (t_last_end == vaddr && t_last_cached == cached) {
        state.mergeable[kind].fetch_add(1, std::memory_order_relaxed);
    }
    t_last_end = vaddr + bytes;
    t_last_cached = cached;
    {
        std::scoped_lock lock{state.pages_mutex};
        for (u64 page = vaddr >> 12; page < (vaddr + bytes + 4095) >> 12; ++page) {
            PageCount& count = state.pages[page];
            if (cached) {
                ++count.protects;
            } else {
                ++count.unprotects;
            }
        }
    }
    Report(state, end);
}

} // namespace Core::Nx2ProtectStats
