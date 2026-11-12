// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>

#include <fmt/format.h>
#include <boost/container/static_vector.hpp>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/formatter.h"

// EDEN_NX2_FENCE_RELEASE_STATS=1 (diagnostic, read once; fable52_answer_chicken 5.1): what the
// fencing thread's "fence: release" spends its time on. Each release (ReleaseThreadFunc) is split
// into texture pops, buffer pops, the fences' operations and query pops, with time and bytes, plus
// how long it waits for and holds the texture and buffer caches' mutexes; one line per 5 s. Every
// single pop over 1 ms is logged with what it landed (images: GPU address, size, format,
// forced_flushed; buffers: range and bytes), at most 30 such lines per 5 s.
// Off: one test of g_enabled per release.
namespace VideoCommon::FenceReleaseStats {

inline const bool g_enabled = [] {
    const char* value = std::getenv("EDEN_NX2_FENCE_RELEASE_STATS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}();

using Clock = std::chrono::steady_clock;

inline u64 NsBetween(Clock::time_point start, Clock::time_point end) {
    return static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

inline u64 NsSince(Clock::time_point start) {
    return NsBetween(start, Clock::now());
}

/// One thing a pop writes to guest memory.
struct LandedItem {
    bool image = false;       ///< false: a buffer range
    u64 address = 0;          ///< image: GPU address; buffer: start of the range
    u64 end = 0;              ///< buffer: end of the range
    u64 bytes = 0;            ///< image: unswizzled bytes; buffer: bytes copied
    u32 width = 0;            ///< image only
    u32 height = 0;           ///< image only
    VideoCore::Surface::PixelFormat format{};
    bool forced_flushed = false;
};

/// What one pop lands, described before it runs: totals and its four largest items.
struct PopLanding {
    static constexpr size_t MaxItems = 4;
    u64 bytes = 0;
    u32 images = 0;
    u32 buffers = 0;
    u64 begin = ~u64{0}; ///< buffer cache pops: hull of the copies
    u64 end = 0;
    boost::container::static_vector<LandedItem, MaxItems> items;

    void Add(const LandedItem& item) {
        bytes += item.bytes;
        if (item.image) {
            ++images;
        } else {
            ++buffers;
            begin = (std::min)(begin, item.address);
            end = (std::max)(end, item.end);
        }
        if (items.size() < MaxItems) {
            items.push_back(item);
            return;
        }
        const auto smallest = std::min_element(
            items.begin(), items.end(),
            [](const LandedItem& a, const LandedItem& b) { return a.bytes < b.bytes; });
        if (smallest->bytes < item.bytes) {
            *smallest = item;
        }
    }
};

/// The fencing thread's sums (only that thread touches them).
struct Totals {
    u64 releases = 0;
    u64 fences = 0;
    u64 release_ns = 0;
    u64 texture_pops = 0;
    u64 texture_ns = 0;
    u64 texture_bytes = 0;
    u64 buffer_pops = 0;
    u64 buffer_ns = 0;
    u64 buffer_bytes = 0;
    u64 operations = 0;
    u64 operations_ns = 0;
    u64 query_pops = 0;
    u64 query_ns = 0;
    u64 texture_wait_ns = 0;
    u64 texture_held_ns = 0;
    u64 buffer_wait_ns = 0;
    u64 buffer_held_ns = 0;
    u64 long_pops = 0;
    u64 long_logged = 0;
    Clock::time_point last = Clock::now();
};

inline Totals& Get() {
    static Totals totals;
    return totals;
}

inline double Ms(u64 ns) {
    return static_cast<double>(ns) / 1e6;
}

/// After a pop: count it, and log it when it took 1 ms or more (30 lines per 5 s at most).
inline void Pop(bool texture, u64 ns, const PopLanding& landing) {
    Totals& t = Get();
    if (texture) {
        ++t.texture_pops;
        t.texture_ns += ns;
        t.texture_bytes += landing.bytes;
    } else {
        ++t.buffer_pops;
        t.buffer_ns += ns;
        t.buffer_bytes += landing.bytes;
    }
    if (ns < 1'000'000) {
        return;
    }
    ++t.long_pops;
    if (t.long_logged >= 30) {
        return;
    }
    ++t.long_logged;
    std::string items;
    for (const LandedItem& item : landing.items) {
        if (item.image) {
            items += fmt::format("; image gpu {:#x} {}x{} {} {} KiB forced_flushed={}",
                                 item.address, item.width, item.height, item.format,
                                 item.bytes >> 10, item.forced_flushed ? 1 : 0);
        } else {
            items += fmt::format("; buffer [{:#x}, {:#x}) {} KiB", item.address, item.end,
                                 item.bytes >> 10);
        }
    }
    if (texture) {
        LOG_INFO(HW_GPU,
                 "Fence release: texture pop {:.2f} ms landed {} images, {} buffers, {} KiB "
                 "(largest {}){}",
                 Ms(ns), landing.images, landing.buffers, landing.bytes >> 10,
                 landing.items.size(), items);
    } else {
        LOG_INFO(HW_GPU,
                 "Fence release: buffer pop {:.2f} ms landed {} copies, {} KiB over [{:#x}, "
                 "{:#x}) (largest {}){}",
                 Ms(ns), landing.buffers, landing.bytes >> 10,
                 landing.buffers != 0 ? landing.begin : 0, landing.end, landing.items.size(),
                 items);
    }
}

/// After each release: its total time; logs the split every 5 s.
inline void Release(u64 fences, u64 ns) {
    Totals& t = Get();
    ++t.releases;
    t.fences += fences;
    t.release_ns += ns;
    const auto now = Clock::now();
    if (now - t.last < std::chrono::seconds{5}) {
        return;
    }
    LOG_INFO(HW_GPU,
             "Fence release split (5 s): {} releases of {} fences, {:.2f} ms; texture pops {} "
             "{:.2f} ms {} KiB; buffer pops {} {:.2f} ms {} KiB; operations {} {:.2f} ms; query "
             "pops {} {:.2f} ms; texture mutex waited {:.2f} ms held {:.2f} ms; buffer mutex "
             "waited {:.2f} ms held {:.2f} ms; pops over 1 ms {} ({} logged)",
             t.releases, t.fences, Ms(t.release_ns), t.texture_pops, Ms(t.texture_ns),
             t.texture_bytes >> 10, t.buffer_pops, Ms(t.buffer_ns), t.buffer_bytes >> 10,
             t.operations, Ms(t.operations_ns), t.query_pops, Ms(t.query_ns),
             Ms(t.texture_wait_ns), Ms(t.texture_held_ns), Ms(t.buffer_wait_ns),
             Ms(t.buffer_held_ns), t.long_pops, t.long_logged);
    t = Totals{};
    t.last = now;
}

} // namespace VideoCommon::FenceReleaseStats
