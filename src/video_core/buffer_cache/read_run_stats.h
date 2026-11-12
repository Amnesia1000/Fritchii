// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <unordered_set>

#include "common/common_types.h"
#include "common/logging.h"

// FRITCHII_READ_RUNS=1 (diagnostic, read once; fable52_answer_fountain 4): the guest reads of
// GPU-written memory that go to the back or wait for a download (GPU::OnCPURead past its
// preemptive answer), one line per 600 frames: whether each read is on the page after the
// previous one, how long the GPU-written run around its page is (BufferCache::GetFlushArea,
// from gpu_modified_ranges inside the read's buffer), whether its bytes were already on their
// way, how long the reader waited, and how many distinct pages a frame reads. It also counts
// EXP112's run flushes. Off: one test of g_enabled at each site.
namespace VideoCommon::ReadRunStats {

inline const bool g_enabled = [] {
    const char* value = std::getenv("FRITCHII_READ_RUNS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}();

/// Pages the run is counted up to (one more than the histogram's last bucket).
inline constexpr u64 RunCountPages = 65;

/// Set by BufferCache::GetFlushArea on the reading thread: the GPU-written pages in a row around
/// the read's pages inside its buffer (capped at RunCountPages); 0 when the buffer cache had no
/// GPU-written byte there (an image, or nothing). GPU::OnCPURead clears it before asking.
inline thread_local u64 t_run_pages = 0;

enum class Kind : u8 {
    Flushed,       ///< a new download through the back (one round trip)
    InFlight,      ///< its bytes were in a read download on its way: only waited
    FenceDownload, ///< its bytes were in a fence download on its way (EXP72): only waited
};

/// What the line reports, reset after each line.
struct Counts {
    u64 reads[3]{};
    u64 next_page{};  ///< on the page after the previous read's
    u64 near_pages{}; ///< 2 to 16 pages after or before it
    u64 same_page{};  ///< on the previous read's page
    u64 runs[5]{};    ///< not in the buffer cache, 1 page, 2-16, 17-64, more
    u64 wait_ns{};
    u64 max_wait_ns{};
    u64 lowest{std::numeric_limits<u64>::max()};
    u64 highest{};
    u64 distinct_pages{}; ///< summed over the window's frames
    u64 repeat_reads{};   ///< reads of a page already read that frame
};

struct State {
    std::mutex mutex;
    Counts counts;
    u64 previous_page{std::numeric_limits<u64>::max()};
    /// Pages read in the frame `frame`.
    std::unordered_set<u64> frame_pages;
    u64 frame{};
};

inline State& Get() {
    static State state;
    return state;
}

/// BufferCache::TickFrame advances it once a frame.
inline std::atomic<u64> g_frame{};
/// EXP112's run flushes and the bytes they took beyond the read's own pages.
inline std::atomic<u64> g_run_flushes{};
inline std::atomic<u64> g_run_bytes{};

/// GPU::OnCPURead, once per read past the preemptive answer.
inline void NoteRead(Kind kind, u64 address, u64 run_pages, u64 wait_ns) {
    State& s = Get();
    const u64 page = address >> 12;
    const u64 frame = g_frame.load(std::memory_order_relaxed);
    std::scoped_lock lock{s.mutex};
    Counts& c = s.counts;
    ++c.reads[static_cast<size_t>(kind)];
    if (s.previous_page != std::numeric_limits<u64>::max()) {
        const u64 distance =
            page >= s.previous_page ? page - s.previous_page : s.previous_page - page;
        if (distance == 0) {
            ++c.same_page;
        } else if (page == s.previous_page + 1) {
            ++c.next_page;
        } else if (distance <= 16) {
            ++c.near_pages;
        }
    }
    s.previous_page = page;
    const size_t bucket = run_pages == 0    ? 0
                          : run_pages == 1  ? 1
                          : run_pages <= 16 ? 2
                          : run_pages <= 64 ? 3
                                            : 4;
    ++c.runs[bucket];
    c.wait_ns += wait_ns;
    c.max_wait_ns = (std::max)(c.max_wait_ns, wait_ns);
    c.lowest = (std::min)(c.lowest, address);
    c.highest = (std::max)(c.highest, address);
    if (frame != s.frame) {
        c.distinct_pages += s.frame_pages.size();
        s.frame_pages.clear();
        s.frame = frame;
    }
    if (!s.frame_pages.insert(page).second) {
        ++c.repeat_reads;
    }
}

/// BufferCache::TickFrame, every 600 frames (after advancing g_frame).
inline void Log(int run_flush_mode) {
    State& s = Get();
    std::scoped_lock lock{s.mutex};
    Counts& c = s.counts;
    c.distinct_pages += s.frame_pages.size();
    s.frame_pages.clear();
    s.frame = g_frame.load(std::memory_order_relaxed);
    const u64 reads = c.reads[0] + c.reads[1] + c.reads[2];
    constexpr double Frames = 600.0;
    LOG_INFO(HW_GPU,
             "Read runs (FRITCHII_READ_RUNS, EXP112={}) last 600 frames: {} reads to the back "
             "({:.1f} a frame): {} flushed, {} only waited for a read download on its way, {} "
             "waited for a fence download | next page {}, within 16 pages {}, same page {} | "
             "GPU-written run around the page: not in the buffer cache {}, 1 page {}, 2-16 {}, "
             "17-64 {}, more {} | waited {:.2f} ms a frame, {:.3f} ms a read, longest {:.2f} ms | "
             "{:.1f} distinct pages a frame, {} reads of a page already read that frame | span "
             "[{:#x}, {:#x}] | run flushes {} ({} KiB beyond the reads' pages)",
             run_flush_mode, reads, static_cast<double>(reads) / Frames, c.reads[0], c.reads[1],
             c.reads[2], c.next_page, c.near_pages, c.same_page, c.runs[0], c.runs[1], c.runs[2],
             c.runs[3], c.runs[4], static_cast<double>(c.wait_ns) / 1e6 / Frames,
             reads != 0 ? static_cast<double>(c.wait_ns) / 1e6 / static_cast<double>(reads) : 0.0,
             static_cast<double>(c.max_wait_ns) / 1e6,
             static_cast<double>(c.distinct_pages) / Frames, c.repeat_reads,
             reads != 0 ? c.lowest : 0, c.highest,
             g_run_flushes.exchange(0, std::memory_order_relaxed),
             g_run_bytes.exchange(0, std::memory_order_relaxed) >> 10);
    c = Counts{};
}

} // namespace VideoCommon::ReadRunStats
