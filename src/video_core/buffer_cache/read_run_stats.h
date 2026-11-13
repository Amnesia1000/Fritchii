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
// A second line, "Read overlap", says per read whether its own bytes [address, address + size)
// overlap the GPU-written bytes it waits for, or only share their page(s) (see ByteCheck).
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

/// The read's own bytes against GPU-written byte ranges inside the read's pages.
enum class Overlap : u8 {
    Unknown,  ///< not answered from the buffer cache's ranges (an image, or no range there)
    Bytes,    ///< at least one of the read's bytes is GPU-written
    PageOnly, ///< none is: the read only shares the page(s) with GPU-written bytes
};

/// One answer: what the read overlaps and, for PageOnly, the bytes between the read and the
/// nearest GPU-written byte in its pages (0 = adjacent).
struct ByteAnswer {
    Overlap overlap{Overlap::Unknown};
    u64 gap{};
};

/// Fed the GPU-written ranges inside the read's pages (clipped to them), one by one.
struct ByteCheck {
    u64 begin{}; ///< the read's first byte
    u64 end{};   ///< one past its last byte
    bool any{};
    bool overlap{};
    u64 gap{std::numeric_limits<u64>::max()};

    void Add(u64 lower, u64 upper) {
        if (lower >= upper) {
            return;
        }
        any = true;
        if (lower < end && begin < upper) {
            overlap = true;
        } else {
            gap = (std::min)(gap, upper <= begin ? begin - upper : lower - end);
        }
    }

    ByteAnswer Answer() const {
        if (!any) {
            return {};
        }
        return overlap ? ByteAnswer{Overlap::Bytes, 0} : ByteAnswer{Overlap::PageOnly, gap};
    }
};

/// Set by BufferCache::GetFlushArea on the reading thread (its mutex held), cleared by
/// GPU::OnCPURead before asking. `modified`: against gpu_modified_ranges at the read (what a
/// flush downloads). `on_way`: against the download already on its way that the read waits for:
/// pending_download_ranges up to the newest one the read needs (EXP72's fence download), or
/// read_download_ranges (EXP49's read download, GetFlushArea's in_flight answer).
struct ReadBytes {
    ByteAnswer modified;
    ByteAnswer on_way;
};
inline thread_local ReadBytes t_read_bytes{};

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
    // The "Read overlap" line.
    u64 overlap[3][3]{};  ///< [Kind][Overlap]
    u64 overlap_wait_ns[3]{}; ///< [Overlap], all kinds
    u64 gaps[5]{};        ///< PageOnly: < 64 B, < 256 B, < 1 KiB, < 4 KiB, more
    u64 sizes[5]{};       ///< <= 16 B, <= 64, <= 256, <= 4096, more
    u64 modified_differs{}; ///< waited for a download, and gpu_modified_ranges answers otherwise
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
/// EXP112: read flushes it looked at (BufferCache::DownloadMemoryForRead), and of them those
/// whose own pages were Preflushable by then (GetFlushArea marks them first with preflushing on).
inline std::atomic<u64> g_run_asked{};
inline std::atomic<u64> g_run_own_marked{};

/// Why EXP112 took no run for a read flush.
enum class NoRun : u8 {
    NotGpuModified,        ///< the read's pages held no GPU-modified byte by the flush
    NoBuffer,              ///< no buffer holds the read's pages
    NeighbourPreflushable, ///< the run would go on, but the next page is Preflushable
    NoNeighbour,           ///< no GPU-written page next to the read's inside its buffer
    Count,
};
inline std::atomic<u64> g_no_run[static_cast<size_t>(NoRun::Count)]{};

inline void CountNoRun(NoRun why) {
    if (g_enabled) [[unlikely]] {
        g_no_run[static_cast<size_t>(why)].fetch_add(1, std::memory_order_relaxed);
    }
}

/// GPU::OnCPURead, once per read past the preemptive answer. `address` is the read's area start
/// (page aligned), `size` the read's own size; `bytes` is t_read_bytes as GetFlushArea left it.
inline void NoteRead(Kind kind, u64 address, u64 run_pages, u64 wait_ns, u64 size,
                     const ReadBytes& bytes) {
    State& s = Get();
    const u64 page = address >> 12;
    const u64 frame = g_frame.load(std::memory_order_relaxed);
    // A flush downloads gpu_modified_ranges' bytes; a wait waits for the download on its way.
    const ByteAnswer& answer = kind == Kind::Flushed ? bytes.modified : bytes.on_way;
    std::scoped_lock lock{s.mutex};
    Counts& c = s.counts;
    ++c.reads[static_cast<size_t>(kind)];
    ++c.overlap[static_cast<size_t>(kind)][static_cast<size_t>(answer.overlap)];
    c.overlap_wait_ns[static_cast<size_t>(answer.overlap)] += wait_ns;
    if (answer.overlap == Overlap::PageOnly) {
        ++c.gaps[answer.gap < 64     ? 0
                 : answer.gap < 256  ? 1
                 : answer.gap < 1024 ? 2
                 : answer.gap < 4096 ? 3
                                     : 4];
    }
    ++c.sizes[size <= 16 ? 0 : size <= 64 ? 1 : size <= 256 ? 2 : size <= 4096 ? 3 : 4];
    if (kind != Kind::Flushed && bytes.modified.overlap != answer.overlap) {
        ++c.modified_differs;
    }
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
             "[{:#x}, {:#x}] | run flushes {} ({} KiB beyond the reads' pages) of {} read "
             "flushes looked at ({} with their own pages Preflushable); no run: not GPU-modified "
             "{}, no buffer {}, next page Preflushable {}, no GPU-written neighbour {}",
             run_flush_mode, reads, static_cast<double>(reads) / Frames, c.reads[0], c.reads[1],
             c.reads[2], c.next_page, c.near_pages, c.same_page, c.runs[0], c.runs[1], c.runs[2],
             c.runs[3], c.runs[4], static_cast<double>(c.wait_ns) / 1e6 / Frames,
             reads != 0 ? static_cast<double>(c.wait_ns) / 1e6 / static_cast<double>(reads) : 0.0,
             static_cast<double>(c.max_wait_ns) / 1e6,
             static_cast<double>(c.distinct_pages) / Frames, c.repeat_reads,
             reads != 0 ? c.lowest : 0, c.highest,
             g_run_flushes.exchange(0, std::memory_order_relaxed),
             g_run_bytes.exchange(0, std::memory_order_relaxed) >> 10,
             g_run_asked.exchange(0, std::memory_order_relaxed),
             g_run_own_marked.exchange(0, std::memory_order_relaxed),
             g_no_run[0].exchange(0, std::memory_order_relaxed),
             g_no_run[1].exchange(0, std::memory_order_relaxed),
             g_no_run[2].exchange(0, std::memory_order_relaxed),
             g_no_run[3].exchange(0, std::memory_order_relaxed));
    constexpr size_t B = static_cast<size_t>(Overlap::Bytes);
    constexpr size_t P = static_cast<size_t>(Overlap::PageOnly);
    constexpr size_t U = static_cast<size_t>(Overlap::Unknown);
    const auto ms_frame = [&](size_t o) {
        return static_cast<double>(c.overlap_wait_ns[o]) / 1e6 / Frames;
    };
    const auto ms_read = [&](size_t o) {
        const u64 n = c.overlap[0][o] + c.overlap[1][o] + c.overlap[2][o];
        return n != 0 ? static_cast<double>(c.overlap_wait_ns[o]) / 1e6 / static_cast<double>(n)
                      : 0.0;
    };
    LOG_INFO(HW_GPU,
             "Read overlap (FRITCHII_READ_RUNS) last 600 frames, the read's own bytes against "
             "the GPU-written bytes in its pages it waits for (fence download: its pending "
             "download ranges; read download: read_download_ranges; flushed: gpu_modified_ranges) "
             "as overlap/page only/unknown: fence download {}/{}/{}, read download {}/{}/{}, "
             "flushed {}/{}/{} | page only, gap to the nearest GPU-written byte: <64 B {}, "
             "<256 B {}, <1 KiB {}, <4 KiB {}, more {} | read size <=16 B {}, <=64 {}, <=256 "
             "{}, <=4096 {}, more {} | waited: overlap {:.2f} ms a frame ({:.3f} ms a read), "
             "page only {:.2f} ({:.3f}), unknown {:.2f} ({:.3f}) | gpu_modified_ranges answers "
             "otherwise for {} of the waits",
             c.overlap[2][B], c.overlap[2][P], c.overlap[2][U], c.overlap[1][B], c.overlap[1][P],
             c.overlap[1][U], c.overlap[0][B], c.overlap[0][P], c.overlap[0][U], c.gaps[0],
             c.gaps[1], c.gaps[2], c.gaps[3], c.gaps[4], c.sizes[0], c.sizes[1], c.sizes[2],
             c.sizes[3], c.sizes[4], ms_frame(B), ms_read(B), ms_frame(P), ms_read(P),
             ms_frame(U), ms_read(U), c.modified_differs);
    c = Counts{};
}

} // namespace VideoCommon::ReadRunStats
