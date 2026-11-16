// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/early_compute.h"

// FRITCHII_READ_RUNS=1 (diagnostic, read once; fable52_answer_fountain 4): the guest reads of
// GPU-written memory that go to the back or wait for a download (GPU::OnCPURead past its
// preemptive answer), one line per 600 frames: whether each read is on the page after the
// previous one, how long the GPU-written run around its page is (BufferCache::GetFlushArea,
// from gpu_modified_ranges inside the read's buffer), whether its bytes were already on their
// way, how long the reader waited, and how many distinct pages a frame reads. It also counts
// EXP112's run flushes. Off: one test of g_enabled at each site.
// A second line, "Read overlap", says per read whether its own bytes [address, address + size)
// overlap the GPU-written bytes it waits for, or only share their page(s) (see ByteCheck).
// "Read writers" and "Read writer" lines (fable52_answer_readback 3): which dispatch wrote the
// bytes each read waits for. Every GPU write mark (BufferCache::MarkWrittenBuffer) goes into a
// ring with the writer the bind set (t_writer: the compute program's hash and binding, or a
// draw's stage and binding); at a read, GetFlushArea takes the newest mark over the read's bytes
// (FindWriter) and OnCPURead counts it per writer (NoteWriter). The "Read runs for B1" line
// sums what the GPU writes into the runs read in the last 16 frames.
// FRITCHII_READ_RUNS=2 adds the "Read writer ticks" line (helper_cu_answer.md): per read kind,
// the newest write mark over the read's pages (the bytes its download takes), where the host GPU
// was with that write's submission when the read asked (not submitted, submitted and not run,
// run), how many submissions the read's own tick is behind it, and, for every read that waits
// for the host GPU, the wait cut at the writer's tick: first until the writer's submission ran,
// then the rest. Same timeline, so the total wait is what it was; only the cut is new.
namespace VideoCommon::ReadRunStats {

/// A boot switch read at its first test, not by a static initializer: those run before main()
/// applies nx2_switches.txt, so the file's value was never seen (only a real environment
/// variable was). Constant-initialized; the first tests may race, they read the same value.
class FirstUseSwitch {
public:
    using Reader = bool (*)();
    constexpr explicit FirstUseSwitch(Reader reader_) noexcept : reader{reader_} {}

    operator bool() const noexcept {
        int value = state.load(std::memory_order_relaxed);
        if (value < 0) [[unlikely]] {
            value = reader() ? 1 : 0;
            state.store(value, std::memory_order_relaxed);
        }
        return value != 0;
    }

private:
    Reader reader;
    mutable std::atomic<int> state{-1};
};

inline bool ReadRunsValue(bool writer_ticks) {
    // FRITCHII_EARLY_COMPUTE and FRITCHII_READ_SIDE_COPY count on this ring and its stamps.
    if (EarlyCompute::g_early || EarlyCompute::g_side_copy) {
        return true;
    }
    const char* value = std::getenv("FRITCHII_READ_RUNS");
    if (value == nullptr || value[0] == '\0' || value[0] == '0') {
        return false;
    }
    return !writer_ticks || value[0] == '2';
}

inline FirstUseSwitch g_enabled{[] { return ReadRunsValue(false); }};

/// FRITCHII_READ_RUNS=2: the write marks carry their stamp and reads cut their host waits there.
inline FirstUseSwitch g_writer_ticks{[] { return ReadRunsValue(true); }};

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

[[nodiscard]] inline u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

/// What made a GPU write mark.
enum class WriterKind : u8 {
    Other,          ///< no bind set one: copies, queries, transform feedback, indirect buffers
    ComputeStorage, ///< a dispatch's storage buffer
    ComputeTexture, ///< a dispatch's texture or image buffer
    DrawStorage,    ///< a draw's storage buffer
    DrawTexture,    ///< a draw's texture or image buffer
};

struct Writer {
    u64 program{};  ///< the compute program's hash (0 for draws and Other)
    u32 binding{};  ///< the binding index; draws: stage * 64 + index
    WriterKind kind{WriterKind::Other};

    bool operator==(const Writer&) const = default;
};

/// Set on the thread that binds a dispatch's or draw's written buffers while it marks them
/// (BufferCache::BindHost*StorageBuffers and *TextureBuffers), Other otherwise.
inline thread_local Writer t_writer{};

/// Dispatches bound so far (BufferCache::BindHostComputeBuffers): a write mark's ring position.
inline std::atomic<u64> g_dispatches{};

struct WriteMark {
    u64 begin{};
    u64 end{};
    u64 time_ns{};
    u64 dispatch{}; ///< g_dispatches when it was made
    u64 stamp{};    ///< the buffer cache's clock stamp then (g_writer_ticks, else 0)
    Writer writer{};
    /// FRITCHII_EARLY_COMPUTE: the writing dispatch's buffer verdict at its bind
    EarlyCompute::Verdict verdict{EarlyCompute::Verdict::NotJudged};
};

/// Marks kept (the "Read writers" line says how many a frame, so how many frames this covers).
inline constexpr size_t RingSize = 16384;

/// B1 (fable52_answer_readback 2): a GPU-written run around a read that went to the back, kept
/// until 16 frames after its last read. Runs do not overlap (merged when added).
struct ReadRun {
    u64 begin{};
    u64 end{};
    u64 frame{};
};
inline constexpr u64 RunKeepFrames = 16;

struct Ring {
    std::mutex mutex;
    std::vector<WriteMark> marks; ///< RingSize entries once the first mark comes
    u64 count{};                  ///< marks made; the newest is marks[(count - 1) % RingSize]
    std::vector<ReadRun> runs;
    u64 run_bytes{};         ///< bytes marked GPU-written inside the runs, all writers
    u64 run_compute_bytes{}; ///< of them by dispatches
    u64 run_count_sum{};     ///< runs kept, summed over the window's frames
    u64 run_size_sum{};      ///< their bytes, summed over the window's frames
    u64 run_size_max{};      ///< the most bytes kept at one frame
};

inline Ring& GetRing() {
    static Ring ring;
    return ring;
}

/// BufferCache::MarkWrittenBuffer (the thread that binds: the back, or the memory stage).
/// `stamp`: the runtime's CurrentTick for the write (g_writer_ticks), else 0.
inline void NoteWrite(u64 begin, u64 size, u64 stamp = 0) {
    if (size == 0) {
        return;
    }
    const u64 end = begin + size;
    const WriteMark mark{.begin = begin,
                         .end = end,
                         .time_ns = NowNs(),
                         .dispatch = g_dispatches.load(std::memory_order_relaxed),
                         .stamp = stamp,
                         .writer = t_writer,
                         .verdict = EarlyCompute::t_verdict};
    const bool compute = mark.writer.kind == WriterKind::ComputeStorage ||
                         mark.writer.kind == WriterKind::ComputeTexture;
    Ring& r = GetRing();
    std::scoped_lock lock{r.mutex};
    if (r.marks.empty()) {
        r.marks.resize(RingSize);
    }
    r.marks[r.count % RingSize] = mark;
    ++r.count;
    for (const ReadRun& run : r.runs) {
        const u64 lower = (std::max)(begin, run.begin);
        const u64 upper = (std::min)(end, run.end);
        if (lower < upper) {
            r.run_bytes += upper - lower;
            if (compute) {
                r.run_compute_bytes += upper - lower;
            }
        }
    }
}

/// Bytes of the read compared before and after they land.
inline constexpr u32 CompareBytes = 64;

/// Set by BufferCache::GetFlushArea on the reading thread when the read's pages hold GPU-written
/// bytes (before anything lands), cleared by GPU::OnCPURead before asking.
struct WriterAnswer {
    bool asked{};     ///< GetFlushArea looked
    bool found{};     ///< a mark in the ring covers at least one of the read's bytes
    bool page_only{}; ///< none does, but one is on the read's page(s)
    WriteMark mark{};
    u64 read_ns{};          ///< when the read asked
    u64 dispatches_after{}; ///< dispatches bound between the mark and the read
    u64 address{};          ///< the read's first byte
    u64 run_begin{};        ///< the GPU-written run around the read's pages (B1)
    u64 run_end{};
    u32 compared{}; ///< bytes in `before` (0: not taken)
    std::array<u8, CompareBytes> before{};
    // FRITCHII_READ_RUNS=2 ("Read writer ticks"): the newest mark over the read's pages, the
    // bytes its download takes (GetFlushArea fills the ticks, the waits fill the cut).
    bool area_found{};
    WriteMark area_mark{};
    u64 write_tick{};     ///< the scheduler tick the area mark's stamp is recorded in
    u64 read_tick{};      ///< the scheduler's current tick when the read asked
    u64 known_tick{};     ///< the host GPU's completed tick then (refreshed)
    bool cut{};           ///< a host wait was cut at the writer's tick
    u64 writer_wait_ns{}; ///< until the writer's submission ran
    u64 rest_wait_ns{};   ///< then until the read's download (or fence download) was there
};
inline thread_local WriterAnswer t_writer_answer{};

/// FRITCHII_EARLY_COMPUTE: whether an unrun write mark overlaps one of `ranges` ({begin, end}),
/// newest first, stopping at the first run one (stamps grow with the ring, near enough for a
/// count). `is_free(stamp)`: the buffer cache clock's test.
template <typename Ranges, typename IsFree>
[[nodiscard]] bool AnyPendingMark(const Ranges& ranges, IsFree&& is_free) {
    Ring& r = GetRing();
    std::scoped_lock lock{r.mutex};
    const u64 n = (std::min<u64>)(r.count, RingSize);
    for (u64 i = 0; i < n; ++i) {
        const WriteMark& mark = r.marks[(r.count - 1 - i) % RingSize];
        if (is_free(mark.stamp)) {
            return false;
        }
        for (const auto& [begin, end] : ranges) {
            if (mark.begin < end && begin < mark.end) {
                return true;
            }
        }
    }
    return false;
}

/// FRITCHII_READ_SIDE_COPY: the stamp of the newest write mark over any of `ranges`
/// ({begin, size}); false when none is in the ring.
template <typename Ranges>
[[nodiscard]] bool NewestMarkOver(const Ranges& ranges, u64& stamp) {
    Ring& r = GetRing();
    std::scoped_lock lock{r.mutex};
    const u64 n = (std::min<u64>)(r.count, RingSize);
    for (u64 i = 0; i < n; ++i) {
        const WriteMark& mark = r.marks[(r.count - 1 - i) % RingSize];
        for (const auto& [begin, size] : ranges) {
            if (mark.begin < begin + size && begin < mark.end) {
                stamp = mark.stamp;
                return true;
            }
        }
    }
    return false;
}

/// GetFlushArea: the newest mark over [address, address + size), else note one on its pages.
inline void FindWriter(u64 address, u64 size, u64 page_begin, u64 page_end) {
    WriterAnswer& a = t_writer_answer;
    a.asked = true;
    a.read_ns = NowNs();
    a.address = address;
    const u64 end = address + size;
    const u64 dispatches = g_dispatches.load(std::memory_order_relaxed);
    Ring& r = GetRing();
    std::scoped_lock lock{r.mutex};
    const u64 n = (std::min<u64>)(r.count, RingSize);
    for (u64 i = 0; i < n; ++i) {
        const WriteMark& mark = r.marks[(r.count - 1 - i) % RingSize];
        if (!a.area_found && mark.begin < page_end && page_begin < mark.end) {
            a.area_found = true;
            a.area_mark = mark;
        }
        if (mark.begin < end && address < mark.end) {
            a.found = true;
            a.page_only = false;
            a.mark = mark;
            a.dispatches_after = dispatches - mark.dispatch;
            return;
        }
        if (!a.page_only && mark.begin < page_end && page_begin < mark.end) {
            a.page_only = true;
        }
    }
}

/// Per writer (program, binding, kind), reset after each line.
struct WriterStats {
    u64 reads[3]{}; ///< [Kind]
    u64 ages[5]{};  ///< the write's age at the read: < 0.5 ms, < 1, < 2, < 4, more
    u64 wait_ns{};
    u64 lowest_offset{std::numeric_limits<u64>::max()};
    u64 highest_offset{};
    u64 previous_offset{std::numeric_limits<u64>::max()};
    std::unordered_set<u64> offsets;           ///< capped at 256
    std::unordered_map<s64, u64> steps;        ///< offset minus the previous read's; capped at 64
    u64 compared{};
    u64 differed{};
    u64 dispatches_after{};
};

struct WriterHash {
    size_t operator()(const Writer& w) const noexcept {
        return std::hash<u64>{}(w.program ^ (u64{w.binding} << 48) ^
                                (u64{static_cast<u8>(w.kind)} << 56));
    }
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
    // The "Read overlap" line.
    u64 overlap[3][3]{};  ///< [Kind][Overlap]
    u64 overlap_wait_ns[3]{}; ///< [Overlap], all kinds
    u64 gaps[5]{};        ///< PageOnly: < 64 B, < 256 B, < 1 KiB, < 4 KiB, more
    u64 sizes[5]{};       ///< <= 16 B, <= 64, <= 256, <= 4096, more
    u64 modified_differs{}; ///< waited for a download, and gpu_modified_ranges answers otherwise
    // The "Read writers" line.
    u64 writer_not_asked{}; ///< GetFlushArea found no GPU-written byte in the read's pages
    u64 writer_none{};      ///< no mark in the ring over the read's bytes nor its pages
    u64 writer_page_only{}; ///< none over its bytes, one on its pages
    u64 writer_ages[3][3]{}; ///< [Kind][age < 1 ms, 1 to 1.5 ms, 1.5 ms or more]
    u64 writer_compared{};
    u64 writer_differed{};
};

/// The "Read writer ticks" line (g_writer_ticks), per Kind; reset after each line.
struct TickCounts {
    u64 reads[3]{};      ///< asked, with a mark over the read's pages in the ring
    u64 no_mark[3]{};    ///< asked, none in the ring (written before its window)
    u64 state[3][3]{};   ///< the writer's submission at the read: not submitted, submitted, run
    u64 behind[3][5]{};  ///< submissions from the writer's to the read's: 0, 1-2, 3-5, 6-10, more
    u64 behind_sum[3]{};
    u64 unrun_sum[3]{};  ///< submitted after the writer's and not run when the read asked
    u64 kinds[3][5]{};   ///< [Kind][WriterKind] of the mark
    u64 wait_ns[3]{};    ///< the reads' whole wait (as the other lines count it)
    u64 cuts[3]{};       ///< host waits cut at the writer's tick
    u64 writer_ns[3]{};  ///< of them, until the writer's submission ran
    u64 rest_ns[3]{};    ///< then until the download was there
    u64 rest_max_ns[3]{};
};

struct State {
    std::mutex mutex;
    Counts counts;
    TickCounts ticks;
    /// The "Read writer" lines: reads whose bytes a mark in the ring covers, by its writer.
    std::unordered_map<Writer, WriterStats, WriterHash> writers;
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

/// GPU::OnCPURead, after NoteRead: the read's writer (t_writer_answer as GetFlushArea left it)
/// and, when `after` is given, the read's first a.compared bytes in guest memory now (landed).
inline void NoteWriter(Kind kind, u64 wait_ns, const WriterAnswer& a, const u8* after) {
    const u64 frame = g_frame.load(std::memory_order_relaxed);
    if (a.run_begin < a.run_end) {
        // B1: this read's run, merged with the kept runs it touches or adjoins.
        Ring& r = GetRing();
        std::scoped_lock lock{r.mutex};
        ReadRun run{.begin = a.run_begin, .end = a.run_end, .frame = frame};
        for (size_t i = 0; i < r.runs.size();) {
            const ReadRun& kept = r.runs[i];
            if (kept.begin <= run.end && run.begin <= kept.end) {
                run.begin = (std::min)(run.begin, kept.begin);
                run.end = (std::max)(run.end, kept.end);
                r.runs[i] = r.runs.back();
                r.runs.pop_back();
            } else {
                ++i;
            }
        }
        r.runs.push_back(run);
    }
    const bool compared = a.compared != 0 && after != nullptr;
    const bool differed = compared && std::memcmp(a.before.data(), after, a.compared) != 0;
    const u64 age_ns = a.found && a.read_ns > a.mark.time_ns ? a.read_ns - a.mark.time_ns : 0;
    // FRITCHII_EARLY_COMPUTE: the programs whose writes are read back, and each read by its
    // writer's verdict, with what running that writer early would have saved this read.
    if (EarlyCompute::g_early && a.asked && a.area_found) [[unlikely]] {
        if (a.area_mark.writer.kind == WriterKind::ComputeStorage ||
            a.area_mark.writer.kind == WriterKind::ComputeTexture) {
            EarlyCompute::LearnFeeder(a.area_mark.writer.program);
        }
        EarlyCompute::NoteRead(static_cast<size_t>(kind), a.area_mark.verdict, a.cut,
                               a.writer_wait_ns);
    }
    State& s = Get();
    std::scoped_lock lock{s.mutex};
    if (g_writer_ticks && a.asked) {
        TickCounts& t = s.ticks;
        const size_t k = static_cast<size_t>(kind);
        t.wait_ns[k] += wait_ns;
        if (!a.area_found) {
            ++t.no_mark[k];
        } else {
            ++t.reads[k];
            const size_t state = a.write_tick >= a.read_tick    ? 0
                                 : a.write_tick <= a.known_tick ? 2
                                                                : 1;
            ++t.state[k][state];
            const u64 behind = a.read_tick > a.write_tick ? a.read_tick - a.write_tick : 0;
            ++t.behind[k][behind == 0    ? 0
                          : behind <= 2  ? 1
                          : behind <= 5  ? 2
                          : behind <= 10 ? 3
                                         : 4];
            t.behind_sum[k] += behind;
            const u64 floor = (std::max)(a.write_tick, a.known_tick);
            t.unrun_sum[k] += a.read_tick > floor + 1 ? a.read_tick - floor - 1 : 0;
            ++t.kinds[k][static_cast<size_t>(a.area_mark.writer.kind)];
            if (a.cut) {
                ++t.cuts[k];
                t.writer_ns[k] += a.writer_wait_ns;
                t.rest_ns[k] += a.rest_wait_ns;
                t.rest_max_ns[k] = (std::max)(t.rest_max_ns[k], a.rest_wait_ns);
            }
        }
    }
    Counts& c = s.counts;
    if (!a.asked) {
        ++c.writer_not_asked;
        return;
    }
    if (!a.found) {
        ++(a.page_only ? c.writer_page_only : c.writer_none);
        return;
    }
    const size_t k = static_cast<size_t>(kind);
    ++c.writer_ages[k][age_ns < 1'000'000 ? 0 : age_ns < 1'500'000 ? 1 : 2];
    c.writer_compared += compared ? 1 : 0;
    c.writer_differed += differed ? 1 : 0;
    WriterStats& w = s.writers[a.mark.writer];
    ++w.reads[k];
    ++w.ages[age_ns < 500'000     ? 0
             : age_ns < 1'000'000 ? 1
             : age_ns < 2'000'000 ? 2
             : age_ns < 4'000'000 ? 3
                                  : 4];
    w.wait_ns += wait_ns;
    const u64 offset = a.address >= a.mark.begin ? a.address - a.mark.begin : 0;
    w.lowest_offset = (std::min)(w.lowest_offset, offset);
    w.highest_offset = (std::max)(w.highest_offset, offset);
    if (w.offsets.size() < 256) {
        w.offsets.insert(offset);
    }
    if (w.previous_offset != std::numeric_limits<u64>::max()) {
        const s64 step = static_cast<s64>(offset - w.previous_offset);
        if (const auto it = w.steps.find(step); it != w.steps.end()) {
            ++it->second;
        } else if (w.steps.size() < 64) {
            w.steps.emplace(step, 1);
        }
    }
    w.previous_offset = offset;
    w.compared += compared ? 1 : 0;
    w.differed += differed ? 1 : 0;
    w.dispatches_after += a.dispatches_after;
}

/// BufferCache::TickFrame, once a frame (after advancing g_frame): drops the runs not read for
/// RunKeepFrames frames and sums what is kept.
inline void FrameTick() {
    const u64 frame = g_frame.load(std::memory_order_relaxed);
    Ring& r = GetRing();
    std::scoped_lock lock{r.mutex};
    u64 bytes = 0;
    for (size_t i = 0; i < r.runs.size();) {
        if (r.runs[i].frame + RunKeepFrames < frame) {
            r.runs[i] = r.runs.back();
            r.runs.pop_back();
        } else {
            bytes += r.runs[i].end - r.runs[i].begin;
            ++i;
        }
    }
    r.run_count_sum += r.runs.size();
    r.run_size_sum += bytes;
    r.run_size_max = (std::max)(r.run_size_max, bytes);
}

inline const char* WriterKindName(WriterKind kind) {
    switch (kind) {
    case WriterKind::ComputeStorage:
        return "dispatch storage buffer";
    case WriterKind::ComputeTexture:
        return "dispatch texture buffer";
    case WriterKind::DrawStorage:
        return "draw storage buffer";
    case WriterKind::DrawTexture:
        return "draw texture buffer";
    case WriterKind::Other:
        break;
    }
    return "other (copy, query, transform feedback)";
}

/// Log: the "Read writers", "Read writer" and "Read runs for B1" lines (s.mutex held).
inline void LogWriters(State& s, double frames) {
    Counts& c = s.counts;
    const u64 found = c.writer_ages[0][0] + c.writer_ages[0][1] + c.writer_ages[0][2] +
                      c.writer_ages[1][0] + c.writer_ages[1][1] + c.writer_ages[1][2] +
                      c.writer_ages[2][0] + c.writer_ages[2][1] + c.writer_ages[2][2];
    u64 marks = 0;
    double runs_a_frame = 0.0;
    double run_kib_a_frame = 0.0;
    u64 run_kib_max = 0;
    u64 run_bytes = 0;
    u64 run_compute_bytes = 0;
    {
        Ring& r = GetRing();
        std::scoped_lock lock{r.mutex};
        static u64 previous_count = 0;
        marks = r.count - previous_count;
        previous_count = r.count;
        runs_a_frame = static_cast<double>(r.run_count_sum) / frames;
        run_kib_a_frame = static_cast<double>(r.run_size_sum) / 1024.0 / frames;
        run_kib_max = r.run_size_max >> 10;
        run_bytes = r.run_bytes;
        run_compute_bytes = r.run_compute_bytes;
        r.run_count_sum = 0;
        r.run_size_sum = 0;
        r.run_size_max = 0;
        r.run_bytes = 0;
        r.run_compute_bytes = 0;
    }
    LOG_INFO(HW_GPU,
             "Read writers (FRITCHII_READ_RUNS) last 600 frames: {} reads with a write mark over "
             "their bytes, {} with one only on their page, {} with none in the ring ({} marks), "
             "{} not GPU-written by the buffer cache | the write's age at the read as <1 ms/1-1.5/"
             ">=1.5: flushed {}/{}/{}, read download {}/{}/{}, fence download {}/{}/{} | read "
             "bytes differed from guest memory at landing in {} of {} compared | {:.1f} marks a "
             "frame, {} writers",
             found, c.writer_page_only, c.writer_none, RingSize, c.writer_not_asked,
             c.writer_ages[0][0], c.writer_ages[0][1], c.writer_ages[0][2], c.writer_ages[1][0],
             c.writer_ages[1][1], c.writer_ages[1][2], c.writer_ages[2][0], c.writer_ages[2][1],
             c.writer_ages[2][2], c.writer_differed, c.writer_compared,
             static_cast<double>(marks) / frames, s.writers.size());
    std::vector<std::pair<const Writer*, const WriterStats*>> order;
    order.reserve(s.writers.size());
    for (const auto& [writer, stats] : s.writers) {
        order.emplace_back(&writer, &stats);
    }
    constexpr size_t Top = 8;
    const size_t shown = (std::min)(Top, order.size());
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(shown),
                      order.end(), [](const auto& a, const auto& b) {
                          return a.second->wait_ns > b.second->wait_ns;
                      });
    for (size_t i = 0; i < shown; ++i) {
        const Writer& writer = *order[i].first;
        const WriterStats& w = *order[i].second;
        const u64 reads = w.reads[0] + w.reads[1] + w.reads[2];
        s64 step = 0;
        u64 step_count = 0;
        for (const auto& [value, count] : w.steps) {
            if (count > step_count) {
                step = value;
                step_count = count;
            }
        }
        const bool draw =
            writer.kind == WriterKind::DrawStorage || writer.kind == WriterKind::DrawTexture;
        const std::string binding = draw ? fmt::format("stage {} binding {}", writer.binding / 64,
                                                       writer.binding % 64)
                                         : fmt::format("binding {}", writer.binding);
        LOG_INFO(HW_GPU,
                 "Read writer {}/{}: {} program {:016x} {} | reads flushed {}, read download {}, "
                 "fence download {} | the write's age at the read <0.5 ms {}, <1 {}, <2 {}, <4 "
                 "{}, more {} | waited {:.2f} ms a frame ({:.3f} ms a read) | the read's offset "
                 "in the marked range {:#x} to {:#x}, {} distinct, most common step {:+#x} ({} "
                 "times) | bytes differed at landing in {} of {} | {:.1f} dispatches bound after "
                 "the write",
                 i + 1, shown, WriterKindName(writer.kind), writer.program, binding, w.reads[0],
                 w.reads[1], w.reads[2], w.ages[0], w.ages[1], w.ages[2], w.ages[3], w.ages[4],
                 static_cast<double>(w.wait_ns) / 1e6 / frames,
                 reads != 0 ? static_cast<double>(w.wait_ns) / 1e6 / static_cast<double>(reads)
                            : 0.0,
                 w.lowest_offset, w.highest_offset, w.offsets.size(), step, step_count,
                 w.differed, w.compared,
                 reads != 0 ? static_cast<double>(w.dispatches_after) / static_cast<double>(reads)
                            : 0.0);
    }
    LOG_INFO(HW_GPU,
             "Read runs for B1 (FRITCHII_READ_RUNS) last 600 frames: {:.1f} distinct GPU-written "
             "runs read in the last {} frames ({:.0f} KiB, at most {} KiB) | marked GPU-written "
             "in them {:.0f} KiB a frame, {:.0f} KiB of it by dispatches",
             runs_a_frame, RunKeepFrames, run_kib_a_frame, run_kib_max,
             static_cast<double>(run_bytes) / 1024.0 / frames,
             static_cast<double>(run_compute_bytes) / 1024.0 / frames);
    s.writers.clear();
}

/// Log: the "Read writer ticks" lines, one per read kind that had reads (s.mutex held).
inline void LogTicks(State& s, double frames) {
    TickCounts& t = s.ticks;
    static constexpr const char* KindNames[3] = {"flushed (new download)", "read download on its way",
                                                 "fence download"};
    for (size_t k = 0; k < 3; ++k) {
        const u64 reads = t.reads[k];
        if (reads == 0 && t.no_mark[k] == 0) {
            continue;
        }
        const auto per_read = [reads](u64 value) {
            return reads != 0 ? static_cast<double>(value) / static_cast<double>(reads) : 0.0;
        };
        const auto ms_per_cut = [&](u64 ns) {
            return t.cuts[k] != 0 ? static_cast<double>(ns) / 1e6 / static_cast<double>(t.cuts[k])
                                  : 0.0;
        };
        LOG_INFO(HW_GPU,
                 "Read writer ticks (FRITCHII_READ_RUNS=2) last 600 frames, {}: {} reads with a "
                 "write mark over their pages, {} without one in the ring | the mark's writer: "
                 "dispatch storage {}, dispatch texture {}, draw storage {}, draw texture {}, "
                 "other (copy, query, transform feedback) {} | its submission when the read "
                 "asked: not submitted {}, submitted not run {}, run {} | the read's tick is "
                 "{:.1f} submissions after the writer's (0: {}, 1-2: {}, 3-5: {}, 6-10: {}, more: "
                 "{}), {:.1f} of them submitted and not run | waited {:.2f} ms a frame | host "
                 "waits cut at the writer's tick {}: until the writer ran {:.3f} ms each, then "
                 "until the download was there {:.3f} ms each (longest {:.2f})",
                 KindNames[k], reads, t.no_mark[k],
                 t.kinds[k][static_cast<size_t>(WriterKind::ComputeStorage)],
                 t.kinds[k][static_cast<size_t>(WriterKind::ComputeTexture)],
                 t.kinds[k][static_cast<size_t>(WriterKind::DrawStorage)],
                 t.kinds[k][static_cast<size_t>(WriterKind::DrawTexture)],
                 t.kinds[k][static_cast<size_t>(WriterKind::Other)], t.state[k][0], t.state[k][1],
                 t.state[k][2], per_read(t.behind_sum[k]), t.behind[k][0], t.behind[k][1],
                 t.behind[k][2], t.behind[k][3], t.behind[k][4], per_read(t.unrun_sum[k]),
                 static_cast<double>(t.wait_ns[k]) / 1e6 / frames, t.cuts[k],
                 ms_per_cut(t.writer_ns[k]), ms_per_cut(t.rest_ns[k]),
                 static_cast<double>(t.rest_max_ns[k]) / 1e6);
    }
    t = TickCounts{};
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
    LogWriters(s, Frames);
    if (g_writer_ticks) {
        LogTicks(s, Frames);
    }
    if (EarlyCompute::g_early || EarlyCompute::g_side_copy) {
        EarlyCompute::Log(Frames);
    }
    c = Counts{};
}

} // namespace VideoCommon::ReadRunStats
