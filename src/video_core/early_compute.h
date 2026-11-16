// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"

// Two answers to the read-back waits of a fight (work\helper_cu_answer.md, Rounds 2 and 3).
// Early compute stays a dry count (Round 3: chained inputs rule it out); the side copy has its
// verify and on modes.
//
// FRITCHII_EARLY_COMPUTE (read at first use, default 0; 1 = dry count; 2 and 3 not built, they
// count as 1): a dispatch whose output the guest CPU reads back (its
// program learned from the reads' writers, FRITCHII_READ_RUNS' ring) could run on a second queue
// at once instead of behind the ~10 submissions the host GPU has not run yet, if nothing pending
// on the main queue touches its bound ranges: no unrun GPU write over any of them (read after
// write, write after write) and no unrun use of the ranges it writes (write after read). The
// dry count evaluates each feeder dispatch where its buffers are bound
// (BufferCache::BindHostComputeBuffers, the back or the one follower's E), keeps the verdict in
// its write marks, and at each read of those bytes adds the read's "until the writer ran" time
// to the verdict's saving. Images are judged where the dispatch's views are filled
// (ComputePipeline::Configure, the back or R): a sampled or storage image whose last GPU write is
// not run yet. The buffer verdict and the image verdict are counted apart (with dispatches
// posted, the two run on different threads); the line says both.
//
// FRITCHII_READ_SIDE_COPY (read at first use, default 0; 1 = dry count; 2 = verify, 3 = on; the
// side queue itself needs the value at boot, Device::WantsSideQueue): a read's new download
// (EXP49) copied on a second queue (vk_side_queue.h) that waits only for the newest write over
// the copied bytes. 2 records both copies, lands the main one and compares (must be 0); 3 records
// the side copy alone and its reader lands it as soon as it is there. At RecordReadDownload the newest write mark over every copied
// range gives that write's scheduler tick: eligible when that tick is submitted already (a write
// in the open chunk keeps today's path). The reader's wait is cut at it: the part after it is the
// saving (less the side copy's own run).
//
// Both turn FRITCHII_READ_RUNS=2's ring and stamps on (read_run_stats.h ReadRunsValue).
namespace VideoCommon::EarlyCompute {

[[nodiscard]] inline int ReadMode(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] < '0' || value[0] > '9') {
        return 0;
    }
    return value[0] - '0';
}

/// A mode switch read at its first test (static initializers run before main() applies
/// nx2_switches.txt). Constant-initialized; racing first tests read the same value.
class FirstUseMode {
public:
    using Reader = int (*)();
    constexpr explicit FirstUseMode(Reader reader_) noexcept : reader{reader_} {}

    [[nodiscard]] int Get() const noexcept {
        int value = state.load(std::memory_order_relaxed);
        if (value < 0) [[unlikely]] {
            value = reader();
            state.store(value, std::memory_order_relaxed);
        }
        return value;
    }

    explicit operator bool() const noexcept {
        return Get() != 0;
    }

private:
    Reader reader;
    mutable std::atomic<int> state{-1};
};

inline FirstUseMode g_early{[] { return ReadMode("FRITCHII_EARLY_COMPUTE"); }};
inline FirstUseMode g_side_copy{[] { return ReadMode("FRITCHII_READ_SIDE_COPY"); }};
/// FRITCHII_SUBMIT_TIMES (vk_submit_times.h): GPU timestamps per submission, for the cuts.
inline FirstUseMode g_submit_times{[] { return ReadMode("FRITCHII_SUBMIT_TIMES"); }};

/// The buffer verdict of a dispatch, kept in its write marks.
enum class Verdict : u8 {
    NotJudged,    ///< not a feeder when bound (or the switch off)
    Qualifies,    ///< nothing unrun on the main queue touches its bound buffer ranges
    PendingWrite, ///< an unrun GPU write over one of its bound ranges (RAW or WAW)
    PendingUse,   ///< an unrun use of a range it writes (WAR)
    Count,
};

/// Set around a dispatch's binds (BindHostComputeBuffers), read by its write marks.
inline thread_local Verdict t_verdict{Verdict::NotJudged};

// ---- Uses: every bound buffer range (Buffer::MarkUsage), for the write-after-read test.

struct Use {
    u64 begin{};
    u64 end{};
    u64 stamp{}; ///< the buffer cache clock's stamp of the use
};

inline constexpr size_t UseRingSize = 65536;

struct UseRing {
    std::mutex mutex;
    std::vector<Use> uses;
    u64 count{};
};

inline UseRing& GetUseRing() {
    static UseRing ring;
    return ring;
}

/// Buffer::MarkUsage (g_early only).
inline void NoteUse(u64 begin, u64 size, u64 stamp) {
    if (size == 0) {
        return;
    }
    UseRing& r = GetUseRing();
    std::scoped_lock lock{r.mutex};
    if (r.uses.empty()) {
        r.uses.resize(UseRingSize);
    }
    r.uses[r.count % UseRingSize] = Use{begin, begin + size, stamp};
    ++r.count;
}

/// Whether an unrun use overlaps [begin, end): newest first, stopping at the first run one
/// (stamps grow with the ring, near enough for a count).
template <typename IsFree>
[[nodiscard]] bool PendingUse(u64 begin, u64 end, IsFree&& is_free) {
    UseRing& r = GetUseRing();
    std::scoped_lock lock{r.mutex};
    const u64 n = (std::min<u64>)(r.count, UseRingSize);
    for (u64 i = 0; i < n; ++i) {
        const Use& use = r.uses[(r.count - 1 - i) % UseRingSize];
        if (is_free(use.stamp)) {
            return false;
        }
        if (use.begin < end && begin < use.end) {
            return true;
        }
    }
    return false;
}

// ---- Feeders: programs whose writes the guest reads back.

struct Feeders {
    std::mutex mutex;
    std::unordered_set<u64> programs;
    std::atomic<u64> count{};
};

inline Feeders& GetFeeders() {
    static Feeders feeders;
    return feeders;
}

inline void LearnFeeder(u64 program) {
    Feeders& f = GetFeeders();
    std::scoped_lock lock{f.mutex};
    if (f.programs.insert(program).second) {
        f.count.fetch_add(1, std::memory_order_relaxed);
    }
}

[[nodiscard]] inline bool IsFeeder(u64 program) {
    Feeders& f = GetFeeders();
    if (f.count.load(std::memory_order_relaxed) == 0) {
        return false;
    }
    std::scoped_lock lock{f.mutex};
    return f.programs.contains(program);
}

// ---- Counts, logged with FRITCHII_READ_RUNS' lines every 600 frames.

struct ProgramImages {
    u64 dispatches{};    ///< feeder dispatches configured
    u64 with_images{};   ///< of them with a sampled or storage image
    u64 image_pending{}; ///< of them with an image whose last GPU write was not run yet
    u64 writes_images{}; ///< of them writing a storage image (its WAR is not tested)
};

struct Counts {
    std::mutex mutex;
    // Binds (the buffer verdict).
    u64 judged[static_cast<size_t>(Verdict::Count)]{};
    u64 uploads{}; ///< qualifying dispatches whose bind uploaded CPU data (moves with it)
    // Reads, per read kind (flushed, read download, fence download), by the writer's verdict.
    u64 reads[3][static_cast<size_t>(Verdict::Count)]{};
    u64 saving_ns[3]{}; ///< "until the writer ran" of the reads whose writer qualified
    u64 cut_reads[3]{}; ///< of them with that cut
    std::unordered_map<u64, ProgramImages> images;
    // FRITCHII_READ_SIDE_COPY dry: new downloads (EXP49) by their newest writer's tick.
    u64 side_downloads{};
    u64 side_eligible{};     ///< the writer's tick submitted when the download was recorded
    u64 side_open{};         ///< in the chunk the download closes: today's path is as good
    u64 side_no_mark{};      ///< no mark over the copied bytes in the ring
    u64 side_cuts{};         ///< reader waits cut at that tick
    u64 side_writer_ns{};    ///< until the writer's submission ran
    u64 side_saving_ns{};    ///< then until the download was there
    u64 side_saving_max_ns{};
    // FRITCHII_READ_SIDE_COPY >= 2.
    u64 side_submits{};        ///< side copies submitted
    u64 side_submit_failed{};  ///< failed (the main copy was recorded instead)
    u64 verify_downloads{};    ///< =2: downloads compared at the landing
    u64 verify_copies{};
    u64 verify_differing{};    ///< must be 0
    u64 verify_bytes{};
    u64 verify_waits{};        ///< =2: readers that waited for both copies
    u64 verify_side_ns{};      ///< until the side copy was there
    u64 verify_main_ns{};      ///< until the main copy was there (today's wait)
    u64 landed{};              ///< =3: reads landed from the side copy alone
    u64 landed_waits{};
    u64 landed_ns{};
    u64 landed_max_ns{};
};

inline Counts& Get() {
    static Counts counts;
    return counts;
}

inline void NoteJudged(Verdict verdict, bool uploaded) {
    Counts& c = Get();
    std::scoped_lock lock{c.mutex};
    ++c.judged[static_cast<size_t>(verdict)];
    if (verdict == Verdict::Qualifies && uploaded) {
        ++c.uploads;
    }
}

/// ReadRunStats::NoteWriter: a read whose pages' newest write mark came from a judged dispatch.
inline void NoteRead(size_t kind, Verdict verdict, bool cut, u64 writer_wait_ns) {
    Counts& c = Get();
    std::scoped_lock lock{c.mutex};
    ++c.reads[kind][static_cast<size_t>(verdict)];
    if (verdict == Verdict::Qualifies && cut) {
        ++c.cut_reads[kind];
        c.saving_ns[kind] += writer_wait_ns;
    }
}

/// ComputePipeline::Configure, for a feeder program.
inline void NoteImages(u64 program, bool with_images, bool pending, bool writes_images) {
    Counts& c = Get();
    std::scoped_lock lock{c.mutex};
    ProgramImages& p = c.images[program];
    ++p.dispatches;
    p.with_images += with_images ? 1 : 0;
    p.image_pending += pending ? 1 : 0;
    p.writes_images += writes_images ? 1 : 0;
}

/// BufferCache::RecordReadDownload: what the newest write over a new download's bytes was.
enum class SideWriter : u8 { NoMark, Open, Submitted };
inline void NoteSideDownload(SideWriter writer) {
    Counts& c = Get();
    std::scoped_lock lock{c.mutex};
    ++c.side_downloads;
    c.side_eligible += writer == SideWriter::Submitted ? 1 : 0;
    c.side_open += writer == SideWriter::Open ? 1 : 0;
    c.side_no_mark += writer == SideWriter::NoMark ? 1 : 0;
}

/// BufferCache::FinishReadDownloads: a reader's wait for an eligible download, cut there.
inline void NoteSideCut(u64 writer_ns, u64 rest_ns) {
    Counts& c = Get();
    std::scoped_lock lock{c.mutex};
    ++c.side_cuts;
    c.side_writer_ns += writer_ns;
    c.side_saving_ns += rest_ns;
    c.side_saving_max_ns = (std::max)(c.side_saving_max_ns, rest_ns);
}

/// BufferCache::RecordReadDownload: a side copy submitted (or failed).
inline void NoteSideSubmit(bool submitted) {
    Counts& c = Get();
    std::scoped_lock lock{c.mutex};
    ++(submitted ? c.side_submits : c.side_submit_failed);
}

/// BufferCache::LandReadDownload (=2): the side copy's bytes against the main copy's.
inline void NoteSideVerify(u64 copies, u64 differing, u64 bytes) {
    Counts& c = Get();
    std::scoped_lock lock{c.mutex};
    ++c.verify_downloads;
    c.verify_copies += copies;
    c.verify_differing += differing;
    c.verify_bytes += bytes;
    if (differing != 0) {
        LOG_ERROR(HW_GPU, "Read side copy (FRITCHII_READ_SIDE_COPY=2): {} of {} copies differ "
                          "from the main queue's (must be 0)",
                  differing, copies);
    }
}

/// BufferCache::FinishReadDownloads (=2): a reader's wait for the side copy, then for both.
inline void NoteSideVerifyWait(u64 side_ns, u64 total_ns) {
    Counts& c = Get();
    std::scoped_lock lock{c.mutex};
    ++c.verify_waits;
    c.verify_side_ns += side_ns;
    c.verify_main_ns += total_ns;
}

/// BufferCache::FinishReadDownloads (=3): a read landed from its side copy (`ns` waited, 0: it
/// was there).
inline void NoteSideLanded(u64 ns) {
    Counts& c = Get();
    std::scoped_lock lock{c.mutex};
    ++c.landed;
    if (ns != 0) {
        ++c.landed_waits;
        c.landed_ns += ns;
        c.landed_max_ns = (std::max)(c.landed_max_ns, ns);
    }
}

/// ReadRunStats::Log, every 600 frames.
inline void Log(double frames) {
    Counts& c = Get();
    std::scoped_lock lock{c.mutex};
    constexpr auto Q = static_cast<size_t>(Verdict::Qualifies);
    constexpr auto W = static_cast<size_t>(Verdict::PendingWrite);
    constexpr auto U = static_cast<size_t>(Verdict::PendingUse);
    constexpr auto N = static_cast<size_t>(Verdict::NotJudged);
    if (g_early) {
        u64 dispatches = 0, with_images = 0, image_pending = 0, writes_images = 0;
        for (const auto& [program, p] : c.images) {
            dispatches += p.dispatches;
            with_images += p.with_images;
            image_pending += p.image_pending;
            writes_images += p.writes_images;
        }
        const auto ms_frame = [frames](u64 ns) { return static_cast<double>(ns) / 1e6 / frames; };
        const auto ms_read = [](u64 ns, u64 n) {
            return n != 0 ? static_cast<double>(ns) / 1e6 / static_cast<double>(n) : 0.0;
        };
        LOG_INFO(HW_GPU,
                 "Early compute (FRITCHII_EARLY_COMPUTE={}, dry) last 600 frames: {} feeder "
                 "programs | feeder dispatches' buffers: qualify {} ({} uploaded CPU data), an "
                 "unrun write over a bound range {}, an unrun use of a written range {} | their "
                 "images (configured {}): with images {}, an image's last write unrun {}, writing "
                 "a storage image {} | reads by the writer's buffer verdict as qualify/unrun "
                 "write/unrun use/not judged: flushed {}/{}/{}/{}, fence download {}/{}/{}/{} | "
                 "predicted saving (the qualifying reads' 'until the writer ran'): flushed {:.2f} "
                 "ms a frame ({:.3f} a read), fence download {:.2f} ({:.3f})",
                 g_early.Get(), GetFeeders().count.load(std::memory_order_relaxed), c.judged[Q],
                 c.uploads, c.judged[W], c.judged[U], dispatches, with_images, image_pending,
                 writes_images, c.reads[0][Q], c.reads[0][W], c.reads[0][U], c.reads[0][N],
                 c.reads[2][Q], c.reads[2][W], c.reads[2][U], c.reads[2][N],
                 ms_frame(c.saving_ns[0]), ms_read(c.saving_ns[0], c.cut_reads[0]),
                 ms_frame(c.saving_ns[2]), ms_read(c.saving_ns[2], c.cut_reads[2]));
        // The feeders with the most dispatches: whether images rule them out.
        std::vector<std::pair<u64, ProgramImages>> order(c.images.begin(), c.images.end());
        const size_t shown = (std::min<size_t>)(6, order.size());
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(shown),
                          order.end(), [](const auto& a, const auto& b) {
                              return a.second.dispatches > b.second.dispatches;
                          });
        for (size_t i = 0; i < shown; ++i) {
            const ProgramImages& p = order[i].second;
            LOG_INFO(HW_GPU,
                     "Early compute feeder {}/{}: program {:016x}, {} dispatches, with images {}, "
                     "an image's last write unrun {}, writing a storage image {}",
                     i + 1, shown, order[i].first, p.dispatches, p.with_images, p.image_pending,
                     p.writes_images);
        }
    }
    if (g_side_copy) {
        const u64 cuts = c.side_cuts;
        LOG_INFO(HW_GPU,
                 "Read side copy (FRITCHII_READ_SIDE_COPY={}, dry) last 600 frames: {} new "
                 "downloads; newest write over their bytes: submitted already {} (eligible), in "
                 "the chunk the download closes {}, no mark in the ring {} | eligible reader waits "
                 "cut there {}: until the writer ran {:.3f} ms each, then (the saving) {:.3f} ms "
                 "each, longest {:.2f}, {:.2f} ms a frame",
                 g_side_copy.Get(), c.side_downloads, c.side_eligible, c.side_open, c.side_no_mark,
                 cuts,
                 cuts != 0 ? static_cast<double>(c.side_writer_ns) / 1e6 / static_cast<double>(cuts)
                           : 0.0,
                 cuts != 0 ? static_cast<double>(c.side_saving_ns) / 1e6 / static_cast<double>(cuts)
                           : 0.0,
                 static_cast<double>(c.side_saving_max_ns) / 1e6,
                 static_cast<double>(c.side_saving_ns) / 1e6 / frames);
        if (g_side_copy.Get() >= 2) {
            const auto per = [](u64 ns, u64 n) {
                return n != 0 ? static_cast<double>(ns) / 1e6 / static_cast<double>(n) : 0.0;
            };
            LOG_INFO(HW_GPU,
                     "Read side copy (FRITCHII_READ_SIDE_COPY={}) last 600 frames: {} side copies "
                     "submitted, {} failed | verify: {} downloads compared ({} copies, {} KiB), {} "
                     "copies differ (must be 0); {} readers waited for both: side copy there "
                     "after {:.3f} ms, main copy after {:.3f} ms | on: {} reads landed from the "
                     "side copy, {} waited {:.3f} ms each (longest {:.2f}), {:.2f} ms a frame",
                     g_side_copy.Get(), c.side_submits, c.side_submit_failed, c.verify_downloads,
                     c.verify_copies, c.verify_bytes >> 10, c.verify_differing, c.verify_waits,
                     per(c.verify_side_ns, c.verify_waits), per(c.verify_main_ns, c.verify_waits),
                     c.landed, c.landed_waits, per(c.landed_ns, c.landed_waits),
                     static_cast<double>(c.landed_max_ns) / 1e6,
                     static_cast<double>(c.landed_ns) / 1e6 / frames);
        }
    }
    std::fill(std::begin(c.judged), std::end(c.judged), 0);
    c.uploads = 0;
    for (auto& kind : c.reads) {
        std::fill(std::begin(kind), std::end(kind), 0);
    }
    std::fill(std::begin(c.saving_ns), std::end(c.saving_ns), 0);
    std::fill(std::begin(c.cut_reads), std::end(c.cut_reads), 0);
    c.images.clear();
    c.side_downloads = c.side_eligible = c.side_open = c.side_no_mark = 0;
    c.side_cuts = c.side_writer_ns = c.side_saving_ns = c.side_saving_max_ns = 0;
    c.side_submits = c.side_submit_failed = 0;
    c.verify_downloads = c.verify_copies = c.verify_differing = c.verify_bytes = 0;
    c.verify_waits = c.verify_side_ns = c.verify_main_ns = 0;
    c.landed = c.landed_waits = c.landed_ns = c.landed_max_ns = 0;
}

} // namespace VideoCommon::EarlyCompute
