// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>

#include "common/logging.h"
#include "core/core.h"
#include "core/device_memory_manager.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_runahead.h"

namespace VideoCommon::Runahead {

namespace {

using Clock = std::chrono::steady_clock;

u64 NowNs() {
    return static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count());
}

std::atomic<u64> g_program_id{0};
std::atomic<const Hooks*> g_hooks{nullptr};

// claim_mutex orders a boundary's gather and frontier move against the back's own gathers
// (HandleInvalidate): the back gathers only while it is past the frontier, and the frontier can't
// move while it does. queue_mutex guards the queues; it is held across a gather until the chunk
// is queued, so the gathered writes never leave ForEachUnappliedRange's sight (a read download
// landing meanwhile would write the GPU's older bytes over them). Order: claim, queue, then the
// dirty managers' locks. ForEachPendingCpuWrite takes queue_mutex last and alone.
std::mutex claim_mutex;
std::mutex queue_mutex;
std::deque<std::unique_ptr<Chunk>> exact_queue;  ///< Mode 1 chunks and the back's own gathers
std::deque<std::unique_ptr<Chunk>> verify_queue; ///< Mode 2 chunks
u64 verify_frontier = 0;                         ///< Front: mode 2's frontier (nothing written)

/// Front: early answers (and words written early) the back has not reached, for the bound.
struct EarlyAnswer {
    u64 position;
    u64 time_ns;
};
std::deque<EarlyAnswer> early_answers;
constexpr size_t MaxEarlyAnswers = 8192;

/// The newest release whose word the front wrote early, per word (LandingSuperseded).
struct ShadowSlot {
    std::atomic<u64> device{~u64{0}};
    std::atomic<u64> position{0};
};
std::array<ShadowSlot, 4096> shadow;

struct Stats {
    std::array<std::atomic<u64>, static_cast<size_t>(Unmet::Count)> unmet{};
    std::atomic<u64> boundaries{};
    std::atomic<u64> early_drain_path{};
    std::atomic<u64> words_written{};
    std::atomic<u64> refused_bound{};
    std::atomic<u64> refused_images{};
    std::atomic<u64> refused_no_word{};
    std::atomic<u64> chunks{};
    std::atomic<u64> chunk_ranges{};
    std::atomic<u64> chunk_bytes{};
    std::atomic<u64> applied{};
    std::atomic<u64> own_gathers{};
    std::atomic<u64> own_bytes{};
    std::atomic<u64> gathers_skipped{};
    std::array<std::atomic<u64>, static_cast<size_t>(Hold::Count)> holds{};
    std::atomic<u64> landings_skipped{};
    std::atomic<u64> verify_chunks{};
    std::atomic<u64> verify_ranges{};
    std::atomic<u64> verify_bytes{};
    std::atomic<u64> verify_differ{};
    std::atomic<u64> verify_differ_bytes{};
    std::atomic<u64> last_report_ns{};
};
Stats stats;
std::atomic<u32> logged_differences{};

u64 Take(std::atomic<u64>& counter) {
    return counter.exchange(0, std::memory_order_relaxed);
}

int BoundMs() {
    return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_runahead_ms,
                                            "EDEN_NX2_RUNAHEAD_MS", 16);
}

/// One line every 10 s, from whichever thread gets there first.
void MaybeReport() {
    const u64 now = NowNs();
    u64 last = stats.last_report_ns.load(std::memory_order_relaxed);
    if (last == 0) {
        stats.last_report_ns.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < 10'000'000'000ULL ||
        !stats.last_report_ns.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
        return;
    }
    const double window = static_cast<double>(now - last) / 1e9;
    const auto rate = [window](std::atomic<u64>& counter) {
        return static_cast<double>(Take(counter)) / window;
    };
    const auto kib = [window](std::atomic<u64>& counter) {
        return static_cast<double>(Take(counter)) / 1024.0 / window;
    };
    const auto unmet = [&](Unmet kind) { return rate(stats.unmet[static_cast<size_t>(kind)]); };
    const auto hold = [&](Hold kind) { return rate(stats.holds[static_cast<size_t>(kind)]); };
    const int value = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_runahead,
                                                       "EDEN_NX2_RUNAHEAD");
    const double queued_later = unmet(Unmet::QueuedLaterWriter);
    const double ran = unmet(Unmet::ReleaseRan);
    const double wrong = unmet(Unmet::WrongPayload);
    const double untracked = unmet(Unmet::UntrackedWriter);
    const double nothing = unmet(Unmet::Nothing);
    const double boundaries = rate(stats.boundaries);
    const double early = rate(stats.early_drain_path);
    const double written = rate(stats.words_written);
    const double refused_bound = rate(stats.refused_bound);
    const double refused_images = rate(stats.refused_images);
    const double refused_no_word = rate(stats.refused_no_word);
    const double chunks = rate(stats.chunks);
    const double ranges = rate(stats.chunk_ranges);
    const double chunk_kib = kib(stats.chunk_bytes);
    const double applied = rate(stats.applied);
    const double own = rate(stats.own_gathers);
    const double own_kib = kib(stats.own_bytes);
    const double skipped = rate(stats.gathers_skipped);
    const double bank = hold(Hold::BankCompare);
    const double fast = hold(Hold::FastUniform);
    const double aligned = hold(Hold::AlignedStream);
    const double landings = rate(stats.landings_skipped);
    const double verify_chunks = rate(stats.verify_chunks);
    const double verify_ranges = rate(stats.verify_ranges);
    const double verify_kib = kib(stats.verify_bytes);
    const u64 differ = Take(stats.verify_differ);
    const u64 differ_bytes = Take(stats.verify_differ_bytes);
    LOG_INFO(HW_GPU,
             "Run-ahead (EDEN_NX2_RUNAHEAD={}, bound {} ms, {:.1f} s): unmet readable acquires/s: "
             "queued release with a later writer {:.0f}, release ran {:.0f}, wrong payload "
             "{:.0f}, untracked writer queued {:.0f}, nothing queued {:.0f}; boundaries {:.0f}/s, "
             "early answers instead of a drain {:.0f}/s; words written early {:.0f}/s (not "
             "written: bound {:.0f}, image pages {:.0f}, no word {:.0f}); chunks {:.0f}/s, "
             "{:.0f} ranges/s, {:.1f} KiB/s, applied {:.0f}/s; back's own gathers {:.0f}/s "
             "({:.1f} KiB/s), skipped behind the frontier {:.0f}/s; held: bank compares "
             "{:.0f}/s, uniform streams {:.0f}/s; misaligned uniform streams from guest memory "
             "behind the frontier {:.0f}/s; payload landings skipped {:.0f}/s; verify {:.0f} "
             "chunks/s, {:.0f} ranges/s, {:.1f} KiB/s, {} ranges differ ({} bytes)",
             value, BoundMs(), window, queued_later, ran, wrong, untracked, nothing, boundaries,
             early, written, refused_bound, refused_images, refused_no_word, chunks, ranges,
             chunk_kib, applied, own, own_kib, skipped, bank, fast, aligned, landings,
             verify_chunks, verify_ranges, verify_kib, differ, differ_bytes);
}

/// Copies the guest bytes of `writes` into `chunk` (split at device pages).
void CopyRanges(const Hooks& hooks, const std::vector<std::pair<u64, u64>>& writes,
                Chunk& chunk) {
    size_t total = 0;
    for (const auto& [address, size] : writes) {
        total += size;
    }
    chunk.bytes.reserve(chunk.bytes.size() + total);
    for (const auto& [address, size] : writes) {
        u64 at = address;
        const u64 end = address + size;
        while (at < end) {
            const u64 page_end = (at & ~(u64{Core::DEVICE_PAGESIZE} - 1)) + Core::DEVICE_PAGESIZE;
            const u64 piece = (std::min)(end, page_end) - at;
            if (const u8* const guest = hooks.guest_pointer(hooks.argument, at); guest != nullptr) {
                chunk.ranges.push_back(Range{.address = at,
                                             .size = static_cast<u32>(piece),
                                             .offset = static_cast<u32>(chunk.bytes.size())});
                chunk.bytes.insert(chunk.bytes.end(), guest, guest + piece);
            }
            at += piece;
        }
    }
}

/// queue_mutex held: the oldest stamp left in either queue.
void RecomputeApplyAt() {
    u64 at = ~u64{0};
    if (!exact_queue.empty()) {
        at = (std::min)(at, exact_queue.front()->stamp);
    }
    if (!verify_queue.empty()) {
        at = (std::min)(at, verify_queue.front()->stamp);
    }
    g_apply_at.store(at, std::memory_order_release);
}

/// Mode 2, on the back at the chunk's stamp: the chunk's bytes against guest memory now, which is
/// what the old path uploads for the draws from here on.
void Verify(const Hooks& hooks, const Chunk& chunk, u64 position) {
    stats.verify_chunks.fetch_add(1, std::memory_order_relaxed);
    stats.verify_ranges.fetch_add(chunk.ranges.size(), std::memory_order_relaxed);
    stats.verify_bytes.fetch_add(chunk.bytes.size(), std::memory_order_relaxed);
    for (const Range& range : chunk.ranges) {
        const u8* const guest = hooks.guest_pointer(hooks.argument, range.address);
        if (guest == nullptr) {
            continue;
        }
        const u8* const bytes = chunk.bytes.data() + range.offset;
        if (std::memcmp(bytes, guest, range.size) == 0) {
            continue;
        }
        u32 differing = 0;
        u32 first = range.size;
        for (u32 i = 0; i < range.size; ++i) {
            if (bytes[i] != guest[i]) {
                first = (std::min)(first, i);
                ++differing;
            }
        }
        stats.verify_differ.fetch_add(1, std::memory_order_relaxed);
        stats.verify_differ_bytes.fetch_add(differing, std::memory_order_relaxed);
        if (logged_differences.fetch_add(1, std::memory_order_relaxed) >= 16) {
            continue;
        }
        const std::array<u64, 2> programs = hooks.bound_programs != nullptr
                                                ? hooks.bound_programs(hooks.argument)
                                                : std::array<u64, 2>{};
        u32 chunk_word = 0;
        u32 guest_word = 0;
        const u32 word_at = first & ~3U;
        const u32 word_bytes = (std::min)(4U, range.size - word_at);
        std::memcpy(&chunk_word, bytes + word_at, word_bytes);
        std::memcpy(&guest_word, guest + word_at, word_bytes);
        LOG_INFO(HW_GPU,
                 "Run-ahead verify: device {:#x}+{} differs at +{:#x} ({} bytes): chunk word "
                 "{:#010x}, guest {:#010x}; chunk of release {} applied at back position {} "
                 "(stamp {}, frontier {}, taken {:.2f} ms ago); bound programs vs {:#x} ps {:#x}",
                 range.address, range.size, first, differing, chunk_word, guest_word,
                 chunk.release, position, chunk.stamp, chunk.frontier,
                 static_cast<double>(NowNs() - chunk.created_ns) / 1e6, programs[0], programs[1]);
    }
}

} // Anonymous namespace

Mode CurrentMode() {
    const int value = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_runahead,
                                                       "EDEN_NX2_RUNAHEAD");
    if (value <= 0 || value > 3) [[likely]] {
        return Mode::Off;
    }
    if (g_program_id.load(std::memory_order_relaxed) != OcarinaOfTime) {
        return Mode::Off;
    }
    // Delayed fences (and EXP33's fence-held reports) land releases when the host GPU passes
    // them, not when the back runs them: LandingSuperseded can't tell those landings apart.
    if (VideoCore::Nx2Debug::DelayedFences() || VideoCore::Nx2Debug::Experiment(33) != 0) {
        return Mode::Off;
    }
    // Chunks hold the collected writes: only EXP35=2 collects every guest write of cached memory.
    if (value == 1 && VideoCore::Nx2Debug::Experiment(35) != 2) {
        return Mode::Off;
    }
    return static_cast<Mode>(value);
}

void SetProgramId(u64 program_id) {
    g_program_id.store(program_id, std::memory_order_relaxed);
}

void SetHooks(const Hooks* hooks) {
    g_hooks.store(hooks, std::memory_order_release);
}

void Reset() {
    std::scoped_lock lock{claim_mutex, queue_mutex};
    exact_queue.clear();
    verify_queue.clear();
    verify_frontier = 0;
    early_answers.clear();
    g_apply_at.store(~u64{0}, std::memory_order_release);
    g_frontier.store(0, std::memory_order_release);
    g_words_written.store(false, std::memory_order_release);
    for (ShadowSlot& slot : shadow) {
        slot.device.store(~u64{0}, std::memory_order_relaxed);
        slot.position.store(0, std::memory_order_relaxed);
    }
}

void CountUnmet(Unmet kind) {
    stats.unmet[static_cast<size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
    MaybeReport();
}

bool WithinBound(u64 consumed) {
    while (!early_answers.empty() && early_answers.front().position <= consumed) {
        early_answers.pop_front();
    }
    if (early_answers.empty()) {
        return true;
    }
    if (early_answers.size() >= MaxEarlyAnswers) {
        return false;
    }
    const u64 bound_ns = static_cast<u64>((std::max)(BoundMs(), 0)) * 1'000'000ULL;
    return NowNs() - early_answers.front().time_ns < bound_ns;
}

void NoteEarlyAnswer(u64 sent, bool by_drain_path) {
    if (early_answers.empty() || early_answers.back().position < sent) {
        early_answers.push_back(EarlyAnswer{.position = sent, .time_ns = NowNs()});
    }
    if (by_drain_path) {
        stats.early_drain_path.fetch_add(1, std::memory_order_relaxed);
    }
}

bool AtBoundary(Core::System& system, u64 device, u8* word, u32 payload, u64 release_position,
                u64 consumed) {
    const Mode mode = CurrentMode();
    if (mode != Mode::On && mode != Mode::Verify) {
        return false;
    }
    const Hooks* const hooks = g_hooks.load(std::memory_order_acquire);
    if (hooks == nullptr || hooks->guest_pointer == nullptr) {
        return false;
    }
    stats.boundaries.fetch_add(1, std::memory_order_relaxed);
    MaybeReport();
    if (mode == Mode::Verify) {
        // Only looks: the back still gathers and applies these writes the old way.
        auto chunk = std::make_unique<Chunk>();
        chunk->verify = true;
        chunk->release = release_position;
        chunk->created_ns = NowNs();
        std::vector<std::pair<u64, u64>> pending;
        system.GPU().ForEachPendingCpuWrite(
            [&pending](DAddr address, size_t size) { pending.emplace_back(address, size); });
        CopyRanges(*hooks, pending, *chunk);
        if (chunk->ranges.empty()) {
            return false;
        }
        std::scoped_lock lock{queue_mutex};
        chunk->stamp = verify_frontier;
        verify_frontier = (std::max)(verify_frontier, release_position);
        chunk->frontier = verify_frontier;
        verify_queue.push_back(std::move(chunk));
        RecomputeApplyAt();
        return false;
    }
    if (!WithinBound(consumed)) {
        stats.refused_bound.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (word == nullptr) {
        stats.refused_no_word.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    std::scoped_lock claim{claim_mutex};
    bool images = false;
    u64 frontier = g_frontier.load(std::memory_order_relaxed);
    // A release at or before the frontier: the chunks already protect every draw before it, so
    // the word goes out without a new one (several acquires often wait on one release).
    if (release_position > frontier) {
        std::scoped_lock queue{queue_mutex};
        auto chunk = std::make_unique<Chunk>();
        chunk->release = release_position;
        chunk->created_ns = NowNs();
        std::vector<std::pair<u64, u64>> gathered;
        std::function<void(PAddr, size_t)> collect = [&gathered](PAddr address, size_t size) {
            gathered.emplace_back(address, size);
        };
        g_exact_used.store(true, std::memory_order_seq_cst);
        system.GatherGPUDirtyMemory(collect);
        CopyRanges(*hooks, gathered, *chunk);
        if (hooks->may_hold_images != nullptr) {
            images = std::ranges::any_of(chunk->ranges, [hooks](const Range& range) {
                return hooks->may_hold_images(hooks->argument, range.address, range.size);
            });
        }
        // Taken whether or not the word is written: the writes are out of the dirty managers.
        chunk->stamp = frontier;
        if (!images) {
            frontier = (std::max)(frontier, release_position);
        }
        chunk->frontier = frontier;
        if (!chunk->ranges.empty()) {
            stats.chunks.fetch_add(1, std::memory_order_relaxed);
            stats.chunk_ranges.fetch_add(chunk->ranges.size(), std::memory_order_relaxed);
            stats.chunk_bytes.fetch_add(chunk->bytes.size(), std::memory_order_relaxed);
            exact_queue.push_back(std::move(chunk));
            RecomputeApplyAt();
        }
    }
    if (images) {
        // An image's upload reads guest memory whenever the texture cache gets to it: the CPU
        // must not run ahead of those bytes (the chunk still reaches the buffers in order).
        stats.refused_images.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    // The back must see the new frontier before the CPU can see the word (its guest reads check
    // the frontier without the claim lock), and the landing check must know the word is taken.
    ShadowSlot& slot = shadow[(device >> 2) % shadow.size()];
    if (slot.device.load(std::memory_order_relaxed) != device ||
        slot.position.load(std::memory_order_relaxed) < release_position) {
        slot.device.store(~u64{0}, std::memory_order_relaxed);
        slot.position.store(release_position, std::memory_order_relaxed);
        slot.device.store(device, std::memory_order_release);
    }
    g_words_written.store(true, std::memory_order_relaxed);
    g_frontier.store(frontier, std::memory_order_seq_cst);
    std::memcpy(word, &payload, sizeof(payload));
    NoteEarlyAnswer(GpuSplit::Get().SentPosition(), false);
    stats.words_written.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void ApplyDue(u64 position) {
    if (!GpuSplit::OnBackThread() || position < g_apply_at.load(std::memory_order_acquire)) {
        return;
    }
    const Hooks* const hooks = g_hooks.load(std::memory_order_acquire);
    for (;;) {
        Chunk* chunk = nullptr;
        bool from_verify = false;
        {
            std::scoped_lock lock{queue_mutex};
            if (!exact_queue.empty() && exact_queue.front()->stamp <= position) {
                chunk = exact_queue.front().get();
            } else if (!verify_queue.empty() && verify_queue.front()->stamp <= position) {
                chunk = verify_queue.front().get();
                from_verify = true;
            }
            if (chunk == nullptr) {
                RecomputeApplyAt();
                break;
            }
        }
        // Applied while still queued: its ranges stay pending for read downloads until done.
        // Only the back pops, so the element stays put while the front pushes behind it.
        if (hooks != nullptr) {
            if (chunk->verify) {
                if (hooks->guest_pointer != nullptr) {
                    Verify(*hooks, *chunk, position);
                }
            } else if (hooks->apply != nullptr) {
                hooks->apply(hooks->argument, *chunk);
            }
        }
        {
            std::scoped_lock lock{queue_mutex};
            (from_verify ? verify_queue : exact_queue).pop_front();
            RecomputeApplyAt();
        }
        stats.applied.fetch_add(1, std::memory_order_relaxed);
    }
    MaybeReport();
}

bool HandleInvalidate(Core::System& system) {
    auto& split = GpuSplit::Get();
    const bool on_back = GpuSplit::OnBackThread();
    const u64 position = on_back ? split.BackEntryPosition() : split.ConsumedNow();
    const bool mode_on = CurrentMode() == Mode::On;
    if (!mode_on && g_apply_at.load(std::memory_order_acquire) == ~u64{0} &&
        g_frontier.load(std::memory_order_acquire) <= position) [[likely]] {
        return false;
    }
    if (!on_back) {
        // Another thread (a host1x read): no inline uploads here (the scheduler is the back's).
        // Behind the frontier, guest memory may hold the next frame: leave the writes pending.
        if (g_frontier.load(std::memory_order_acquire) > position) {
            stats.gathers_skipped.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        return false;
    }
    ApplyDue(position);
    const Hooks* const hooks = g_hooks.load(std::memory_order_acquire);
    if (hooks == nullptr || hooks->guest_pointer == nullptr || hooks->apply == nullptr) {
        return false;
    }
    std::scoped_lock claim{claim_mutex};
    if (g_frontier.load(std::memory_order_acquire) > position) {
        // The front's next boundary takes them, with their bytes as they are then.
        stats.gathers_skipped.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    if (!mode_on) {
        // Switched off and caught up: the old path from here on.
        return false;
    }
    // Past the frontier: what guest memory holds now is what the draws from here on must see.
    // Applied at once, so no CPU-modified mark is left for a later boundary to race.
    {
        std::scoped_lock queue{queue_mutex};
        auto chunk = std::make_unique<Chunk>();
        chunk->own = true;
        chunk->created_ns = NowNs();
        std::vector<std::pair<u64, u64>> gathered;
        std::function<void(PAddr, size_t)> collect = [&gathered](PAddr address, size_t size) {
            gathered.emplace_back(address, size);
        };
        g_exact_used.store(true, std::memory_order_seq_cst);
        system.GatherGPUDirtyMemory(collect);
        CopyRanges(*hooks, gathered, *chunk);
        if (chunk->ranges.empty()) {
            return true;
        }
        stats.own_gathers.fetch_add(1, std::memory_order_relaxed);
        stats.own_bytes.fetch_add(chunk->bytes.size(), std::memory_order_relaxed);
        chunk->stamp = 0;
        exact_queue.push_back(std::move(chunk));
        RecomputeApplyAt();
    }
    ApplyDue(position);
    return true;
}

bool BackHolds(Hold what) {
    const u64 frontier = g_frontier.load(std::memory_order_acquire);
    if (frontier == 0 || !GpuSplit::OnBackThread() ||
        GpuSplit::Get().BackEntryPosition() >= frontier) [[likely]] {
        return false;
    }
    stats.holds[static_cast<size_t>(what)].fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool LandingSuperseded(u64 device) {
    if (!g_words_written.load(std::memory_order_relaxed) || !GpuSplit::OnBackThread())
        [[likely]] {
        return false;
    }
    ShadowSlot& slot = shadow[(device >> 2) % shadow.size()];
    if (slot.device.load(std::memory_order_acquire) != device) {
        return false;
    }
    const u64 position = slot.position.load(std::memory_order_acquire);
    if (slot.device.load(std::memory_order_acquire) != device) {
        return false;
    }
    // The release being run starts before the position the front noted it at (the end of its
    // entry); a later release starts at or after it.
    if (GpuSplit::Get().BackEntryPosition() >= position) {
        return false;
    }
    stats.landings_skipped.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void ForEachUnappliedRange(void (*callback)(void* argument, u64 address, u64 size),
                           void* argument) {
    std::scoped_lock lock{queue_mutex};
    for (const auto& chunk : exact_queue) {
        for (const Range& range : chunk->ranges) {
            callback(argument, range.address, range.size);
        }
    }
}

} // namespace VideoCommon::Runahead
