// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <optional>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <thread>
#include <vector>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "common/common_types.h"
#include "video_core/engines/engine_interface.h"
#include "video_core/split_write_tracker.h"

namespace Core {
class System;
}

namespace Tegra {
class MemoryManager;
}

namespace Tegra::Engines {
class EngineInterface;
}

namespace VideoCommon {

/// Writer ranges compared by GpuSplit::WritersMayWriteSince scans (logged with the front DMA
/// snapshot line): what the larger scan limit costs the front.
inline std::atomic<u64> g_writer_ranges_compared{0};

/// Why the split's front waited for its back (EDEN_NX2_GPU_PROFILE counts each).
enum class DrainSite : u32 {
    Other,
    FetchDrain,     ///< EDEN_NX2_FETCH_DRAIN: before every entry
    Acquire,        ///< A semaphore acquire reads the value the back may release
    SemaphoreFlush, ///< The acquired semaphore is in the buffer cache
    FenceRelease,   ///< The puller's acquires release the pending fences
    ListWriters,    ///< GPU-written command lists: the dispatches and copies sent before them
    ListSync,       ///< Command lists on draw-written pages: the work before the last host sync
    ListFlush,      ///< Command lists on GPU-written pages: drain and flush the pages
    MacroDmaRead,   ///< The front's macros read memory (MME DMA)
    MacroDmaWrites, ///< The same, waiting only for the queued work that may write it
    MacroDmaFlush,  ///< Those words are in the buffer cache
    GpSyncFence,    ///< A GP entry's sync bit (sync_memory_operations)
    Map,            ///< GPU memory maps before later command reads
    FrontStart,     ///< The front takes an engine's macros over
    FrontVerify,    ///< EDEN_NX2_FRONT_VERIFY
    MacroDmaCheck,  ///< EDEN_NX2_CMDPAGE_DIAG: a macro's memory read repeated after a full drain
    DispatchWait,   ///< EXP10: a macro memory read waits for the dispatches sent since it flushed
    FenceWriters,   ///< EXP12: a fence release waits for the writers a semaphore read can't see
    QmdRead,        ///< EXP13=2: the front's QMD scheduler copy reads a QMD queued work writes
    ListCopies,     ///< Lists on GPU-written pages: the dispatches and copies before the last sync
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(DrainSite::Count)> drain_site_names{
    "other",       "fetch drain",     "acquire",    "semaphore flush", "fence release",
    "list writers", "list sync",      "list flush", "macro dma read",  "macro dma writes",
    "macro dma flush",
    "gp sync fence", "map",           "front start", "front verify",  "macro dma check",
    "macro dma dispatch", "fence writers", "qmd read", "list copy writers",
};

/// EXP13=2: which queued writer set GpuSplit::QueuedGuestWriters' position (a wait cause).
enum class GuestWriterCause : u8 {
    None,
    Pending,          ///< A noted write over the bytes (NoteMemoryWrite, NoteMemoryWriteData)
    WriterRange,      ///< A compute or copy range over them (NoteWriterRange, NoteReleaseWriter)
    Ranged,           ///< A 3D ranged write over them (RangeLastWrite)
    UnboundedWriter,  ///< Compute or copy work without a range (NoteUnboundedWriter)
    Unranged,         ///< KeplerMemory, Fermi2D, syncpoint operations (NoteUnrangedWriter)
    Function,         ///< A queued function (GPU memory maps)
    MethodWrite,      ///< A 3D memory-writing method without a range (NoteMethodWrite)
    UnknownPending,   ///< A noted write of unknown range (its `what` says which)
    PendingScanLimit, ///< More noted writes queued than are remembered
    WriterScanLimit,  ///< More writer ranges queued than are compared
    OverlayEvicted,   ///< KeplerCompute's QMD copy: an evicted inline QMD may be the writer
    Count,
};

// Splits the GPU thread in two. The GPU thread (the front) takes the
// CPU's commands, parses command lists and runs the puller; engine method calls and everything
// behind them (register state, macros, draws, the rasterizer and its caches) run in order on a
// second thread (the back), fed through a single-producer single-consumer ring. Work the front
// would do on back-owned state is deferred into the same ring; the front waits for the back to
// catch up (Drain) only where it needs a result from that state, e.g. a semaphore it acquires.
// On for Switch 2 titles with the 3D and compute engines' macros on the front, against copies of
// the registers they read (Maxwell3D and KeplerCompute Front*), which sends the writes they make.
// EDEN_NX2_SPLIT=1 keeps the macros on the back; EDEN_NX2_SPLIT=0 runs everything on one thread.
class GpuSplit {
public:
    static GpuSplit& Get() {
        return instance;
    }

    /// Whether the front sends work to the back (front thread only).
    [[nodiscard]] bool Active() const {
        return active;
    }
    /// Counts starts of the split (EXP13=2: the front's QMD scheduler copy starts over).
    [[nodiscard]] u64 Session() const {
        return session;
    }
    /// Whether the front runs the engines' macros (EDEN_NX2_SPLIT=2).
    [[nodiscard]] bool FrontMacros() const {
        return front_macros;
    }

    /// EDEN_NX2_MACRO_SINK=1 (default off): the state writes the front's 3D macros make
    /// (registers outside execution_mask) go to the back's method sink like the command list's
    /// own state writes, instead of one CallMethod each; the back applies its sink, in order,
    /// before the engine's next executable method.
    [[nodiscard]] static bool MacroSink() {
        return macro_sink;
    }
    /// EDEN_NX2_COMPUTE_SINK=1 (default off): native compute writes that only store their value
    /// (native_plain_methods), from macros or the command list, are dropped when the front's
    /// register copy shows the back holds the value already, else sent batched (SinkMethod).
    [[nodiscard]] static bool ComputeSink() {
        return compute_sink;
    }
    /// EDEN_NX2_RING_PAIRS=1 (default off): the state writes that would go to the back's method
    /// sink (the command list's 3D state writes; the 3D macros' with EDEN_NX2_MACRO_SINK, the
    /// compute ones with EDEN_NX2_COMPUTE_SINK) go as (method, value) pairs in Pairs blocks
    /// (PairWrite), which the back applies in one loop per block (EngineInterface::ApplyPairs),
    /// after its method sink.
    [[nodiscard]] static bool RingPairs() {
        return ring_pairs;
    }

    /// The front's writes that took a batched path (EDEN_NX2_MACRO_SINK, EDEN_NX2_RING_PAIRS),
    /// logged every 10 s.
    enum class Batched : u32 {
        MacroSunk,   ///< 3D macro state writes sent to the method sink (or as pairs)
        ComputeSunk, ///< Compute state writes sent to the method sink (or as pairs)
        ComputeSame, ///< Compute state writes dropped: the register held the value
        Pairs,       ///< Writes sent as pairs (any engine, any source)
        PairBlocks,  ///< Pairs blocks opened
        DeltaWrites, ///< EXP22: 3D macro state writes sent in StateDelta blocks
        DeltaBlocks, ///< EXP22: StateDelta blocks opened
        DrawBatches, ///< EXP22: DrawBatch entries sent
        DrawRepeats, ///< EXP22: instance-loop groups they stand for past their first
        SameIndexFirst, ///< EXP2 or EXP22 bit 4: same-value index_buffer.first writes not sent
        Count,
    };
    void CountBatched(Batched counter, u64 amount = 1) {
        batched[static_cast<size_t>(counter)] += amount;
    }

    /// Whether the calling thread is the back.
    /// Back: true while its idle hook runs because it ran out of work (EXP99=1 destroys only then).
    [[nodiscard]] static bool InIdleWait() {
        return in_idle_wait;
    }
    [[nodiscard]] static bool OnBackThread() {
        return on_back_thread;
    }
    /// Back: the ring position of the entry being run.
    [[nodiscard]] u64 BackEntryPosition() const {
        return back_entry_position;
    }
    /// Called on the back before it sleeps waiting for work (the rasterizer submits deferred work).
    void SetIdleHook(void (*hook)(void*), void* argument) {
        idle_hook = hook;
        idle_argument = argument;
    }
    /// Experiment(3): asked on the back whenever it has run everything published, whether a
    /// ReleaseFences there would only invalidate the GPU cache (RasterizerInterface::FencesQuiet).
    using FenceQuietCheck = bool (*)(void* argument, bool strict, u64* epoch);
    void SetFenceQuietCheck(FenceQuietCheck check, void* argument) {
        fence_quiet_check = check;
        fence_quiet_argument = argument;
    }
    enum class FenceQuiet : u32 {
        Quiet,     ///< The back found nothing to release with everything sent executed
        Behind,    ///< Work sent is still queued (or unpublished)
        Busy,      ///< The back found fence work with everything sent executed
        Unchecked, ///< The back has not checked at this position (or its idle hook ran since)
    };
    /// Front: what the back's last check found, if it was made with everything sent executed.
    /// `epoch`: the fence work epoch the back saw (compare with FenceWorkEpoch()).
    [[nodiscard]] FenceQuiet FencesQuietNow(u64& epoch) const {
        if (consumed.load(std::memory_order_acquire) != write_position) {
            return FenceQuiet::Behind;
        }
        if (fence_quiet_position.load(std::memory_order_acquire) == write_position) {
            epoch = fence_quiet_epoch.load(std::memory_order_relaxed);
            return FenceQuiet::Quiet;
        }
        return fence_busy_position.load(std::memory_order_relaxed) == write_position
                   ? FenceQuiet::Busy
                   : FenceQuiet::Unchecked;
    }

    /// Any thread: whether the back thread runs (false before it starts and once it stopped).
    [[nodiscard]] bool BackRunning() const {
        return back_running.load(std::memory_order_acquire);
    }
    /// Any thread: has the back call the idle hook soon, between entries or at once if it sleeps.
    void RequestIdleHook() {
        hook_requested.store(true, std::memory_order_seq_cst);
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }

    /// Starts the back thread once the split is wanted (EDEN_NX2_SPLIT=1, a Switch 2 title,
    /// commands read without cache flushes). Called by the front between commands.
    void MaybeStart(Core::System& system);
    void Stop();

    /// An engine method write, executed as DmaPusher would (method sink, ConsumeSink, call).
    /// `segment`: the engine's current_dma_segment for the call, when the engine reads it.
    /// `effect`: what the entry may write (SplitWriteTracker; unknown unless the caller knows).
    using Effect = SplitWriteTracker::Effect;
    void Method(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument, bool is_last,
                const GPUVAddr* segment, Effect effect = Effect::Unknown) {
        SelectEngine(engine);
        const size_t words = segment != nullptr ? 2 : 1;
        u64* const entry = Reserve(words);
        entry[0] = static_cast<u64>(Kind::Method) | (is_last ? IsLast : 0) |
                   (segment != nullptr ? HasSegment : 0) | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        if (segment != nullptr) {
            entry[1] = *segment;
        }
        Commit(words, effect);
        if (engine->memory_write_mask[method]) {
            NoteMethodWrite(engine, method);
        }
    }
    void MultiMethod(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                     u32 amount, u32 methods_pending, const GPUVAddr* segment,
                     Effect effect = Effect::Unknown);
    /// A write made by a macro: CallMethod without the method sink, as the engines' macros do.
    void MacroWrite(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument,
                    Effect effect = Effect::Unknown) {
        SelectEngine(engine);
        u64* const entry = Reserve(1);
        entry[0] = static_cast<u64>(Kind::Method) | IsLast | Direct | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        Commit(1, effect);
        if (engine->memory_write_mask[method]) {
            NoteMethodWrite(engine, method);
        }
    }
    /// EDEN_NX2_MACRO_SINK: a write that only stores a register's value, to an engine whose
    /// execution_mask has the register (native compute): the back puts it in the engine's method
    /// sink tagged EngineInterface::SplitPlainTag, applied before the engine's next other write.
    void SinkMethod(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument) {
        SelectEngine(engine);
        u64* const entry = Reserve(1);
        entry[0] = static_cast<u64>(Kind::Method) | IsLast | Sink | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        Commit(1, Effect::None);
    }
    /// EDEN_NX2_RING_PAIRS: a write that only stores a register's value (no execution_mask side
    /// effect; native compute: native_plain_methods), appended to the open Pairs block. A block
    /// is open only until anything else is reserved or the ring is published (Reserve, Publish),
    /// so published entries never change and a block never crosses the ring's end.
    void PairWrite(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument) {
        SelectEngine(engine);
        const u64 pair = u64{method & 0xffff} | (u64{argument} << 32);
        const size_t index = write_position & RingMask;
        if (pairs_header != nullptr && index != 0 && pairs_count < MaxPairs &&
            write_position + 1 - consumed_cache <= RingWords) [[likely]] {
            ring[index] = pair;
            ++pairs_count;
            *pairs_header = static_cast<u64>(Kind::Pairs) | (u64{pairs_count} << 32);
            CountBatched(Batched::Pairs);
            Commit(1, Effect::None);
            return;
        }
        u64* const entry = Reserve(2);
        entry[0] = static_cast<u64>(Kind::Pairs) | (u64{1} << 32);
        entry[1] = pair;
        pairs_header = entry;
        pairs_count = 1;
        CountBatched(Batched::Pairs);
        CountBatched(Batched::PairBlocks);
        Commit(2, Effect::None);
    }

    // EDEN_NX2_EXP22, from a ported change (EDEN_NX2_FRONT_STATE, EDEN_NX2_DRAW_BATCH).
    /// Bit 1: a 3D state register write (not executable, writes no memory), which would go as a
    /// Direct MacroWrite or a PairWrite, appended to the open StateDelta block instead. The back
    /// applies the block in one loop, write by write as CallMethod and ApplyPairs would
    /// (Maxwell3D::ApplyFrontDelta). Open as a Pairs block is, and each write notes the effect
    /// its old path noted at its position.
    void DeltaWrite(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument,
                    Effect effect) {
        SelectEngine(engine);
        const u64 word = u64{method & 0xffff} | (u64{argument} << 32);
        const size_t index = write_position & RingMask;
        if (delta_header != nullptr && index != 0 && delta_count < MaxPairs &&
            write_position + 1 - consumed_cache <= RingWords) [[likely]] {
            ring[index] = word;
            ++delta_count;
            *delta_header = static_cast<u64>(Kind::StateDelta) | (u64{delta_count} << 32);
            CountBatched(Batched::DeltaWrites);
            Commit(1, effect);
            return;
        }
        u64* const entry = Reserve(2);
        entry[0] = static_cast<u64>(Kind::StateDelta) | (u64{1} << 32);
        entry[1] = word;
        delta_header = entry;
        delta_count = 1;
        CountBatched(Batched::DeltaWrites);
        CountBatched(Batched::DeltaBlocks);
        Commit(2, effect);
    }
    /// Bit 2: `count` (method | argument << 32) writes of an instance-loop group (draw.begin
    /// Subsequent and the instance loop stores after it), run once, then `repeats - 1` more
    /// identical groups counted as instances (Maxwell3D::RunDrawBatch).
    void DrawBatch(Tegra::Engines::EngineInterface* engine, const u64* group_writes, u32 count,
                   u32 repeats, Effect effect) {
        SelectEngine(engine);
        u64* const entry = Reserve(1 + size_t{count});
        entry[0] = static_cast<u64>(Kind::DrawBatch) | (u64{count & 0xff} << 8) |
                   (u64{repeats} << 32);
        std::memcpy(entry + 1, group_writes, size_t{count} * sizeof(u64));
        CountBatched(Batched::DrawBatches);
        CountBatched(Batched::DrawRepeats, repeats - 1);
        Commit(1 + size_t{count}, effect);
    }
    /// Work the front holds back (EXP22's draw batches): `hook(argument)` sends it before any
    /// other entry is reserved, before an engine is selected, before a publish and before a
    /// drain, so it keeps its place in the method order. Another holder's work is sent first.
    using PendingFlush = void (*)(void*);
    void SetPendingFlush(PendingFlush hook, void* argument) {
        if (pending_flush != hook || pending_flush_argument != argument) {
            RunPendingFlush();
            pending_flush = hook;
            pending_flush_argument = argument;
        }
    }
    void RunPendingFlush() {
        if (pending_flush != nullptr) [[unlikely]] {
            const PendingFlush hook = pending_flush;
            pending_flush = nullptr;
            hook(pending_flush_argument);
        }
    }
    /// `argument` sends its held work itself, or goes away: forget its hook.
    void DropPendingFlush(void* argument) {
        if (pending_flush_argument == argument) {
            pending_flush = nullptr;
            pending_flush_argument = nullptr;
        }
    }
    /// Instance folding (EDEN_NX2_FOLD_INSTANCES), front: one more
    /// instance of the instance loop `context` draws. Nothing is sent: `flush(context, count)`
    /// sends the instances counted so far as one entry before the next entry, engine selection,
    /// Publish or Drain. Held as the pending flush above, so other held work (EXP22's batches,
    /// EDEN_NX2_DRAW_PACKET's packet) keeps its order against it.
    void FoldInstance(void (*flush)(void*, u32), void* context) {
        if (fold_count != 0 && fold_context != context) {
            RunPendingFlush(); // The other loop's count (FlushFoldThunk) goes first
        }
        SetPendingFlush(&FlushFoldThunk, this);
        fold_flush = flush;
        fold_context = context;
        ++fold_count;
    }
    /// Sends the instances FoldInstance counted, if any (GpuSplit clears the hook before it).
    void FlushFold() {
        if (fold_count == 0) {
            return;
        }
        const u32 count = std::exchange(fold_count, 0);
        fold_flush(fold_context, count);
    }

    /// Runs `func` on the back after everything sent before it. Must be trivially copyable.
    /// `flags`: Nx2FastDrawCount::Neutral() for a call that can't change what a draw resolves to
    /// (bit 7 of a Call entry, set only while EDEN_NX2_FASTDRAW_COUNT counts; Execute ignores it).
    template <typename Func>
    void Defer(const Func& func, Effect effect = Effect::Unknown, u64 flags = 0) {
        static_assert(std::is_trivially_copyable_v<Func> && alignof(Func) <= alignof(u64));
        constexpr size_t payload = (sizeof(Func) + sizeof(u64) - 1) / sizeof(u64);
        u64* const entry = Reserve(2 + payload);
        entry[0] = static_cast<u64>(Kind::Call) | flags | (u64{payload} << 8);
        entry[1] = reinterpret_cast<u64>(&CallThunk<Func>);
        std::memcpy(entry + 2, &func, sizeof(Func));
        Commit(2 + payload, effect);
    }
    void DeferFunction(std::function<void()>&& func);
    /// Runs `func(payload)` on the back after everything sent before it, with `words` payload
    /// words that `fill(u64* payload)` writes here (a Call entry with a variable payload, for
    /// EDEN_NX2_DRAW_PACKET's write lists). `fill` must not call into the split.
    template <typename Fill>
    void CallWithPayload(void (*func)(const u64*), size_t words, Effect effect, const Fill& fill) {
        u64* const entry = Reserve(2 + words);
        entry[0] = static_cast<u64>(Kind::Call) | (u64{words} << 8);
        entry[1] = reinterpret_cast<u64>(func);
        fill(entry + 2);
        Commit(2 + words, effect);
    }
    /// Runs `func` on the back and waits for it: for work on back-owned state, such as a buffer
    /// cache download that records into the Vulkan scheduler. A drained back still submits from
    /// its idle hook, so the front never touches that state itself.
    /// `flags` as Defer's.
    template <typename Func>
    void Run(const Func& func, DrainSite site = DrainSite::Other, u64 flags = 0) {
        if (on_back_thread || !active) {
            func();
            return;
        }
        Defer(func, Effect::Unknown, flags);
        Drain(site);
    }

    /// EDEN_NX2_SPLIT_CHECK=1 (debug): reports, with its stack, a thread other than the back
    /// using back-owned state (the Vulkan scheduler) while the back runs entries.
    static inline bool check_owner{};
    static void CheckOwner(const char* where);

    /// Makes what was sent visible to the back.
    void Publish();
    /// EDEN_NX2_EXP27 & 1 (the original's EDEN_NX2_PUBLISH_WAITS): before the front yields
    /// or sleeps polling suspended channels, it publishes what they sent (held-back work too),
    /// so the back doesn't idle on work the front already decoded.
    void PublishBeforeWait();
    /// Waits until the back executed everything sent so far.
    void Drain(DrainSite site = DrainSite::Other);

    /// What sent compute or copy work that may write any memory (EXP10's log line).
    enum class WriterKind : u32 {
        Upload,  ///< Compute inline uploads the front could not bound
        Release, ///< Compute report semaphores the front could not translate
        Compute, ///< Other compute methods (unknown methods)
        Copy,    ///< Copies the front could not bound
        Other,   ///< Writers no engine front classified
        Count,
    };
    /// Compute or copy work was just sent: what can write GPU-written command lists.
    /// `classified`: the engine's front noted what that work writes (NoteWriterRange,
    /// NoteUnboundedWriter, NoteDispatchWriter); otherwise it may write anything.
    void NoteWriter(bool classified = false) {
        last_writer_position = write_position;
        if (!classified) {
            NoteUnboundedWriter(WriterKind::Other);
        }
    }
    /// Compute or copy work just sent may write any memory.
    void NoteUnboundedWriter(WriterKind kind) {
        last_unbounded_writer_position = write_position;
        ++unbounded_writers[static_cast<size_t>(kind)];
    }
    /// Compute or copy work just sent writes only [address, address + size) of `memory` (a copy's
    /// destination, a release, an inline upload); counts as unbounded when the front can't
    /// translate it as the back will.
    void NoteWriterRange(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size,
                         WriterKind kind);
    /// How the front noted a compute report semaphore (NoteReleaseWriter, EXP10's log line).
    enum class ReleaseNote : u32 {
        Ranged,       ///< Its bytes, as one device range
        Unmapped,     ///< Translated when the back runs it, to nothing: writes nothing
        NoWrite,      ///< An operation the back ignores (KeplerCompute::FrontNoteWriter)
        PendingMap,   ///< Unbounded: a queued function may still remap it
        Sparse,       ///< Unbounded: in a lazily backed sparse span
        LateUnmapped, ///< Unbounded: unmapped now, translated only when its fence signals
        Split,        ///< Unbounded: not one device range
        Count,
    };
    /// A compute report semaphore just sent writes [address, address + size). `at_run`: the back
    /// translates it when it runs the method and writes nothing when that fails (the query
    /// cache's payload reports), so an address without a mapping (Ocarina of Time releases to GPU
    /// address 0) needs no range; otherwise as NoteWriterRange.
    void NoteReleaseWriter(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size,
                           bool at_run);
    void CountRelease(ReleaseNote note) {
        ++release_notes[static_cast<size_t>(note)];
    }
    [[nodiscard]] const std::array<u64, static_cast<size_t>(ReleaseNote::Count)>&
    ReleaseNotes() const {
        return release_notes;
    }
    /// The noted ranges so far: a check made now covers those below it (WritersMayWriteSince).
    [[nodiscard]] u64 WriterRangeCount() const {
        writer_range_sealed = writer_range_count;
        return writer_range_count;
    }
    /// EXP10 +8: the writes found since a macro DMA read snapshot that overlap the range it
    /// checks (a 4 KiB page), as offsets into that range, so a read of other bytes of it can
    /// still skip the drain (Maxwell3D::FrontDmaRead). Past Capacity the two closest listed
    /// ranges become their union: never smaller than what was written, only coarser.
    struct ListedWrites {
        static constexpr u32 Capacity = 8;
        std::array<std::pair<u32, u32>, Capacity> ranges{};
        u32 count{};
        /// Unions made to stay within Capacity (for the log line).
        u32 merges{};
        /// Lists [begin, end) of the range at `base` (clipped to `size` bytes) unless a listed
        /// one covers it.
        void Add(u64 base, u64 size, u64 begin, u64 end) {
            const u32 first = static_cast<u32>((std::max)(begin, base) - base);
            const u32 last = static_cast<u32>((std::min)(end, base + size) - base);
            for (u32 i = 0; i < count; ++i) {
                if (ranges[i].first <= first && last <= ranges[i].second) {
                    return;
                }
            }
            if (count < Capacity) {
                ranges[count++] = {first, last};
                return;
            }
            // Sorted by start, the neighbours with the smallest gap (or the most overlap) merge.
            std::array<std::pair<u32, u32>, Capacity + 1> all{};
            std::copy(ranges.begin(), ranges.end(), all.begin());
            all[Capacity] = {first, last};
            std::sort(all.begin(), all.end());
            size_t best = 0;
            s64 best_gap = (std::numeric_limits<s64>::max)();
            for (size_t i = 0; i + 1 < all.size(); ++i) {
                const s64 gap = static_cast<s64>(all[i + 1].first) - all[i].second;
                if (gap < best_gap) {
                    best_gap = gap;
                    best = i;
                }
            }
            all[best].second = (std::max)(all[best].second, all[best + 1].second);
            for (size_t i = 0, out = 0; i < all.size(); ++i) {
                if (i != best + 1) {
                    ranges[out++] = all[i];
                }
            }
            ++merges;
        }
        /// Whether a listed write overlaps device bytes [begin, end) (the range at `base`).
        [[nodiscard]] bool Overlaps(u64 base, u64 begin, u64 end) const {
            for (u32 i = 0; i < count; ++i) {
                if (base + ranges[i].first < end && begin < base + ranges[i].second) {
                    return true;
                }
            }
            return false;
        }
    };
    /// Why WritersMayWriteSince answered true (EXP10's log line).
    enum class WriterCheck : u32 {
        Miss,      ///< Answered false
        Unbounded, ///< An unbounded writer since
        Overlap,   ///< A noted range overlaps (without `listed`)
        ScanLimit, ///< More than MaxWriterScan ranges noted since `checked`
        Forgotten, ///< Ranges since `checked` left the ring before they were compared
        Count,
    };
    /// Whether compute or copy work sent after position `since` may write device bytes
    /// [begin, begin + size): an unbounded writer, or a noted range that overlaps them.
    /// `checked` (optional): the noted ranges below it were found to miss these bytes before;
    /// set to WriterRangeCount() when nothing does. `listed` (optional, EXP10 +8): overlapping
    /// ranges are added to it instead of answering true; it and `checked` change only when the
    /// answer is false (the ranges below `checked` then miss or are listed). `why` (optional):
    /// set to the reason.
    [[nodiscard]] bool WritersMayWriteSince(u64 since, u64 begin, u64 size,
                                            u64* checked = nullptr,
                                            ListedWrites* listed = nullptr,
                                            WriterCheck* why = nullptr) const {
        u64 compared = 0;
        const auto answer = [why, &compared](WriterCheck reason) {
            if (why != nullptr) {
                *why = reason;
            }
            // Ranges compared by these scans: the front time a larger MaxWriterScan costs.
            g_writer_ranges_compared.fetch_add(compared, std::memory_order_relaxed);
            return reason != WriterCheck::Miss;
        };
        if (last_unbounded_writer_position > since) {
            return answer(WriterCheck::Unbounded);
        }
        const u64 known = checked != nullptr ? *checked : 0;
        ListedWrites found = listed != nullptr ? *listed : ListedWrites{};
        // EXP122: a watched page answers from its own list (gpu_split.cpp).
        if (watched_pages != 0) {
            const WatchAnswer watch = WatchedWritesSince(
                since, begin, size, known, listed != nullptr ? &found : nullptr);
            if (watch == WatchAnswer::Write) {
                return answer(WriterCheck::Overlap);
            }
            if (watch == WatchAnswer::NoWrite) {
                if (listed != nullptr) {
                    *listed = found;
                }
                if (checked != nullptr) {
                    *checked = WriterRangeCount();
                }
                return answer(WriterCheck::Miss);
            }
        }
        // At most MaxWriterScan ranges are compared (a drained read takes a new snapshot), and
        // more than are remembered count as overlapping.
        const u64 end = begin + size;
        const u64 oldest =
            writer_range_count - (std::min<u64>)(writer_range_count, writer_ranges.size());
        u64 index = writer_range_count;
        bool reached = false;
        for (u64 scanned = 0; index > (std::max)(known, oldest); --index, ++scanned) {
            if (scanned == MaxWriterScan) {
                return answer(WriterCheck::ScanLimit);
            }
            const WriterRange& range = writer_ranges[(index - 1) % writer_ranges.size()];
            ++compared;
            if (range.position <= since) {
                reached = true;
                break;
            }
            if (range.begin < end && begin < range.end) {
                if (listed == nullptr) {
                    return answer(WriterCheck::Overlap);
                }
                found.Add(begin, size, range.begin, range.end);
            }
        }
        if (!reached && oldest > known) {
            return answer(WriterCheck::Forgotten);
        }
        if (listed != nullptr) {
            *listed = found;
        }
        if (checked != nullptr) {
            *checked = WriterRangeCount();
        }
        return answer(WriterCheck::Miss);
    }
    /// EXP122 page watch (gpu_split.cpp): 0 off, 1 on, 2 on with a full-scan verify. Front only.
    void SetPageWatch(int mode) {
        if (mode != page_watch_mode) {
            page_watch_mode = mode;
            if (mode == 0) {
                ClearPageWatch();
            }
        }
    }
    /// EXP122: the front just took a snapshot of device page `page` (Maxwell3D::FrontDmaRead).
    void WatchPage(u64 page);
    struct PageWatchStats {
        u64 answered;       ///< WritersMayWriteSince answers from a watched page
        u64 answered_write; ///< of them "may write"
        u64 watched;        ///< Pages watched now
        u64 overflows;      ///< Byte lists that overflowed (unioned)
        u64 verified;       ///< EXP122=2: "no write" answers checked by a full scan
        u64 mismatches;     ///< of them the full scan found an overlap (must stay 0)
    };
    [[nodiscard]] PageWatchStats GetPageWatchStats() const;
    /// The ranges noted so far, without sealing the last one (EXP10's log line).
    [[nodiscard]] u64 NotedWriterRanges() const {
        return writer_range_count;
    }
    [[nodiscard]] u64 LastUnboundedWriterPosition() const {
        return last_unbounded_writer_position;
    }
    [[nodiscard]] const std::array<u64, static_cast<size_t>(WriterKind::Count)>&
    UnboundedWriters() const {
        return unbounded_writers;
    }

    // EXP10: what native compute dispatches write. The front can't tell (their storage buffers
    // and images come from constant buffers and descriptor pools queued work may still write),
    // so the back records, for each dispatch it runs, the device ranges it marks GPU-written in
    // the buffer cache (BufferCache::MarkWrittenBuffer: the storage and texture buffers' write
    // marks, all a flush or fence download brings to guest memory), its written images' guest
    // ranges, its indirect arguments and its QMD releases. A front read checks those once the
    // back ran every dispatch sent since its snapshot, waiting for that when asked (only that
    // far, not for what was sent after).

    /// Native compute work that may dispatch was just sent (a QMD launch or an inline QMD).
    void NoteDispatchWriter();
    /// Why a dispatch the back ran counts as writing anything.
    enum class DispatchUnbounded : u32 {
        Unconfirmed, ///< The rasterizer did not report its writes (another path or backend)
        Replaced,    ///< Replaced by other work (a network bypass, an upscaler)
        Image,       ///< A written image without a guest range
        Count,
    };
    enum class DispatchCheck : u32 {
        Miss,      ///< Every dispatch sent since ran, and none writes the bytes
        Queued,    ///< One is still queued (without the wait)
        Overlap,   ///< One writes the bytes
        Unbounded, ///< One may write anything (DispatchUnbounded, or sent with EXP10 off)
        ScanLimit, ///< Too many writes recorded since to compare
        Count,
    };
    /// Whether the native compute dispatches sent after position `since` may write device bytes
    /// [begin, begin + size). `checked`: the recorded writes below it were found to miss these
    /// bytes before (DispatchWriteCount() at the snapshot); updated on a miss. With `wait`, waits
    /// for the back to run the newest of those dispatches first (`waited` then set). `listed`
    /// (optional, EXP10 +8): as WritersMayWriteSince's, bounded overlapping writes are listed
    /// instead of answering Overlap; changed only on a Miss.
    DispatchCheck DispatchesMayWriteSince(u64 since, u64 begin, u64 size, bool wait, u64& checked,
                                          bool& waited, ListedWrites* listed = nullptr);
    /// The writes the back recorded so far.
    [[nodiscard]] u64 DispatchWriteCount() const {
        return dispatch_write_count.load(std::memory_order_acquire);
    }
    /// For EXP10's log line: native compute work sent that may dispatch, the dispatches the back
    /// ran whose writes it recorded, and why the others count as writing anything.
    struct DispatchStats {
        u64 sent;
        u64 bounded;
        std::array<u64, static_cast<size_t>(DispatchUnbounded::Count)> unbounded;
    };
    [[nodiscard]] DispatchStats GetDispatchStats() const;

    /// Back: whether the dispatches being run record their writes (BeginDispatchRecord).
    [[nodiscard]] static bool RecordingDispatch() {
        return recording_dispatch;
    }
    /// Back, around a QMD action that may dispatch and release (KeplerCompute).
    void BeginDispatchRecord() {
        recording_dispatch = on_back_thread && back_dispatch_records;
    }
    void EndDispatchRecord() {
        recording_dispatch = false;
    }
    /// Back, while recording: a dispatch starts; the rasterizer recorded all it writes; it ended
    /// (`ran`: it reached the rasterizer, `replaced`: other work ran instead).
    void StartDispatch() {
        dispatch_confirmed = false;
        dispatch_unbounded = false;
    }
    void ConfirmDispatch() {
        dispatch_confirmed = true;
    }
    void FinishDispatch(bool ran, bool replaced);
    /// Back, while recording: the dispatch writes device bytes [begin, begin + size).
    void RecordDispatchWrite(u64 begin, u64 size) {
        if (size != 0) {
            AppendDispatchWrite(begin, begin + size);
        }
    }
    /// Back, while recording: the dispatch may write anything.
    void RecordDispatchUnbounded(DispatchUnbounded reason);
    /// Waits until the back executed the compute and copy work sent so far (not what came after).
    /// Returns whether any of it was still queued.
    bool WaitForWriters();
    /// Other work that writes guest memory on the back was just sent: inline uploads, reports,
    /// puller methods (semaphores).
    void NoteMemoryWriter() {
        last_memory_writer_position = write_position;
    }
    /// Waits until the back executed every NoteWriter and NoteMemoryWriter work sent so far.
    bool WaitForMemoryWriters();
    /// memory-writing work whose range the front does not note was just sent (KeplerMemory
    /// and Fermi2D methods; DmaPusher is the only caller). Ends every macro DMA read snapshot.
    void NoteUnrangedWriter() {
        last_memory_writer_position = write_position;
        last_unranged_position = write_position;
        unranged_without_exp120 = write_position;
    }
    /// EXP120: a KeplerMemory or Fermi2D method that EXP120 = 0 notes as an unranged writer was
    /// just sent, and EXP120 noted its bytes instead (NoteMemoryWrite) or found it writes no
    /// memory. FrontDmaRead tells the reads this spares a drain apart (their own verify count).
    void NoteUnrangedWriterSpared() {
        unranged_without_exp120 = write_position;
    }
    /// Where LastUnrangedPosition would be with EXP120 = 0.
    [[nodiscard]] u64 LastUnrangedPositionWithoutExp120() const {
        return unranged_without_exp120;
    }
    /// What the KeplerMemory and Fermi2D methods sent were noted as (DmaPusher, EXP120), for the
    /// "DMA snapshot" log. Front only.
    enum class EngineWriteKind : u32 {
        I2mData,      ///< KeplerMemory data: the back writes the upload
        I2mRegister,  ///< KeplerMemory, any other method (exec included): registers only
        BlitLaunch,   ///< Fermi2D's blit launch (pixels_from_memory.src_y0 + 1)
        BlitRegister, ///< Fermi2D, any other method: registers only
        Count,
    };
    enum class EngineWriteNote : u32 {
        Unranged, ///< NoteUnrangedWriter
        Ranged,   ///< EXP120: NoteMemoryWrite of the bytes it may write
        Nothing,  ///< EXP120: writes no memory, not noted
        Count,
    };
    /// [call: 0 CallMethod, 1 CallMultiMethod][kind][note]
    using EngineWriteNotes =
        std::array<std::array<std::array<u64, static_cast<size_t>(EngineWriteNote::Count)>,
                              static_cast<size_t>(EngineWriteKind::Count)>,
                   2>;
    void CountEngineWrite(bool multi, EngineWriteKind kind, EngineWriteNote note) {
        ++engine_write_notes[multi ? 1 : 0][static_cast<size_t>(kind)][static_cast<size_t>(note)];
    }
    [[nodiscard]] const EngineWriteNotes& GetEngineWriteNotes() const {
        return engine_write_notes;
    }
    /// the ring position of everything sent so far.
    [[nodiscard]] u64 SentPosition() const {
        return write_position;
    }
    /// the last compute/copy and unranged writer positions.
    [[nodiscard]] u64 LastWriterPosition() const {
        return last_writer_position;
    }
    [[nodiscard]] u64 LastUnrangedPosition() const {
        return last_unranged_position;
    }
    /// whether work sent after position `since` may write device bytes
    /// [begin, begin + size): compute or copies (any range), unranged writers, or a noted write
    /// that overlaps the range. Also true when more writes were noted since than are remembered.
    [[nodiscard]] bool MayBeWrittenSince(u64 since, u64 begin, u64 size) const {
        if (last_writer_position > since || last_unranged_position > since) {
            return true;
        }
        return NotedWriteSince(since, begin, size);
    }
    /// The noted-write part of MayBeWrittenSince: a NoteMemoryWrite sent after `since` overlaps
    /// [begin, begin + size), or more were noted since than are remembered.
    [[nodiscard]] bool NotedWriteSince(u64 since, u64 begin, u64 size) const {
        if (last_memory_writer_position <= since) {
            return false;
        }
        const u64 end = begin + size;
        const u64 remembered = (std::min<u64>)(pending_write_count, pending_writes.size());
        for (u64 scanned = 0; scanned < remembered; ++scanned) {
            const PendingWrite& write =
                pending_writes[(pending_write_count - 1 - scanned) % pending_writes.size()];
            if (write.position <= since) {
                return false;
            }
            if (write.begin < end && begin < write.end) {
                return true;
            }
        }
        return remembered == pending_writes.size();
    }
    /// The newest NoteMemoryWrite / NoteMemoryWriteData sent after `since` over the u32 at
    /// `device` (what OverlayQueuedWrites(since) applies last to it), whether or not the back ran
    /// it since. `position` 0: none (or more were noted since than are remembered); `words` 0:
    /// its words are unknown, else `value` is the word it writes at `device`.
    struct NotedWord {
        u64 position;
        const char* what;
        u32 words;
        u32 value;
    };
    [[nodiscard]] NotedWord NewestNotedWriteTo(u64 since, u64 device) const {
        const u64 remembered = (std::min<u64>)(pending_write_count, pending_writes.size());
        for (u64 scanned = 0; scanned < remembered; ++scanned) {
            const PendingWrite& write =
                pending_writes[(pending_write_count - 1 - scanned) % pending_writes.size()];
            if (write.position <= since) {
                break;
            }
            if (write.begin < device + sizeof(u32) && device < write.end) {
                NotedWord found{write.position, write.what, write.words, 0};
                if (const u64 offset = device - write.begin;
                    write.words != 0 && device >= write.begin && offset % sizeof(u32) == 0 &&
                    offset / sizeof(u32) < write.words) {
                    found.value = write.data[offset / sizeof(u32)];
                } else {
                    found.words = 0;
                }
                return found;
            }
        }
        return {};
    }
    /// Work that writes the device range [begin, begin + size) was just sent (an inline upload, a
    /// report, a semaphore release). size ~0 means an unknown range.
    void NoteMemoryWrite(u64 begin, u64 size, const char* what = "unknown") {
        if (write_stamps) [[unlikely]] {
            if (size == ~u64{0}) {
                unknown_write_position = write_position;
            } else {
                StampWrite(begin, begin + size, false);
            }
        }
        PendingWrite& write = pending_writes[pending_write_count++ % pending_writes.size()];
        write.what = what;
        write.begin = begin;
        write.end = size == ~u64{0} ? ~u64{0} : begin + size;
        write.position = write_position;
        write.words = 0;
    }
    /// As NoteMemoryWrite, for a write whose words the front knows (a fence release payload,
    /// linear inline upload data): reads overlay them instead of waiting.
    void NoteMemoryWriteData(u64 begin, const u32* data, u32 words) {
        if (write_stamps) [[unlikely]] {
            StampWrite(begin, begin + u64{words} * sizeof(u32), false);
        }
        while (words != 0) {
            const u32 count = (std::min<u32>)(words, PendingWriteWords);
            PendingWrite& write = pending_writes[pending_write_count++ % pending_writes.size()];
            write.what = "known";
            write.begin = begin;
            write.end = begin + u64{count} * sizeof(u32);
            write.position = write_position;
            write.words = count;
            std::memcpy(write.data.data(), data, count * sizeof(u32));
            begin += u64{count} * sizeof(u32);
            data += count;
            words -= count;
        }
    }
    /// The back's position, for OverlayQueuedWrites: take it before reading guest memory.
    [[nodiscard]] u64 ConsumedNow() {
        return consumed.load(std::memory_order_acquire);
    }
    /// Applies to `out` (the words at device address `begin`, read from guest memory after
    /// `since` = ConsumedNow()) the queued writes the front knows the words of. False when an
    /// overlapping queued write's words are unknown (the caller waits instead).
    bool OverlayQueuedWrites(u64 since, u64 begin, u32* out, u32 words);
    /// Waits until the back executed the noted writes that overlap [begin, begin + size).
    /// Returns false when more writes are queued than are remembered (the caller drains).
    bool WaitForWritesTo(u64 begin, u64 size);
    /// Whether a noted write that overlaps [begin, begin + size) is still queued (or too many
    /// are queued to tell).
    bool HasQueuedWritesTo(u64 begin, u64 size);
    /// Front: a write that lands when a host query completes or a fence signals (counter
    /// reports, timed releases), not when the back runs it (Tegra::Nx2AsyncReportsSettled).
    void NoteAsyncWrite(u64 begin, u64 size) {
        for (u64 slot = begin >> 4; slot <= (begin + size - 1) >> 4; ++slot) {
            async_writes.Set(slot, write_position);
        }
    }
    /// Front: the ring position of the newest async write noted to these bytes, or 0.
    [[nodiscard]] u64 HasAsyncWrite(u64 begin, u64 size) const {
        u64 newest = 0;
        if (async_writes.empty()) {
            return newest;
        }
        for (u64 slot = begin >> 4; slot <= (begin + size - 1) >> 4; ++slot) {
            if (const u64* const position = async_writes.Find(slot)) {
                newest = (std::max)(newest, *position);
            }
        }
        return newest;
    }

    /// A host-level wait was just sent (a syncpoint acquire, a host WFI, a GP entry with its sync
    /// bit; semaphore acquires drain): the guest's GPU work before it is complete before the
    /// commands after it are fetched.
    void NoteSync() {
        last_sync_position = write_position;
    }
    /// Waits until the back executed the work sent before the last NoteSync. Returns whether any
    /// of it was still queued.
    bool WaitForSync();
    /// Waits until the back executed the compute and copy work sent before the last NoteSync:
    /// the writers a command list fetched now is ordered after.
    /// Returns whether any of it was still queued. `site`: whose wait it is, for
    /// EDEN_NX2_GPU_PROFILE.
    bool WaitForSyncedWriters(DrainSite site = DrainSite::ListWriters);
    /// Whether the back executed the work sent before `position` (as far as it stored `consumed`,
    /// every few thousand words of a batch and at its end).
    [[nodiscard]] bool Ran(u64 position) const {
        return consumed.load(std::memory_order_acquire) >= position;
    }
    /// Deferred work that can write memory was just sent (puller semaphore releases, maps).
    void NoteMemoryWrite() {
        last_memory_write_position = write_position;
        last_memory_write_method = DeferredWrite;
    }
    /// The 3D method just sent writes only the bytes its front noted (NoteMemoryWrite with a
    /// range, NoteMemoryWriteData): QueuedGuestWriters counts those, not the method as a writer
    /// of anything (as RangeLastWrite does for ranged writes). Only QueuedGuestWriters reads it.
    void RangedMethodWrite() {
        if (last_method_write_position == write_position) {
            last_method_write_position = previous_method_write_position;
        }
    }
    /// A method that writes memory was just sent; the method is kept for the GPU profile.
    void NoteMethodWrite(const Tegra::Engines::EngineInterface* engine, u32 method) {
        previous_memory_write_position = last_memory_write_position;
        previous_memory_write_method = last_memory_write_method;
        last_memory_write_position = write_position;
        // Engines all of whose methods write (compute, DMA, inline-to-memory, 2D) mark method 0.
        last_memory_write_method = engine->memory_write_mask[0] ? OtherEngineWrite : method;
        // QueuedGuestWriters: per-method writers only (3D); the engines all of whose methods
        // write note their ranges or count as unbounded or unranged writers. 3D wait_for_idle
        // (0x44) and sync_info (0xb2) write no guest memory themselves.
        if (!engine->memory_write_mask[0] && method != 0x44 && method != 0xb2) {
            previous_method_write_position = last_method_write_position;
            last_method_write_position = write_position;
        }
    }
    /// The method just sent writes only [begin, end): a macro memory read waits for it only where
    /// they overlap. Without this call its write counts for every read. EDEN_NX2_RANGED_WAITS=0
    /// keeps every write counting.
    /// `data` (optional): the words it writes there, in address space `space`, which a macro
    /// memory read of them takes without waiting (ForwardRead).
    void RangeLastWrite(GPUVAddr begin, GPUVAddr end, const u32* data = nullptr, size_t space = 0);
    /// For a macro memory read of `words` words at `begin` in address space `space`: when the
    /// newest work that may write them is one constant buffer upload still queued for the back,
    /// copies its words to `out` and returns true; the read then needs no wait.
    /// EDEN_NX2_FORWARD_READS=0 turns it off.
    bool ForwardRead(size_t space, GPUVAddr begin, u32 words, u32* out);
    /// For the front's macro memory reads (MME DMA) of [begin, end): waits until the back executed
    /// the methods that write memory in command order (memory_write_mask, NoteMemoryWrite; ranged
    /// writes only where they overlap), the compute and copy work and the work before the last
    /// host-level wait, all sent so far.
    bool WaitForMemoryWriters(DrainSite site, GPUVAddr begin, GPUVAddr end);
    /// Whether WaitForMemoryWriters(begin, end) would wait now: work that may write the range is
    /// still queued for the back.
    bool HasQueuedMemoryWriters(GPUVAddr begin, GPUVAddr end);
    /// Diagnostics: the newest queued work that may write [begin, end): 0 none, 1 compute or
    /// copy, 2 work before a host sync, 3 an unranged memory write (`method`), 4 a ranged write.
    u32 QueuedWriterKind(GPUVAddr begin, GPUVAddr end, u32& method);
    /// A semaphore release that writes `payload` at device address `device` (from GPU address
    /// `address`) was just sent (a puller release, a 3D payload report).
    void NoteRelease(u64 device, GPUVAddr address, u32 payload);
    /// EXP13=2 (diagnostic): the newest work sent after `done` that may write guest memory over
    /// device bytes [device, device + size) (GPU addresses from `address`) when the back runs
    /// it, leaving out native compute launches and inline QMDs (the front's QMD scheduler copy
    /// keeps their words), draws and dispatches (their stores go to the host GPU's buffers; what
    /// a download brings back lands on pages marked GPU-written, which the callers check) and
    /// queued functions other than maps (fence releases: GPU-written pages, semaphore words and
    /// reports). `noted`: the newest write noted over the bytes (inline uploads, reports,
    /// semaphore releases, copies and compute releases with known ranges; those at a position in
    /// `skip` don't count); `unbounded`: the newest that may write anything (compute and copy
    /// writers without a range, unranged writers, 3D memory writes without a range, maps, more
    /// writes than are remembered). 0 for none.
    struct GuestWriters {
        u64 noted;
        u64 unbounded;
        GuestWriterCause noted_cause;
        GuestWriterCause unbounded_cause;
        const char* what; ///< UnknownPending: the noted write's `what`
        /// WriterRange: the range's kind and device bytes [range_begin, range_end).
        WriterKind range_kind;
        u64 range_begin;
        u64 range_end;
    };
    [[nodiscard]] GuestWriters QueuedGuestWriters(u64 done, GPUVAddr address, u64 device, u64 size,
                                                  std::span<const u64> skip = {}) const;
    /// EXP13=2/3, the inverse of QueuedGuestWriters (fable_answer_6 6f (c) a): once enabled,
    /// every write the front notes stamps the 256-byte blocks it covers (4 KB pages for ranges
    /// over 16 KB, everything for ranges over 1 MB) with its position, in hashed tables whose
    /// collisions only make stamps newer. WriteStamp is the newest position that may have
    /// written [device, device + size) or GPU [address, address + size) since: noted writes,
    /// writer ranges, 3D ranged writes, and the writers QueuedGuestWriters counts as unbounded.
    /// Everything sent before EnableWriteStamps counts as a write. Writes noted while a tag is
    /// set (SetWriteStampTag) are not counted by a WriteStamp asking with that tag (one inline
    /// upload patching what it already wrote), unless another writer stamped the block since.
    void EnableWriteStamps();
    void SetWriteStampTag(u64 tag) {
        write_stamp_tag = tag;
    }
    /// A tag no other write has (inline uploads of any engine).
    [[nodiscard]] u64 NewWriteStampTag() {
        return ++write_stamp_tags;
    }
    [[nodiscard]] u64 WriteStamp(u64 device, u64 size, GPUVAddr address, u64 skip_tag = 0) const;
    /// For a semaphore acquire of `device`: the payload a release still queued for the back writes
    /// there, if nothing sent after it may write the word. The back runs the acquire's channel
    /// after that release, so an acquire it meets needs no drain. EDEN_NX2_ACQUIRE_QUEUED=0 off.
    /// `miss` (when no payload): 1 no release recorded, 2 it already ran, 3 a later write.
    std::optional<u32> QueuedReleasePayload(u64 device, GPUVAddr address, u32* miss = nullptr);
    /// EDEN_NX2_ACQUIRE_PRECHECK=2's reset check (puller.cpp PrecheckAudit): the payload of the
    /// newest release recorded for the u32 at `device` (NoteRelease) while the back has not run
    /// it (its ring position is past `done`), whatever was sent after it (unlike
    /// QueuedReleasePayload); `position` gets that ring position. queued_releases keeps one
    /// release per slot: a newer release to another word in the same slot hides it.
    [[nodiscard]] std::optional<u32> QueuedReleaseTo(u64 device, u64 done, u64& position) const {
        const QueuedRelease& release = queued_releases[(device >> 2) % queued_releases.size()];
        if (release.device != device || release.position == 0 || release.position <= done) {
            return std::nullopt;
        }
        position = release.position;
        return release.payload;
    }
    /// EDEN_NX2_RUNAHEAD: the newest release recorded for the u32 at `device` (NoteRelease),
    /// whether or not the back ran it; `position` gets its ring position.
    [[nodiscard]] std::optional<u32> NewestReleaseTo(u64 device, u64& position) const {
        const QueuedRelease& release = queued_releases[(device >> 2) % queued_releases.size()];
        if (release.device != device || release.position == 0) {
            return std::nullopt;
        }
        position = release.position;
        return release.payload;
    }
    /// EDEN_NX2_RUNAHEAD: the ring position of the newest native compute work that may dispatch
    /// (NoteDispatchWriter), 0 for none this session.
    [[nodiscard]] u64 LastDispatchPosition() const {
        return last_dispatch_position;
    }
    /// EXP12: the ring position past the queued work that may write the u32 at `device` without
    /// the front knowing its words, so a semaphore read that waits for or overlays the noted
    /// writes (EDEN_NX2_SEM_FAST) can't account for it: compute and copies (their QMD and report
    /// releases among them), unranged writers, a release recorded for the word, and queued
    /// functions (maps). Draws that store to the word mark its page GPU-written once they ran.
    [[nodiscard]] u64 UntrackedWriterPosition(u64 device) const {
        u64 target =
            (std::max)({last_writer_position, last_unranged_position, last_function_position});
        if (const QueuedRelease& release = queued_releases[(device >> 2) % queued_releases.size()];
            release.device == device) {
            target = (std::max)(target, release.position);
        }
        return (std::min)(target, write_position);
    }
    /// The entry just sent also writes [address, address + size) of `memory`, noted by device
    /// page (other mappings may alias it, and the caches keep state per page).
    void NoteWrite(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size);
    /// The entry just sent writes guest memory the front can't tell.
    void NoteUnknownWrite() {
        writes.Entry(Effect::Unknown, write_position);
    }
    /// Waits for the queued functions (GPU memory maps among them) before the front translates
    /// addresses, then drops the pending writes the back has run. Returns the back's position.
    u64 SettleWrites();
    /// Ring position the back must pass to execute possible writers of [begin, end).
    /// Deferred fence writes can still be pending afterwards.
    [[nodiscard]] u64 ReadTarget(u64 begin, u64 end) const {
        return writes.ReadTarget(begin, end);
    }
    [[nodiscard]] const SplitWriteTracker& Writes() const {
        return writes;
    }
    /// Waits until the back consumed `position` (not what came after). Returns whether it had
    /// not yet.
    bool WaitUntil(u64 position, DrainSite site) {
        return WaitFor((std::min)(position, write_position), site);
    }

private:
    friend struct GpuSplitTestAccess;

    enum class Kind : u64 {
        Wrap,   ///< The rest of the ring is unused: continue at its start
        Engine, ///< The engine the next methods go to: word 1
        Method, ///< Bits 8-23 method, 32-63 argument; flags IsLast, HasSegment (word 1)
        Multi,  ///< Bits 8-23 method, 32-63 amount; word 1 methods pending, [segment], words
        Call,   ///< Bits 8-31 payload size in words; word 1 thunk, then the payload
        Pairs,  ///< Bits 32-63 count; then one word per write: bits 0-15 method, 32-63 value
        StateDelta, ///< As Pairs, Maxwell3D macro writes (DeltaWrite)
        DrawBatch,  ///< Bits 8-15 count, 32-63 repeats; then count words (method | arg << 32)
    };
    static constexpr u64 IsLast = 1 << 4;
    static constexpr u64 HasSegment = 1 << 5;
    static constexpr u64 Direct = 1 << 6; ///< Method: call it even if not executable (macros)
    static constexpr u64 Sink = 1 << 7;   ///< Method: into the method sink, tagged (SinkMethod)

    // Read when the split first starts (MaybeStart), after main() applied nx2_switches.txt: a
    // static initialiser would run before it.
    static inline bool macro_sink{};
    static inline bool compute_sink{};
    static inline bool ring_pairs{};
    static constexpr u32 MaxPairs = 4096;

    /// async_writes' table: 16-byte slot -> ring position, open addressing with linear probing.
    /// As the std::unordered_map it replaces: entries are never erased (only clear(), at Reset),
    /// a note to a known slot overwrites its position. It allocates only when it doubles (at
    /// half full), not a node per new slot (FrontNoteMemoryWrite -> NoteAsyncWrite was 1.4-1.5%
    /// of the front, ~15 ms/s, fable_answer_regression_2 4.4).
    class AsyncWriteTable {
    public:
        [[nodiscard]] bool empty() const {
            return count == 0;
        }
        void clear() {
            if (count != 0) {
                std::fill(entries.begin(), entries.end(), Entry{EmptyKey, 0});
                count = 0;
            }
        }
        void Set(u64 key, u64 position) {
            if ((count + 1) * 2 > entries.size()) [[unlikely]] {
                Grow();
            }
            for (size_t index = Hash(key) & mask;; index = (index + 1) & mask) {
                Entry& entry = entries[index];
                if (entry.key == key) {
                    entry.position = position;
                    return;
                }
                if (entry.key == EmptyKey) {
                    entry = {key, position};
                    ++count;
                    return;
                }
            }
        }
        /// The position noted to `key`, or null.
        [[nodiscard]] const u64* Find(u64 key) const {
            if (count == 0) {
                return nullptr;
            }
            for (size_t index = Hash(key) & mask;; index = (index + 1) & mask) {
                const Entry& entry = entries[index];
                if (entry.key == key) {
                    return &entry.position;
                }
                if (entry.key == EmptyKey) {
                    return nullptr;
                }
            }
        }

    private:
        /// Slots are device addresses >> 4: never all ones.
        static constexpr u64 EmptyKey = ~u64{0};
        struct Entry {
            u64 key;
            u64 position;
        };
        [[nodiscard]] static size_t Hash(u64 key) {
            const u64 mixed = key * 0x9e3779b97f4a7c15ULL;
            return static_cast<size_t>(mixed ^ (mixed >> 29));
        }
        void Grow() {
            std::vector<Entry> old = std::move(entries);
            entries.assign(old.empty() ? size_t{1024} : old.size() * 2, Entry{EmptyKey, 0});
            mask = entries.size() - 1;
            count = 0;
            for (const Entry& entry : old) {
                if (entry.key != EmptyKey) {
                    Set(entry.key, entry.position);
                }
            }
        }
        std::vector<Entry> entries;
        size_t mask{};
        size_t count{};
    };
    AsyncWriteTable async_writes; ///< Front: 16-byte slot -> ring position

    static constexpr size_t RingWords = size_t{1} << 20; // 8 MiB
    static constexpr size_t RingMask = RingWords - 1;
    static constexpr u32 MaxMultiWords = 8192;
    static constexpr size_t PublishEvery = 256;
    /// EXP39 & 1: the fewest words the front publishes to a back that waits for them (Commit).
    /// A back that runs faster than the front would otherwise take one entry per publish.
    static constexpr size_t PublishAfterWait = 8;

    template <typename Func>
    static void CallThunk(const u64* payload) {
        (*reinterpret_cast<const Func*>(payload))();
    }

    /// Room for an entry of `words` at the write position (never across the ring's end).
    u64* Reserve(size_t words) {
        RunPendingFlush(); // Held-back work goes before any other entry
        // What follows ends the open Pairs or StateDelta block.
        pairs_header = nullptr;
        delta_header = nullptr;
        const size_t index = write_position & RingMask;
        if (index + words <= RingWords && write_position + words - consumed_cache <= RingWords)
            [[likely]] {
            return &ring[index];
        }
        return ReserveSlow(words);
    }
    u64* ReserveSlow(size_t words);
    void Commit(size_t words, Effect effect) {
        write_position += words;
        writes.Entry(effect, write_position);
        // EXP39 & 1: the back waits for work at the last
        // position published (it stored it in back_idle_at before it spun), so it waits for
        // exactly the words the front holds: it gets them after PublishAfterWait, not after
        // publish_every. Not while a Pairs or StateDelta block is open (publishing ends it).
        // Replaces EXP27 & 2, which published every entry while the back was idle.
        const u64 unpublished = write_position - published_local;
        if (unpublished >= publish_every ||
            (publish_waiting && unpublished >= PublishAfterWait && pairs_header == nullptr &&
             delta_header == nullptr &&
             back_idle_at.load(std::memory_order_relaxed) == published_local)) {
            Publish();
        }
    }
    void WaitForSpace(size_t words);
    void SelectEngine(Tegra::Engines::EngineInterface* engine) {
        // Before the engine is compared: the held-back work selects its own.
        RunPendingFlush();
        if (engine != front_engine) [[unlikely]] {
            SwitchEngine(engine);
        }
    }
    void SwitchEngine(Tegra::Engines::EngineInterface* engine);
    bool WaitFor(u64 target, DrainSite site);

    /// One step of a front wait for the back to consume `target`.
    void Backoff(u32 spins, u64 target);
    /// After the back stored `consumed`: wakes a front sleeping in Backoff.
    void NotifyFront() {
        if (front_waiting.load(std::memory_order_seq_cst)) {
            front_wake.fetch_add(1, std::memory_order_seq_cst);
            front_wake.notify_one();
        }
    }

    void BackLoop(std::stop_token stop_token);
    /// Runs the entry at `entry` (of the `words` published before the ring's end), returns its size.
    size_t Execute(const u64* entry, size_t words);

    Core::System* system{};
    bool active{};
    bool front_macros{};
    std::unique_ptr<u64[]> ring;

    // Front side.
    u64 write_position{};
    u64 last_writer_position{};
    u64 last_memory_writer_position{};
    u64 last_unranged_position{};
    u64 unranged_without_exp120{};
    EngineWriteNotes engine_write_notes{};
    static constexpr u32 PendingWriteWords = 8;
    struct PendingWrite {
        u64 begin;
        u64 end;
        u64 position;
        u32 words; ///< Known words from begin, or 0 (unknown contents)
        const char* what;
        std::array<u32, PendingWriteWords> data;
    };
    std::array<PendingWrite, 8192> pending_writes{};
    u64 pending_write_count{};
    /// Compute and copy writes with a known device range (NoteWriterRange), newest last.
    struct WriterRange {
        u64 position;
        u64 begin;
        u64 end;
        WriterKind kind; ///< For EXP13=2's wait histogram
    };
    // 8,192 / 4,096 (was 1,024 / 512): at the Castle Town pot room ~11,500 ranges are noted a
    // second (compute inline uploads), so a page read every few frames was always past the
    // old limit and every front DMA read of it drained. Both numbers only bound work: past
    // them the answer is "may write" without looking, so a larger limit is exact.
    std::array<WriterRange, 8192> writer_ranges{};
    static constexpr u64 MaxWriterScan = 4096;
    u64 writer_range_count{};
    u64 last_unbounded_writer_position{};
    std::array<u64, static_cast<size_t>(WriterKind::Count)> unbounded_writers{};
    std::array<u64, static_cast<size_t>(ReleaseNote::Count)> release_notes{};
    /// NoteWriterRange's translation: noted, or why not (a ReleaseNote past Ranged).
    ReleaseNote TryNoteWriterRange(const Tegra::MemoryManager& memory, GPUVAddr address,
                                   u64 size, WriterKind kind);
    /// NoteWriterRange's last range and the function position its translation holds for: the
    /// same range again (an inline upload's data words) only moves the newest range's position,
    /// unless a check may have kept what it found for it (below writer_range_sealed).
    mutable u64 writer_range_sealed{};
    const Tegra::MemoryManager* writer_range_memory{};
    GPUVAddr writer_range_address{};
    u64 writer_range_size{};
    u64 writer_range_function{~u64{0}};
    /// EXP122 page watch (gpu_split.cpp), front only.
    struct PageWatch;
    std::unique_ptr<PageWatch> page_watch;
    u32 watched_pages{};
    int page_watch_mode{};
    enum class WatchAnswer : u8 { Unwatched, NoWrite, Write };
    WatchAnswer WatchedWritesSince(u64 since, u64 begin, u64 size, u64 known,
                                   ListedWrites* found) const;
    void NoteWatchedWrite(u64 begin, u64 end, u64 position);
    void ClearPageWatch();
    /// Front, EXP10 dispatches (NoteDispatchWriter): the newest sent, the newest sent while the
    /// back did not record (EXP10 off), whether the back was last told to record, and how many.
    u64 last_dispatch_position{};
    u64 last_unrecorded_dispatch_position{};
    bool dispatch_records{};
    u64 dispatches_sent{};
    /// Dispatch writes the back recorded (begin, end; 0, ~0 for anything) with the position of
    /// the entry that ran them, newest last; written by the back, read by the front.
    struct DispatchWrite {
        std::atomic<u64> position;
        std::atomic<u64> begin;
        std::atomic<u64> end;
    };
    static constexpr u64 DispatchWriteSlots = 8192;
    static constexpr u64 MaxDispatchScan = 2048;
    std::array<DispatchWrite, DispatchWriteSlots> dispatch_writes{};
    alignas(64) std::atomic<u64> dispatch_write_count{};
    std::atomic<u64> dispatches_bounded{};
    std::array<std::atomic<u64>, static_cast<size_t>(DispatchUnbounded::Count)>
        dispatches_unbounded{};
    void AppendDispatchWrite(u64 begin, u64 end);
    u64 last_sync_position{};
    u64 last_memory_write_position{};
    static constexpr u32 DeferredWrite = ~0U;
    static constexpr u32 OtherEngineWrite = ~0U - 1;
    /// The 3D method behind last_memory_write_position, or one of the two values above.
    u32 last_memory_write_method{DeferredWrite};
    u64 previous_memory_write_position{};
    u32 previous_memory_write_method{DeferredWrite};
    /// The 3D memory-writing methods alone (NoteMethodWrite, RangeLastWrite), without the
    /// deferred work, the all-writing engines and the barriers last_memory_write_position also
    /// counts (QueuedGuestWriters).
    u64 last_method_write_position{};
    u64 previous_method_write_position{};
    u64 session{};
    /// EnableWriteStamps' tables: a slot keeps its newest stamp, the tag that made it, and the
    /// newest stamp of any other tag before it.
    struct StampSlot {
        u64 last;
        u64 last_tag;
        u64 before;
    };
    static constexpr size_t BlockStamps = size_t{1} << 16;
    static constexpr size_t PageStamps = size_t{1} << 14;
    std::unique_ptr<StampSlot[]> block_stamps;
    std::unique_ptr<StampSlot[]> page_stamps;
    bool write_stamps{};
    u64 write_stamp_tag{};
    u64 write_stamp_tags{};
    u64 write_stamp_floor{};
    u64 unknown_write_position{};
    void StampWrite(u64 begin, u64 end, bool gpu_address);
    [[nodiscard]] u64 StampOf(u64 begin, u64 end, bool gpu_address, u64 skip_tag) const;
    /// Writes with a known range (RangeLastWrite) the back may not have run yet, oldest first.
    struct RangedWrite {
        u64 position;
        GPUVAddr begin;
        GPUVAddr end;
        u32 method;
        size_t space;           ///< Address space of `data`; 0 when the words are not known
        std::vector<u32> data;  ///< The words written over [begin, end), when known
    };
    /// Ranged writes kept before the oldest counts for every read again (EDEN_NX2_RANGED_MAX).
    static size_t MaxRangedWrites();
    /// ranged_writes' storage, in send order as the deque it replaces: a ring of slots whose word
    /// vectors are cleared when a slot is reused, never freed (the deque destroyed one vector per
    /// upload and built a new one: ~35 ms/s of the front in vector and deque allocations,
    /// fable_answer_regression_2 4.3). Sized to MaxRangedWrites() (RangeLastWrite never holds
    /// more), grown like a deque past it. A slot's vector over 64 KiB is freed on reuse.
    class RangedRing {
    public:
        [[nodiscard]] bool empty() const {
            return count == 0;
        }
        [[nodiscard]] size_t size() const {
            return count;
        }
        /// Oldest first: 0 is the oldest, size() - 1 the newest.
        [[nodiscard]] RangedWrite& operator[](size_t index) {
            return slots[(head + index) & mask];
        }
        [[nodiscard]] const RangedWrite& operator[](size_t index) const {
            return slots[(head + index) & mask];
        }
        [[nodiscard]] RangedWrite& front() {
            return slots[head];
        }
        [[nodiscard]] RangedWrite& back() {
            return (*this)[count - 1];
        }
        [[nodiscard]] const RangedWrite& back() const {
            return (*this)[count - 1];
        }
        void pop_front() {
            head = (head + 1) & mask;
            --count;
        }
        /// As emplace_back(RangedWrite{position, begin, end, method, 0, {}}).
        RangedWrite& push_back(u64 position, GPUVAddr begin, GPUVAddr end, u32 method) {
            if (count == slots.size()) [[unlikely]] {
                Grow();
            }
            RangedWrite& write = slots[(head + count) & mask];
            ++count;
            write.position = position;
            write.begin = begin;
            write.end = end;
            write.method = method;
            write.space = 0;
            if (write.data.capacity() > MaxKeptWords) [[unlikely]] {
                std::vector<u32>{}.swap(write.data);
            } else {
                write.data.clear();
            }
            return write;
        }
        /// Empties the ring; the slots keep their vectors.
        void clear() {
            head = 0;
            count = 0;
        }

    private:
        static constexpr size_t MaxKeptWords = 16384;
        void Grow();
        std::vector<RangedWrite> slots;
        size_t mask{};
        size_t head{};
        size_t count{};
    };
    RangedRing ranged_writes;
    struct QueuedRelease {
        u64 position;
        u64 device;
        u32 payload;
    };
    std::array<QueuedRelease, 64> queued_releases{};
    /// Drops consumed ranged writes; returns the position a memory read of [begin, end) waits
    /// for and the method behind it.
    u64 MemoryWriterTarget(u64 done, GPUVAddr begin, GPUVAddr end, u32& target_method);
    u64 last_function_position{};
    SplitWriteTracker writes;
    /// NoteWrite's last translation: memory manager, GPU page, device page, the function
    /// position it was made at (a later function may remap).
    const Tegra::MemoryManager* note_memory{};
    GPUVAddr note_page{~GPUVAddr{0}};
    u64 note_device{};
    u64 note_function{};
    u64 published_local{};
    u64 consumed_cache{};
    /// Words between publishes: small right after a drain, while the back has nothing to do.
    size_t publish_every{PublishEvery};
    /// EXP39 & 1, read at each Publish: Commit publishes a few words to a back that waits.
    bool publish_waiting{};
    Tegra::Engines::EngineInterface* front_engine{};
    /// The open Pairs block's head (PairWrite), and its count.
    u64* pairs_header{};
    u32 pairs_count{};
    /// The open StateDelta block's head (DeltaWrite), and its count.
    u64* delta_header{};
    u32 delta_count{};
    /// Held-back front work (SetPendingFlush).
    PendingFlush pending_flush{};
    void* pending_flush_argument{};
    /// Instance folding (FoldInstance): instances counted and not sent yet, and who sends them.
    u32 fold_count{};
    void (*fold_flush)(void*, u32){};
    void* fold_context{};
    static void FlushFoldThunk(void* split) {
        static_cast<GpuSplit*>(split)->FlushFold();
    }
    std::array<u64, static_cast<size_t>(Batched::Count)> batched{};
    std::chrono::steady_clock::time_point batched_since{};
    u32 publishes{};
    void ReportBatched();

    // Back side.
    Tegra::Engines::EngineInterface* back_engine{};
    /// The position of the entry being run (dispatch writes record it).
    u64 back_entry_position{};
    /// EXP10: the front asked the back to record dispatch writes (NoteDispatchWriter).
    bool back_dispatch_records{};
    /// The dispatch being run: the rasterizer recorded its writes, it recorded an unbounded one.
    bool dispatch_confirmed{};
    bool dispatch_unbounded{};
    static inline thread_local bool recording_dispatch{};

    static GpuSplit instance;
    static inline thread_local bool on_back_thread{};
    /// The back is in its wait for work (the idle hook's idle call, not a RequestIdleHook mid-batch).
    static inline thread_local bool in_idle_wait{};
    void (*idle_hook)(void*){};
    void* idle_argument{};
    FenceQuietCheck fence_quiet_check{};
    void* fence_quiet_argument{};
    /// Back: runs fence_quiet_check at `position` (everything published executed).
    void CheckFencesQuiet(u64 position);
    /// Back: forgets the last check (the idle hook submits, which may change the caches).
    void ForgetFencesQuiet() {
        fence_quiet_position.store(~u64{0}, std::memory_order_relaxed);
        fence_busy_position.store(~u64{0}, std::memory_order_relaxed);
    }
    /// Written by the back before `consumed`: the position of its last check that found nothing
    /// to release (with the epoch it saw), or that found work; ~0 for none.
    alignas(128) std::atomic<u64> fence_quiet_position{~u64{0}};
    std::atomic<u64> fence_busy_position{~u64{0}};
    std::atomic<u64> fence_quiet_epoch{};
    alignas(64) std::atomic<bool> hook_requested{};
    std::atomic<bool> back_running{};

    // 128 bytes: x86 spatial prefetchers pull 64-byte lines in pairs. back_waiting and wake are
    // always touched together by both threads, so they share a block.
    alignas(128) std::atomic<u64> published{};
    alignas(128) std::atomic<u64> consumed{};
    /// A position the front waits for (WaitFor): the back stores `consumed` once past it.
    alignas(128) std::atomic<u64> wait_target{~u64{0}};
    /// The back's position when it last ran out of work, stored before it waits for more: while
    /// it equals published_local, the back waits for the words the front holds (Commit, EXP39 &
    /// 1). The back writes it once per wait, the front reads it at every Commit. ~0: none since
    /// the split started (positions start over at 0).
    alignas(128) std::atomic<u64> back_idle_at{~u64{0}};
    alignas(128) std::atomic<bool> back_waiting{};
    std::atomic<u32> wake{};
    /// The front sleeps in Backoff (ported: instead of yielding).
    alignas(128) std::atomic<bool> front_waiting{};
    std::atomic<u32> front_wake{};
    std::jthread back_thread;
};

} // namespace VideoCommon
