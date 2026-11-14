// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// The memory stage's thread, GPUMem (fable_memstage_step6_spec.md, piece 6a: "the thread, by
// hand-off"; piece 6b: "ring-fed, level mode").
//
// The thread exists while the GPU thread split runs (started and stopped with the back,
// GpuSplit::MaybeStart and Stop) and sleeps until the back hands it work. FRITCHII_MEMSTAGE_THREAD
// (live, read at a frame end by MemStage::TickFrame) picks the mode:
//   0: nothing is handed off; the fills run on the back (2b).
//   1: hand-off. The back's two big fills, the draw's (MemStage::FillDraw) and the dispatch's
//      (FillDispatch), and the buffer recycler's idle destroys, are run here: the back posts the
//      function (Call), GPUMem runs it, the back waits. Same order, same timing, another thread.
//   2: ring-fed, level mode (piece 6b, below).
//
// One thread at a time. While a Call runs, the back only waits; outside a Call GPUMem only
// sleeps. So the buffer cache, its allocator and its staging pool have one user at any moment:
// GPUMem inside a Call, the back otherwise. That is "lending by construction" (spec 6.1).
//
// The sync-serving loop (spec 4.3). A fill can reach a sync: SeamClock::Finish, Wait, WaitTick,
// SubmitForHostRead, the multi-range bind, a front-side wait. They need the scheduler, which is
// the back's. On GPUMem they go through Serve: the request is posted to the back, which runs it
// inside its wait (the back is then the cache's user, GPUMem waits), answers, and goes on
// waiting for the Call's end. A request is always posted before GPUMem waits, so there is no
// deadlock: the back is already waiting in the Call.
//
// Waits on both sides spin about 2,000 pauses, then sleep on a futex word (std::atomic::wait).
// The hand-over goes through seq_cst stores and loads of four counters (calls, done, requests,
// answered), which also order everything the two threads write around them.
//
// The cache user (spec 5.1). With the mode at 1, a thread-local flag says who the cache's user is:
// GPUMem while it runs a Call (not while it waits for a served request), the back while it is not
// waiting in a Call (and while it serves a request). Counted when the caller is not the user
// (these must read 0):
//   allocator: any call on the buffer cache's VMA allocator (MemoryAllocator calls, buffer and
//              commit releases; vulkan_memory_allocator.cpp, vulkan_wrapper.cpp);
//   staging:   StagingBufferPool::Request on the buffer cache's pool;
//   push:      any MemStage::Push*;
//   budget:    vmaGetHeapBudgets on an allocator from a thread that is not its user (5.3).
// And, with the mode at 1 (spec 5.4, 5.5, 5.7):
//   stage scheduler:   a scheduler or descriptor queue use on GPUMem (SeamHook);
//   push nowhere:      an entry GPUMem would execute at once (outside a fill bracket): it is
//                      served on the back instead, and counted;
//   push pending:      an entry executed at once on the arena's owner while entries are pending;
//   player on stage:   MemPlayer reached on GPUMem (Exec, AddDescriptors, the list functions, the
//                      runtime's Player() and Seam());
//   front wait:        BufferCacheRuntime::WaitSubmittedTick or WaitForHostRead reached on GPUMem
//                      (the front's waits; served on the back, counted).
// The first 16 of each are logged with the place. "Memory stage link" logs the counters and the
// hand-off rates every 10 s while the mode is 1.
//
// Profile (EDEN_NX2_GPU_PROFILE): GPUMem reports as "GPU memory thread"; its "memory fill" zone
// is the fills it ran. On the back, "memory hand-off" is the time it waited in Calls (served
// requests inside it have their own zones); on GPUMem, "memory sync wait" is its waits for served
// requests.
//
// ---------------------------------------------------------------------------------------------
// FRITCHII_MEMSTAGE_THREAD = 2: the ring-fed stage, level mode (spec 6.3, plan 6.2; AHEAD = 0).
//
// The cursors (GpuSplit, friend). Ring A, the front's ring, gets a second reader: GPUMem, with its
// own position. It publishes `staged`, how far it has read; the back executes only entries below
// `staged` (its limit, where it used `published`). A level run is published as [lent_from,
// lent_until): the back runs those entries "lent", with today's code (its own fills, the cache in
// its hands); every other entry it runs "served" (MemoryStageLink::BackServed): the stage did the
// entry's cache work. Ring B (vk_mem_stage.cpp) carries the stage's draw packets to the back.
//
// The stage's loop (RingLoop, spec 4.1), on GPUMem:
//   - it waits for the front (`published`), then, in level mode, until the back is parked: it has
//     executed everything published to it (`consumed` >= `staged`) and is not in its idle hook
//     (a seq_cst handshake on stage_busy / back_in_hook). From there to its next publish the stage
//     is the only thread working: the back waits for it. So nothing the stage does (fills, the
//     cache, constant buffer words into game memory, its own register copy) ever runs beside the
//     back. That is the whole exclusion rule of level mode;
//   - it runs 5c's code on each entry (MemStageShadow::StageProcess: classify, apply, follow; its
//     copy of the registers is the one the fills read). Draws it serves are filled on the spot
//     (the backend's fill_draw: the facts table, the copy as CacheInputs, the buffer half of the
//     ranges its constant buffer words collected, then the packet into ring B);
//   - after an entry that sent a packet it publishes `staged` past it (the back then plays it); it
//     also publishes every 64 entries and before every wait for the front;
//   - a hard entry starts a run: Lend hands the cache to the back (the stage's uncollected ranges
//     go with it, applied by the back first; the cache's flags 15-48 go to the back's engines, plan
//     4.2), publishes [lent_from, lent_until) and `staged`, then extends the run over the entries
//     that follow and need nothing from the stage (closures it cannot follow, other engines),
//     following them on its copy, then waits for the back to reach the run's end (consumed >=
//     lent_until). At the return it takes the flags back, execute_on or the draw state where 5c
//     says so, and a full copy of whatever an unfollowed entry left stale.
// The back's loop (gpu_split.cpp): its limit is `staged`; BackEntry sets "lent" or "served" per
// entry; at a run's first entry it plays whole the packets of entries before the run (must be 0,
// "packet left unplayed") and applies the stage's ranges; after the run's last entry it consumes
// its method sink and calls FlushCaching (plan 5.2) before it stores `consumed`. While it waits
// for the stage it serves the stage's sync requests (Serve, as in 6a).
//
// The back in served mode never enters the cache. What it would do there goes to GPUMem as a
// served call (ServeOnStage: 6a's Call the other way round, run from the stage's waits): the
// buffer half of a read flush (FlushRegion, FlushRegionForRead) and of the gathers it starts, a
// MustFlushRegion question. Counted by cause. What is left is counted as "back in the cache while
// not lent" (BufferCache entry points), "InnerInvalidation with ranges while not lent", "a packet
// asked for at a draw the stage sent none for" and "a packet left unplayed at a run's start": all
// must read 0.
//
// Deviations (the report has the reasons): the back serves the stage's sync requests where it is
// parked, at the batch's start, instead of at the entry (4.3's "publish before you wait"), so
// nothing ever runs beside a fill; the stage's collected ranges at a lend are applied by the back
// at the run's first entry, not by the stage before it.
//
// ---------------------------------------------------------------------------------------------
// Step 7: the window, FRITCHII_MEMSTAGE_AHEAD (live, read at a frame end, a level run, by
// MemStage::TickThread; 0 is level mode above, 65,536 and more unlimited).
//
// With AHEAD = N the stage no longer waits for the back to park before each entry. It waits only
//   - while N packets are in ring B that the back has not taken ("the window"), and while ring B
//     has no room for another ("room"); both after publishing `staged` up to the entry, so the
//     back can take every one of them (spec 1.4);
//   - at a level run, as before: it lends, then waits for the back to get through the run, so
//     every run still runs on the back with the cache as the back's position has it, and the
//     stage is level after every run (at least once a frame: the frame end is one);
//   - level inside the window (StageLevel: publish up to the entry, wait until the back has run
//     everything before it), where running ahead would change an answer the back gets:
//       full copy:     the entry takes a full copy of the back's engine (a new or stale copy);
//       writing draw:  a draw whose fill marks buffers GPU-written (written storage buffers,
//                      transform feedback): the back's served calls (a read flush's download, a
//                      gather's claims, MustFlushRegion) read those marks, so no packet ahead of
//                      the back may carry new ones;
//       claims:        the collected constant buffer ranges of the next fill cover GPU-written
//                      buffer data (applying them claims it: the same marks).
// So a served call the back makes while packets are ahead is answered by the cache with every
// GPU-written mark it has at the back's position; what the packets ahead changed (uploads, new
// bindings, the stream) does not enter those answers. Counted ("served calls while packets were
// ahead"), not a failure. One change of theirs would: a buffer a fill ahead joined into a new one
// (its old bytes reach the new buffer in that fill's packet), under GPU-written data a served
// download reads. GPUMem keeps the joins of its fills with their entries (NoteBufferJoin) and the
// served calls check them: "a served download over GPU-written data in a buffer a packet ahead of
// the back joined", must be 0.
//
// What the threads still share, and how it is ordered with the window open:
//   - the lend's flag hand-over (plan 4.2) is kept by the stage at the lend and set by the back
//     at the run's first entry, where both are at the same position (StageLendDeferred,
//     BackTakeLend); the ranges a lend hands over already worked that way;
//   - a sync in the stage's own fill (Serve) is recorded where the back is: first the back runs
//     up to the published position (StageSyncCatchUp), so every packet the stage sent before is
//     recorded before the sync's entries and a wait's number is played. If the back cannot (it
//     waits in a served call, or for the buffer cache's lock, which the stage holds in its fill),
//     the sync goes at once and is counted: "a sync of GPUMem's fill sent to the back before it
//     played the packets ahead of it", must be 0 (syncs in served fills: 0/s at the market), and
//     the window is refused for the rest of the session from the next frame end. A join in the
//     stage's fill over GPU-written data catches up the same way (StageJoinCatchUpSlow);
//   - constant buffer words onto GPU-written data: the entry turns hard before the stage writes
//     them (StageWriteWords), so the back writes them in a level run, in stream order: no served
//     download comes in between, and no claim of theirs is recorded before the entry's own
//     earlier packets;
//   - the buffer cache's lock: a back that waits for it while the stage holds it serves the
//     stage's requests from the wait (SpinRecursiveMutex's contended hook; not a request it is
//     serving already), so a sync in the stage's fill never waits on a back that waits on the
//     lock; each such wait is counted with its caller's frames. The other way round, a stage
//     that waits for the lock the back holds (EXP118's delta draws hold it through Configure,
//     where a read flush is a served call) serves the back's call from the wait, at its clean
//     lock sites only (CleanPoint, StageContendedHook);
//   - the back's calls are answered between entries too (StageServeCalls at the loop top);
//   - the ring's start and end clear the stage's unplayed mark (and the end plays the stage's
//     pending entries first), so a fill that threw cannot hold the ticks open;
//   - EDEN_NX2_RUNAHEAD on: the window runs 0 (the back applies its chunks before each entry);
//   - packet ticks: each side writes its own unplayed mark (vk_mem_stage.cpp, "Packet ticks with
//     the stage ahead"), so every buffer, staging and stream region a packet in flight uses
//     outlives its play.
// "Memory stage window" logs every 10 s while the window is open: packets in ring B at each push
// (average, maximum, and a histogram), the back's waits for the stage, the stage's waits for the
// window, for room and to come level (by reason, and at constant buffer writes), served calls
// while packets were ahead, sync and join catch-ups, lends handed over at the run's start, the
// lock waits both ways, served downloads past the join ring's reach, idle hook runs with the
// stage busy, and the stage's idle work.
//
// "Memory stage ring" logs every 10 s (section 8.3): entries/s by class, runs/s with entries and
// microseconds per run (GPUMem from the lend to the return, the back from the lend seen to its
// next packet), packets/s and bytes, the back's waits for the stage, the stage's waits by reason,
// its busy time and fill time, full copies, flag hand-overs, served calls by cause, level draws by
// kind, the verify hashes, and the must-be-0 counters. 5c's lines (runs by first hard kind, full
// copies by cause, ...) come from the stage's own copy of the shadow's state every 10 s too.

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

#include <fmt/format.h>

#include "common/dkb_cpu_trace.h"
#include "common/logging.h"
#include "common/nx2_timeline.h"
#include "common/spin_recursive_mutex.h"
#include "common/thread.h"
#include "core/core.h"
#include "video_core/gpu_split.h"
#include "video_core/gpu_wait_site.h"
#include "video_core/memory_stage_link.h"
#include "video_core/memstage_shadow.h"
#include "video_core/nx2_profile.h"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // RtlCaptureStackBackTrace (the lock waits' call sites)
#endif

namespace VideoCommon {

/// Friend of GpuSplit: the ring's cursors.
struct MemoryStageRingAccess {
    static std::atomic<u64>& Published() {
        return GpuSplit::Get().published;
    }
    static std::atomic<u64>& Consumed() {
        return GpuSplit::Get().consumed;
    }
    static std::atomic<u64>& Staged() {
        return GpuSplit::Get().staged;
    }
    static std::atomic<u64>& LentFrom() {
        return GpuSplit::Get().lent_from;
    }
    static std::atomic<u64>& LentUntil() {
        return GpuSplit::Get().lent_until;
    }
    static std::atomic<u64>& IdleAt() {
        return GpuSplit::Get().back_idle_at;
    }
    static void WakeBack() {
        GpuSplit& split = GpuSplit::Get();
        if (split.back_waiting.load(std::memory_order_seq_cst)) {
            split.wake.fetch_add(1, std::memory_order_seq_cst);
            split.wake.notify_one();
        }
    }
};

namespace {

using Ring = MemoryStageRingAccess;

/// About 2,000 pauses (spec 6.1) before a wait sleeps.
constexpr u32 LinkSpinPauses = 2048;
/// With the window open, how long the stage's wait for the front spins before it sleeps.
constexpr u64 InputSpinNs = 200'000;
/// Entries the stage reads before it publishes them anyway (spec 4.1).
constexpr u64 PublishEvery = 64;

void LinkPause() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#else
    std::this_thread::yield();
#endif
}

u64 LinkNowNs() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

/// Stats counters: a plain load and store (one writer each, or a lost count is fine).
void LinkBump(std::atomic<u64>& counter, u64 amount = 1) noexcept {
    counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}

constexpr std::array<const char*, static_cast<size_t>(MemoryStageLink::Check::Count)>
    LINK_CHECK_NAMES{
    "buffer allocator call by a thread that is not the cache's user",
    "buffer staging request by a thread that is not the cache's user",
    "memory seam push by a thread that is not the cache's user",
    "VMA heap budget read by a thread that is not the allocator's user",
    "scheduler or descriptor queue use on GPUMem",
    "entry executed at once on GPUMem (push with nowhere to go)",
    "entry executed at once while entries are pending",
    "memory player reached on GPUMem",
    "front-side wait reached on GPUMem",
    "fill handed off without the back holding the buffer cache's lock",
    "the back in the buffer cache while not lent",
    "InnerInvalidation with ranges on the back while not lent",
    "a packet asked for at a draw the stage sent none for",
    "a packet left unplayed at a run's start",
    "ring B full",
    "draws issued by the back and the stage differ at a run's end (compare a)",
    "a packet played whole for a draw the back did not prepare",
    "an exception out of GPUMem's ring loop (the ring ended)",
    "a packet whose entry is past the back's draw (dropped unplayed, the back filled)",
    "a served draw whose texture handles the back read from game memory",
    "a sync of GPUMem's fill sent to the back before it played the packets ahead of it",
    "a served download over GPU-written data in a buffer a packet ahead of the back joined",
    "an image registered over a constant buffer write GPUMem made for a later entry",
    "a served download over a constant buffer write GPUMem made for a later entry",
    "facts key collisions (one program key for two programs' registers: told apart, a miss)",
    "a served list draw the back would read back (the stage and the back decided differently)",
    "a served list draw whose packet has no list words buffer (the back asked the cache)",
    "a list draw packet at a back draw that asked for no list arguments (the draws differ)",
};
static_assert(LINK_CHECK_NAMES.back() != nullptr, "a name for every check");
constexpr std::array<const char*, static_cast<size_t>(MemoryStageLink::ListDraw::Count)>
    LIST_DRAW_NAMES{"served", "level with the window closed or the switch off",
                    "level read back on the CPU", "level as the fill refused it",
                    "level for a frame dump or trace"};
static_assert(LIST_DRAW_NAMES.back() != nullptr, "a name for every list draw outcome");
constexpr std::array<const char*, static_cast<size_t>(MemoryStageLink::LevelReason::Count)>
    LEVEL_REASON_NAMES{"full copy", "writing draw", "ranges over GPU-written data"};
constexpr std::array<const char*, static_cast<size_t>(MemoryStageLink::HandOff::Count)>
    LINK_HANDOFF_NAMES{"draw fills", "dispatch fills", "idle destroys", "served for the back"};
constexpr std::array<const char*, static_cast<size_t>(MemoryStageLink::Served::Count)>
    SERVED_NAMES{"FlushRegion", "FlushRegionForRead", "gather", "single invalidation",
                 "InnerInvalidation", "MustFlushRegion", "fallback fill"};
constexpr std::array<const char*, 8> LEVEL_DRAW_NAMES{
    "indirect", "GPU count", "texel buffers", "diagnostic or forced",
    "facts miss",       "no packet room", "late (after a hard method)", "other"};

/// The calling code's frames as module+offset (for a counter "by call site").
std::string CallerFrames() {
    std::string text;
#ifdef _WIN32
    std::array<void*, 12> frames{};
    const USHORT count = RtlCaptureStackBackTrace(2, static_cast<ULONG>(frames.size()),
                                                  frames.data(), nullptr);
    for (USHORT i = 0; i < count; ++i) {
        HMODULE module{};
        char name[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               static_cast<LPCSTR>(frames[i]), &module)) {
            GetModuleFileNameA(module, name, sizeof(name));
        }
        const char* const base_name = std::strrchr(name, '\\');
        text += fmt::format(" {}+{:#x}", base_name != nullptr ? base_name + 1 : name,
                            reinterpret_cast<uintptr_t>(frames[i]) -
                                reinterpret_cast<uintptr_t>(module));
    }
#else
    text = fmt::format(" {}", __builtin_return_address(0));
#endif
    return text;
}

const char* LinkThreadKind() noexcept {
    if (MemoryStageLink::OnStageThread()) {
        return "GPUMem";
    }
    return GpuSplit::OnBackThread() ? "GPUBack" : "another thread";
}

} // Anonymous namespace

MemoryStageLink MemoryStageLink::instance;

void MemoryStageLink::Start(Core::System& system_) {
    if (thread.joinable()) {
        return;
    }
    // The cache's own allocator (step 3) is what lets the cache run on a thread of its own: with
    // one allocator for everything two threads would be inside it (spec 5.1).
    const char* const split = std::getenv("FRITCHII_MEMSTAGE_SPLIT_ALLOC");
    if (split != nullptr && split[0] == '0') {
        LOG_WARNING(HW_GPU, "Memory stage thread (GPUMem) not started: it needs the buffer cache's "
                            "own allocator, and FRITCHII_MEMSTAGE_SPLIT_ALLOC=0 turns it off");
        return;
    }
    system = &system_;
    mode = 0;
    refused_logged = false;
    calls.store(0, std::memory_order_relaxed);
    done.store(0, std::memory_order_relaxed);
    requests.store(0, std::memory_order_relaxed);
    answered.store(0, std::memory_order_relaxed);
    call_sent = 0;
    calls_seen = 0;
    request_sent = 0;
    answered_local = 0;
    ring_wanted.store(false, std::memory_order_relaxed);
    ring_start.store(~u64{0}, std::memory_order_relaxed);
    ring_running.store(false, std::memory_order_relaxed);
    ring_failed.store(false, std::memory_order_relaxed);
    g_ring_active.store(false, std::memory_order_relaxed);
    stage_busy.store(false, std::memory_order_relaxed);
    back_in_hook.store(false, std::memory_order_relaxed);
    if (!stage_writes) {
        stage_writes = std::make_unique<StageWrite[]>(StageWriteSlots);
    }
    // A refusal of the window is for one game session (re-review N3).
    ahead_refused.store(false, std::memory_order_relaxed);
    ahead_blocked_logged = nullptr;
    thread = std::jthread([this](std::stop_token stop) { StageMain(stop); });
    LOG_INFO(HW_GPU, "Memory stage thread (GPUMem) started, sleeping until "
                     "FRITCHII_MEMSTAGE_THREAD=1 or 2 gives it work");
}

void MemoryStageLink::Stop() {
    if (!thread.joinable()) {
        return;
    }
    // The back has stopped: no Call is in flight.
    ring_wanted.store(false, std::memory_order_seq_cst);
    thread.request_stop();
    WakeStage();
    thread.join();
    thread = {};
    mode = 0;
    g_checking.store(false, std::memory_order_relaxed);
    g_ring_active.store(false, std::memory_order_relaxed);
}

u32 MemoryStageLink::SetMode(u32 wanted) {
    u32 next = (std::min)(wanted, 2U);
    if (next == 2 && ring_failed.load(std::memory_order_relaxed)) {
        next = 1; // GPUMem failed in ring mode this session (RingFailed)
    }
    if (next != 0 && !thread.joinable()) {
        if (!refused_logged) {
            refused_logged = true;
            LOG_WARNING(HW_GPU, "FRITCHII_MEMSTAGE_THREAD={}: the memory stage thread does not run "
                                "(no GPU thread split, or FRITCHII_MEMSTAGE_SPLIT_ALLOC=0); fills "
                                "stay on the back",
                        wanted);
        }
        next = 0;
    }
    if (next != mode) {
        mode = next;
        g_checking.store(next != 0, std::memory_order_relaxed);
        // The ring starts and ends at the back's next loop top (BackRingTop) and at the stage's
        // next parked point (RingLoop): both threads are level there.
        ring_wanted.store(next == 2, std::memory_order_seq_cst);
        WakeStage();
        LOG_INFO(HW_GPU, "Memory stage thread (FRITCHII_MEMSTAGE_THREAD={}): {}", next,
                 next == 2   ? "the stage reads the ring itself and fills the draws it serves "
                               "(level mode)"
                 : next == 1 ? "the draw and dispatch fills run on GPUMem by hand-off"
                             : "the fills run on the back");
        last_log_ns = 0;
    }
    return mode;
}

void MemoryStageLink::WakeStage() {
    stage_word.fetch_add(1, std::memory_order_seq_cst);
    if (stage_sleeping.load(std::memory_order_seq_cst)) {
        stage_word.notify_one();
    }
}

void MemoryStageLink::WakeBack() {
    back_word.fetch_add(1, std::memory_order_seq_cst);
    if (back_sleeping.load(std::memory_order_seq_cst)) {
        back_word.notify_one();
    }
    // THREAD = 2: the back waits for the stage on the split's own word.
    if (g_ring_active.load(std::memory_order_relaxed)) {
        Ring::WakeBack();
    }
}

void MemoryStageLink::Call(HandOff kind, Function function, void* argument) {
    // Inline: hand-off off; on GPUMem itself; while the back serves one of GPUMem's requests
    // (GPUMem is waiting then, so this is still one thread at a time). THREAD = 2 hands off only
    // served calls, and only while the stage reads the ring (it runs them from its waits).
    const bool posts = kind == HandOff::Served
                           ? g_ring_active.load(std::memory_order_acquire) &&
                                 ring_running.load(std::memory_order_acquire)
                           : mode == 1;
    if (!posts || t_on_stage || t_back_serving || !GpuSplit::OnBackThread()) {
        function(argument);
        return;
    }
    const u64 start = LinkNowNs();
    call_function = function;
    call_argument = argument;
    call_recording = GpuSplit::RecordingDispatch();
    call_served = kind == HandOff::Served;
    call_error = nullptr;
    const u32 seq = ++call_sent;
    const bool was_lent = t_back_lent;
    t_back_lent = true;
    calls.store(seq, std::memory_order_seq_cst);
    WakeStage();
    WakeLockWaiters(); // GPUMem may sleep on the lock this thread holds (EXP118's delta draws)
    {
        const Nx2Profile::Scope zone{Nx2Profile::Zone::MemoryHandOff};
        const BackWaitScope site{"memory stage hand-off (a fill on GPUMem)"};
        WaitForStage(seq);
    }
    t_back_lent = was_lent;
    LinkBump(counters.calls[static_cast<size_t>(kind)]);
    LinkBump(counters.call_ns, LinkNowNs() - start);
    if (call_error) {
        std::rethrow_exception(std::exchange(call_error, nullptr));
    }
}

void MemoryStageLink::WaitForStage(u32 seq) {
    for (u32 spin = 0;;) {
        if (done.load(std::memory_order_seq_cst) == seq) {
            return;
        }
        if (requests.load(std::memory_order_seq_cst) != answered_local) {
            ServeOne();
            spin = 0;
            continue;
        }
        if (spin < LinkSpinPauses) {
            ++spin;
            LinkPause();
            continue;
        }
        back_sleeping.store(true, std::memory_order_seq_cst);
        const u32 word = back_word.load(std::memory_order_seq_cst);
        if (done.load(std::memory_order_seq_cst) != seq &&
            requests.load(std::memory_order_seq_cst) == answered_local) {
            back_word.wait(word, std::memory_order_seq_cst);
            LinkBump(counters.back_sleeps);
        }
        back_sleeping.store(false, std::memory_order_relaxed);
    }
}

void MemoryStageLink::ServeOne() {
    const u32 seq = requests.load(std::memory_order_seq_cst);
    const u64 start = LinkNowNs();
    const bool was_lent = t_back_lent;
    t_back_lent = false;
    t_back_serving = true;
    try {
        request_function(request_argument);
    } catch (...) {
        request_error = std::current_exception();
    }
    t_back_serving = false;
    t_back_lent = was_lent;
    LinkBump(counters.served);
    LinkBump(counters.served_ns, LinkNowNs() - start);
    answered_local = seq;
    answered.store(seq, std::memory_order_seq_cst);
    WakeStage();
}

void MemoryStageLink::Serve(Function function, void* argument) {
    if (!t_in_call) {
        function(argument);
        return;
    }
    if (ring_window != 0 && t_on_stage && !t_serving_back) [[unlikely]] {
        // Step 7: GPUMem's own fill, the window open. The back records this sync where it is;
        // the packets GPUMem sent before must be played first (StageSyncCatchUp).
        StageSyncCatchUp();
    }
    request_function = function;
    request_argument = argument;
    request_error = nullptr;
    const u32 seq = ++request_sent;
    // Publish before waiting (spec 4.3): the back is in the Call's wait and serves it there. At
    // THREAD = 2 the back is parked at `staged` and serves it in its wait for the stage.
    const bool was_user = t_stage_user;
    t_stage_user = false;
    requests.store(seq, std::memory_order_seq_cst);
    WakeBack();
    WakeLockWaiters(); // the back may sleep on the lock GPUMem holds in its fill
    {
        const Nx2Profile::Scope zone{Nx2Profile::Zone::MemorySyncWait};
        const StageWaitScope site{"a sync the back serves"};
        for (u32 spin = 0; answered.load(std::memory_order_seq_cst) != seq;) {
            if (g_ring_active.load(std::memory_order_relaxed) &&
                !GpuSplit::Get().BackRunning()) [[unlikely]] {
                // THREAD = 2 at shutdown: the back left its loop; nobody else answers.
                request_sent = seq - 1;
                t_stage_user = was_user;
                function(argument);
                return;
            }
            if (spin < LinkSpinPauses) {
                ++spin;
                LinkPause();
                continue;
            }
            stage_sleeping.store(true, std::memory_order_seq_cst);
            const u32 word = stage_word.load(std::memory_order_seq_cst);
            if (answered.load(std::memory_order_seq_cst) != seq &&
                (!g_ring_active.load(std::memory_order_relaxed) ||
                 GpuSplit::Get().BackRunning())) {
                stage_word.wait(word, std::memory_order_seq_cst);
                LinkBump(counters.stage_sleeps);
            }
            stage_sleeping.store(false, std::memory_order_relaxed);
        }
    }
    t_stage_user = was_user;
    if (request_error) {
        std::rethrow_exception(std::exchange(request_error, nullptr));
    }
}

void MemoryStageLink::StageMain(std::stop_token stop) {
    const Common::DkbCpuTrace::Registration cpu_trace{Common::DkbCpuTrace::Role::GpuMem};
    Common::SetCurrentThreadName("GPUMem");
    t_on_stage = true;
    stage_thread.store(std::this_thread::get_id(), std::memory_order_relaxed);
    // As the back (spec 5.8); not pinned in 6a and 6b.
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();
    system->RegisterHostThread();
    // Start reset `calls` to 0 before this thread ran: a Call posted before here is seen.
    calls_seen = 0;
    for (;;) {
        u32 now = calls_seen;
        u64 start = ~u64{0};
        for (u32 spin = 0;;) {
            now = calls.load(std::memory_order_seq_cst);
            start = ring_start.load(std::memory_order_acquire);
            if (now != calls_seen || start != ~u64{0} || stop.stop_requested()) {
                break;
            }
            if (spin < LinkSpinPauses) {
                ++spin;
                LinkPause();
                continue;
            }
            stage_sleeping.store(true, std::memory_order_seq_cst);
            const u32 word = stage_word.load(std::memory_order_seq_cst);
            if (calls.load(std::memory_order_seq_cst) == calls_seen &&
                ring_start.load(std::memory_order_seq_cst) == ~u64{0} &&
                !stop.stop_requested()) {
                stage_word.wait(word, std::memory_order_seq_cst);
                LinkBump(counters.stage_sleeps);
            }
            stage_sleeping.store(false, std::memory_order_relaxed);
        }
        if (stop.stop_requested()) {
            break;
        }
        if (start != ~u64{0}) {
            ring_start.store(~u64{0}, std::memory_order_relaxed);
            RingLoop(stop, start);
            continue;
        }
        calls_seen = now;
        RunCall(now);
        Nx2Profile::Report("GPU memory thread");
    }
}

void MemoryStageLink::RunCall(u32 seq) {
    const u64 start = LinkNowNs();
    const bool was_in_call = t_in_call;
    const bool was_user = t_stage_user;
    const bool was_serving = t_serving_back;
    const bool was_clean = t_stage_clean;
    t_in_call = true;
    t_stage_user = true;
    t_serving_back = call_served;
    t_stage_clean = false;
    // EXP10: the back records the writes of the dispatches it runs; this fill is one of them.
    GpuSplit::AdoptRecordingDispatch(call_recording);
    try {
        call_function(call_argument);
    } catch (...) {
        call_error = std::current_exception();
    }
    GpuSplit::AdoptRecordingDispatch(false);
    t_stage_clean = was_clean;
    t_serving_back = was_serving;
    t_stage_user = was_user;
    t_in_call = was_in_call;
    LinkBump(counters.stage_ns, LinkNowNs() - start);
    done.store(seq, std::memory_order_seq_cst);
    WakeBack();
}

bool MemoryStageLink::IsCacheUserSlow() noexcept {
    if (t_on_stage) {
        return t_stage_user;
    }
    if (!GpuSplit::OnBackThread()) {
        return false;
    }
    if (t_back_serving) {
        return true;
    }
    if (g_ring_active.load(std::memory_order_relaxed)) {
        // THREAD = 2: the back holds the cache only in a level run.
        return t_back_run;
    }
    return !t_back_lent;
}

void MemoryStageLink::CheckCacheUserSlow(Check check) noexcept {
    if (!IsCacheUserSlow()) {
        Note(check, nullptr);
    }
}

void MemoryStageLink::Note(Check check, const char* where) noexcept {
    const size_t index = static_cast<size_t>(check);
    instance.counters.checks[index].fetch_add(1, std::memory_order_relaxed);
    static std::array<std::atomic<u32>, static_cast<size_t>(Check::Count)> logged{};
    if (logged[index].fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_ERROR(HW_GPU, "Memory stage link (must be 0): {} on {}{}{}", LINK_CHECK_NAMES[index],
                  LinkThreadKind(), where != nullptr ? ": " : "", where != nullptr ? where : "");
    }
}

// ---------------------------------------------------------------------------------------------
// THREAD = 2: the back's side

void MemoryStageLink::ServeOnStage(Served cause, Function function, void* argument) {
    if (!BackServed()) {
        function(argument);
        return;
    }
    LinkBump(counters.served_calls[static_cast<size_t>(cause)]);
    if (ring_backend.ring_packets != nullptr &&
        ring_backend.ring_packets(ring_backend.context) != 0) {
        // Step 7: answered by the cache as of GPUMem's position, later than the back's (the
        // switch's documentation has why that answer is the back's: StageLevel's rules).
        LinkBump(counters.served_ahead);
    }
    Call(HandOff::Served, function, argument);
}

bool MemoryStageLink::BackContendedHook(std::thread::id holder, bool first) {
    // The back waits for a lock (the buffer cache's): if GPUMem holds it, it may be waiting for
    // the back in a sync of its fill (Serve), so the back serves it here and does not sleep.
    // StageSyncCatchUp sees the count and stops waiting for the back to play on. Another holder:
    // false, the wait sleeps as usual.
    MemoryStageLink& link = instance;
    if (holder != link.stage_thread.load(std::memory_order_relaxed)) {
        return false;
    }
    if (first) {
        // Counted by call site (the caller's frames, first 16 logged): each site is read once
        // and declared safe for a sync at that line, or becomes a served call.
        LinkBump(link.counters.back_lock_waits);
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_WARNING(HW_GPU,
                        "Memory stage window: the back waited for the buffer cache's lock, held "
                        "by GPUMem, while served; from:{}",
                        CallerFrames());
        }
    }
    link.back_contended.fetch_add(1, std::memory_order_seq_cst);
    if (link.stage_sleeping.load(std::memory_order_seq_cst)) {
        link.WakeStage();
    }
    // Not while the back serves GPUMem already: the request it serves is still "pending" (its
    // answer is stored after the function returns) and would run again, inside itself.
    if (!t_back_serving && link.BackRequestPending()) {
        link.BackServeRequests();
    }
    return true;
}

bool MemoryStageLink::StageContendedHook(std::thread::id holder, bool first) {
    // GPUMem waits for the buffer cache's lock while the back holds it. The back may hold it
    // across a call it posted for GPUMem (EXP118's delta draws hold it through Configure, and a
    // read flush there is a served call): GPUMem serves the call from here, at the lock sites
    // that are clean points (CleanPoint: no fill open, between entries' cache work), and does
    // not sleep while the back holds the lock there. Anything else: false, it sleeps as usual.
    MemoryStageLink& link = instance;
    static_cast<void>(first);
    if (!t_stage_clean || link.ring_window == 0 ||
        holder != link.back_thread.load(std::memory_order_relaxed)) {
        return false;
    }
    if (link.calls.load(std::memory_order_seq_cst) != link.calls_seen) {
        LinkBump(link.counters.stage_lock_serves);
        link.StageServeCalls();
    }
    return true;
}

bool MemoryStageLink::BackRingTop(u64 position) {
    const bool on = BackRingTopImpl(position);
    if (on) {
        back_thread.store(std::this_thread::get_id(), std::memory_order_relaxed);
    }
    // Step 7: with the window open GPUMem fills while the back runs, and a sync in its fill waits
    // for the back with the buffer cache's lock held: a back that waits for that lock serves it.
    Common::SpinRecursiveMutex::t_contended_hook =
        on && Ahead() != 0 ? &MemoryStageLink::BackContendedHook : nullptr;
    return on;
}

bool MemoryStageLink::BackRingTopImpl(u64 position) {
    const bool wanted = ring_wanted.load(std::memory_order_acquire);
    const bool active = g_ring_active.load(std::memory_order_acquire);
    if (wanted && !active && !ring_running.load(std::memory_order_acquire)) {
        // The ring starts here: everything before `position` is executed and consumed, so the
        // stage's copies (all stale) are taken from the back as it is now.
        Ring::Staged().store(position, std::memory_order_relaxed);
        Ring::LentFrom().store(0, std::memory_order_relaxed);
        Ring::LentUntil().store(0, std::memory_order_relaxed);
        back_draw_seq = 0;
        back_run_from = ~u64{0};
        back_drew_in_run.store(nullptr, std::memory_order_relaxed);
        back_state_in_run.store(nullptr, std::memory_order_relaxed);
        lend_ranges.clear();
        if (ring_backend.reset_stage != nullptr) {
            ring_backend.reset_stage(ring_backend.context, false);
        }
        t_back_run = false;
        t_back_served = true;
        MemStageShadow::SetRingBack(true);
        g_ring_active.store(true, std::memory_order_seq_cst);
        ring_start.store(position, std::memory_order_release);
        WakeStage();
        LOG_INFO(HW_GPU, "Memory stage ring (FRITCHII_MEMSTAGE_THREAD=2): GPUMem reads the ring "
                         "from position {:#x}",
                 position);
        return true;
    }
    if (!active) {
        if (t_back_served || t_back_run) {
            // The stage ended the ring (at a parked point): back to `published`, the cache the
            // back's again. Ring B is drained first: a packet left is a draw the back never
            // prepared (must be 0), played whole in order so the cache's uploads are recorded and
            // no stale header meets the next start; then the stage's last ranges.
            t_back_served = false;
            t_back_run = false;
            if (ring_backend.drop_packets != nullptr) {
                const u64 left = ring_backend.drop_packets(ring_backend.context, ~u64{0});
                for (u64 i = 0; i < left; ++i) {
                    Note(Check::PacketOrphan, "left in ring B when the ring ended");
                }
            }
            // GPUMem's unplayed mark must not outlive the ring (a fill that threw leaves its
            // entries pending and its number open): its entries are played, after the packets
            // of earlier entries above (they are later in the stream), and the mark cleared;
            // then the ranges.
            if (ring_backend.reset_stage != nullptr) {
                ring_backend.reset_stage(ring_backend.context, true);
            }
            if (!lend_ranges.empty() && ring_backend.apply_ranges != nullptr) {
                ring_backend.apply_ranges(ring_backend.context, lend_ranges.data(),
                                          lend_ranges.size());
            }
            lend_ranges.clear();
            // Both draw counts start at 0 with the next ring (StageStart, the start above).
            back_draw_seq = 0;
            back_run_from = ~u64{0};
            MemStageShadow::SetRingBack(false);
            LOG_INFO(HW_GPU, "Memory stage ring ended at position {:#x}", position);
        }
        return false;
    }
    return true;
}

MemoryStageLink::BackLimits MemoryStageLink::BackLoad(u64 published) const noexcept {
    if (!g_ring_active.load(std::memory_order_acquire)) {
        return {published, 0, 0};
    }
    const u64 available = Ring::Staged().load(std::memory_order_acquire);
    return {available, Ring::LentFrom().load(std::memory_order_relaxed),
            Ring::LentUntil().load(std::memory_order_relaxed)};
}

void MemoryStageLink::BackEntry(u64 position, const BackLimits& limits) {
    if (!t_back_served && !t_back_run) {
        return; // the ring is off
    }
    const bool lent = position >= limits.lent_from && position < limits.lent_until;
    if (lent && !t_back_run) {
        // A run starts: the cache is the back's until its last entry. A run the stage extended
        // after the back had ended it (same lent_from) goes on: what the back drew in its first
        // part stays noted for the return.
        const bool continued = limits.lent_from == back_run_from;
        t_back_run = true;
        t_back_served = false;
        back_run_from = limits.lent_from;
        if (continued) {
            return;
        }
        back_run_seen_ns = LinkNowNs();
        back_drew_in_run.store(nullptr, std::memory_order_relaxed);
        back_state_in_run.store(nullptr, std::memory_order_relaxed);
        if (std::exchange(lend_flags_deferred, false)) {
            // Step 7 (plan 4.2): the cache's flags 15-48 the stage had at the lend, set here, where
            // the back reaches the same entry (with the window open it was not parked then).
            MemStageShadow::BackTakeLend();
        }
        if (ring_backend.drop_packets != nullptr) {
            // Packets of entries before the run that no draw took (must be 0): played whole, so
            // what the cache believes it uploaded is recorded.
            const u64 left = ring_backend.drop_packets(ring_backend.context, limits.lent_from);
            for (u64 i = 0; i < left; ++i) {
                Note(Check::PacketUnplayed, "at a lend");
            }
        }
        if (!lend_ranges.empty() && ring_backend.apply_ranges != nullptr) {
            // The stage's constant buffer ranges no fill applied yet (plan 6.2 item 2).
            ring_backend.apply_ranges(ring_backend.context, lend_ranges.data(),
                                      lend_ranges.size());
        }
        lend_ranges.clear();
        return;
    }
    if (!lent) {
        t_back_served = true;
    }
}

bool MemoryStageLink::BackAfter(u64 position, const BackLimits& limits) {
    if (!t_back_run || position < limits.lent_until) {
        return false;
    }
    // The stage may have extended the run since the batch's limits were read (it stores
    // lent_until before `staged`): the run ends only at its current end.
    return position >= Ring::LentUntil().load(std::memory_order_acquire);
}

void MemoryStageLink::BackEndRun() {
    // gpu_split.cpp consumed the engine's method sink already (spec 1.5); the ranges the run
    // collected must not outlive it (plan 5.2).
    if (ring_backend.end_run != nullptr) {
        ring_backend.end_run(ring_backend.context);
    }
    t_back_run = false;
    t_back_served = true;
}

bool MemoryStageLink::BackServeRequests() {
    if (requests.load(std::memory_order_seq_cst) == answered_local) {
        return false;
    }
    ServeOne();
    return true;
}

bool MemoryStageLink::BackRequestPending() const noexcept {
    return requests.load(std::memory_order_seq_cst) != answered_local;
}

bool MemoryStageLink::BackEnterHook() {
    if (!g_ring_active.load(std::memory_order_relaxed)) {
        return true;
    }
    back_in_hook.store(true, std::memory_order_seq_cst);
    if (stage_busy.load(std::memory_order_seq_cst)) {
        back_in_hook.store(false, std::memory_order_seq_cst);
        return false;
    }
    return true;
}

void MemoryStageLink::BackLeaveHook() {
    back_in_hook.store(false, std::memory_order_seq_cst);
}

void MemoryStageLink::NoteBackDraw(const void* engine) noexcept {
    ++back_draw_seq;
    if (t_back_run) {
        back_drew_in_run.store(engine, std::memory_order_relaxed);
    }
}

void MemoryStageLink::NoteBackDrawState(const void* engine) noexcept {
    if (t_back_run) {
        back_state_in_run.store(engine, std::memory_order_relaxed);
    }
}

void MemoryStageLink::NoteBackWait(u64 ns) noexcept {
    LinkBump(counters.back_wait);
    LinkBump(counters.back_wait_ns, ns);
}

void MemoryStageLink::NoteConsumed(u64 position) noexcept {
    static_cast<void>(position);
    if (stage_sleeping.load(std::memory_order_seq_cst)) {
        WakeStage();
    }
}

void MemoryStageLink::NotePublished() noexcept {
    if (g_ring_active.load(std::memory_order_relaxed) &&
        stage_sleeping.load(std::memory_order_seq_cst)) {
        WakeStage();
    }
}

void MemoryStageLink::NotePacketSent(u64 bytes, u64 ahead) noexcept {
    LinkBump(counters.packets);
    LinkBump(counters.packet_bytes, bytes);
    // The window's fill (step 7, 8.3's "the stage's lead"): packets in ring B after this push.
    LinkBump(counters.ahead_sum, ahead);
    const size_t bucket = ahead <= 1    ? 0
                          : ahead < 64  ? static_cast<size_t>(std::bit_width(ahead)) - 1
                          : ahead < 256 ? 6
                                        : 7;
    LinkBump(counters.window[(std::min)(bucket, WindowBuckets - 1)]);
    if (ahead > counters.ring_ahead_max.load(std::memory_order_relaxed)) {
        counters.ring_ahead_max.store(ahead, std::memory_order_relaxed);
    }
}

void MemoryStageLink::WakeLockWaiters() noexcept {
    // This thread holds the buffer cache's lock and is about to wait for the other thread: if
    // that one sleeps on the lock (a hooked wait that read another holder), it wakes, sees this
    // thread as the holder, and works for it (re-review N6).
    if (cache_lock != nullptr && cache_lock->IsHeldByCurrentThread()) [[unlikely]] {
        cache_lock->WakeWaiters();
    }
}

void MemoryStageLink::NoteHookFlush() noexcept {
    LinkBump(counters.hook_flushes);
}

void MemoryStageLink::NoteRingFreed() noexcept {
    // The back took or dropped a packet (its count is MemStage's, stored seq_cst before this).
    // Only the window's waits need it (level mode waits for `consumed`), and of them only a
    // GPUMem asleep on the window or on room in ring B: its other sleeps (the back's position,
    // a run, the front) a take does not end, and each wake was a system call in the back's draw.
    // GPUMem stores stage_ring_wait before stage_sleeping, both seq_cst, and tests the window
    // and the room after: either this sees both, or GPUMem sees this take before it sleeps.
    if (Ahead() != 0 && stage_sleeping.load(std::memory_order_seq_cst) &&
        stage_ring_wait.load(std::memory_order_seq_cst)) {
        LinkBump(counters.ring_wakes);
        WakeStage();
    }
}

void MemoryStageLink::NotePacketTaken() noexcept {
    LinkBump(counters.taken);
    if (back_run_seen_ns != 0) {
        LinkBump(counters.back_run_ns, LinkNowNs() - back_run_seen_ns);
        back_run_seen_ns = 0;
    }
}

void MemoryStageLink::NoteLevelDraw(u32 kind) noexcept {
    LinkBump(counters.level_draws[(std::min)(kind, 7U)]);
}

void MemoryStageLink::NoteHash(bool same) noexcept {
    LinkBump(counters.hashes);
    if (!same) {
        LinkBump(counters.hash_mismatches);
    }
}

bool MemoryStageLink::RingVerify() noexcept {
    static const bool on = [] {
        const char* const value = std::getenv("FRITCHII_MEMSTAGE_RING_VERIFY");
        return value != nullptr && value[0] != '0';
    }();
    return on;
}

// ---------------------------------------------------------------------------------------------
// THREAD = 2: the stage's loop

void MemoryStageLink::PublishStaged(u64 position) {
    stage_busy.store(false, std::memory_order_seq_cst);
    if (position == staged_local) {
        return;
    }
    staged_local = position;
    LinkBump(counters.publishes);
    Ring::Staged().store(position, std::memory_order_seq_cst);
    Ring::WakeBack();
}

bool MemoryStageLink::StageServeCalls() {
    const u32 now = calls.load(std::memory_order_seq_cst);
    if (now == calls_seen) {
        return false;
    }
    calls_seen = now;
    RunCall(now);
    return true;
}

bool MemoryStageLink::StageWaitFor(std::stop_token& stop, StageWait reason, u64 target) {
    const u64 start = LinkNowNs();
    static constexpr std::array<const char*, static_cast<size_t>(StageWait::Count)> names{
        "the front",          "the back (level mode)", "the back (level run)",
        "the window (AHEAD)", "room in ring B",        "the back (level inside the window)"};
    const Nx2Profile::Scope zone{Nx2Profile::Zone::StageWait};
    Common::Nx2Timeline::ArgSpan span{"stage wait", {"reason", static_cast<u32>(reason)}};
    const StageWaitScope site{names[static_cast<size_t>(reason)]};
    const auto reached = [&] {
        switch (reason) {
        case StageWait::Input:
            return Ring::Published().load(std::memory_order_acquire) != target ||
                   !ring_wanted.load(std::memory_order_acquire);
        case StageWait::Window:
            // Packets in ring B the back has not taken (the back's count is stored seq_cst). A
            // back parked at the published position takes no more: what is left is a packet no
            // draw asked for (counted when it is dropped), so the stage goes on.
            return ring_backend.ring_packets(ring_backend.context) < target ||
                   Ring::Consumed().load(std::memory_order_acquire) >= staged_local;
        case StageWait::Room:
            // As the window; without room then, the next draw is level (StageFillDraw).
            return ring_backend.ring_room(ring_backend.context) ||
                   Ring::Consumed().load(std::memory_order_acquire) >= staged_local;
        default:
            return Ring::Consumed().load(std::memory_order_acquire) >= target;
        }
    };
    // With the window open, the wait for the front spins about 200 us before it sleeps: a
    // sleeping stage costs the front a wake in its next publish and a wake-up inside the drain
    // that follows. Level mode keeps the short spin.
    const bool long_spin = reason == StageWait::Input && ring_window != 0;
    const bool ring_wait = reason == StageWait::Window || reason == StageWait::Room;
    u64 spin_from = start;
    bool ok = true;
    for (u32 spin = 0;;) {
        if (stop.stop_requested()) {
            ok = false;
            break;
        }
        if (reached()) {
            break;
        }
        if (StageServeCalls()) {
            spin = 0;
            if (long_spin) {
                spin_from = LinkNowNs();
            }
            continue;
        }
        if (spin < LinkSpinPauses ||
            (long_spin && ((spin % 64) != 0 || LinkNowNs() - spin_from < InputSpinNs))) {
            ++spin;
            LinkPause();
            continue;
        }
        // The back's take wakes only these two waits (NoteRingFreed).
        stage_ring_wait.store(ring_wait, std::memory_order_seq_cst);
        stage_sleeping.store(true, std::memory_order_seq_cst);
        const u32 word = stage_word.load(std::memory_order_seq_cst);
        if (!reached() && calls.load(std::memory_order_seq_cst) == calls_seen &&
            !stop.stop_requested()) {
            stage_word.wait(word, std::memory_order_seq_cst);
            LinkBump(counters.stage_sleeps);
            if (reason == StageWait::Input) {
                LinkBump(counters.stage_input_sleeps);
            }
        }
        stage_sleeping.store(false, std::memory_order_relaxed);
        if (ring_wait) {
            stage_ring_wait.store(false, std::memory_order_relaxed);
        }
    }
    LinkBump(counters.stage_wait_ns[static_cast<size_t>(reason)], LinkNowNs() - start);
    return ok;
}

bool MemoryStageLink::BecomeBusy(std::stop_token& stop) {
    // The seq_cst handshake with BackEnterHook: either the back sees the stage busy and skips its
    // idle hook, or the stage sees the hook and waits for it to end.
    for (;;) {
        stage_busy.store(true, std::memory_order_seq_cst);
        if (!back_in_hook.load(std::memory_order_seq_cst)) {
            return true;
        }
        stage_busy.store(false, std::memory_order_seq_cst);
        while (back_in_hook.load(std::memory_order_seq_cst)) {
            if (stop.stop_requested()) {
                return false;
            }
            if (!StageServeCalls()) {
                LinkPause();
            }
        }
    }
}

void MemoryStageLink::StageIdle(std::stop_token& stop) {
    // The cache's idle work (spec 5.2: the buffer recycler's doomed buffers), on the cache's
    // thread, only while the back is parked.
    if (ring_backend.idle == nullptr ||
        Ring::Consumed().load(std::memory_order_acquire) < staged_local) {
        return;
    }
    if (!BecomeBusy(stop)) {
        return;
    }
    LinkBump(counters.stage_idles);
    ring_backend.idle(ring_backend.context);
    stage_busy.store(false, std::memory_order_seq_cst);
}

void MemoryStageLink::SetAhead(u32 packets, const char* blocked) {
    // Refused for the session after a sync of GPUMem's fill went to a back that had not played
    // the packets ahead of it (StageSyncCatchUp), and while `blocked` names a reason (logged once
    // per reason).
    if (packets != 0 && ahead_refused.load(std::memory_order_relaxed)) {
        packets = 0;
    }
    if (packets != 0 && blocked != nullptr) {
        if (blocked != ahead_blocked_logged) {
            ahead_blocked_logged = blocked;
            LOG_WARNING(HW_GPU, "FRITCHII_MEMSTAGE_AHEAD={}: runs as 0 while {} is on", packets,
                        blocked);
        }
        packets = 0;
    }
    const u32 next = (std::min)(packets, AheadUnlimited);
    if (g_ahead.exchange(next, std::memory_order_relaxed) != next) {
        LOG_INFO(HW_GPU, "Memory stage window (FRITCHII_MEMSTAGE_AHEAD={}): {}", next,
                 next == 0               ? "level mode, GPUMem fills only while the back is parked"
                 : next >= AheadUnlimited ? "unlimited, GPUMem fills ahead of the back up to its "
                                            "next level run"
                                          : "GPUMem fills up to that many packets ahead of the "
                                            "back's take");
    }
}

void MemoryStageLink::SetListDraws(bool on) {
    // At a frame end, a level run (GPUMem is parked on it), as SetAhead: GPUMem reads it at its
    // next list draw. Served only while the window is open too (ListDrawsServed).
    g_list_draws.store(on, std::memory_order_relaxed);
    if (mode == 2 && list_draws_logged != (on ? 1 : 0)) {
        list_draws_logged = on ? 1 : 0;
        LOG_INFO(HW_GPU, "Memory stage list draws (FRITCHII_MEMSTAGE_LIST_DRAWS={}): {}", on ? 1 : 0,
                 on ? "served by GPUMem while the window is open (FRITCHII_MEMSTAGE_AHEAD > 0)"
                    : "level draws, as before");
    }
}

void MemoryStageLink::SetFastLoop(bool on) {
    // At a frame end, a level run (GPUMem is parked on it), as SetListDraws: GPUMem reads it at
    // its next entry. Used only while the window is open too (FastLoop).
    g_fast_loop.store(on, std::memory_order_relaxed);
    if (mode == 2 && fast_loop_logged != (on ? 1 : 0)) {
        fast_loop_logged = on ? 1 : 0;
        LOG_INFO(HW_GPU, "Memory stage fast entry loop (FRITCHII_MEMSTAGE_FAST_LOOP={}): {}",
                 on ? 1 : 0,
                 on ? "GPUMem applies a Pairs block's writes in one loop and reads the input's end "
                      "once per stretch, while the window is open (FRITCHII_MEMSTAGE_AHEAD > 0)"
                    : "GPUMem's entry loop as before");
    }
}

void MemoryStageLink::CloseBusyBatch() noexcept {
    if (busy_batch_start != 0) {
        LinkBump(counters.stage_busy_ns, LinkNowNs() - busy_batch_start);
        busy_batch_start = 0;
    }
}

bool MemoryStageLink::StageBeforeFill(u64 position) {
    // fps fix 1: the window and the room, tested where a packet may follow (StageFillDraw, before
    // the fill and outside any lock), not before every entry. At most `ring_window` packets in
    // ring B the back has not taken, and room for the next (spec 1.4). Publish before you wait:
    // the back runs up to this entry and can take every packet of the entries before it (the
    // waits also end when it is parked there).
    if (!t_on_stage || ring_window == 0 || ring_stop == nullptr) {
        return true;
    }
    bool waited = false;
    if (ring_backend.ring_packets(ring_backend.context) >= ring_window) {
        CloseBusyBatch();
        PublishStaged(position);
        waited = true;
        if (!StageWaitFor(*ring_stop, StageWait::Window, ring_window)) {
            return false;
        }
    }
    if (!ring_backend.ring_room(ring_backend.context)) {
        CloseBusyBatch();
        PublishStaged(position);
        waited = true;
        if (!StageWaitFor(*ring_stop, StageWait::Room, 0)) {
            return false;
        }
    }
    if (waited && !stage_busy.load(std::memory_order_relaxed) && !BecomeBusy(*ring_stop)) {
        return false;
    }
    return true;
}

bool MemoryStageLink::StageLevel(u64 position, LevelReason reason) {
    if (!t_on_stage || ring_window == 0 || ring_stop == nullptr) {
        return true; // level mode: the stage is level before each entry already
    }
    LinkBump(counters.levels[static_cast<size_t>(reason)]);
    if (Ring::Consumed().load(std::memory_order_acquire) < position) {
        // Publish before you wait (spec 4.3): the back runs up to this entry, not into it.
        CloseBusyBatch();
        PublishStaged(position);
        if (!StageWaitFor(*ring_stop, StageWait::Level, position)) {
            return false; // stopping
        }
    }
    if (!stage_busy.load(std::memory_order_relaxed)) {
        return BecomeBusy(*ring_stop);
    }
    return true;
}

void MemoryStageLink::NoteStageWrite(u64 begin, u64 size, u64 position) noexcept {
    if (!stage_writes || size == 0) {
        return;
    }
    const u64 number = stage_write_count.load(std::memory_order_relaxed);
    StageWrite& slot = stage_writes[number % StageWriteSlots];
    slot.seq.store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    slot.begin.store(begin, std::memory_order_relaxed);
    slot.end.store(begin + size, std::memory_order_relaxed);
    slot.position.store(position, std::memory_order_relaxed);
    slot.seq.store(number + 1, std::memory_order_release);
    stage_write_count.store(number + 1, std::memory_order_release);
}

void MemoryStageLink::NoteImageRegisteredSlow(u64 address, u64 size) {
    // Risk 9.1 of the spec: the stage wrote constant buffer words for an entry the back has not
    // reached, on pages an image is registered over now; that image's upload reads the later
    // bytes. Only the back registers images at its position.
    if (!GpuSplit::OnBackThread() || size == 0) {
        return;
    }
    const u64 back = GpuSplit::Get().BackEntryPosition();
    const u64 end = address + size;
    bool hit = false;
    ForEachStageWriteSince(0, [&](u64 write_begin, u64 write_end, u64 position) {
        if (position <= back) {
            return true; // older writes are at or before the back's entry
        }
        hit = write_begin < end && address < write_end;
        return hit;
    });
    if (hit) [[unlikely]] {
        Note(Check::FutureWriteUnderImage, "the texture cache's image registration");
    }
}

void MemoryStageLink::NoteBufferJoinSlow(u64 address, u64 size) noexcept {
    joins[join_count % JoinSlots] = Join{address, address + size, stage_entry};
    ++join_count;
}

MemoryStageLink::Joined MemoryStageLink::JoinedAhead(u64 address, u64 size,
                                                     u64 back_position) const noexcept {
    // A join copies the old buffer's bytes into the new one in its packet: a download the back
    // records before that packet reads the new buffer before the copy. The served calls' answers
    // stay the back's only while no such join is ahead of it over GPU-written data (step 7; joins
    // over GPU-written data catch up first, so this is the check).
    const u64 end = address + size;
    const u64 kept = (std::min)(join_count, u64{JoinSlots});
    for (u64 i = 0; i < kept; ++i) {
        const Join& join = joins[(join_count - 1 - i) % JoinSlots];
        if (join.position <= back_position) {
            return Joined::No; // the fills are in stream order: the rest are played
        }
        if (join.begin < end && address < join.end) {
            return Joined::Yes;
        }
    }
    // Every kept join is ahead of the back: older ones are not known.
    return kept == JoinSlots ? Joined::Unknown : Joined::No;
}


void MemoryStageLink::StageSyncCatchUp() {
    // GPUMem's fill reaches a sync with the window open. The back records the sync (and the
    // entries a sync plays) where it is now; the packets GPUMem sent before belong earlier in the
    // stream, and a wait may be on one of their numbers (their staging and stream regions would
    // look free). So the back first runs up to the published position (every packet of an entry
    // before it played) by itself.
    LinkBump(counters.catchups);
    StageCatchUp("a sync in GPUMem's fill");
}

void MemoryStageLink::StageJoinCatchUpSlow() {
    // A join in GPUMem's fill over a buffer holding GPU-written data (BufferCache::JoinOverlap):
    // the back plays the packets ahead first, so no download it records later in a served call
    // reads the new buffer before the join's copy (Fable's review, B).
    LinkBump(counters.join_catchups);
    StageCatchUp("a buffer join over GPU-written data in GPUMem's fill");
}

bool MemoryStageLink::StageCatchUp(const char* what) {
    // The back cannot run on when it waits in a served call (GPUMem runs those only between
    // entries) or for the buffer cache's lock, which GPUMem holds in its fill: then GPUMem goes on
    // at once, counted (must be 0), and the window is refused for the rest of the session.
    if (Ring::Consumed().load(std::memory_order_acquire) >= staged_local) {
        return true;
    }
    // A back asleep on the lock GPUMem holds wakes, sees GPUMem as the holder and counts
    // (back_contended): the wait below then stops at once instead of waiting forever.
    WakeLockWaiters();
    const u64 start = LinkNowNs();
    bool caught_up = true;
    const u64 contended = back_contended.load(std::memory_order_seq_cst);
    const auto stuck = [&] {
        return calls.load(std::memory_order_seq_cst) != calls_seen ||
               back_contended.load(std::memory_order_seq_cst) != contended;
    };
    const auto done = [&] {
        return Ring::Consumed().load(std::memory_order_acquire) >= staged_local ||
               (ring_stop != nullptr && ring_stop->stop_requested()) ||
               !GpuSplit::Get().BackRunning();
    };
    {
        const StageWaitScope site{"the back's catch-up (a sync or a join in GPUMem's fill)"};
        for (u32 spin = 0;;) {
            if (done()) {
                break;
            }
            if (stuck()) {
                caught_up = false;
                Note(Check::SyncAhead, what);
                if (!ahead_refused.exchange(true, std::memory_order_relaxed)) {
                    LOG_CRITICAL(HW_GPU, "Memory stage window: {} went on before the back played "
                                         "the packets ahead of it; FRITCHII_MEMSTAGE_AHEAD runs "
                                         "as 0 from the next frame end, for this session",
                                 what);
                }
                break;
            }
            if (spin < LinkSpinPauses) {
                ++spin;
                LinkPause();
                continue;
            }
            stage_sleeping.store(true, std::memory_order_seq_cst);
            const u32 word = stage_word.load(std::memory_order_seq_cst);
            if (!done() && !stuck()) {
                stage_word.wait(word, std::memory_order_seq_cst);
                LinkBump(counters.stage_sleeps);
            }
            stage_sleeping.store(false, std::memory_order_relaxed);
        }
    }
    LinkBump(counters.catchup_ns, LinkNowNs() - start);
    return caught_up;
}

void MemoryStageLink::Lend(u64 from, u64 until) {
    // The ranges the stage's constant buffer words collected and no fill applied: the back
    // applies their buffer half before the run's first entry (it reads the cache there).
    std::vector<std::pair<u64, std::size_t>>& ranges = MemStageShadow::StageRanges();
    lend_ranges.swap(ranges);
    ranges.clear();
    // Plan 4.2: the cache's flags (15-48) go with the cache. In level mode the back is parked
    // here; with the window open it may still be running entries before the run, so the stage
    // keeps what it lends and the back takes it at the run's first entry (BackEntry), where both
    // are at the same position.
    if (ring_window != 0) {
        MemStageShadow::StageLendDeferred();
        lend_flags_deferred = true;
        LinkBump(counters.deferred_lends);
    } else {
        MemStageShadow::StageLend();
    }
    LinkBump(counters.flag_handovers);
    LinkBump(counters.runs);
    Ring::LentFrom().store(from, std::memory_order_relaxed);
    Ring::LentUntil().store(until, std::memory_order_relaxed);
    t_stage_user = false;
    t_in_call = false;
    PublishStaged(until);
}

u64 MemoryStageLink::RingFailed(std::stop_token& stop, const char* what) {
    // An exception out of the stage's work (a fill, the cache): the ring ends cleanly where the
    // back is parked, the entries past `staged` run on the back as at THREAD 1 or 0, and THREAD 2
    // is refused for the rest of the session (SetMode). Counted, must be 0.
    t_in_call = false;
    t_stage_user = false;
    stage_busy.store(false, std::memory_order_seq_cst);
    ring_failed.store(true, std::memory_order_seq_cst);
    ring_wanted.store(false, std::memory_order_seq_cst);
    Note(Check::StageException, what);
    LOG_CRITICAL(HW_GPU, "Memory stage ring: GPUMem failed ({}); the ring ends at {:#x} and "
                         "FRITCHII_MEMSTAGE_THREAD=2 runs as 1 from here",
                 what, staged_local);
    StageWaitFor(stop, StageWait::Back, staged_local);
    return staged_local;
}

void MemoryStageLink::RingLoop(std::stop_token& stop, u64 start) {
    ring_running.store(true, std::memory_order_seq_cst);
    staged_local = start;
    ring_stop = &stop;
    ring_window = 0;
    MemStageShadow::StageStart(ring_backend.back_channel != nullptr
                                   ? ring_backend.back_channel(ring_backend.context)
                                   : 0);
    u64 position = start;
    u64 known_available = 0; ///< The input's end as last read (FRITCHII_MEMSTAGE_FAST_LOOP)
    u64 since_publish = 0;
    u64 last_tick_ns = LinkNowNs();
    // GPUMem's busy time, timed per publish batch (fps fix 1: two clock reads an entry were a
    // good part of the loop's cost). A batch starts at the first entry after a publish or a wait
    // and ends at the next publish, wait or lend (CloseBusyBatch; the waits inside an entry close
    // it too).
    busy_batch_start = 0;
    const auto close_batch = [this] { CloseBusyBatch(); };
    t_stage_user = true;
    try {
        for (;;) {
            if (stop.stop_requested()) {
                break;
            }
            if (!ring_wanted.load(std::memory_order_acquire)) {
                // The ring ends where the back is parked at the stage's position.
                PublishStaged(position);
                if (!StageWaitFor(stop, StageWait::Back, position)) {
                    break;
                }
                break;
            }
            // FRITCHII_MEMSTAGE_FAST_LOOP: the input's end is read again only where GPUMem has
            // reached the last one it read (the front writes it at every publish: a load per entry
            // mostly missed in GPUMem's cache). Entries before it stay published, and its acquire
            // made them visible. Off, or with the window closed, it is read at every entry.
            if (!FastLoop() || position >= known_available) {
                known_available = Ring::Published().load(std::memory_order_acquire);
            }
            const u64 available = known_available;
            if (available == position) {
                close_batch();
                PublishStaged(position);
                // EXP39: the front publishes a few words to a reader that waits for them; the stage
                // is the reader the front feeds now.
                Ring::IdleAt().store(position, std::memory_order_relaxed);
                StageIdle(stop);
                const u64 now = LinkNowNs();
                if (now - last_tick_ns >= 1'000'000'000ULL) {
                    last_tick_ns = now;
                    MemStageShadow::StageTick();
                    Nx2Profile::Report("GPU memory thread");
                }
                if (!StageWaitFor(stop, StageWait::Input, position)) {
                    break;
                }
                continue;
            }
            // The window (step 7), read here: it changes only at a frame end, where the stage is
            // parked in a level run.
            ring_window = Ahead();
            // Step 7: GPUMem serves the back's call while it waits for the buffer cache's lock
            // the back holds, at its clean lock sites (StageContendedHook). Only with the window
            // open: in level mode a lock wait stays the plain spin-then-sleep.
            Common::SpinRecursiveMutex::t_contended_hook =
                ring_window != 0 ? &MemoryStageLink::StageContendedHook : nullptr;
            if (ring_window == 0) {
                // Level mode (AHEAD = 0): the stage runs only while the back is parked.
                if (Ring::Consumed().load(std::memory_order_acquire) < staged_local) {
                    close_batch();
                    if (!StageWaitFor(stop, StageWait::Back, staged_local)) {
                        break;
                    }
                }
            } else {
                // The back's served calls are answered between entries too: with the window
                // open the stage may not wait for a long time.
                StageServeCalls();
                // A full copy of the back's engine (a new or stale copy) is only exact where the
                // back is at this entry. (Two loads and a switch: it must run before the entry,
                // where the copy is taken.) The window and the room are tested before a fill
                // only (StageBeforeFill, fps fix 1): an entry that sends no packet needs neither.
                if (MemStageShadow::StageMayCopy(position)) [[unlikely]] {
                    close_batch();
                    if (!StageLevel(position, LevelReason::Copy)) {
                        break;
                    }
                }
            }
            if (!stage_busy.load(std::memory_order_relaxed) && !BecomeBusy(stop)) {
                break;
            }
            if (busy_batch_start == 0) {
                busy_batch_start = LinkNowNs();
            }
            t_in_call = true;
            t_stage_user = true;
            stage_entry = position;
            // fps fix 4 (profiling): the entries in a fine zone of their own (with their constant
            // buffer words in another, inside it), so GPUMem's "other" splits.
            const MemStageShadow::StageResult result = [&] {
                const Nx2Profile::FineScope zone{Nx2Profile::Zone::StageEntries};
                return MemStageShadow::StageProcess(position, false);
            }();
            LinkBump(counters.entries[(std::min)(static_cast<size_t>(result.cls), size_t{3})]);
            if (!result.hard) {
                t_in_call = false;
                position += result.size;
                if (result.packets != 0 || ++since_publish >= PublishEvery) {
                    since_publish = 0;
                    close_batch();
                    PublishStaged(position);
                }
                continue;
            }
            // A hard entry: a level run [position, until), extended over what follows that needs
            // nothing of the stage.
            u64 until = position + result.size;
            u64 run_entries = 1;
            u64 run_launches = result.launches;
            close_batch();
            Lend(position, until);
            const u64 run_start = LinkNowNs();
            {
                const Nx2Profile::Scope zone{Nx2Profile::Zone::LevelRun};
                Common::Nx2Timeline::ArgSpan span{"level run", {"entries", 1}};
                // With the window open the run's end is published once, after GPUMem followed
                // every entry it extends over (fps fix 1: one publish per run entry made the
                // back's batches short). Level mode publishes each, as before.
                const bool publish_once = ring_window != 0;
                for (;;) {
                    if (until == Ring::Published().load(std::memory_order_acquire)) {
                        if (!publish_once) {
                            break; // level mode: the run ends at the input's end, as before
                        }
                        // fps fix 3: the run reached the end of what the front published. Ending
                        // it here and lending again at the next hard entry cut one run in two
                        // (about 4,700 lends a second at the market). With the window open it
                        // stays lent: the back gets what is followed so far, GPUMem waits for
                        // more input, and the run ends at the first entry that needs GPUMem.
                        if (until != staged_local) {
                            Ring::LentUntil().store(until, std::memory_order_relaxed);
                            PublishStaged(until);
                        }
                        LinkBump(counters.run_input_waits);
                        if (Ring::Consumed().load(std::memory_order_acquire) >= until) {
                            // The back finished the run's part and is parked: GPUMem's idle
                            // work runs here too, or it would starve while runs are kept lent
                            // (the buffer recycler's doomed buffers, EXP104), and so do its
                            // once-a-second lines. The cache is GPUMem's again until the run
                            // is extended (the back ended its part).
                            t_stage_user = true;
                            StageIdle(stop);
                            t_stage_user = false;
                            const u64 now = LinkNowNs();
                            if (now - last_tick_ns >= 1'000'000'000ULL) {
                                last_tick_ns = now;
                                MemStageShadow::StageTick();
                                Nx2Profile::Report("GPU memory thread");
                            }
                        }
                        if (!StageWaitFor(stop, StageWait::Input, until) ||
                            !ring_wanted.load(std::memory_order_acquire)) {
                            break;
                        }
                        continue;
                    }
                    if (!MemStageShadow::StageExtendsRun(until)) {
                        break;
                    }
                    const MemStageShadow::StageResult more = MemStageShadow::StageProcess(until, true);
                    LinkBump(counters.entries[(std::min)(static_cast<size_t>(more.cls), size_t{3})]);
                    until += more.size;
                    ++run_entries;
                    run_launches += more.launches;
                    if (!publish_once) {
                        Ring::LentUntil().store(until, std::memory_order_relaxed);
                        PublishStaged(until);
                    }
                }
                if (publish_once && until != staged_local) {
                    // lent_until before `staged` (BackAfter reads it after the batch's limits).
                    Ring::LentUntil().store(until, std::memory_order_relaxed);
                    PublishStaged(until);
                }
                span.Set(0, {"entries", run_entries});
                LinkBump(counters.run_entries, run_entries);
                if (!StageWaitFor(stop, StageWait::Run, until)) {
                    break;
                }
            }
            {
                const u64 run_time = LinkNowNs() - run_start;
                LinkBump(counters.run_ns, run_time);
                NoteRunShape(result.channel, run_entries, run_launches, run_time);
            }
            // The return (back parked at `until`): flags, takes, full copies.
            if (!BecomeBusy(stop)) {
                break;
            }
            const u64 copies = MemStageShadow::StageReturn(
                back_drew_in_run.exchange(nullptr, std::memory_order_relaxed),
                back_state_in_run.exchange(nullptr, std::memory_order_relaxed),
                ring_backend.back_channel != nullptr ? ring_backend.back_channel(ring_backend.context)
                                                     : 0);
            LinkBump(counters.copies, copies);
            // Compare a with two threads: both sides counted every draw since the ring started. A
            // difference would mispair every later packet; the back's count is taken (it is parked).
            if (const u64 stage_seq = MemStageShadow::StageDrawSeq(back_draw_seq);
                stage_seq != back_draw_seq) [[unlikely]] {
                Note(Check::DrawCount, "the stage's count set to the back's");
            }
            t_stage_user = true;
            position = until;
            since_publish = 0;
        }
    } catch (const std::exception& error) {
        position = RingFailed(stop, error.what());
    } catch (...) {
        position = RingFailed(stop, "an exception of no standard type");
    }
    t_in_call = false;
    t_stage_user = false;
    ring_window = 0;
    ring_stop = nullptr;
    Common::SpinRecursiveMutex::t_contended_hook = nullptr;
    stage_busy.store(false, std::memory_order_seq_cst);
    if (!stop.stop_requested()) {
        // The cache goes back to the back for good, with its flags 15-48 (plan 4.2, as at a
        // lend; the back is parked).
        MemStageShadow::StageLend();
    }
    // The ranges the stage's words collected that no fill applied and no lend handed over: the
    // back applies their buffer half when it sees the ring end (BackRingTop), before anything
    // else, as at a lend. (StageStart wipes the stage's list at the next start.)
    {
        std::vector<std::pair<u64, std::size_t>>& ranges = MemStageShadow::StageRanges();
        lend_ranges.insert(lend_ranges.end(), ranges.begin(), ranges.end());
        ranges.clear();
    }
    // The back's limit goes back to `published`; it is parked at `position` (or stopping).
    g_ring_active.store(false, std::memory_order_seq_cst);
    ring_running.store(false, std::memory_order_seq_cst);
    Ring::WakeBack();
    LOG_INFO(HW_GPU, "Memory stage ring: GPUMem stopped reading the ring at {:#x}", position);
}

// ---------------------------------------------------------------------------------------------
// Stats

namespace {
/// The run shape's buckets: 0, 1, 2-7, 8-63, 64-511, 512 and more.
size_t RunBucket(u64 count) noexcept {
    return count == 0 ? 0 : count == 1 ? 1 : count < 8 ? 2 : count < 64 ? 3 : count < 512 ? 4 : 5;
}
} // Anonymous namespace

void MemoryStageLink::NoteRunShape(s32 channel, u64 entries, u64 launches, u64 ns) noexcept {
    // GPUMem, once a run: a few relaxed adds, with the GPU profile only.
    if (!Nx2Profile::Enabled()) {
        return;
    }
    RunShape& shape = run_shape;
    const size_t slot = channel >= 0 && static_cast<size_t>(channel) < RunChannels - 1
                            ? static_cast<size_t>(channel)
                            : RunChannels - 1;
    LinkBump(shape.runs[slot]);
    LinkBump(shape.entries[slot], entries);
    LinkBump(shape.launches[slot], launches);
    LinkBump(shape.ns[slot], ns);
    const size_t by_entries = RunBucket(entries);
    LinkBump(shape.by_entries[by_entries]);
    LinkBump(shape.by_entries_ns[by_entries], ns);
    const size_t by_launches = RunBucket(launches);
    LinkBump(shape.by_launches[by_launches]);
    LinkBump(shape.by_launches_ns[by_launches], ns);
}

void MemoryStageLink::LogRunShape(double seconds) {
    RunShape& shape = run_shape;
    const auto take = [](std::atomic<u64>& value) {
        return value.exchange(0, std::memory_order_relaxed);
    };
    const auto rate = [seconds](u64 value) { return static_cast<double>(value) / seconds; };
    const auto ms = [seconds](u64 ns) { return static_cast<double>(ns) / 1e6 / seconds; };
    const auto each = [](u64 total, u64 count) {
        return count != 0 ? static_cast<double>(total) / static_cast<double>(count) : 0.0;
    };
    std::string channels;
    for (size_t i = 0; i < RunChannels; ++i) {
        const u64 runs = take(shape.runs[i]);
        const u64 entries = take(shape.entries[i]);
        const u64 launches = take(shape.launches[i]);
        const u64 ns = take(shape.ns[i]);
        if (runs == 0) {
            continue;
        }
        channels += fmt::format("{}channel {}: {:.0f}/s, {:.1f} entries and {:.1f} compute "
                                "launches each, {:.1f} ms/s",
                                channels.empty() ? "" : "; ",
                                i == RunChannels - 1 ? std::string{"other"} : fmt::format("{}", i),
                                rate(runs), each(entries, runs), each(launches, runs), ms(ns));
    }
    static constexpr std::array<const char*, RunBuckets> bucket_names{"0",     "1",       "2-7",
                                                                      "8-63",  "64-511",  "512+"};
    const auto buckets = [&](auto& counts, auto& times, size_t first) {
        std::string text;
        for (size_t i = first; i < RunBuckets; ++i) {
            const u64 runs = take(counts[i]);
            const u64 ns = take(times[i]);
            text += fmt::format("{}{} {:.0f}/s {:.1f} ms/s", text.empty() ? "" : ", ",
                                bucket_names[i], rate(runs), ms(ns));
        }
        return text;
    };
    const std::string by_entries = buckets(shape.by_entries, shape.by_entries_ns, 1);
    const std::string by_launches = buckets(shape.by_launches, shape.by_launches_ns, 0);
    LOG_INFO(HW_GPU,
             "Memory stage runs by channel {:.0f} s: {} | by entries a run: {} | by compute "
             "launches a run: {}",
             seconds, channels.empty() ? std::string{"none"} : channels, by_entries,
             by_launches);
}

void MemoryStageLink::Take(Snapshot& out) const noexcept {
    const auto load = [](const std::atomic<u64>& value) {
        return value.load(std::memory_order_relaxed);
    };
    for (size_t i = 0; i < out.calls.size(); ++i) {
        out.calls[i] = load(counters.calls[i]);
    }
    out.call_ns = load(counters.call_ns);
    out.stage_ns = load(counters.stage_ns);
    out.served = load(counters.served);
    out.served_ns = load(counters.served_ns);
    out.back_sleeps = load(counters.back_sleeps);
    out.stage_sleeps = load(counters.stage_sleeps);
    out.stage_input_sleeps = load(counters.stage_input_sleeps);
    for (size_t i = 0; i < out.checks.size(); ++i) {
        out.checks[i] = load(counters.checks[i]);
    }
    for (size_t i = 0; i < out.served_calls.size(); ++i) {
        out.served_calls[i] = load(counters.served_calls[i]);
    }
    for (size_t i = 0; i < out.entries.size(); ++i) {
        out.entries[i] = load(counters.entries[i]);
    }
    out.runs = load(counters.runs);
    out.run_entries = load(counters.run_entries);
    out.run_ns = load(counters.run_ns);
    out.back_run_ns = load(counters.back_run_ns);
    out.back_wait = load(counters.back_wait);
    out.back_wait_ns = load(counters.back_wait_ns);
    for (size_t i = 0; i < out.stage_wait_ns.size(); ++i) {
        out.stage_wait_ns[i] = load(counters.stage_wait_ns[i]);
    }
    out.stage_busy_ns = load(counters.stage_busy_ns);
    out.fill_ns = load(counters.fill_ns);
    out.packets = load(counters.packets);
    out.packet_bytes = load(counters.packet_bytes);
    out.taken = load(counters.taken);
    out.publishes = load(counters.publishes);
    out.copies = load(counters.copies);
    out.flag_handovers = load(counters.flag_handovers);
    for (size_t i = 0; i < out.level_draws.size(); ++i) {
        out.level_draws[i] = load(counters.level_draws[i]);
    }
    out.hashes = load(counters.hashes);
    out.hash_mismatches = load(counters.hash_mismatches);
    for (size_t i = 0; i < out.window.size(); ++i) {
        out.window[i] = load(counters.window[i]);
    }
    out.ahead_sum = load(counters.ahead_sum);
    for (size_t i = 0; i < out.levels.size(); ++i) {
        out.levels[i] = load(counters.levels[i]);
    }
    out.served_ahead = load(counters.served_ahead);
    out.catchups = load(counters.catchups);
    out.catchup_ns = load(counters.catchup_ns);
    out.deferred_lends = load(counters.deferred_lends);
    out.back_lock_waits = load(counters.back_lock_waits);
    out.stage_lock_serves = load(counters.stage_lock_serves);
    out.join_catchups = load(counters.join_catchups);
    out.joins_unknown = load(counters.joins_unknown);
    out.write_levels = load(counters.write_levels);
    out.staging_fallbacks = load(counters.staging_fallbacks);
    out.hook_flushes = load(counters.hook_flushes);
    out.ring_wakes = load(counters.ring_wakes);
    out.stage_idles = load(counters.stage_idles);
    out.run_input_waits = load(counters.run_input_waits);
    for (size_t i = 0; i < out.list_draws.size(); ++i) {
        out.list_draws[i] = load(counters.list_draws[i]);
    }
    out.list_drawn = load(counters.list_drawn);
}

void MemoryStageLink::LogStats() {
    if (mode == 0) {
        return;
    }
    const u64 now = LinkNowNs();
    if (last_log_ns == 0) {
        last_log_ns = now;
        Take(reported);
        return;
    }
    if (now - last_log_ns < 10'000'000'000ULL) {
        return;
    }
    const double seconds = static_cast<double>(now - last_log_ns) / 1e9;
    last_log_ns = now;
    Snapshot current;
    Take(current);
    const auto rate = [seconds](u64 value) { return static_cast<double>(value) / seconds; };
    const auto ms = [seconds](u64 ns) { return static_cast<double>(ns) / 1e6 / seconds; };
    const auto per = [](u64 ns, u64 count) {
        return count != 0 ? static_cast<double>(ns) / 1e3 / static_cast<double>(count) : 0.0;
    };
    std::string calls_text;
    u64 total_calls = 0;
    for (size_t i = 0; i < current.calls.size(); ++i) {
        const u64 n = current.calls[i] - reported.calls[i];
        total_calls += n;
        calls_text += fmt::format("{}{} {:.0f}", calls_text.empty() ? "" : ", ", LINK_HANDOFF_NAMES[i],
                                  rate(n));
    }
    std::string checks_text;
    u64 total_checks = 0;
    for (size_t i = 0; i < current.checks.size(); ++i) {
        const u64 n = current.checks[i] - reported.checks[i];
        total_checks += n;
        if (n != 0) {
            checks_text += fmt::format("; {} {}", LINK_CHECK_NAMES[i], n);
        }
    }
    const u64 call_ns = current.call_ns - reported.call_ns;
    LOG_INFO(HW_GPU,
             "Memory stage link (FRITCHII_MEMSTAGE_THREAD={}) {:.0f} s: hand-offs {:.0f}/s ({}); "
             "the back waited {:.1f} ms/s ({:.2f} us a hand-off, slept {:.0f}/s), GPUMem ran "
             "{:.1f} ms/s and slept {:.0f}/s ({:.0f}/s waiting for the front); served syncs "
             "{:.0f}/s ({:.2f} ms/s on the back); must be 0: {}{}",
             mode, seconds, rate(total_calls), calls_text, ms(call_ns), per(call_ns, total_calls),
             rate(current.back_sleeps - reported.back_sleeps),
             ms(current.stage_ns - reported.stage_ns),
             rate(current.stage_sleeps - reported.stage_sleeps),
             rate(current.stage_input_sleeps - reported.stage_input_sleeps),
             rate(current.served - reported.served), ms(current.served_ns - reported.served_ns),
             total_checks, checks_text);
    if (mode == 2) {
        // Section 8.3's line.
        const u64 runs = current.runs - reported.runs;
        std::string served_text;
        for (size_t i = 0; i < current.served_calls.size(); ++i) {
            served_text += fmt::format("{}{} {:.0f}", i == 0 ? "" : ", ", SERVED_NAMES[i],
                                       rate(current.served_calls[i] - reported.served_calls[i]));
        }
        std::string level_text;
        for (size_t i = 0; i < current.level_draws.size(); ++i) {
            level_text += fmt::format("{}{} {:.0f}", i == 0 ? "" : ", ", LEVEL_DRAW_NAMES[i],
                                      rate(current.level_draws[i] - reported.level_draws[i]));
        }
        const u64 packets = current.packets - reported.packets;
        const u64 hashes = current.hashes - reported.hashes;
        LOG_INFO(HW_GPU,
                 "Memory stage ring {:.0f} s (active {}): entries/s pass {:.0f} work {:.0f} level "
                 "{:.0f} neutral {:.0f}; runs/s {:.0f} ({:.1f} entries each, {:.2f} us each on "
                 "GPUMem, {:.2f} us from the lend to the next packet on the back); packets/s "
                 "{:.0f} ({:.0f} B each), taken/s {:.0f}, publishes/s {:.0f}; the back waited for "
                 "the stage {:.0f}/s, {:.1f} ms/s; GPUMem busy {:.1f} ms/s, waited for the front "
                 "{:.1f} ms/s, the back {:.1f} ms/s, runs {:.1f} ms/s; full copies/s {:.0f}, flag "
                 "hand-overs/s {:.0f}; served calls/s ({}); level draws/s ({}); verify hashes "
                 "{} ({} differ, FRITCHII_MEMSTAGE_RING_VERIFY {})",
                 seconds, RingActive() ? "yes" : "no",
                 rate(current.entries[0] - reported.entries[0]),
                 rate(current.entries[1] - reported.entries[1]),
                 rate(current.entries[2] - reported.entries[2]),
                 rate(current.entries[3] - reported.entries[3]), rate(runs),
                 runs != 0 ? static_cast<double>(current.run_entries - reported.run_entries) /
                                 static_cast<double>(runs)
                           : 0.0,
                 per(current.run_ns - reported.run_ns, runs),
                 per(current.back_run_ns - reported.back_run_ns, runs), rate(packets),
                 packets != 0 ? static_cast<double>(current.packet_bytes - reported.packet_bytes) /
                                    static_cast<double>(packets)
                              : 0.0,
                 rate(current.taken - reported.taken),
                 rate(current.publishes - reported.publishes),
                 rate(current.back_wait - reported.back_wait),
                 ms(current.back_wait_ns - reported.back_wait_ns),
                 ms(current.stage_busy_ns - reported.stage_busy_ns),
                 ms(current.stage_wait_ns[0] - reported.stage_wait_ns[0]),
                 ms(current.stage_wait_ns[1] - reported.stage_wait_ns[1]),
                 ms(current.stage_wait_ns[2] - reported.stage_wait_ns[2]),
                 rate(current.copies - reported.copies),
                 rate(current.flag_handovers - reported.flag_handovers), served_text, level_text,
                 hashes, current.hash_mismatches - reported.hash_mismatches,
                 RingVerify() ? "on" : "off");
        {
            // FRITCHII_MEMSTAGE_LIST_DRAWS: what became of the list draws GPUMem's copy issued
            // outside a hard entry. Each level one made its entry the first hard entry of a run
            // (it starts the run); each served one is a run not started.
            std::string list_text;
            u64 list_level = 0;
            for (size_t i = 0; i < current.list_draws.size(); ++i) {
                const u64 n = current.list_draws[i] - reported.list_draws[i];
                if (static_cast<ListDraw>(i) != ListDraw::Served) {
                    list_level += n;
                }
                list_text += fmt::format("{}{} {:.0f}", i == 0 ? "" : ", ", LIST_DRAW_NAMES[i],
                                         rate(n));
            }
            LOG_INFO(HW_GPU,
                     "Memory stage list draws (FRITCHII_MEMSTAGE_LIST_DRAWS={}, window {}) {:.0f} "
                     "s: per second {}; runs started by a list draw {:.0f}/s of {:.0f}/s; the back "
                     "drew {:.0f}/s from the packet's list words buffer",
                     g_list_draws.load(std::memory_order_relaxed) ? 1 : 0,
                     Ahead() != 0 ? "open" : "closed", seconds, list_text, rate(list_level),
                     rate(runs), rate(current.list_drawn - reported.list_drawn));
        }
        if (Nx2Profile::Enabled()) {
            LogRunShape(seconds);
        }
        // Step 7's window (spec 8.3: the stage's lead; 6.4's counters).
        static constexpr std::array<const char*, WindowBuckets> bucket_names{
            "1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-255", "256+"};
        std::string window_text;
        for (size_t i = 0; i < WindowBuckets; ++i) {
            window_text += fmt::format("{}{} {:.0f}", i == 0 ? "" : ", ", bucket_names[i],
                                       rate(current.window[i] - reported.window[i]));
        }
        std::string levels_text;
        for (size_t i = 0; i < current.levels.size(); ++i) {
            levels_text += fmt::format("{}{} {:.0f}", i == 0 ? "" : ", ", LEVEL_REASON_NAMES[i],
                                       rate(current.levels[i] - reported.levels[i]));
        }
        const u64 ahead_max = counters.ring_ahead_max.exchange(0, std::memory_order_relaxed);
        const u32 window = Ahead();
        const auto delta = [&](u64 Snapshot::*member) { return current.*member - reported.*member; };
        // Not at AHEAD 0: everything on it is the window's (level mode has the ring line).
        if (window != 0) {
            LOG_INFO(
                HW_GPU,
                "Memory stage window (FRITCHII_MEMSTAGE_AHEAD={}) {:.0f} s: packets in ring B at "
                "each push {:.1f} on average, {} at most; per second by packets ahead ({}); the "
                "back waited for the stage {:.0f}/s, {:.1f} ms/s; GPUMem waited for the window "
                "{:.1f} ms/s, for room in ring B {:.1f} ms/s (woken by the back's takes "
                "{:.0f}/s), level inside the window {:.1f} ms/s ({}/s; entries made hard by "
                "constant buffer words onto GPU-written data {:.0f}/s); new staging buffers for GPUMem, the stream busy {:.0f}/s; served calls "
                "while packets were ahead {:.0f}/s; sync catch-ups {:.0f}/s ({:.2f} ms/s); join "
                "catch-ups {:.0f}/s; lends handed over at the run's start {:.0f}/s; the back "
                "waited for the cache's lock held by GPUMem {:.0f}/s; GPUMem served the back's "
                "call while waiting for the lock {:.0f}/s; served downloads past the join ring's "
                "reach {:.0f}/s; idle hook runs with the stage busy {:.0f}/s; GPUMem idle work "
                "{:.0f}/s; runs kept lent at the end of the input {:.0f}/s",
                window >= AheadUnlimited ? std::string{"unlimited"} : fmt::format("{}", window),
                seconds,
                packets != 0 ? static_cast<double>(delta(&Snapshot::ahead_sum)) /
                                   static_cast<double>(packets)
                             : 0.0,
                ahead_max, window_text, rate(delta(&Snapshot::back_wait)),
                ms(delta(&Snapshot::back_wait_ns)),
                ms(current.stage_wait_ns[3] - reported.stage_wait_ns[3]),
                ms(current.stage_wait_ns[4] - reported.stage_wait_ns[4]),
                rate(delta(&Snapshot::ring_wakes)),
                ms(current.stage_wait_ns[5] - reported.stage_wait_ns[5]), levels_text,
                rate(delta(&Snapshot::write_levels)), rate(delta(&Snapshot::staging_fallbacks)),
                rate(delta(&Snapshot::served_ahead)),
                rate(delta(&Snapshot::catchups)), ms(delta(&Snapshot::catchup_ns)),
                rate(delta(&Snapshot::join_catchups)), rate(delta(&Snapshot::deferred_lends)),
                rate(delta(&Snapshot::back_lock_waits)), rate(delta(&Snapshot::stage_lock_serves)),
                rate(delta(&Snapshot::joins_unknown)), rate(delta(&Snapshot::hook_flushes)),
                rate(delta(&Snapshot::stage_idles)), rate(delta(&Snapshot::run_input_waits)));
        }
    }
    reported = current;
}

} // namespace VideoCommon
