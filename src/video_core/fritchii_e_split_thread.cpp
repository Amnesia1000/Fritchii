// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// FRITCHII_ESPLIT: the E split's thread (fritchii_e_split_thread.h; work\E_SPLIT_DESIGN.md 2b).
//
// S1, lockstep (FRITCHII_ESPLIT_DEPTH=0). Who owns what:
//   E1 (GpuSplit's back thread): ring A, engine A (registers, dirty flags, draw manager, sink),
//      the pipeline lookup, the records (fritchii_e_split.cpp's pending sets), the hand-over ring
//      below (it writes jobs and `posted`).
//   E2 (this file's thread): E2's view, every job's run, `done`.
// A hand-over: E1 takes a job slot, builds the record (when the work reads E2's view), moves its
// per-call thread state into the job (one follower's, the renderer's fill state), hands a lock it
// holds over, publishes `posted` and waits for `done` to reach the job. E2 takes the job, installs
// the state, applies the record to its view (1 job in 64: the view is compared with the engine),
// runs the work, takes the state back into the job, publishes `done`. E1 installs the state the
// work left and goes on. One of the two threads runs at a time and each piece of work runs on the
// same objects in the same order as on one thread: the result is today's.
//
// S2, async (FRITCHII_ESPLIT_DEPTH=N > 0). E1 posts draw tails and binds and goes on, up to N
// draw tails ahead of E2. The rules that keep the two threads off each other's state:
//   - Everything that is not an async job is still a lockstep hand-over or a join, and both first
//     drain E2 (E1 waits for `done` to reach `posted`).
//   - The buffer flags (IndexBuffer, VertexBuffers, VertexBuffer0..31) are E2's: E1 takes the
//     ones raised on its engine at each async draw record and clears them there (the record
//     carries them, E2 ORs them into its view: the take of fritchii_e_split.h). Wherever E2's
//     work runs on the engine (a lockstep hand-over, a join, E1's own lend or push) the view's
//     flags are copied to the engine first and back after (SyncFlagsToEngine / ToView).
//   - An async fill reads only E2's view (SetInputsOverride, the bindings, a list draw's indirect
//     parameters, the render target formats); the repack layout's VertexInput raise goes to E1
//     (DeferVertexInput, raised at E1's next draw, join or push).
//   - R's draw packet is packed by E1 at the record (OneFollower::PackDraw: the spans are E1's
//     registers at the draw) and pushed by E2 after the fill (PushDrawPacked): R's packet queue
//     keeps one producer, E2, while E2 is behind; E1 drains E2 before any lend or push of its own.
//   - A lend E2 makes while E1 runs ahead holds E1 at its next safe point (each ring entry, and
//     every wait for E2 counts as held): R's lent call then sees no E1 work in flight. Such a lend
//     packs no new spans for R from E1 (its view stays at E2's place: OneFollower::Lend skips the
//     pack on E2 in S2) and the view keeps its own flags after it (EndLend's sync skipped).
//   - E2's position: async jobs carry the ring position of their entry; GpuSplit::
//     BackEntryPosition returns it on E2; `consumed` is stored by an async job (the front's waits
//     and ring A's room then follow E2), and at a batch's end after the drain.
// Async draws need: rule 3's pipeline (FRITCHII_ONEF_E_PIPELINE), no texel buffers, ring B by
// value, no marks ahead, no inline index bytes, no handle program keep on a pipeline with handle
// programs, no price timers, a record (no seed due). The rest go lockstep.
//
// Waits: both sides poll first (the haltpoll rule, Common::NextPollWindow) and then sleep on a
// wake word (std::atomic::wait), with a "sleeping" flag each so a publish wakes only a sleeper.
//
// S4, async constant buffer words (FRITCHII_ESPLIT_CB_ASYNC, with DEPTH > 0). E1 runs a constant
// buffer entry itself: the register half (ConsumeSink, CB_DATA through the shadow RAM, the dirty
// flags, a deferred draw the first word issues, which takes E1's own draw path, the offset
// advance). Each memory half (Maxwell3D::WriteConstantWords) goes into an open record by value
// (CbTarget: CB_ADDRESS, CB_SIZE, the offset, the pools; the words' values after the shadow RAM),
// with a running number. The record stays open across entries and is posted (no wait) before any
// other hand-over, drain, join, push or wait of E1's, at a batch's end, or when full. Why the
// result is lockstep's:
//   - E2 applies the writes in ring order: one queue, records in seq order, writes numbered
//     (ESplitCbOrder). Everything before a write in ring order (fills reading the old words,
//     binds, other E2 entries) was posted before it, everything after it is posted after it.
//   - The half reads nothing E1 changes later: no register (the target is by value), not E's
//     pools (E1 tested them), not one follower's E state. What it reads (memory, the page table,
//     the buffer cache, the invalidation accumulator) is E2's and at E2's place.
//   - E1 reads no game memory outside a join (E1Touch, and ESplitCbRead with writes queued);
//     every join drains E2 after posting the open record. A guest-visible signal (semaphores,
//     sync points, reports, fences) is an E2 entry or a join: after the writes.
//   - The front reads written ranges after a wait on `consumed`, stored only at E2's place (a
//     Store record after the writes, or after a drain; FrontAheadOfE2).
//   - The stream-write guard acts (R drained, the next draw lent) only on texture or shader pages
//     or E's pools: E1 tests those when it queues a write (StreamPagesClear) and makes such a
//     write itself in a join, so the guard runs at today's place. A page marked between E1's
//     test and E2's write is ESplitCbGuard.
//   - The pipeline lookup on E1 runs before E2's flush of queued writes: they are on no shader
//     page (above). Writes from anything else (a lockstep hand-over, a join, a push or lend, an
//     E2 lend) make the next draw's flush owed: it runs before the lookup, E2 drained first
//     (FlushPlace). A flush on E2 marks the view's buffer flags (the draw tail sets the inputs).
//   - The values: the same code computes them from the same registers at the same ring place;
//     1 write in 64 is read back after E2 wrote it (ESplitCbBytes).

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

#include <fmt/format.h>

#include "common/logging.h"
#include "common/spin_recursive_mutex.h"
#include "common/thread.h"
#include "common/thread_cpu_clock.h"
#include "core/core.h"
#include "video_core/back_read_audit.h"
#include "video_core/fritchii_e_split.h"
#include "video_core/fritchii_e_split_thread.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_profile.h"
#include "video_core/one_follower.h"
#include "video_core/rasterizer_interface.h"

namespace VideoCommon::ESplit {

namespace {

namespace OneF = VideoCommon::OneFollower;
namespace Profile = VideoCommon::Nx2Profile;
using Maxwell3D = Tegra::Engines::Maxwell3D;

void Pause() noexcept {
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

constexpr u64 Slots = 64;          ///< hand-over ring
constexpr int MaxDepth = 48;       ///< FRITCHII_ESPLIT_DEPTH's cap (draw tails ahead)
constexpr u64 SampleMask = 63;     ///< 1 hand-over in 64 compared (or hashed)
constexpr size_t HandCount = static_cast<size_t>(Hand::Count);
constexpr size_t JoinCount = static_cast<size_t>(Join::Count);
constexpr size_t TouchCount = static_cast<size_t>(Touch::Count);
constexpr std::array<const char*, HandCount> HandNames{
    "entries", "flushes", "draw tails", "binds", "stores", "cb records", "queued entries",
    "raster ops"};
constexpr std::array<const char*, JoinCount> JoinNames{
    "entries", "decisions", "conditions", "idle",  "pushes", "lockstep",
    "cb writes", "lookups", "flushes",  "tails", "seeds"};

/// S4: why a constant buffer write went to a join instead of E2's queue.
enum class CbJoin : u8 { Off, Debug, Size, Pages, Sparse, Unmapped, Guarded, Count };
constexpr size_t CbJoinCount = static_cast<size_t>(CbJoin::Count);
constexpr std::array<const char*, CbJoinCount> CbJoinNames{
    "not async", "fast draw count", "too long", "across a page",
    "sparse",    "unmapped",                         "texture, shader pages or pools"};
/// A record's room for constant buffer writes (each a CbSegment and its words).
constexpr size_t CbRoom = 4096;
struct CbSegment {
    CbTarget target;
    u64 seq; ///< E1's running number of the write (ESplitCbOrder)
    u32 bytes;
    u8 block;
    u8 sample; ///< ESplitCbBytes: read back after the write
};
static_assert(sizeof(CbSegment) % 8 == 0 && std::is_trivially_copyable_v<CbSegment>);
/// The record's payload.
struct CbRecord {
    Maxwell3D* engine;
    u32 used;
    u32 segments;
};
[[nodiscard]] constexpr u32 CbPadded(u32 bytes) noexcept {
    return (bytes + 7) & ~u32{7};
}
constexpr std::array<const char*, TouchCount> TouchNames{"cache lock", "memory read",
                                                         "memory write"};

/// The queued mode's join entries by kind (E1's, per 10 s window): open addressing on the key.
struct JoinKindSlot {
    u64 key;         ///< 0: empty
    u64 count;
    const void* function;
    u32 kind;
    u32 engine;
    u32 method_or_tag;
};
constexpr size_t JoinKindSlots = 256;
std::array<JoinKindSlot, JoinKindSlots> g_join_kinds{};
u64 g_join_kinds_lost{};
constexpr std::array<const char*, 4> EngineNames{"3D", "compute", "other engine", "no engine"};

struct Job {
    void (*function)(void*);
    void* argument; ///< lockstep: E1's; async: the payload below
    u64 seq;
    u64 position; ///< ring A position of the entry the work belongs to
    u64 hash;     ///< async draw, sampled: E1's hash of what the fill reads
    HandInfo info;
    RecordKind record_kind;
    u32 record_bytes;
    bool async;
    bool sample;  ///< lockstep: compare the view; async draw: compare the hash
    bool view_ok; ///< E2: the fill may read the view
    OneF::EngineTls tls;
    alignas(16) std::array<u8, FillStateBytes> fill_state;
    alignas(16) std::array<u8, AsyncPayloadBytes> payload;
};

/// Per-window counters. E1's: plain (E1 writes and reports). E2's: single writer, relaxed.
struct E1Counts {
    std::array<u64, HandCount> hands{};
    std::array<u64, JoinCount> joins{};
    std::array<u64, TouchCount> touches{};
    u64 wait_ticks{}; ///< E1 waiting for E2 (lockstep runs, drains, room)
    u64 hold_ticks{}; ///< E1 held for E2's lends
    u64 seeds{};
    u64 lost{};
    u64 samples{};
    u64 async_draws{};
    u64 visible{};
    u64 front_ahead{};
    u64 depth_sum{}; ///< draw tails outstanding at each async post
    u64 lookups{};      ///< E1's lookups (thread mode)
    u64 lookup_reads{}; ///< their game memory reads (shader code), allowed
    u64 tail_reasons[4]{}; ///< draw tails that stayed with E1: no fit, record lost, a join, S1
    // S4
    u64 cb_records{};
    u64 cb_segments{};
    u64 cb_bytes{};
    std::array<u64, CbJoinCount> cb_joins{};
    u64 cb_reads{};      ///< ESplitCbRead
    u64 flush_before{};  ///< a draw's flush before the lookup, E2 idle
    u64 flush_drained{}; ///< ... owed, E2 drained first
    u64 flush_tail{};    ///< ... at the head of the draw's tail (after the lookup)
};
struct E2Counts {
    std::atomic<u64> job_ticks{};
    std::atomic<u64> wait_ticks{};
    std::atomic<u64> view_differs{};
    std::atomic<u64> compares{};
    std::atomic<u64> view_fills{};
    std::atomic<u64> after_checks{};
    std::atomic<u64> after_differs{};
    std::atomic<u64> order{};
    std::atomic<u64> hash_checks{};
    std::atomic<u64> hash_differs{};
    std::atomic<u64> holds{};
    std::atomic<u64> hold_spin_ticks{};
    std::atomic<u64> room_waits{};
    std::atomic<u64> late_flushes{};
    std::atomic<u64> refused{};
    // S4
    std::atomic<u64> cb_order{};
    std::atomic<u64> cb_checks{};
    std::atomic<u64> cb_differs{};
    std::atomic<u64> cb_guard{};
    std::atomic<u64> async_writes{}; ///< ESplitAsyncWrite
};

void Bump(std::atomic<u64>& counter, u64 amount = 1) noexcept {
    counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}
u64 TakeCount(std::atomic<u64>& counter) noexcept {
    return counter.exchange(0, std::memory_order_relaxed);
}

/// S4: the cached writes E's engine's memory manager has taken so far (FlushPlace's owed test).
u64 WriteMark() noexcept {
    if (Maxwell3D* const engine = OneF::Engine()) {
        return engine->GetMemoryManager().CachedWriteCount();
    }
    return 0;
}

struct SlotBuffers {
    std::unique_ptr<u8[]> record;
    std::unique_ptr<u8[]> packet;
    std::unique_ptr<u8[]> cb; ///< S4: a constant buffer record's writes (CbRoom)
};

struct Shared {
    // E1 -> E2
    alignas(128) std::atomic<u64> posted{};
    std::atomic<bool> e2_sleeping{};
    std::atomic<u32> e2_wake{};
    std::atomic<bool> stop{};
    // E2 -> E1
    alignas(128) std::atomic<u64> done{};
    std::atomic<u64> draws_done{}; ///< async draw tails E2 finished
    std::atomic<u64> cb_done{};    ///< S4: the last constant buffer write E2 applied
    /// S4: an async job other than a cb record left cached writes (the next flush is owed).
    std::atomic<bool> async_wrote{};
    std::atomic<bool> e1_sleeping{};
    std::atomic<u32> e1_wake{};
    /// E1 is at a safe point (waiting for E2, or held): an E2 lend may run.
    alignas(128) std::atomic<bool> e1_parked{};
    /// E2's async fills raised this flag (index + 1) for E1's engine (VertexInput).
    std::atomic<u32> back_raise{};
    alignas(128) std::array<Job, Slots> jobs{};
    std::array<SlotBuffers, Slots> buffers;
    Core::System* system{};
    void (*adopt)(void*){};
    void* adopt_argument{};
    std::thread::id e2_id{};
    std::thread::id e1_id{};
    std::jthread thread;
    std::atomic<bool> e2_ready{};
    /// S3: E2's hard affinity (0: leave it), and E2's thread CPU time (ns, E2 writes it).
    u64 e2_mask{};
    std::atomic<u64> e2_cpu_ns{};
};

Shared g;
E1Counts g_e1;
E2Counts g_e2;
// E1's own
u64 g_next_seq = 1;
u64 g_draws_posted{};
int g_depth{};
Job* g_open{}; ///< a slot BeginAsync* gave, not posted yet
u32 g_rng = 0x9e3779b9u;
u64 g_batches{};
u64 g_since_ns{};
u64 g_since_ticks{};
u64 g_touch_logged{};
u64 g_visible_logged{};
u64 g_front_logged{};
u64 g_e1_cpu_last{};
u64 g_e2_cpu_last{};
// S4, E1's: the open constant buffer record, the writes numbered, the flush owed
Job* g_cb_open{};
Maxwell3D* g_cb_engine{};
u32 g_cb_used{};
u32 g_cb_segments{};
u64 g_cb_staged{};
bool g_flush_owed{true};
bool g_flush_async{};
u64 g_join_mark{}; ///< the cached write count at the outer join's start (E2 drained)
u64 g_push_mark{}; ///< ... at E1's push or lend (E2 drained)
std::chrono::nanoseconds g_e1_poll{std::chrono::microseconds{50}};
std::chrono::nanoseconds g_hold_poll{std::chrono::microseconds{20}};
// E2's own
u64 g_e2_last_seq{};
u64 g_e2_last_position{};
u64 g_order_logged{};
u64 g_hash_logged{};
u64 g_refused_logged{};
Job* g_current{}; ///< E2: the job it runs
u64 g_e2_cb_seq{};
u64 g_cb_order_logged{};
u64 g_cb_bytes_logged{};
u64 g_cb_guard_logged{};
std::chrono::nanoseconds g_e2_poll{std::chrono::microseconds{20}};

/// E1 polls up to 200 us for a lockstep hand-over (the work is a few us; a sleep and a wake
/// would cost more than the work), up to 20 us in S2 (where it waits only for room or a drain:
/// sleeping waits, so the rest of the machine does not grow). E2 polls up to 50 us for the next
/// job in S1, 20 us in S2.
constexpr std::chrono::nanoseconds E1PollLimit{std::chrono::microseconds{200}};
constexpr std::chrono::nanoseconds AsyncPollLimit{std::chrono::microseconds{20}};
constexpr std::chrono::nanoseconds E2PollLimit{std::chrono::microseconds{50}};

[[nodiscard]] bool Sample() noexcept {
    u32 x = g_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rng = x;
    return (x & SampleMask) == 0;
}

/// Waits until `ready()`: polls for `poll`, then sleeps on `wake` with `sleeping` set (the other
/// side bumps `wake` after its store when it sees `sleeping`; both seq_cst, so one of the two
/// sees the other). The poll adapts (haltpoll). Returns the ticks waited.
template <typename Ready>
u64 WaitFor(Ready&& ready, std::atomic<u32>& wake, std::atomic<bool>& sleeping,
            std::chrono::nanoseconds& poll, std::chrono::nanoseconds limit) {
    if (ready()) {
        return 0;
    }
    const u64 start_ticks = Profile::Ticks();
    const auto start = std::chrono::steady_clock::now();
    const auto poll_end = start + poll;
    u32 spins = 0;
    while (!ready()) {
        Pause();
        if ((++spins & 63) == 0 && std::chrono::steady_clock::now() >= poll_end) {
            while (!ready()) {
                const u32 seen = wake.load(std::memory_order_seq_cst);
                sleeping.store(true, std::memory_order_seq_cst);
                if (!ready()) {
                    wake.wait(seen, std::memory_order_seq_cst);
                }
                sleeping.store(false, std::memory_order_relaxed);
            }
            break;
        }
    }
    const auto waited = std::chrono::steady_clock::now() - start;
    poll = Common::NextPollWindow(poll, std::chrono::duration_cast<std::chrono::nanoseconds>(waited),
                                  limit);
    return Profile::Ticks() - start_ticks;
}

void WakeE1() noexcept {
    if (g.e1_sleeping.load(std::memory_order_seq_cst)) {
        g.e1_wake.fetch_add(1, std::memory_order_seq_cst);
        g.e1_wake.notify_all();
    }
}

std::string Site(const void* address) {
#ifdef _WIN32
    if (address != nullptr) {
        HMODULE module{};
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               static_cast<LPCSTR>(address), &module)) {
            char name[MAX_PATH] = "?";
            GetModuleFileNameA(module, name, MAX_PATH);
            const char* base = std::strrchr(name, '\\');
            return fmt::format("{}+{:#x}", base != nullptr ? base + 1 : name,
                               reinterpret_cast<uintptr_t>(address) -
                                   reinterpret_cast<uintptr_t>(module));
        }
    }
#endif
    return fmt::format("{}", address);
}

std::string Stack() {
#ifdef _WIN32
    void* frames[10];
    const USHORT count = RtlCaptureStackBackTrace(2, 10, frames, nullptr);
    std::string text;
    for (USHORT i = 0; i < count; ++i) {
        text += (i == 0 ? "" : " < ") + Site(frames[i]);
    }
    return text;
#else
    return "(no stack)";
#endif
}

/// BackReadAudit's watcher: E's game memory reads.
void ReadTouch(const void* caller) {
    if (t_role == 1 && t_join == 0 && g_thread) [[unlikely]] {
        if (t_lookup) {
            ++g_e1.lookup_reads; // shader code, no shader page write pending (LookupScope)
            return;
        }
        NoteTouchImpl(Touch::MemoryRead, caller);
        // S4: with constant buffer writes queued the read may see memory before them.
        if (g_cb_open != nullptr || g_cb_staged > g.cb_done.load(std::memory_order_acquire)) {
            ++g_e1.cb_reads;
        }
    }
}

// --- E2 --------------------------------------------------------------------------------------------

void RunJob(Job& job) {
    const Profile::Scope zone{Profile::Zone::ESplitJob, Profile::Enabled(), false};
    const u64 start = Profile::Ticks();
    // E2Order: hand-overs run in the order E1 made them, at non-decreasing ring positions.
    if (job.seq != g_e2_last_seq + 1 || job.position < g_e2_last_position) [[unlikely]] {
        Bump(g_e2.order);
        if (++g_order_logged <= 16) {
            LOG_ERROR(HW_GPU,
                      "E split thread: E2Order: job {} at ring position {:#x} after job {} at "
                      "{:#x} ({} since boot)",
                      job.seq, job.position, g_e2_last_seq, g_e2_last_position, g_order_logged);
        }
    }
    g_e2_last_seq = job.seq;
    g_e2_last_position = job.position;
    t_e2_position = job.position;
    // A queued write or bind runs on what it carries: no per-call state to move.
    const bool moves_state = !job.async || job.info.hand == Hand::DrawTail;
    if (moves_state) {
        OneF::PutEngineTls(job.tls);
        if (g_fill_state.install != nullptr) {
            g_fill_state.install(job.fill_state.data(), false);
        }
    }
    job.view_ok = false;
    if (job.record_kind != RecordKind::None) {
        if (const Maxwell3D* const engine = OneF::Engine()) {
            const u8* const record = g.buffers[job.seq % Slots].record.get();
            // Async: the engine is past E2's place, nothing to compare with (the hash instead).
            const bool compare = job.sample && !job.async;
            const u64 differs = ThreadConsume(job.record_kind, record, *engine, compare,
                                              job.info.draw, job.info.list_args);
            if (compare && job.record_kind == RecordKind::Record && ThreadView().seeded) {
                Bump(g_e2.compares);
                Bump(g_e2.view_differs, differs);
                job.view_ok = differs == 0;
            }
            if (job.async && job.info.draw) {
                job.view_ok = ThreadView().seeded && ThreadView().engine == engine;
                if (job.sample) {
                    Bump(g_e2.hash_checks);
                    const u64 hash = HashView(job.info.list_args);
                    if (hash != job.hash) [[unlikely]] {
                        Bump(g_e2.hash_differs);
                        if (++g_hash_logged <= 16) {
                            LOG_ERROR(HW_GPU,
                                      "E split thread: ESplitHash: the view's hash {:#x} differs "
                                      "from E1's {:#x} at ring position {:#x} ({} since boot)",
                                      hash, job.hash, job.position, g_hash_logged);
                        }
                    }
                }
            }
        }
    }
    g_current = &job;
    // S4: async draw tails, binds and stores leave no cached writes (ESplitAsyncWrite, must be
    // 0): FlushPlace's "not owed" counts on it, since E1 may decide before E2 ran them. One that
    // does still makes a later flush owed.
    const bool watch_writes = job.async && job.info.hand != Hand::CbWords;
    const u64 mark = watch_writes ? WriteMark() : 0;
    job.function(job.argument);
    if (watch_writes && WriteMark() != mark) [[unlikely]] {
        Bump(g_e2.async_writes);
        g.async_wrote.store(true, std::memory_order_release);
    }
    g_current = nullptr;
    if (job.info.cb_entry) {
        const u64 words = job.info.cb_words != 0 ? job.info.cb_words
                          : job.info.cb_size != nullptr ? *job.info.cb_size
                                                        : 0;
        ThreadCbAdvance(words);
    }
    // Lockstep: the state goes back to E1 with the job. Async: E1 moved on; E2's copy is dropped
    // (the next job installs its own).
    if (moves_state) {
        OneF::TakeEngineTls(job.tls);
        if (g_fill_state.capture != nullptr) {
            g_fill_state.capture(job.fill_state.data(), false);
        }
    }
    if (job.async && job.info.hand == Hand::DrawTail) {
        g.draws_done.store(g.draws_done.load(std::memory_order_relaxed) + 1,
                           std::memory_order_release);
    }
    Bump(g_e2.job_ticks, Profile::Ticks() - start);
}

/// E2, S4, 1 write in 64: the memory after E2's write holds the words (what lockstep writes).
void CheckCbBytes(Maxwell3D& engine, const CbSegment& segment, const u32* values) {
    auto& memory = engine.GetMemoryManager();
    const GPUVAddr at = segment.target.address + segment.target.offset;
    if (segment.bytes == 0 || !memory.IsFullyMappedRange(at, segment.bytes)) {
        return;
    }
    std::array<u8, CbRoom> read;
    memory.ReadBlockUnsafe(at, read.data(), segment.bytes);
    Bump(g_e2.cb_checks);
    if (std::memcmp(read.data(), values, segment.bytes) != 0) [[unlikely]] {
        Bump(g_e2.cb_differs);
        if (++g_cb_bytes_logged <= 16) {
            LOG_ERROR(HW_GPU,
                      "E split thread: ESplitCbBytes: {} bytes at {:#x}+{:#x} read back different "
                      "after E2's write (write {}, {} since boot)",
                      segment.bytes, segment.target.address, segment.target.offset, segment.seq,
                      g_cb_bytes_logged);
        }
    }
}

/// E2, S4: a constant buffer record, its writes in E1's order.
void RunCbRecord(void* raw) {
    CbRecord record;
    std::memcpy(&record, raw, sizeof(record));
    const u8* at = g.buffers[g_current->seq % Slots].cb.get();
    t_async_cb = true;
    for (u32 i = 0; i < record.segments; ++i) {
        CbSegment segment;
        std::memcpy(&segment, at, sizeof(segment));
        const auto* const values = reinterpret_cast<const u32*>(at + sizeof(CbSegment));
        if (segment.seq != g_e2_cb_seq + 1) [[unlikely]] {
            Bump(g_e2.cb_order);
            if (++g_cb_order_logged <= 16) {
                LOG_ERROR(HW_GPU,
                          "E split thread: ESplitCbOrder: cb write {} applied after {} ({} since "
                          "boot)",
                          segment.seq, g_e2_cb_seq, g_cb_order_logged);
            }
        }
        g_e2_cb_seq = segment.seq;
        if (segment.block != 0) {
            record.engine->WriteConstantWords(segment.target, values, segment.bytes, true);
        } else {
            // ProcessCBMultiData's zone (the block write opens its own).
            const Profile::FineScope zone{Profile::Zone::CbUpload};
            record.engine->WriteConstantWords(segment.target, values, segment.bytes, false);
        }
        if (segment.sample != 0) [[unlikely]] {
            CheckCbBytes(*record.engine, segment, values);
        }
        at += sizeof(CbSegment) + CbPadded(segment.bytes);
    }
    t_async_cb = false;
    g.cb_done.store(g_e2_cb_seq, std::memory_order_release);
}

void E2Main(std::stop_token) {
    Common::SetCurrentThreadName("GPUBackE2");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();
    g.system->RegisterHostThread();
    if (g.adopt != nullptr) {
        g.adopt(g.adopt_argument); // GpuSplit: OnBackThread
    }
    if (g.e2_mask != 0) {
        Common::SetCurrentThreadCpuMask(g.e2_mask); // FRITCHII_ESPLIT_PIN (S3)
    }
    OneF::MarkEngineHelperThread();
    ThreadMark();
    t_role = 2;
    g.e2_id = std::this_thread::get_id();
    g.e2_ready.store(true, std::memory_order_release);
    g.e2_ready.notify_all();
    u64 next = 1;
    while (true) {
        if (g.posted.load(std::memory_order_acquire) < next) {
            if (g.stop.load(std::memory_order_acquire)) {
                break;
            }
            const Profile::Scope zone{Profile::Zone::ESplitWait, Profile::Enabled(), false};
            Bump(g_e2.wait_ticks,
                 WaitFor(
                     [next] {
                         return g.posted.load(std::memory_order_acquire) >= next ||
                                g.stop.load(std::memory_order_acquire);
                     },
                     g.e2_wake, g.e2_sleeping, g_e2_poll,
                     g_async ? AsyncPollLimit : E2PollLimit));
            continue;
        }
        RunJob(g.jobs[next % Slots]);
        g.done.store(next, std::memory_order_seq_cst);
        WakeE1();
        if ((next & 1023) == 0) {
            g.e2_cpu_ns.store(Common::CurrentThreadCpuTimeNs().value_or(0),
                              std::memory_order_relaxed);
            Profile::Report("GPU E2 thread");
        }
        ++next;
    }
}

/// E2's nesting of holds (a guard inside a lend's way, say): only the outermost holds.
thread_local u32 t_hold_depth{};

/// E2, S2: E1 held at a safe point for an E2 lend.
void HoldE1() noexcept {
    if (t_hold_depth++ != 0) {
        return;
    }
    g_hold.store(1, std::memory_order_seq_cst);
    const u64 start = Profile::Ticks();
    u32 spins = 0;
    while (!g.e1_parked.load(std::memory_order_seq_cst)) {
        if (++spins < 4096) {
            Pause();
        } else {
            std::this_thread::yield();
        }
    }
    Bump(g_e2.holds);
    Bump(g_e2.hold_spin_ticks, Profile::Ticks() - start);
}

void ReleaseE1() noexcept {
    if (--t_hold_depth != 0) {
        return;
    }
    g_hold.store(0, std::memory_order_seq_cst);
    g.e1_wake.fetch_add(1, std::memory_order_seq_cst);
    g.e1_wake.notify_all();
}

// --- E1 --------------------------------------------------------------------------------------------

void Publish(u64 seq) {
    g.posted.store(seq, std::memory_order_seq_cst);
    if (g.e2_sleeping.load(std::memory_order_seq_cst)) {
        g.e2_wake.fetch_add(1, std::memory_order_seq_cst);
        g.e2_wake.notify_all();
    }
}

/// E1 waits for E2 (a safe point: an E2 lend may run meanwhile), then holds if E2 asks.
template <typename Ready>
void E1Wait(Ready&& ready) {
    if (ready()) {
        return;
    }
    const Profile::Scope zone{Profile::Zone::ESplitHandOver, Profile::Enabled(), false};
    g.e1_parked.store(true, std::memory_order_seq_cst);
    g_e1.wait_ticks += WaitFor(ready, g.e1_wake, g.e1_sleeping, g_e1_poll,
                               g_async ? AsyncPollLimit : E1PollLimit);
    g.e1_parked.store(false, std::memory_order_seq_cst);
    if (g_hold.load(std::memory_order_seq_cst) != 0) [[unlikely]] {
        HoldHereImpl();
    }
}

void WaitDone(u64 seq) {
    E1Wait([seq] { return g.done.load(std::memory_order_acquire) >= seq; });
}

void WaitDrained() {
    PostCbWrites(); // S4: the open record first, so "drained" includes it
    WaitDone(g_next_seq - 1);
}

/// E1 in S2 before E2's work runs on the engine (E2 drained): VertexInput raises, the flags.
void ToEngine() {
    if (Maxwell3D* const engine = OneF::Engine()) {
        ApplyBackRaises(*engine);
    }
    SyncFlagsToEngine();
}

void ToView() {
    SyncFlagsToView();
}

Job& TakeSlot(bool draw_tail) {
    PostCbWrites(); // S4: the open record keeps its place in ring order
    E1Wait([draw_tail] {
        const u64 done = g.done.load(std::memory_order_acquire);
        if (g_next_seq - 1 - done >= Slots - 1) {
            return false;
        }
        return !draw_tail ||
               g_draws_posted - g.draws_done.load(std::memory_order_acquire) <
                   static_cast<u64>(g_depth);
    });
    const u64 seq = g_next_seq++;
    Job& job = g.jobs[seq % Slots];
    job.seq = seq;
    job.position = GpuSplit::Get().BackEntryPosition();
    job.record_kind = RecordKind::None;
    job.record_bytes = 0;
    job.sample = false;
    job.async = false;
    job.hash = 0;
    return job;
}

void SetDepth(int depth) {
    const bool async = depth > 0;
    if (async != g_async) {
        WaitDrained();
        if (!async) {
            ToEngine(); // the lockstep fills read the engine's flags again
        } else {
            ToView(); // from now the engine holds only E1's raises between transfers
        }
        g_async = async;
        LOG_INFO(HW_GPU, "E split thread: FRITCHII_ESPLIT_DEPTH={} ({})", depth,
                 async ? "queued: E1 appends E2's work and waits only at joins"
                       : "lockstep: E1 waits for each hand-over (the check mode)");
    }
    g_depth = depth;
}

int PinMode() noexcept {
    static const int mode = std::clamp(
        VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_esplit_pin, "FRITCHII_ESPLIT_PIN"),
        0, 3);
    return mode;
}

/// E1's hard affinity now (group 0), 0 when unknown.
u64 OwnMask() noexcept {
#ifdef _WIN32
    GROUP_AFFINITY affinity{};
    if (GetThreadGroupAffinity(GetCurrentThread(), &affinity) && affinity.Group == 0) {
        return static_cast<u64>(affinity.Mask);
    }
#endif
    return 0;
}

/// S3, E1 before E2 starts: E2's mask; E1 moves where the mode says once E2 runs (Place).
void PlanPlacement(u64& e1_mask, bool& e1_unpin) {
    const int mode = PinMode();
    const u64 back = OwnMask(); // E's pin (PinnedRole::GpuBack), or the process mask unpinned
    e1_mask = 0;
    e1_unpin = false;
    if (mode == 0 || back == 0) {
        return;
    }
    switch (mode) {
    case 1:
        g.e2_mask = back;
        e1_unpin = true;
        break;
    case 2:
        if (std::popcount(back) > 1) {
            const u64 high = u64{1} << (63 - std::countl_zero(back));
            g.e2_mask = high;
            e1_mask = back & ~high;
        }
        break;
    case 3:
        g.e2_mask = back;
        break;
    default:
        break;
    }
    LOG_INFO(HW_GPU, "E split thread: FRITCHII_ESPLIT_PIN={}: E's core {:#x}, E2 {:#x}, E1 {}",
             mode, back, g.e2_mask,
             e1_unpin       ? std::string{"unpinned (performance cores)"}
             : mode == 3    ? std::string{"on R's core (PinnedRole::GpuMem)"}
             : e1_mask != 0 ? fmt::format("{:#x}", e1_mask)
                            : std::string{"unchanged"});
}

void Report(u64 now) {
    const double window = static_cast<double>(now - g_since_ns) / 1e9;
    const double ns_per_tick =
        static_cast<double>(now - g_since_ns) /
        static_cast<double>((std::max)(Profile::Ticks() - g_since_ticks, u64{1}));
    const auto rate = [window](u64 count) { return static_cast<double>(count) / window; };
    const auto ms = [&](u64 ticks) {
        return static_cast<double>(ticks) * ns_per_tick / 1e6 / window;
    };
    const E1Counts c = g_e1;
    g_e1 = {};
    const u64 job_ticks = TakeCount(g_e2.job_ticks);
    const u64 e2_wait = TakeCount(g_e2.wait_ticks);
    const u64 view_differs = TakeCount(g_e2.view_differs);
    const u64 compares = TakeCount(g_e2.compares);
    const u64 view_fills = TakeCount(g_e2.view_fills);
    const u64 after_checks = TakeCount(g_e2.after_checks);
    const u64 after_differs = TakeCount(g_e2.after_differs);
    const u64 order = TakeCount(g_e2.order);
    const u64 hash_checks = TakeCount(g_e2.hash_checks);
    const u64 hash_differs = TakeCount(g_e2.hash_differs);
    const u64 holds = TakeCount(g_e2.holds);
    const u64 hold_spin = TakeCount(g_e2.hold_spin_ticks);
    const u64 room_waits = TakeCount(g_e2.room_waits);
    const u64 late_flushes = TakeCount(g_e2.late_flushes);
    const u64 refused = TakeCount(g_e2.refused);
    u64 hands = 0;
    std::string hand_text;
    for (size_t i = 0; i < HandCount; ++i) {
        hands += c.hands[i];
        hand_text += fmt::format("{}{} {:.0f}", i == 0 ? "" : ", ", HandNames[i], rate(c.hands[i]));
    }
    u64 joins = 0;
    std::string join_text;
    for (size_t i = 0; i < JoinCount; ++i) {
        joins += c.joins[i];
        join_text += fmt::format("{}{} {:.0f}", i == 0 ? "" : ", ", JoinNames[i], rate(c.joins[i]));
    }
    const double e1_wait = ms(c.wait_ticks);
    const double e2_busy = ms(job_ticks);
    const double cost = e1_wait - e2_busy;
    const std::string depth = g_async ? fmt::format("{}, async", g_depth) : "0, lockstep";
    LOG_INFO(HW_GPU,
             "E split thread (FRITCHII_ESPLIT=1, depth {}) 10 s: hand-overs {:.0f}/s ({}); E2 "
             "busy {:.1f} ms/s, E2 waiting {:.1f} ms/s; E1 waiting for E2 {:.1f} ms/s{}; joins "
             "{:.0f}/s ({})",
             depth, rate(hands), hand_text, e2_busy, ms(e2_wait), e1_wait,
             g_async ? std::string{}
                     : fmt::format(" = E2's runs + hand-over cost {:.1f} ms/s ({:.2f} us each)",
                                   cost,
                                   hands != 0 ? cost * 1000.0 * window / static_cast<double>(hands)
                                              : 0.0),
             rate(joins), join_text);
    const u64 touches = c.touches[0] + c.touches[1] + c.touches[2];
    LOG_INFO(HW_GPU,
             "E split thread 10 s: checks (must be 0): ESplitView {} in {} compares (1 lockstep "
             "hand-over in 64), ESplitInputs {} in {} fills from the view, E1Touch {} ({} {}, {} "
             "{}, {} {}), E2Order {}, E1Visible {}, FrontAheadOfE2 {}, ESplitHash {} in {} async "
             "draws hashed, E2Refused {}; seeds {}, untracked hand-overs {}",
             view_differs, compares, after_differs, after_checks, touches, TouchNames[0],
             c.touches[0], TouchNames[1], c.touches[1], TouchNames[2], c.touches[2], order,
             c.visible, c.front_ahead, hash_differs, hash_checks, refused, c.seeds, c.lost);
    LOG_INFO(HW_GPU,
             "E split thread 10 s: lookups {:.0f}/s, their shader code reads {:.0f}/s (allowed: no "
             "shader page write pending; lookup joins in the join counts); draw tails E1 ran "
             "itself: no async fit {:.0f}/s, view not seedable {:.0f}/s, S1 {:.0f}/s",
             rate(c.lookups), rate(c.lookup_reads), rate(c.tail_reasons[0]),
             rate(c.tail_reasons[1]), rate(c.tail_reasons[3]));
    LOG_INFO(HW_GPU,
             "E split thread 10 s: async draw tails {:.0f}/s (mean {:.1f} ahead of E2 when "
             "posted), E2 lends holding E1 {:.0f}/s (E2 waited {:.2f} ms/s for E1 to park, E1 "
             "held {:.1f} ms/s), ring B room waits {:.0f}/s, flushes after E1's lookup with work "
             "{:.0f}/s; fills from E2's view {:.0f}/s",
             rate(c.async_draws),
             c.async_draws != 0 ? static_cast<double>(c.depth_sum) / static_cast<double>(c.async_draws)
                                : 0.0,
             rate(holds), ms(hold_spin), ms(c.hold_ticks), rate(room_waits), rate(late_flushes),
             rate(view_fills));
    // S4: constant buffer records, the writes that joined by cause, the checks, the flushes.
    u64 cb_joins = 0;
    std::string cb_join_text;
    for (size_t i = 0; i < CbJoinCount; ++i) {
        cb_joins += c.cb_joins[i];
        cb_join_text += fmt::format("{}{} {:.0f}", i == 0 ? "" : ", ", CbJoinNames[i],
                                    rate(c.cb_joins[i]));
    }
    const u64 cb_order = TakeCount(g_e2.cb_order);
    const u64 cb_checks = TakeCount(g_e2.cb_checks);
    const u64 cb_differs = TakeCount(g_e2.cb_differs);
    const u64 cb_guard = TakeCount(g_e2.cb_guard);
    const u64 async_writes = TakeCount(g_e2.async_writes);
    LOG_INFO(HW_GPU,
             "E split thread 10 s: cb words (FRITCHII_ESPLIT_CB_ASYNC={}): queued writes {:.0f}/s "
             "in {:.0f} records/s ({:.1f} KiB/s), writes in joins {:.0f}/s ({}); checks (must be "
             "0): ESplitCbOrder {}, ESplitCbBytes {} in {} read back (1 write in 64), "
             "ESplitCbGuard {}, ESplitCbRead {}, ESplitAsyncWrite {}; draw flushes "
             "(FRITCHII_ESPLIT_FLUSH_ASYNC={}): before the lookup {:.0f}/s, owed (E2 drained "
             "first) {:.0f}/s, in the tail {:.0f}/s",
             g_cb_async ? 1 : 0, rate(c.cb_segments), rate(c.cb_records),
             rate(c.cb_bytes) / 1024.0, rate(cb_joins), cb_join_text, cb_order, cb_differs,
             cb_checks, cb_guard, c.cb_reads, async_writes, g_flush_async ? 1 : 0,
             rate(c.flush_before), rate(c.flush_drained), rate(c.flush_tail));
    // The queued mode: the entries that still join, by kind (the top 8).
    {
        std::array<JoinKindSlot, JoinKindSlots> slots = g_join_kinds;
        g_join_kinds = {};
        std::sort(slots.begin(), slots.end(),
                  [](const JoinKindSlot& a, const JoinKindSlot& b) { return a.count > b.count; });
        std::string text;
        u64 total = 0;
        for (const JoinKindSlot& slot : slots) {
            total += slot.count;
        }
        for (size_t i = 0; i < 8 && slots[i].count != 0; ++i) {
            const JoinKindSlot& slot = slots[i];
            static constexpr std::array<const char*, 16> KindNames{
                "wrap", "engine", "method", "multi", "call", "pairs", "delta", "draw batch",
                "cb words", "kind 9", "kind 10", "kind 11", "kind 12", "kind 13", "kind 14",
                "kind 15"};
            std::string what = fmt::format("{} {}", EngineNames[slot.engine & 3],
                                           KindNames[slot.kind & 15]);
            if (slot.function != nullptr) {
                what += fmt::format(" tag {} {}", slot.method_or_tag, Site(slot.function));
            } else {
                what += fmt::format(" {:#x}", slot.method_or_tag);
            }
            text += fmt::format("{}{} {:.0f}", i == 0 ? "" : "; ", what, rate(slot.count));
        }
        LOG_INFO(HW_GPU,
                 "E split thread 10 s: entries that join {:.0f}/s (FRITCHII_ESPLIT_ASYNC_KINDS={}), "
                 "top kinds: {}{}",
                 rate(total), g_async_kinds, text.empty() ? std::string{"none"} : text,
                 g_join_kinds_lost != 0 ? fmt::format(" ({} uncounted: table full)",
                                                      std::exchange(g_join_kinds_lost, 0))
                                        : std::string{});
    }
    // S3: each thread's CPU time against its busy time (a spinning wait shows as CPU time).
    const u64 e1_cpu = Common::CurrentThreadCpuTimeNs().value_or(0);
    const u64 e2_cpu = g.e2_cpu_ns.load(std::memory_order_relaxed);
    const auto cpu_ms = [window](u64 now_ns, u64 last_ns) {
        return now_ns > last_ns ? static_cast<double>(now_ns - last_ns) / 1e6 / window : 0.0;
    };
    LOG_INFO(HW_GPU,
             "E split thread 10 s: placement FRITCHII_ESPLIT_PIN={}: E1 CPU {:.0f} ms/s (its "
             "waits for E2 {:.1f}, held {:.1f}), E2 CPU {:.0f} ms/s (busy {:.1f}, waiting {:.1f})",
             PinMode(), cpu_ms(e1_cpu, g_e1_cpu_last), e1_wait, ms(c.hold_ticks),
             cpu_ms(e2_cpu, g_e2_cpu_last), e2_busy, ms(e2_wait));
    g_e1_cpu_last = e1_cpu;
    g_e2_cpu_last = e2_cpu;
}

} // Anonymous namespace

bool ThreadWanted() noexcept {
    static const bool wanted =
        VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_esplit, "FRITCHII_ESPLIT") == 1;
    return wanted;
}

void StartE2(Core::System& system, void (*adopt)(void*), void* adopt_argument) {
    if (!OneF::OnEngineThread()) {
        LOG_ERROR(HW_GPU, "E split thread: FRITCHII_ESPLIT=1 needs FRITCHII_ONEF=1; E2 not started");
        return;
    }
    g.system = &system;
    g.adopt = adopt;
    g.adopt_argument = adopt_argument;
    g.e1_id = std::this_thread::get_id();
    for (SlotBuffers& buffers : g.buffers) {
        buffers.record = std::make_unique<u8[]>(RecordRoom());
        buffers.packet = std::make_unique<u8[]>(OneF::PackedDrawRoom());
        buffers.cb = std::make_unique<u8[]>(CbRoom);
    }
    t_role = 1;
    ThreadMark();
    ThreadRecordsOn();
    // E1's reads of game memory, through the back read audit's funnel (its own tables count only
    // with its switch).
    BackReadAudit::g_touch_hook = &ReadTouch;
    BackReadAudit::enabled = true;
    u64 e1_mask = 0;
    bool e1_unpin = false;
    PlanPlacement(e1_mask, e1_unpin);
    g.thread = std::jthread{E2Main};
    g.e2_ready.wait(false, std::memory_order_acquire);
    // S3: E1's place, once E2 runs on its own.
    if (e1_unpin) {
        Common::SetCurrentThreadCpuMask(0);
        Common::SetCurrentThreadToPerformanceCores();
    } else if (e1_mask != 0) {
        Common::SetCurrentThreadCpuMask(e1_mask);
    } else if (PinMode() == 3) {
        Common::PinCurrentThread(Common::PinnedRole::GpuMem);
    }
    g_e1_cpu_last = Common::CurrentThreadCpuTimeNs().value_or(0);
    g_since_ns = Profile::Now();
    g_since_ticks = Profile::Ticks();
    g_thread = true;
    LOG_INFO(HW_GPU, "E split thread: FRITCHII_ESPLIT=1, E2 (GPUBackE2) started; "
                     "FRITCHII_ESPLIT_DEPTH=0 runs lockstep, N > 0 lets E1 run N draws ahead");
}

void StopE2() {
    if (!g_thread) {
        return;
    }
    WaitDrained();
    g_thread = false;
    g_async = false;
    g_cb_async = false;
    g_flush_async = false;
    g.stop.store(true, std::memory_order_seq_cst);
    g.e2_wake.fetch_add(1, std::memory_order_seq_cst);
    g.e2_wake.notify_all();
    g.thread = {};
    BackReadAudit::g_touch_hook = nullptr;
}

void ThreadBatch() noexcept {
    if (!g_thread) {
        return;
    }
    // Default 16: the queued mode; 0 = S1's lockstep (the check mode).
    int depth = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_esplit_depth,
                                                 "FRITCHII_ESPLIT_DEPTH", 16);
    depth = std::clamp(depth, 0, MaxDepth);
    if (depth != g_depth) [[unlikely]] {
        SetDepth(depth);
    }
    // FRITCHII_ESPLIT_ASYNC_KINDS (live): which E2 entries queue; changed with E2 drained (a
    // compute engine's sink moves with its entries).
    {
        const u32 kinds = static_cast<u32>(VideoCore::Nx2Debug::SwitchValue(
                              VideoCore::Nx2Debug::g_esplit_async_kinds,
                              "FRITCHII_ESPLIT_ASYNC_KINDS", static_cast<int>(KindsDefault))) &
                          7;
        if (kinds != g_async_kinds) [[unlikely]] {
            WaitDrained();
            LOG_INFO(HW_GPU,
                     "E split thread: FRITCHII_ESPLIT_ASYNC_KINDS={} (cache calls {}, raster ops "
                     "{}, compute entries {})",
                     kinds, (kinds & KindCacheCalls) != 0 ? "queued" : "join",
                     (kinds & KindRasterOps) != 0 ? "queued" : "join",
                     (kinds & KindCompute) != 0 ? "queued" : "join");
            g_async_kinds = kinds;
        }
    }
    // S4: async constant buffer words and the draw flush after the lookup (both live, with
    // DEPTH > 0). The open record goes first; nothing else waits on the change.
    // The queued mode's default: constant buffer writes queued (FRITCHII_ESPLIT_CB_ASYNC=0: every
    // one a join, S4's check value).
    const bool cb_async =
        g_async && VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_esplit_cb_async,
                                                    "FRITCHII_ESPLIT_CB_ASYNC", 1) != 0;
    const bool flush_async =
        g_async && VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_esplit_flush_async,
                                                    "FRITCHII_ESPLIT_FLUSH_ASYNC") != 0;
    if (cb_async != g_cb_async || flush_async != g_flush_async) [[unlikely]] {
        PostCbWrites();
        LOG_INFO(HW_GPU,
                 "E split thread: FRITCHII_ESPLIT_CB_ASYNC {}, FRITCHII_ESPLIT_FLUSH_ASYNC {} "
                 "(both need FRITCHII_ESPLIT_DEPTH > 0)",
                 cb_async ? "on: constant buffer writes queued for E2" : "off",
                 flush_async ? "on: a flush owed nothing goes after the lookup" : "off");
        g_cb_async = cb_async;
        g_flush_async = flush_async;
    }
    if ((++g_batches & 63) == 0) [[unlikely]] {
        const u64 now = Profile::Now();
        if (now - g_since_ns >= 10'000'000'000ULL) {
            Report(now);
            g_since_ns = now;
            g_since_ticks = Profile::Ticks();
        }
    }
}

void Run(const HandInfo& info, void (*function)(void*), void* argument) noexcept {
    if (g_async) {
        // A lockstep hand-over in S2: E2 drained, its flags on the engine for the run.
        WaitDrained();
        ToEngine();
        ++g_e1.joins[static_cast<size_t>(Join::Lockstep)];
    }
    Job& job = TakeSlot(false);
    job.function = function;
    job.argument = argument;
    job.info = info;
    ++g_e1.hands[static_cast<size_t>(info.hand)];
    if (info.record) {
        if (const Maxwell3D* const engine = OneF::Engine()) {
            job.record_kind = ThreadBuild(*engine, info.draw, info.list_args,
                                          g.buffers[job.seq % Slots].record.get(),
                                          job.record_bytes);
            job.sample = Sample();
            g_e1.samples += job.sample ? 1 : 0;
            g_e1.seeds += job.record_kind == RecordKind::Seed ? 1 : 0;
            g_e1.lost += job.record_kind == RecordKind::Lost ? 1 : 0;
        }
    }
    OneF::TakeEngineTls(job.tls);
    if (g_fill_state.capture != nullptr) {
        g_fill_state.capture(job.fill_state.data(), true);
    }
    if (info.held != nullptr) {
        info.held->HandOver(g.e2_id);
    }
    const u64 mark = WriteMark(); // S4: E2 drained (async) or idle (S1): only this job counts
    Publish(job.seq);
    WaitDone(job.seq);
    if (info.held != nullptr) {
        info.held->HandOver(g.e1_id);
    }
    OneF::PutEngineTls(job.tls);
    if (g_fill_state.install != nullptr) {
        g_fill_state.install(job.fill_state.data(), true);
    }
    if (g_async) {
        ToView();
    }
    if (info.hand != Hand::Flush && WriteMark() != mark) {
        g_flush_owed = true; // S4: E2's work on the engine left cached writes
    }
}

bool E2Idle() noexcept {
    return g.done.load(std::memory_order_acquire) >= g_next_seq - 1;
}

bool BeginAsyncDraw(Maxwell3D& engine, bool list_args, AsyncSlot& slot) noexcept {
    if (!g_async || t_role != 1 || t_join != 0) {
        return false;
    }
    if (!ThreadRecordReady(engine)) {
        // The first draw, a channel switch, or untracked writes: E1 seeds E2's view itself, E2
        // drained (the view's old engine gets its whole flag state back first, the new one's
        // goes to the view after: BeginJoin and EndJoin).
        BeginJoin(Join::Seed);
        u32 bytes = 0;
        const RecordKind kind = ThreadBuild(engine, false, false,
                                            g.buffers[g_next_seq % Slots].record.get(), bytes);
        (void)ThreadConsume(kind, nullptr, engine, false, false, false);
        g_e1.seeds += kind == RecordKind::Seed ? 1 : 0;
        g_e1.lost += kind == RecordKind::Lost ? 1 : 0;
        EndJoin();
        if (!ThreadRecordReady(engine)) {
            ++g_e1.tail_reasons[1];
            return false;
        }
    }
    const u64 ahead = g_draws_posted - g.draws_done.load(std::memory_order_acquire);
    Job& job = TakeSlot(true);
    job.async = true;
    job.info = HandInfo{.hand = Hand::DrawTail, .record = true, .draw = true, .list_args = list_args};
    job.record_kind = RecordKind::Record;
    job.record_bytes = ThreadBuildAsync(engine, list_args, g.buffers[job.seq % Slots].record.get());
    job.sample = Sample();
    if (job.sample) {
        job.hash = HashEngine(engine, list_args);
    }
    slot.payload = job.payload.data();
    slot.packet = g.buffers[job.seq % Slots].packet.get();
    g_open = &job;
    ++g_draws_posted;
    ++g_e1.async_draws;
    g_e1.depth_sum += ahead;
    ++g_e1.hands[static_cast<size_t>(Hand::DrawTail)];
    return true;
}

void* BeginAsync(Hand hand) noexcept {
    if (!g_async || t_role != 1 || t_join != 0) {
        return nullptr;
    }
    Job& job = TakeSlot(false);
    job.async = true;
    job.info = HandInfo{.hand = hand};
    g_open = &job;
    ++g_e1.hands[static_cast<size_t>(hand)];
    return job.payload.data();
}

void PostAsync(void (*function)(void*)) noexcept {
    Job& job = *std::exchange(g_open, nullptr);
    job.function = function;
    job.argument = job.payload.data();
    if (job.info.hand == Hand::DrawTail) {
        OneF::TakeEngineTls(job.tls); // the draw values go with the draw (PushPacked takes them)
    } else {
        job.tls = OneF::EngineTls{};
        job.tls.query_kind = OneF::QueryKind::Count;
    }
    if (g_fill_state.capture != nullptr) {
        g_fill_state.capture(job.fill_state.data(), true);
    }
    Publish(job.seq);
}

void NoteTailStayed(int reason) noexcept {
    ++g_e1.tail_reasons[reason & 3];
}

void BeginLookupImpl() noexcept {
    ++g_e1.lookups;
    if (g_shader_pending.exchange(false, std::memory_order_acq_rel)) [[unlikely]] {
        // A write onto shader pages (queued or run in a join) may not be flushed yet: E2 drained,
        // the invalidation run here, then the lookup.
        BeginJoin(Join::Lookup);
        if (Maxwell3D* const engine = OneF::Engine()) {
            auto& memory_manager = engine->GetMemoryManager();
            if (memory_manager.HasPendingCachedWrites()) {
                memory_manager.FlushCaching();
            }
        }
        EndJoin();
    }
    t_lookup = true;
}

bool E2ExclusiveBegin() noexcept {
    if (t_role != 2 || !g_async || t_async_cb) {
        return false;
    }
    HoldE1();
    return true;
}

void E2ExclusiveEnd() noexcept {
    ReleaseE1();
}

void QueueRasterOpImpl(RasterOp op, VideoCore::RasterizerInterface& rasterizer, u64 address,
                       u32 a, u32 b, u32 c, u32 d) noexcept {
    struct Raster {
        VideoCore::RasterizerInterface* rasterizer;
        u64 address;
        u32 a, b, c, d;
        RasterOp op;
    } const raster{&rasterizer, address, a, b, c, d, op};
    static_assert(sizeof(Raster) <= AsyncPayloadBytes && std::is_trivially_copyable_v<Raster>);
    void* const payload = BeginAsync(Hand::Raster);
    std::memcpy(payload, &raster, sizeof(raster));
    PostAsync([](void* raw) {
        Raster r;
        std::memcpy(&r, raw, sizeof(r));
        auto& rast = *r.rasterizer;
        switch (r.op) {
        case RasterOp::WaitForIdle:
            rast.WaitForIdle();
            break;
        case RasterOp::FragmentBarrier:
            rast.FragmentBarrier();
            break;
        case RasterOp::InvalidateTextureDataCache:
            rast.InvalidateTextureDataCache();
            break;
        case RasterOp::TiledCacheBarrier:
            rast.TiledCacheBarrier();
            break;
        case RasterOp::Query:
            rast.Query(r.address, static_cast<VideoCommon::QueryType>(r.a),
                       static_cast<VideoCommon::QueryPropertiesFlags>(r.b), r.c, r.d);
            break;
        case RasterOp::ResetCounter:
            rast.ResetCounter(static_cast<VideoCommon::QueryType>(r.a));
            break;
        case RasterOp::SignalSyncPoint:
            rast.SignalSyncPoint(r.a);
            break;
        }
    });
}

void NoteJoinEntry(u32 kind, u32 engine, u32 method_or_tag, const void* function) noexcept {
    const u64 key = function != nullptr
                        ? reinterpret_cast<u64>(function) ^ (u64{method_or_tag} << 56)
                        : (u64{1} << 63) | (u64{kind} << 40) | (u64{engine} << 32) | method_or_tag;
    size_t index = static_cast<size_t>((key * 0x9E3779B97F4A7C15ULL) >> 56) % JoinKindSlots;
    for (size_t probe = 0; probe < JoinKindSlots; ++probe) {
        JoinKindSlot& slot = g_join_kinds[index];
        if (slot.key == key) {
            ++slot.count;
            return;
        }
        if (slot.key == 0) {
            slot = {key, 1, function, kind, engine, method_or_tag};
            return;
        }
        index = (index + 1) % JoinKindSlots;
    }
    ++g_join_kinds_lost;
}

void* E2FlagsFor(const Maxwell3D* engine) noexcept {
    if (t_role != 2) {
        return nullptr;
    }
    View& view = ThreadView();
    return view.seeded && view.engine == engine ? &view.flags : nullptr;
}

bool FillAsync() noexcept {
    return t_role == 2 && g_current != nullptr && g_current->async;
}

void ApplyBackRaises(Maxwell3D& engine) noexcept {
    if (const u32 raise = g.back_raise.exchange(0, std::memory_order_acquire); raise != 0)
        [[unlikely]] {
        engine.dirty.flags[raise - 1] = true;
    }
}

void DeferFlag(u32 flag) noexcept {
    g.back_raise.store(flag + 1, std::memory_order_release);
}

void NoteRoomWait() noexcept {
    Bump(g_e2.room_waits);
}

void NoteLateFlush() noexcept {
    Bump(g_e2.late_flushes);
}

void NoteAsyncRefused() noexcept {
    Bump(g_e2.refused);
    if (++g_refused_logged <= 16) {
        LOG_ERROR(HW_GPU,
                  "E split thread: E2Refused: an async draw's fill refused at ring position {:#x} "
                  "after E1 moved on; the draw is dropped ({} since boot)",
                  t_e2_position, g_refused_logged);
    }
}

void CheckStoreOnE1() noexcept {
    if (g_thread && g_async && t_role == 1 && !E2Idle()) [[unlikely]] {
        ++g_e1.front_ahead;
        if (++g_front_logged <= 16) {
            LOG_ERROR(HW_GPU,
                      "E split thread: FrontAheadOfE2: E1 stored `consumed` with E2 behind ({} "
                      "since boot)",
                      g_front_logged);
        }
    }
}

void HoldHereImpl() noexcept {
    const u64 start = Profile::Ticks();
    const u64 mark = WriteMark(); // S4: R's lent call may write game memory
    while (g_hold.load(std::memory_order_seq_cst) != 0) {
        g.e1_parked.store(true, std::memory_order_seq_cst);
        (void)WaitFor([] { return g_hold.load(std::memory_order_seq_cst) == 0; }, g.e1_wake,
                      g.e1_sleeping, g_hold_poll, AsyncPollLimit);
        g.e1_parked.store(false, std::memory_order_seq_cst);
    }
    if (WriteMark() != mark) {
        g_flush_owed = true;
    }
    g_e1.hold_ticks += Profile::Ticks() - start;
}

bool BeforeEnginePushImpl(bool lend, bool visible) noexcept {
    if (t_role == 1 && t_join == 0) {
        if (visible) [[unlikely]] {
            ++g_e1.visible;
            if (++g_visible_logged <= 16) {
                LOG_ERROR(HW_GPU,
                          "E split thread: E1Visible: a fence or query op from E1 outside a join "
                          "at ring position {:#x} ({} since boot): {}",
                          GpuSplit::Get().BackEntryPosition(), g_visible_logged, Stack());
            }
        }
        if (g_async) {
            WaitDrained();
            ToEngine();
            ++g_e1.joins[static_cast<size_t>(Join::Push)];
            g_push_mark = WriteMark();
            ++t_join; // E2's work on the engine until the push is over
            return true;
        }
        return false;
    }
    if (t_role == 2 && lend && g_async) {
        HoldE1();
        return true;
    }
    return false;
}

void AfterEnginePushImpl(bool lend, bool acted) noexcept {
    (void)lend;
    if (!acted) {
        return;
    }
    if (t_role == 1) {
        --t_join;
        ToView();
        if (WriteMark() != g_push_mark) {
            g_flush_owed = true; // S4: the push or lend left cached writes
        }
    } else {
        ReleaseE1();
    }
}

void HandBind(VideoCore::RasterizerInterface& rasterizer, std::size_t stage, u32 slot,
              bool enabled, u64 address, u32 size) noexcept {
    struct Bind {
        VideoCore::RasterizerInterface* rasterizer;
        std::size_t stage;
        u64 address;
        u32 slot;
        u32 size;
        bool enabled;
    } bind{&rasterizer, stage, address, slot, size, enabled};
    static_assert(sizeof(Bind) <= AsyncPayloadBytes);
    constexpr auto run = [](void* raw) {
        Bind b;
        std::memcpy(&b, raw, sizeof(b));
        if (!b.enabled) {
            b.rasterizer->DisableGraphicsUniformBuffer(b.stage, b.slot);
        } else {
            b.rasterizer->BindGraphicsUniformBuffer(b.stage, b.slot, b.address, b.size);
        }
    };
    // S2: by value, async (it reads nothing of the engine).
    if (void* const payload = BeginAsync(Hand::Bind)) {
        std::memcpy(payload, &bind, sizeof(bind));
        PostAsync(run);
        return;
    }
    Run(HandInfo{.hand = Hand::Bind}, run, &bind);
}

void BeginJoin(Join why) noexcept {
    if (t_join++ == 0) {
        ++g_e1.joins[static_cast<size_t>(why)];
        if (g_async) {
            WaitDrained();
            ToEngine();
        }
        g_join_mark = WriteMark(); // S4: E2 drained (async) or idle (S1)
    }
}

void EndJoin() noexcept {
    if (--t_join == 0) {
        if (WriteMark() != g_join_mark) {
            g_flush_owed = true; // S4: E1 ran E2's work and it left cached writes
        }
        if (g_async) {
            ToView();
        }
    }
}

View* FillView(const Maxwell3D& engine) noexcept {
    if (t_role != 2 || g_current == nullptr || g_current->info.hand != Hand::DrawTail ||
        !g_current->view_ok) {
        return nullptr;
    }
    if (!g_current->async && !engine.draw_manager.draw_state.inline_index_draw_indexes.empty()) {
        return nullptr;
    }
    Bump(g_e2.view_fills);
    return &ThreadView();
}

void FillDone(const Maxwell3D& engine, const View* used) noexcept {
    if (FillAsync()) {
        return; // the fill kept the flags in the view, E2's own
    }
    if (t_role == 2 || (t_role == 1 && t_join != 0)) {
        const u64 before = ThreadAfterDiffers();
        AfterFill(engine, used);
        if (used != nullptr && t_role == 2) {
            Bump(g_e2.after_checks);
            Bump(g_e2.after_differs, ThreadAfterDiffers() - before);
        }
    }
}

// --- S4 ---------------------------------------------------------------------------------------------

namespace {

/// E1: where a constant buffer write can go. Count: E2's queue.
CbJoin CbCause(Maxwell3D& engine, const CbTarget& target, u32 bytes) {
    if (!g_cb_async) {
        return CbJoin::Off;
    }
    if (!target.may_defer) {
        return CbJoin::Debug;
    }
    if (sizeof(CbSegment) + CbPadded(bytes) > CbRoom) {
        return CbJoin::Size;
    }
    const u64 destination = target.address + target.offset;
    if ((destination & 0xfff) + bytes > 0x1000) {
        return CbJoin::Pages; // one GPU page: one device range for the page test
    }
    // Switch 2 sparse memory: the write may map a block (the page table changes under E1).
    if (destination + bytes > Tegra::g_nx2_sparse_low.load(std::memory_order_relaxed) &&
        destination < Tegra::g_nx2_sparse_high.load(std::memory_order_relaxed)) {
        return CbJoin::Sparse;
    }
    const std::optional<DAddr> device = engine.GetMemoryManager().GpuToCpuAddress(destination);
    if (!device) {
        return CbJoin::Unmapped;
    }
    if (!OneF::StreamPagesClear(*device, bytes)) {
        return CbJoin::Guarded;
    }
    return CbJoin::Count;
}

/// E1: the write into the open record (opened in ring order, posted later without a wait).
void QueueCb(Maxwell3D& engine, const CbTarget& target, const u32* values, u32 bytes,
             bool block) {
    const u32 need = static_cast<u32>(sizeof(CbSegment)) + CbPadded(bytes);
    if (g_cb_open != nullptr && (g_cb_engine != &engine || g_cb_used + need > CbRoom)) {
        PostCbWrites();
    }
    if (g_cb_open == nullptr) {
        Job& job = TakeSlot(false);
        job.async = true;
        job.info = HandInfo{.hand = Hand::CbWords};
        g_cb_open = &job;
        g_cb_engine = &engine;
        g_cb_used = 0;
        g_cb_segments = 0;
        ++g_e1.hands[static_cast<size_t>(Hand::CbWords)];
    }
    u8* const at = g.buffers[g_cb_open->seq % Slots].cb.get() + g_cb_used;
    const CbSegment segment{
        .target = target,
        .seq = ++g_cb_staged,
        .bytes = bytes,
        .block = static_cast<u8>(block ? 1 : 0),
        .sample = static_cast<u8>(Sample() ? 1 : 0),
    };
    std::memcpy(at, &segment, sizeof(segment));
    std::memcpy(at + sizeof(segment), values, bytes);
    g_cb_used += need;
    ++g_cb_segments;
    ++g_e1.cb_segments;
    g_e1.cb_bytes += bytes;
}

} // Anonymous namespace

bool CbWriteImpl(Maxwell3D& engine, const CbTarget& target, const u32* values, u32 bytes,
                 bool block) noexcept {
    // E1 advances the offset itself after this: E2's view takes it from E1's next record.
    NoteWord(MAXWELL3D_REG_INDEX(const_buffer.offset));
    if (bytes == 0) {
        return true; // all clipped: no memory half
    }
    const CbJoin why = CbCause(engine, target, bytes);
    if (why == CbJoin::Count) {
        QueueCb(engine, target, values, bytes, block);
        return true;
    }
    ++g_e1.cb_joins[static_cast<size_t>(why)];
    const JoinScope join{Join::CbWrite};
    engine.WriteConstantWords(target, values, bytes, block);
    return true;
}

void PostCbWrites() noexcept {
    if (g_cb_open == nullptr) {
        return;
    }
    Job& job = *std::exchange(g_cb_open, nullptr);
    const CbRecord record{g_cb_engine, g_cb_used, g_cb_segments};
    static_assert(sizeof(CbRecord) <= AsyncPayloadBytes);
    std::memcpy(job.payload.data(), &record, sizeof(record));
    job.function = &RunCbRecord;
    job.argument = job.payload.data();
    job.tls = OneF::EngineTls{};
    job.tls.query_kind = OneF::QueryKind::Count;
    if (g_fill_state.capture != nullptr) {
        g_fill_state.capture(job.fill_state.data(), true);
    }
    ++g_e1.cb_records;
    Publish(job.seq);
}

void NoteCbGuardHit() noexcept {
    Bump(g_e2.cb_guard);
    if (++g_cb_guard_logged <= 16) {
        LOG_ERROR(HW_GPU,
                  "E split thread: ESplitCbGuard: a queued cb write at ring position {:#x} lands "
                  "on pages tracked since E1 queued it ({} since boot)",
                  t_e2_position, g_cb_guard_logged);
    }
}

FlushAt FlushPlace() noexcept {
    if (!g_async) {
        return FlushAt::Before; // S1: E2 idle at every draw
    }
    if (g.async_wrote.exchange(false, std::memory_order_acquire)) [[unlikely]] {
        g_flush_owed = true;
    }
    if (g_flush_owed) {
        // Writes from work that is not a queued cb write may be unflushed (on shader pages,
        // say): the flush before the lookup, as on one thread.
        g_flush_owed = false;
        if (E2Idle()) {
            ++g_e1.flush_before;
            return FlushAt::Before;
        }
        ++g_e1.flush_drained;
        return FlushAt::Drained;
    }
    if (E2Idle() && !g_flush_async) {
        ++g_e1.flush_before;
        return FlushAt::Before;
    }
    ++g_e1.flush_tail;
    return FlushAt::Tail;
}

void NoteTouchImpl(Touch kind, const void* caller) noexcept {
    ++g_e1.touches[static_cast<size_t>(kind)];
    if (++g_touch_logged <= 16) {
        LOG_ERROR(HW_GPU, "E split thread: E1Touch ({}, {} since boot) at {:#x}: {}",
                  TouchNames[static_cast<size_t>(kind)], g_touch_logged,
                  GpuSplit::Get().BackEntryPosition(),
                  caller != nullptr ? Site(caller) : Stack());
    }
}

} // namespace VideoCommon::ESplit
