// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// One follower, part (a): the render thread R, the packet ring E -> R, lent packets, the ahead
// window and the owner asserts (one_follower.h has the design; the counters and log lines are
// part (d)'s, one_follower_stats.h).
//
// The ring: one writer (E), one reader (R), variable-size packets, never split at the ring's end
// (a pad packet fills the tail). E commits a packet by raising `sent` (release); R frees its bytes
// by raising `freed` and finishes it by raising `played` to its number. E waits for `played` at a
// lend (the park) and for `freed` or `played` at a full ring or window.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <string>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/dkb_cpu_trace.h"
#include "common/fs/path_util.h"
#include "common/spin_recursive_mutex.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/thread.h"
#include "core/core.h"
#include "video_core/gpu_split.h"
#include "video_core/gpu_wait_site.h"
#include "video_core/nx2.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/one_follower.h"
#include "video_core/nx2_profile.h"
#include "video_core/one_follower_stats.h"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif
#ifdef _WIN32
#include <windows.h> // RtlCaptureStackBackTrace (R's waits for E's lock)
#endif

namespace VideoCommon::OneFollower {

namespace {

void Pause() {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

constexpr u64 RingBytes = u64{2} << 20;
/// Ring A positions and push times of packets by number (EngineConsumed's floor, LentRan's reach).
/// Packets are at least sizeof(PacketHeader) bytes, so fewer than this many are ever in flight.
constexpr u64 PositionSlots = RingBytes / sizeof(PacketHeader) + 1;
/// A pad packet's kind (not in PacketKind: never pushed, only skipped).
constexpr u8 PadKind = 0xff;
/// Spins before a thread sleeps on its wake word (about as long as the back's).
constexpr u32 SpinLimit = 8192;

struct Impl {
    bool on{};   ///< FRITCHII_ONEF=1 (or FRITCHII_ONEF_LEAN=1) at boot (sticks for the process)
    bool lean{}; ///< FRITCHII_ONEF_LEAN=1 at boot: step 4 from boot, no compares
    bool read{}; ///< ReadSwitches ran
    bool pin{};  ///< FRITCHII_ONEF_PIN
    Core::System* system{};
    std::jthread thread;

    // Live switches, refreshed by Refresh when the switch generation moved.
    std::mutex refresh_mutex;
    std::atomic<u32> seen_generation{~u32{0}};
    std::atomic<u8> step{};
    std::atomic<u8> asked{}; ///< the step the switches ask for before nesting
    std::atomic<u32> unlend_mask{};
    std::atomic<int> verify_every{DefaultVerifyEvery};
    std::atomic<int> ahead{DefaultAhead};
    int logged_view{-1};
    int logged_build{-1};
    int logged_unlend{-1};

    std::unique_ptr<u8[]> ring;
    std::unique_ptr<u64[]> positions; ///< ring A position of packet n at n % PositionSlots
    std::unique_ptr<u64[]> push_ns;   ///< E's push time of packet n (with the GPU profile)

    // E's side (its own cache line).
    alignas(64) u64 write{}; ///< bytes written
    u64 pushed{};            ///< packets pushed
    u64 draw_seq{};
    std::atomic<u64> sent{};          ///< bytes committed
    std::atomic<u64> sent_packets{};  ///< packets committed
    std::atomic<u64> intended{};      ///< the consumed position E last stored (EngineConsumed)

    // R's side.
    alignas(64) u64 read_at{};
    std::atomic<u64> freed{};
    std::atomic<u64> played{};

    alignas(64) std::atomic<u32> r_wake{};
    std::atomic<bool> r_sleeping{};
    alignas(64) std::atomic<u32> e_wake{};
    std::atomic<bool> e_waiting{};

    std::exception_ptr lent_error; ///< thrown by a lent call on R, rethrown on E
    std::exception_ptr op_error;   ///< thrown by a render op on R, rethrown at E's next lend

    // The pricing counter of fable_b 11.5 (E writes; the frame end logs on R).
    std::array<std::atomic<u64>, static_cast<size_t>(QueryKind::Count)> query_count{};
    std::array<std::atomic<u64>, static_cast<size_t>(QueryKind::Count)> query_parked_ns{};
    std::array<u64, static_cast<size_t>(QueryKind::Count)> query_count_logged{};
    std::array<u64, static_cast<size_t>(QueryKind::Count)> query_ns_logged{};
    u64 query_log_ns{};
    std::atomic<bool> query_ops_logged{};
    std::atomic<u64> guest_write_op{}; ///< GuestWriteOpMark
    /// Lends E made holding a watched lock (handed to R for the park), the op packets' line.
    std::atomic<u64> lends_holding_lock{};
    u64 lends_holding_lock_logged{};

    // FRITCHII_ONEF_INVAL_E (item 2): E's page write stamps, the next packet number (low 32
    // bits) at each write of an invalidation E ran itself, by device page (2^22 pages = 16 GiB
    // direct, higher pages share slots: a shared slot only answers "ahead" more often).
    std::unique_ptr<std::atomic<u32>[]> stamps;
    std::atomic<bool> stamps_on{};
    std::atomic<bool> inval_refused_logged{};
    std::array<std::atomic<u64>, static_cast<size_t>(InvalKind::Count)> inval_count{};
    std::array<u64, static_cast<size_t>(InvalKind::Count)> inval_logged{};
    std::atomic<u64> ahead_checks{};
    u64 ahead_checks_logged{};

    // fable_b 12.3's dry prices (E counts; the frame-end line reads them).
    std::array<std::atomic<u64>, static_cast<size_t>(OpCount::Count)> op_count{};
    std::atomic<u64> op_bytes{};
    std::array<u64, static_cast<size_t>(OpCount::Count)> op_logged{};
    u64 op_bytes_logged{};
    std::atomic<u64> fences_lent{};
    std::atomic<u64> fences_with_flush{};
    std::atomic<u64> gathers_lent{};
    std::atomic<u64> gather_ranges{};
    std::array<u64, 4> posting_logged{};

    // The hang watchdog's view: written relaxed by E and R (no clock), read once a second.
    std::jthread watchdog;
    std::atomic<u8> e_wait{};         ///< EWait: what E waits for
    std::atomic<u64> e_wait_seq{};    ///< E's waits begun (one that never ends keeps it still)
    std::atomic<u8> e_wait_op{};      ///< the lend's LentOp (EWait::Park)
    std::atomic<u64> e_wait_number{}; ///< the packet E waits for (park, drain) or pushes (room)
    std::atomic<u64> r_packet{};      ///< the packet R plays, 0 between packets
    std::atomic<u8> r_kind{};         ///< its PacketKind
    std::atomic<u8> r_op{};           ///< its LentOp, for a lent packet
    std::atomic<u64> r_progress{};    ///< packets and pads R finished
    std::atomic<u64> rescued{};       ///< wakes the watchdog gave
    std::atomic<std::thread::id> e_thread{};
    std::atomic<std::thread::id> r_thread{};
    std::array<std::pair<const char*, const Common::SpinRecursiveMutex*>, 4> locks{};
    std::atomic<u32> lock_count{};

    std::atomic<Maxwell3D*> engine{};
    std::atomic<bool> unlend_disabled{}; ///< DisableUnlend: step 4 off for the session
};

Impl g;

using Detail::t_render;
thread_local u32 t_lent_depth{};
/// R: inside PlayDraw of an unlent draw packet.
thread_local bool t_playing_draw{};
/// E: the kind of the query being lent or pushed (SetQueryKind), consumed by it.
thread_local QueryKind t_query_kind{QueryKind::Count};
/// fable_b 14.4 cut 1: the hot-path counters (hand-overs, behind, lends by site, query kinds,
/// invalidations) run unless LEAN without the GPU profile.
bool CountersOn() noexcept {
    return !g.lean || Nx2Profile::Enabled();
}

/// E: SetLendSite's name for the next Lend, consumed by it.
thread_local const char* t_lend_site{};

// FRITCHII_ONEF_TRACE (boot; fable_b 15.5): one record per lend and per posted op, by packet
// number in a ring. E writes its half (site, op, posted, its arrival, its work since its last
// record, its resume), R its own (the call's start and end, its busy time on other packets since
// its last record), GPUWatch writes the finished ones to <log>/onef_lend_trace.bin every second
// (and the site names to onef_lend_trace_sites.txt). lend_replay.py reads them.
struct TraceRecord {
    u64 number;   ///< the packet number (0: no record in this slot yet)
    u64 site;     ///< the site name's address (onef_lend_trace_sites.txt), 0 = none
    u64 e_arrive; ///< E at the push (ns, Nx2Hitch::NowNs)
    u64 r_start;  ///< R starts the packet
    u64 r_end;    ///< R ends it
    u64 e_resume; ///< E released (lent), 0 for a posted op
    u64 e_work;   ///< E's time since its previous record ended (its resume or post)
    u64 r_busy;   ///< R's time on other packets since its previous record
    u8 op;        ///< LentOp
    u8 posted;    ///< 1: a posted op (E did not wait)
    u8 pad[6];
};
static_assert(sizeof(TraceRecord) == 72);
constexpr u64 TraceSlots = u64{1} << 20;
bool g_trace{};
std::unique_ptr<TraceRecord[]> g_trace_ring;
u64 g_trace_written{}; ///< GPUWatch: the next packet number to write
u64 g_trace_records{};
u64 g_trace_lost{};
std::ofstream g_trace_file;
std::mutex g_trace_sites_mutex;
std::vector<const char*> g_trace_sites;
bool g_trace_sites_dirty{};
/// E: the end of its previous record (resume or post).
thread_local u64 t_e_last{};
/// R: its time on other packets since its previous record.
thread_local u64 t_r_busy{};

/// E: a record's E half at the push (before the packet can be played: R writes other fields).
void TraceArrive(u64 number, const char* site, LentOp op, bool posted, u64 now) {
    TraceRecord& record = g_trace_ring[number % TraceSlots];
    record.site = reinterpret_cast<u64>(site);
    record.op = static_cast<u8>(op);
    record.posted = posted ? 1 : 0;
    record.e_arrive = now;
    record.e_work = t_e_last != 0 && now > t_e_last ? now - t_e_last : 0;
    record.e_resume = 0;
    t_e_last = now;
    if (site != nullptr) {
        std::scoped_lock lock{g_trace_sites_mutex};
        if (std::ranges::find(g_trace_sites, site) == g_trace_sites.end()) {
            g_trace_sites.push_back(site);
            g_trace_sites_dirty = true;
        }
    }
    // Last: the slot is this packet's now (GPUWatch tests it).
    std::atomic_ref<u64>{record.number}.store(number, std::memory_order_release);
}

/// GPUWatch: writes every finished record (both halves in), and the site names when new.
void TraceFlush(bool final) {
    if (!g_trace || !g_trace_file) {
        return;
    }
    // A record is finished once R played it and E pushed past it (its resume was written
    // before its next push); a margin covers a posted op's E half, written right after its push.
    const u64 played = g.played.load(std::memory_order_acquire);
    const u64 sent = g.sent_packets.load(std::memory_order_acquire);
    u64 safe = (std::min)(played, sent);
    safe = final ? safe : (safe > 64 ? safe - 64 : 0);
    if (g_trace_written == 0) {
        g_trace_written = safe > TraceSlots ? safe - TraceSlots + 1 : 1;
    }
    if (safe >= g_trace_written + TraceSlots) {
        g_trace_lost += safe - TraceSlots + 1 - g_trace_written;
        g_trace_written = safe - TraceSlots + 1;
    }
    for (; g_trace_written <= safe; ++g_trace_written) {
        const TraceRecord& record = g_trace_ring[g_trace_written % TraceSlots];
        if (std::atomic_ref<u64>{const_cast<u64&>(record.number)}.load(std::memory_order_acquire) !=
            g_trace_written) {
            continue; // a draw or delta packet: no record
        }
        g_trace_file.write(reinterpret_cast<const char*>(&record), sizeof(record));
        ++g_trace_records;
    }
    g_trace_file.flush();
    std::vector<const char*> sites;
    {
        std::scoped_lock lock{g_trace_sites_mutex};
        if (!std::exchange(g_trace_sites_dirty, false) && !final) {
            return;
        }
        sites = g_trace_sites;
    }
    std::ofstream names{Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) /
                            "onef_lend_trace_sites.txt",
                        std::ios::trunc};
    for (const char* const site : sites) {
        names << fmt::format("{:#x}\t{}\n", reinterpret_cast<u64>(site), site);
    }
    if (final) {
        LOG_INFO(HW_GPU, "One follower lend trace: {} records written, {} lost (ring overrun)",
                 g_trace_records, g_trace_lost);
    }
}
/// E's lends by call site (written on E; the 10 s line reads them on R in a frame-end lend,
/// E parked). Keyed by the name's pointer (one literal per site); the rest counted together.
struct LendSite {
    const char* name{};
    u8 op{};
    u64 count{};
    u64 ns{};
    u64 count_logged{};
    u64 ns_logged{};
};
constexpr size_t LendSiteSlots = 96;
std::array<LendSite, LendSiteSlots> g_lend_sites{};
LendSite g_lend_sites_rest{};
u64 g_lend_sites_log_ns{};
/// R: the number of the packet it plays (or played last): its place for the page write stamps.
thread_local u32 t_play_number{};
constexpr u64 StampSlots = u64{1} << 22;
/// The most a packet's spans can take (every register word, the flags, the draw, the programs
/// and the pools), and an op packet's payload after them.
constexpr std::size_t MaxSpanBytes = sizeof(RegPair) * Maxwell3D::Regs::NUM_REGS +
                                     sizeof(FlagBits) + sizeof(DrawValues) + sizeof(Programs) +
                                     sizeof(Pools) + 64;
constexpr std::size_t MaxPayloadBytes = 64;

/// E's packet buffer (fable_b 11.1 item 0): allocated once per thread, written at a cursor, never
/// resized or cleared on the hot path (the spans' bytes are written before they are sent).
u8* PackBuffer() {
    thread_local std::unique_ptr<u8[]> buffer;
    if (!buffer) [[unlikely]] {
        buffer.reset(new u8[sizeof(PacketHeader) + MaxSpanBytes + MaxPayloadBytes]);
    }
    return buffer.get();
}
/// E: the draw the next Lend or PackSpans is for (SetNextDraw), consumed by them.
thread_local DrawValues t_next_draw{};
thread_local bool t_next_draw_set{};

/// Applies the nesting: each step needs the one before; a refused switch is logged once per value.
void Refresh(u32 generation) {
    std::scoped_lock lock{g.refresh_mutex};
    using namespace VideoCore::Nx2Debug;
    // FRITCHII_ONEF_LEAN: steps 2 and 3 are asked for by the lean mode itself (their compares
    // stay off: no PacketVerify, VERIFY_EVERY 0); step 4 by FRITCHII_ONEF_UNLEND (lean's 3).
    const int view = g.lean ? 1 : SwitchValue(g_onef_view, "FRITCHII_ONEF_VIEW");
    const int build = g.lean ? 1 : SwitchValue(g_onef_build, "FRITCHII_ONEF_BUILD");
    const int unlend = SwitchValue(g_onef_unlend, "FRITCHII_ONEF_UNLEND");
    g.verify_every.store(SwitchValue(g_onef_verify_every, "FRITCHII_ONEF_VERIFY_EVERY",
                                     DefaultVerifyEvery),
                         std::memory_order_relaxed);
    g.ahead.store(SwitchValue(g_onef_ahead, "FRITCHII_ONEF_AHEAD", DefaultAhead),
                  std::memory_order_relaxed);
    const u8 asked = !g.on ? 0 : unlend != 0 ? 4 : build != 0 ? 3 : view != 0 ? 2 : 1;
    StepLevel level = g.on ? StepLevel::Lent : StepLevel::Off;
    const auto refuse = [](int& logged, int value, const char* name, const char* needs) {
        if (logged != value) {
            logged = value;
            LOG_WARNING(HW_GPU, "{}={} needs {}; running 0", name, value, needs);
        }
    };
    if (view != 0) {
        if (level == StepLevel::Lent) {
            level = StepLevel::View;
        } else {
            refuse(g.logged_view, view, "FRITCHII_ONEF_VIEW", "FRITCHII_ONEF=1 at boot");
        }
    }
    if (build != 0) {
        if (level == StepLevel::View) {
            level = StepLevel::Build;
        } else {
            refuse(g.logged_build, build, "FRITCHII_ONEF_BUILD", "FRITCHII_ONEF_VIEW=1");
        }
    }
    u32 mask = 0;
    if (unlend != 0) {
        if (g.unlend_disabled.load(std::memory_order_relaxed)) {
            refuse(g.logged_unlend, unlend, "FRITCHII_ONEF_UNLEND",
                   "a session without an unlent draw R could not build (it was disabled)");
        } else if (level != StepLevel::Build) {
            refuse(g.logged_unlend, unlend, "FRITCHII_ONEF_UNLEND", "FRITCHII_ONEF_BUILD=1");
        } else if (GetDrawPlayer() == nullptr) {
            refuse(g.logged_unlend, unlend, "FRITCHII_ONEF_UNLEND", "R's draw player (part c)");
        } else {
            level = StepLevel::Unlend;
            mask = static_cast<u32>(unlend) & AllKinds;
        }
    }
    const auto previous = static_cast<StepLevel>(g.step.exchange(static_cast<u8>(level)));
    g.asked.store(asked, std::memory_order_relaxed);
    g.unlend_mask.store(mask, std::memory_order_relaxed);
    if (previous >= StepLevel::View && level < StepLevel::View) {
        // Writes stop being tracked: the view is seeded again when step 2 comes back.
        Detail::ViewTrackingStopped();
    }
    g_track_writes.store(level >= StepLevel::View, std::memory_order_relaxed);
    if (previous != level && g.on) {
        LOG_INFO(HW_GPU, "One follower: step {} (unlend mask {:#x}, verify every {}, ahead {})",
                 static_cast<int>(level), mask, g.verify_every.load(std::memory_order_relaxed),
                 g.ahead.load(std::memory_order_relaxed));
    }
    if (g.lean && level != StepLevel::Unlend && unlend != 0 &&
        !g.unlend_disabled.load(std::memory_order_relaxed) && GetDrawPlayer() == nullptr) {
        return; // LEAN before R's draw player exists (boot): ask again at the next Step()
    }
    g.seen_generation.store(generation, std::memory_order_release);
}

/// What E waits for (the watchdog's report).
enum class EWait : u8 { None, Park, Room, Drain };
constexpr std::array<const char*, 4> e_wait_names{"running", "parked in a lend",
                                                  "waiting for ring room or the window",
                                                  "draining R"};

/// E waits until `ready` holds; returns the nanoseconds waited (0 at once, or without the profile).
template <typename Ready>
u64 EngineWait(Ready&& ready, EWait kind = EWait::Room, u64 number = 0) {
    if (ready()) {
        return 0;
    }
    g.e_wait_number.store(number, std::memory_order_relaxed);
    g.e_wait.store(static_cast<u8>(kind), std::memory_order_relaxed);
    g.e_wait_seq.store(g.e_wait_seq.load(std::memory_order_relaxed) + 1,
                       std::memory_order_relaxed);
    SCOPE_EXIT {
        g.e_wait.store(static_cast<u8>(EWait::None), std::memory_order_relaxed);
    };
    const bool timing = Stats::Timing();
    const u64 begin = timing ? Stats::Now() : 0;
    for (u32 spins = 0;; ++spins) {
        if (ready()) {
            break;
        }
        if (spins < SpinLimit) {
            Pause();
            continue;
        }
        g.e_waiting.store(true, std::memory_order_seq_cst);
        // The flag before `ready`'s loads (they are acquire loads): WakeEngine's fence pairs.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const u32 seen = g.e_wake.load(std::memory_order_seq_cst);
        if (!ready()) {
            g.e_wake.wait(seen, std::memory_order_seq_cst);
        }
        g.e_waiting.store(false, std::memory_order_relaxed);
    }
    return timing ? Stats::Now() - begin : 1;
}

// The wake handshakes are Dekker's: the sleeper stores its flag and then reads the condition; the
// waker stores the condition and then reads the flag. The waker's condition stores (`sent`,
// `freed`) are release stores, which x86 may hold in its store buffer past the flag's load
// (store-load reordering): both sides then read the other's old value, the sleeper sleeps, the
// waker sees no sleeper, and E (parked in a lend) and R (asleep with the lend in the ring) wait
// for each other for good. The full fence before the flag's load closes it.
void WakeRender() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (g.r_sleeping.load(std::memory_order_seq_cst)) {
        g.r_wake.fetch_add(1, std::memory_order_seq_cst);
        g.r_wake.notify_one();
    }
}

void WakeEngine() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (g.e_waiting.load(std::memory_order_seq_cst)) {
        g.e_wake.fetch_add(1, std::memory_order_seq_cst);
        g.e_wake.notify_one();
    }
}

/// E: copies a finished packet (its header first, `bytes` in all) into the ring. `windowed`: an
/// unparked packet, held to FRITCHII_ONEF_AHEAD (a lent packet waits for its own play anyway).
void Commit(const u8* packet, u32 bytes, u64 number, u64 position, bool windowed) {
    u64 waited = 0;
    // A packet never wraps: the tail goes to a pad when it does not fit.
    const u64 offset = g.write % RingBytes;
    if (offset + bytes > RingBytes) {
        const u64 pad = RingBytes - offset;
        waited += EngineWait(
            [&] { return g.write + pad - g.freed.load(std::memory_order_acquire) <= RingBytes; },
            EWait::Room, number);
        PacketHeader header{};
        header.bytes = static_cast<u32>(pad);
        header.kind = static_cast<PacketKind>(PadKind);
        std::memcpy(g.ring.get() + offset, &header, sizeof(header));
        g.write += pad;
        g.sent.store(g.write, std::memory_order_release);
        WakeRender();
    }
    const int ahead = g.ahead.load(std::memory_order_relaxed);
    waited += EngineWait([&] {
        if (g.write + bytes - g.freed.load(std::memory_order_acquire) > RingBytes) {
            return false;
        }
        // The window: packets pushed before this one that R has not played.
        return !windowed || ahead <= 0 ||
               number - 1 - g.played.load(std::memory_order_acquire) < static_cast<u64>(ahead);
    }, EWait::Room, number);
    if (waited != 0) {
        Stats::WindowWait(waited);
    }
    std::memcpy(g.ring.get() + g.write % RingBytes, packet, bytes);
    g.positions[number % PositionSlots] = position;
    if (Stats::Timing()) {
        g.push_ns[number % PositionSlots] = Stats::Now();
    }
    g.write += bytes;
    // `positions` before `sent_packets`: R and EngineConsumed read it after an acquire of it.
    g.sent_packets.store(number, std::memory_order_release);
    g.sent.store(g.write, std::memory_order_release);
    WakeRender();
}

/// The oldest unplayed packet's ring A position, ~0 when R has played everything committed.
u64 Floor(u64 played, u64 sent_packets) {
    return played < sent_packets ? g.positions[(played + 1) % PositionSlots] : ~u64{0};
}

/// R (after each unparked packet) and E (EngineConsumed): GpuSplit's `played` (8.2) =
/// min(what E ran, R's oldest unplayed packet). The front's waits read it.
void RaisePlayed() {
    // A Dekker pair: E stores `intended` then reads `played`; R stores `played` then reads
    // `intended`. Without this fence E's load could pass its store (x86 reorders a store with a
    // later load) while R read the old `intended`: both raised GpuSplit's `played` short of the
    // truth and neither raised it again, the front waiting for a position E and R had both passed
    // (the FRITCHII_ONEF_QUERY_OPS hang: E and R idle, GPUWatch silent; render ops made E end its
    // batches with R's packets unplayed, which opened the window). With it at least one side
    // reads the other's store.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    // `intended` before `sent_packets` (acquire both): an `intended` E stored after committing a
    // packet comes with that packet's count.
    const u64 intended = g.intended.load(std::memory_order_acquire);
    const u64 played = g.played.load(std::memory_order_acquire);
    const u64 sent_packets = g.sent_packets.load(std::memory_order_acquire);
    GpuSplit::Get().RaisePlayed((std::min)(intended, Floor(played, sent_packets)));
}

/// E, after a lend's park: its site's count and time.
void NoteLendSite(LentOp op, u64 ns) {
    const char* const name = std::exchange(t_lend_site, nullptr);
    LendSite* slot = &g_lend_sites_rest;
    for (auto& site : g_lend_sites) {
        if (site.name == name && site.op == static_cast<u8>(op)) {
            slot = &site;
            break;
        }
        if (site.name == nullptr && site.count == 0) {
            site.name = name; // null: Lend called directly (draws, verify compares)
            site.op = static_cast<u8>(op);
            slot = &site;
            break;
        }
    }
    ++slot->count;
    slot->ns += ns;
}

/// The lends by site, every 10 s (R, at frame-end lends: E parked), the 16 costliest.
void LogLendSites() {
    const u64 now = Common::Nx2Hitch::NowNs();
    if (g_lend_sites_log_ns == 0) {
        g_lend_sites_log_ns = now;
        return;
    }
    if (now - g_lend_sites_log_ns < 10'000'000'000ULL) {
        return;
    }
    const double seconds = static_cast<double>(now - g_lend_sites_log_ns) / 1e9;
    g_lend_sites_log_ns = now;
    struct Row {
        const LendSite* site;
        u64 count;
        u64 ns;
    };
    std::vector<Row> rows;
    const auto take = [&](LendSite& site) {
        const Row row{&site, site.count - site.count_logged, site.ns - site.ns_logged};
        site.count_logged = site.count;
        site.ns_logged = site.ns;
        if (row.count != 0) {
            rows.push_back(row);
        }
    };
    for (auto& site : g_lend_sites) {
        take(site);
    }
    take(g_lend_sites_rest);
    std::ranges::sort(rows, [](const Row& a, const Row& b) { return a.ns > b.ns; });
    std::string text;
    for (size_t i = 0; i < rows.size() && i < 16; ++i) {
        const LendSite& site = *rows[i].site;
        const char* const op = site.op < lent_op_names.size() ? lent_op_names[site.op] : "?";
        text += fmt::format("{}{} ({}) {:.0f}/s {:.1f} ms/s", text.empty() ? "" : "; ",
                            &site == &g_lend_sites_rest ? "other sites"
                            : site.name != nullptr      ? site.name
                                                        : "direct",
                            op, static_cast<double>(rows[i].count) / seconds,
                            static_cast<double>(rows[i].ns) / 1e6 / seconds);
    }
    LOG_INFO(HW_GPU, "One follower lends by site (E parked, 10 s, costliest first): {}", text);
}

/// The query kinds' 10 s line (R, at frame-end lends).
void LogQueryKinds() {
    const u64 now = Common::Nx2Hitch::NowNs();
    if (g.query_log_ns == 0) {
        g.query_log_ns = now;
        return;
    }
    if (now - g.query_log_ns < 10'000'000'000ULL) {
        return;
    }
    const double seconds = static_cast<double>(now - g.query_log_ns) / 1e9;
    g.query_log_ns = now;
    std::string text;
    for (size_t i = 0; i < query_kind_names.size(); ++i) {
        const u64 count = g.query_count[i].load(std::memory_order_relaxed);
        const u64 ns = g.query_parked_ns[i].load(std::memory_order_relaxed);
        text += fmt::format("{}{} {:.0f}/s {:.1f} ms/s", i == 0 ? "" : ", ", query_kind_names[i],
                            static_cast<double>(count - g.query_count_logged[i]) / seconds,
                            static_cast<double>(ns - g.query_ns_logged[i]) / 1e6 / seconds);
        g.query_count_logged[i] = count;
        g.query_ns_logged[i] = ns;
    }
    // fable_b 11.3: the acquire latches by thread (session totals): R's is the expected one.
    static constexpr std::array<const char*, static_cast<size_t>(InvalKind::Count)> inval_names{
        "run on E", "lent: texture page", "lent: shader page", "lent: query page",
        "lent: hit at the second look", "inline uploads on E",
        "inline uploads with R's half lent", "inline uploads lent: query page", "gathers on E",
        "gathers lent: texture or shader page"};
    std::string inval;
    for (size_t i = 0; i < inval_names.size(); ++i) {
        const u64 count = g.inval_count[i].load(std::memory_order_relaxed);
        inval += fmt::format("{}{} {:.0f}/s", i == 0 ? "" : ", ", inval_names[i],
                             static_cast<double>(count - g.inval_logged[i]) / seconds);
        g.inval_logged[i] = count;
    }
    const u64 checks = g.ahead_checks.load(std::memory_order_relaxed);
    LOG_INFO(HW_GPU,
             "One follower invalidations (FRITCHII_ONEF_INVAL_E={}) {:.0f} s: {}; image uploads "
             "checked against the page write stamps {:.0f}/s (\"texture read ahead of R\" in the "
             "checks line, must be 0)",
             InvalOnEngine() ? 1 : 0, seconds, inval,
             static_cast<double>(checks - g.ahead_checks_logged) / seconds);
    g.ahead_checks_logged = checks;
    const std::array<u64, 4> posting{g.fences_lent.load(std::memory_order_relaxed),
                                     g.fences_with_flush.load(std::memory_order_relaxed),
                                     g.gathers_lent.load(std::memory_order_relaxed),
                                     g.gather_ranges.load(std::memory_order_relaxed)};
    const auto rate = [&](size_t i) {
        return static_cast<double>(posting[i] - g.posting_logged[i]) / seconds;
    };
    const double gathers = rate(2);
    LOG_INFO(HW_GPU,
             "One follower posting (dry, FRITCHII_ONEF_POSTED={}) {:.0f} s: lent fences {:.0f}/s, "
             "{:.0f}/s with a buffer flush pending (the rest need no E half); lent gathers "
             "{:.0f}/s, {:.1f} ranges each",
             Posted() ? 1 : 0, seconds, rate(0), rate(1), gathers,
             gathers > 0 ? rate(3) / gathers : 0.0);
    g.posting_logged = posting;
    static constexpr std::array<const char*, static_cast<size_t>(OpCount::Count)> op_names{
        "posted", "lent: a sync in a held fill", "lent: inside a draw fill", "lent: no room",
        "lent: no packet ticks"};
    std::string ops;
    for (size_t i = 0; i < op_names.size(); ++i) {
        const u64 count = g.op_count[i].load(std::memory_order_relaxed);
        ops += fmt::format("{}{} {:.0f}/s", i == 0 ? "" : ", ", op_names[i],
                           static_cast<double>(count - g.op_logged[i]) / seconds);
        g.op_logged[i] = count;
    }
    const u64 op_bytes = g.op_bytes.load(std::memory_order_relaxed);
    const double posted = static_cast<double>(g.op_count[0].load(std::memory_order_relaxed));
    (void)posted;
    LOG_INFO(HW_GPU,
             "One follower op packets (fable_b 13.1; FRITCHII_ONEF_OP_INLINE={}) {:.0f} s: {}; "
             "{:.1f} KiB/s of entries posted; lends parked holding a lock {:.0f}/s (handed to R)",
             PostOpSite(15) ? 1 : 0, seconds, ops,
             static_cast<double>(op_bytes - g.op_bytes_logged) / 1024.0 / seconds,
             static_cast<double>(g.lends_holding_lock.load(std::memory_order_relaxed) -
                                 g.lends_holding_lock_logged) /
                 seconds);
    g.lends_holding_lock_logged = g.lends_holding_lock.load(std::memory_order_relaxed);
    g.op_bytes_logged = op_bytes;
    LOG_INFO(HW_GPU,
             "One follower queries by kind (FRITCHII_ONEF_QUERY_OPS={}) {:.0f} s: {}; acquire "
             "latches this session by R {}, E {}, other threads {}",
             QueryOps() ? 1 : 0, seconds, text, g_latches[0].load(std::memory_order_relaxed),
             g_latches[1].load(std::memory_order_relaxed),
             g_latches[2].load(std::memory_order_relaxed));
}

void LogMaybe() {
    LogQueryKinds();
    LogLendSites();
    Stats::Logged now{};
    now.step = static_cast<StepLevel>(g.step.load(std::memory_order_relaxed));
    now.asked = g.asked.load(std::memory_order_relaxed);
    now.unlend_mask = g.unlend_mask.load(std::memory_order_relaxed);
    now.ahead = g.ahead.load(std::memory_order_relaxed);
    now.verify_every = g.verify_every.load(std::memory_order_relaxed);
    Stats::MaybeLog(now);
}

/// R: one packet. Returns where its time goes (Stats::RenderTime).
Stats::RenderTime Play(const PacketHeader& header, const u8* spans) {
    t_play_number = static_cast<u32>(header.number);
    RegisterView& view = View();
    Maxwell3D* const engine = g.engine.load(std::memory_order_acquire);
    const bool view_on = Step() >= StepLevel::View;
    if (header.kind == PacketKind::Lent && view_on && engine != nullptr &&
        ((header.flags & PacketSeed) != 0 || !view.seeded)) {
        SeedView(view, *engine); // E is parked: its engine is at this packet's place
    }
    if (header.spans != 0 && view.seeded) {
        ApplySpans(view, header, spans);
    }
    // Part (c)'s view engine (from step 3): every packet after the spans, a lent one before its
    // call runs (E parked). Below step 3 it is idle and costs nothing.
    DrawPlayer* const player = Step() >= StepLevel::Build ? GetDrawPlayer() : nullptr;
    if (player != nullptr) {
        player->OnPacket(header, spans);
    } else if (DrawPlayer* const idle = GetDrawPlayer()) {
        idle->ViewOff(); // below step 3 the view is not kept: R reads E's engine (lent only)
    }
    switch (header.kind) {
    case PacketKind::Lent: {
        const bool around = view_on && view.seeded && engine != nullptr;
        if (player != nullptr) {
            player->BeforeLent(); // rules 1 and 2: the lent call reads E's engine
        }
        if (around) {
            BeforeLentCall(view, *engine);
            if ((header.flags & PacketVerify) != 0) {
                VerifyView(view, *engine);
            }
        }
        const bool timing = Stats::Timing();
        const u64 begin = timing ? Stats::Now() : 0;
        ++t_lent_depth;
        try {
            header.call.function(header.call.argument);
        } catch (...) {
            g.lent_error = std::current_exception();
        }
        --t_lent_depth;
        if (around) {
            AfterLentCall(view, *engine);
        }
        if (player != nullptr) {
            player->AfterLent(); // the views take the engine's flags; R is on the view again
        }
        const LentOp op = static_cast<LentOp>(header.op);
        if (timing) {
            const u64 end = Stats::Now();
            const u64 pushed_at = g.push_ns[header.number % PositionSlots];
            Stats::LentRan(op, begin > pushed_at && pushed_at != 0 ? begin - pushed_at : 0,
                           end - begin);
        } else {
            Stats::LentRan(op, 0, 0);
        }
        if (op == LentOp::FrameEnd) {
            LogMaybe();
        }
        return Stats::RenderTime::LentCalls;
    }
    case PacketKind::Draw:
        if (DrawPlayer* const draws = GetDrawPlayer(); draws != nullptr) {
            t_playing_draw = true;
            SCOPE_EXIT {
                t_playing_draw = false;
            };
            draws->PlayDraw(header);
        } else {
            Note(Check::RenderPacketMissing, "a draw packet with no player");
        }
        return Stats::RenderTime::DrawPackets;
    case PacketKind::Delta:
        break;
    case PacketKind::Op:
        // A render op at its stream place (E not parked): R's own code on R's state.
        try {
            header.call.function(const_cast<u8*>(spans) + header.reserved2);
        } catch (...) {
            if (!g.op_error) {
                g.op_error = std::current_exception();
            }
        }
        break;
    }
    return Stats::RenderTime::OtherPackets;
}

/// R's contended-lock hook (SpinRecursiveMutex::t_contended_hook): R waiting for a lock E holds
/// outside a lent call is R reaching into E's cache. Counted with the call stack (first reports):
/// the deadlock of 12 Nov had R in a draw packet waiting for the buffer cache's lock while E
/// was parked in a lend holding it.
bool RenderContended(std::thread::id holder, bool first) {
    if (!first || holder != g.e_thread.load(std::memory_order_relaxed) || t_lent_depth != 0) {
        return false;
    }
    std::string stack;
#ifdef _WIN32
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1, std::memory_order_relaxed) < 12) {
        void* frames[32];
        const USHORT count = RtlCaptureStackBackTrace(1, 32, frames, nullptr);
        for (USHORT i = 0; i < count; ++i) {
            HMODULE module{};
            char name[MAX_PATH] = "?";
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   static_cast<LPCSTR>(frames[i]), &module)) {
                GetModuleFileNameA(module, name, sizeof(name));
            }
            const char* const base_name = std::strrchr(name, '\\');
            stack += fmt::format(" {}+{:#x}", base_name != nullptr ? base_name + 1 : name,
                                 reinterpret_cast<uintptr_t>(frames[i]) -
                                     reinterpret_cast<uintptr_t>(module));
        }
    }
#endif
    thread_local std::string note;
    note = fmt::format("R (packet {}) waits for a lock E holds; stack:{}",
                       g.r_packet.load(std::memory_order_relaxed),
                       stack.empty() ? std::string{" (not logged)"} : stack);
    Note(Check::RenderWaitsEngineLock, note.c_str());
    return false;
}

void RenderMain(std::stop_token stop) {
    const Common::DkbCpuTrace::Registration cpu_trace{Common::DkbCpuTrace::Role::GpuMem};
    Common::SetCurrentThreadName("GPURender");
    t_render = true;
    Common::SpinRecursiveMutex::t_contended_hook = &RenderContended;
    GpuSplit::MarkRenderThread();
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();
    if (g.pin) {
        // FRITCHII_ONEF_PIN (default 1): the core ThreadPinning keeps for GPUMem (thread.cpp
        // reserves it when FRITCHII_ONEF=1).
        Common::PinCurrentThread(Common::PinnedRole::GpuMem);
    }
    g.system->RegisterHostThread();
    g.r_thread.store(std::this_thread::get_id(), std::memory_order_relaxed);
    for (;;) {
        u64 sent = g.sent.load(std::memory_order_acquire);
        if (sent == g.read_at && !stop.stop_requested()) {
            const bool timing = Stats::Timing();
            const u64 idle_begin = timing ? Stats::Now() : 0;
            for (u32 spins = 0; sent == g.read_at && !stop.stop_requested(); ++spins) {
                if (spins < SpinLimit) {
                    Pause();
                } else {
                    g.r_sleeping.store(true, std::memory_order_seq_cst);
                    const u32 seen = g.r_wake.load(std::memory_order_seq_cst);
                    if (g.sent.load(std::memory_order_seq_cst) == g.read_at &&
                        !stop.stop_requested()) {
                        Stats::RenderSlept();
                        g.r_wake.wait(seen, std::memory_order_seq_cst);
                    }
                    g.r_sleeping.store(false, std::memory_order_relaxed);
                    spins = 0;
                }
                sent = g.sent.load(std::memory_order_acquire);
            }
            if (timing) {
                Stats::RenderTimeSpent(Stats::RenderTime::Idle, Stats::Now() - idle_begin);
            }
        }
        if (sent == g.read_at) {
            break; // stop requested with nothing left
        }
        const u8* const at = g.ring.get() + g.read_at % RingBytes;
        PacketHeader header;
        std::memcpy(&header, at, sizeof(header));
        if (static_cast<u8>(header.kind) == PadKind) {
            g.read_at += header.bytes;
            g.freed.store(g.read_at, std::memory_order_release);
            g.r_progress.store(g.r_progress.load(std::memory_order_relaxed) + 1,
                               std::memory_order_relaxed);
            WakeEngine();
            continue;
        }
        const bool timing = Stats::Timing();
        const u64 begin = timing ? Stats::Now() : 0;
        g.r_kind.store(static_cast<u8>(header.kind), std::memory_order_relaxed);
        g.r_op.store(header.op, std::memory_order_relaxed);
        g.r_packet.store(header.number, std::memory_order_relaxed);
        const u64 trace_start = g_trace ? Common::Nx2Hitch::NowNs() : 0;
        const Stats::RenderTime where = Play(header, at + sizeof(PacketHeader));
        if (g_trace) [[unlikely]] {
            const u64 trace_end = Common::Nx2Hitch::NowNs();
            if (header.kind == PacketKind::Lent || header.kind == PacketKind::Op) {
                TraceRecord& record = g_trace_ring[header.number % TraceSlots];
                record.r_start = trace_start;
                record.r_end = trace_end;
                record.r_busy = std::exchange(t_r_busy, 0);
            } else {
                t_r_busy += trace_end - trace_start;
            }
        }
        g.r_packet.store(0, std::memory_order_relaxed);
        g.r_progress.store(g.r_progress.load(std::memory_order_relaxed) + 1,
                           std::memory_order_relaxed);
        if (timing && where != Stats::RenderTime::LentCalls) {
            Stats::RenderTimeSpent(where, Stats::Now() - begin);
        }
        g.read_at += header.bytes;
        g.freed.store(g.read_at, std::memory_order_release);
        g.played.store(header.number, std::memory_order_seq_cst);
        WakeEngine();
        if (header.kind != PacketKind::Lent) {
            // Step 4: E raised `played` only to this packet while it was unplayed.
            RaisePlayed();
        }
    }
}

/// The watchdog's names for the threads it knows.
std::string ThreadName(std::thread::id id) {
    if (id == std::thread::id{}) {
        return "free";
    }
    if (id == g.e_thread.load(std::memory_order_relaxed)) {
        return "GPUBack (E)";
    }
    if (id == g.r_thread.load(std::memory_order_relaxed)) {
        return "GPURender (R)";
    }
    return fmt::format("another thread ({:#x})", std::hash<std::thread::id>{}(id));
}

/// The watchdog's second line: the front's wait and the acquires it may be held by.
void ReportFrontAndAcquires(bool front_stuck, bool e_outside, u64 now_ns) {
    auto& split = GpuSplit::Get();
    const auto& front = split.FrontWaitNow();
    const u64 begin = front.begin_ns.load(std::memory_order_acquire);
    const u32 site = front.site.load(std::memory_order_relaxed);
    std::string waiters;
    u32 count = 0;
    Tegra::Nx2SemaphoreWaiters::Get().ForEach(
        [&](u64 address, u32 value, u32 operation, bool latched, const u32* word) {
            if (++count > 16) {
                return;
            }
            waiters += fmt::format(
                "{}{:#x} {} {:#x}{}{}", waiters.empty() ? "" : "; ", address,
                operation == Tegra::Nx2SemaphoreWaiters::AcquireEqual    ? "=="
                : operation == Tegra::Nx2SemaphoreWaiters::AcquireGequal ? ">="
                : operation == Tegra::Nx2SemaphoreWaiters::AcquireMask   ? "&"
                                                                          : "?",
                value, latched ? " latched" : "",
                word != nullptr ? fmt::format(" (word now {:#x})",
                                              std::atomic_ref<u32>{*const_cast<u32*>(word)}
                                                  .load(std::memory_order_relaxed))
                                : std::string{});
        });
    LOG_CRITICAL(
        HW_GPU,
        "One follower hang watchdog: front {} (site {}, target {}, {}, for {:.1f} s, wait #{}); "
        "published {}, consumed (E) {}, played (R) {}; E {}; acquire waiters {}: {}; latches by "
        "R {}, E {}, other {}",
        begin != 0 ? (front_stuck ? "stuck in a wait" : "in a wait") : "not in a wait",
        site < drain_site_names.size() ? drain_site_names[site] : "none",
        front.target.load(std::memory_order_relaxed),
        front.on_engine.load(std::memory_order_relaxed) ? "on E's position" : "on R's position",
        begin != 0 ? static_cast<double>(now_ns - begin) / 1e9 : 0.0,
        front.seq.load(std::memory_order_relaxed), split.PublishedNow(),
        split.EngineConsumedNow(), split.PlayedNow(std::memory_order_acquire),
        e_outside ? "owes work and has not moved for 2 s outside its one-follower waits"
                  : "not stuck outside its waits",
        count, waiters.empty() ? std::string{"none"} : waiters,
        g_latches[0].load(std::memory_order_relaxed), g_latches[1].load(std::memory_order_relaxed),
        g_latches[2].load(std::memory_order_relaxed));
}

/// The hang watchdog (FRITCHII_ONEF=1): one wake a second. Two seconds without progress where
/// progress is owed (R has a packet it does not finish, or packets it does not take; E in one wait
/// that does not end) logs the whole state once, at Critical, then gives both sides a wake (a
/// lost wakeup ends there, and the log says so when progress resumes).
void WatchMain(std::stop_token stop) {
    Common::SetCurrentThreadName("GPUWatch");
    struct Seen {
        u64 played{};
        u64 r_packet{};
        u64 r_progress{};
        u64 e_wait_seq{};
        u64 front_seq{};
        u64 consumed{};
    } last{};
    u32 r_strikes = 0;
    u32 e_strikes = 0;
    u32 e_out_strikes = 0;
    bool reported = false;
    bool lost_wakeup = false;
    for (;;) {
        for (int i = 0; i < 10; ++i) {
            if (stop.stop_requested()) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
        }
        TraceFlush(false);
        const u64 played = g.played.load(std::memory_order_acquire);
        const u64 sent_packets = g.sent_packets.load(std::memory_order_acquire);
        const u64 r_packet = g.r_packet.load(std::memory_order_relaxed);
        const u64 r_progress = g.r_progress.load(std::memory_order_relaxed);
        const u8 e_wait = g.e_wait.load(std::memory_order_relaxed);
        const u64 e_wait_seq = g.e_wait_seq.load(std::memory_order_relaxed);
        const u64 sent = g.sent.load(std::memory_order_acquire);
        const u64 freed = g.freed.load(std::memory_order_acquire);
        // R owes progress when it holds a packet, or when packets wait in the ring.
        const bool r_owes = r_packet != 0 || sent != freed || sent_packets > played;
        const bool r_still = r_owes && r_packet == last.r_packet && r_progress == last.r_progress;
        const bool e_still = e_wait != 0 && e_wait_seq == last.e_wait_seq;
        r_strikes = r_still ? r_strikes + 1 : 0;
        e_strikes = e_still ? e_strikes + 1 : 0;
        // The front in one wait for over 2 s (GpuSplit's WaitFor or Drain), and E owing work
        // (the front published past its position) without moving and outside its own waits: both
        // were invisible to the R and E strikes (the first QUERY_OPS hang).
        auto& split_now = GpuSplit::Get();
        const auto& front = split_now.FrontWaitNow();
        const u64 front_begin = front.begin_ns.load(std::memory_order_acquire);
        const u64 front_seq = front.seq.load(std::memory_order_relaxed);
        const u64 now_ns = Common::Nx2Hitch::NowNs();
        const bool front_stuck = front_begin != 0 && now_ns - front_begin > 2'000'000'000ULL;
        const u64 consumed_now = split_now.EngineConsumedNow();
        const bool e_out_still = e_wait == 0 && split_now.PublishedNow() > consumed_now &&
                                 consumed_now == last.consumed;
        e_out_strikes = e_out_still ? e_out_strikes + 1 : 0;
        last = Seen{played, r_packet, r_progress, e_wait_seq, front_seq, consumed_now};
        const bool hung = r_strikes >= 2 || e_strikes >= 2 || front_stuck || e_out_strikes >= 2;
        if (!hung) {
            if (reported) {
                LOG_CRITICAL(HW_GPU, "One follower hang watchdog: progress again ({}; {} wakes "
                                     "given this session)",
                             lost_wakeup ? "a lost wakeup, ended by the watchdog's wake"
                                         : "a long wait that ended",
                             g.rescued.load(std::memory_order_relaxed));
                reported = false;
            }
            continue;
        }
        if (!reported) {
            reported = true;
            // A lost wakeup: a side asleep although what it waits for is there.
            lost_wakeup = (g.r_sleeping.load(std::memory_order_seq_cst) && sent != freed) ||
                          (e_wait == static_cast<u8>(EWait::Park) &&
                           g.e_waiting.load(std::memory_order_seq_cst) &&
                           played >= g.e_wait_number.load(std::memory_order_relaxed));
            auto& split = GpuSplit::Get();
            std::string locks;
            const u32 lock_count = g.lock_count.load(std::memory_order_acquire);
            for (u32 i = 0; i < lock_count; ++i) {
                locks += fmt::format("{}{} held by {}", i == 0 ? "" : ", ", g.locks[i].first,
                                     ThreadName(g.locks[i].second->Owner()));
            }
            const char* const back_site = back_wait_site.load(std::memory_order_relaxed);
            ReportFrontAndAcquires(front_stuck, e_out_strikes >= 2, now_ns);
            // A lost raise: R has nothing unplayed and E is idle, yet `played` is short of
            // E's position. Raise it (the min of the two, as RaisePlayed always takes) and say so.
            const u64 intended = g.intended.load(std::memory_order_seq_cst);
            if (!r_owes && split.PlayedNow(std::memory_order_seq_cst) < intended) {
                LOG_CRITICAL(HW_GPU,
                             "One follower hang watchdog: a lost raise of `played` ({} short of "
                             "E's {}, R with nothing unplayed): raised now",
                             split.PlayedNow(std::memory_order_relaxed), intended);
                RaisePlayed();
            }
            const u8 r_kind = g.r_kind.load(std::memory_order_relaxed);
            const u8 r_op = g.r_op.load(std::memory_order_relaxed);
            const u8 e_op = g.e_wait_op.load(std::memory_order_relaxed);
            LOG_CRITICAL(
                HW_GPU,
                "One follower hang watchdog: no progress for 2 s ({}{}{}{}{}); step {}; "
                "positions: published {}, consumed (E) {}, played (R) {}; packets: pushed {}, "
                "played {}, ring bytes sent {} freed {}; E: {} (op {}, packet {}), e_waiting {}, "
                "back wait site {}; R: {} (packet {}, kind {}, op {}), r_sleeping {}; locks: {}",
                r_strikes >= 2 ? "R" : "", e_strikes >= 2 ? (r_strikes >= 2 ? " and E" : "E") : "",
                front_stuck ? " the front in one wait" : "",
                e_out_strikes >= 2 ? " E owing work outside its waits" : "",
                lost_wakeup ? ", a lost wakeup: a side asleep with its condition met" : "",
                static_cast<int>(g.step.load(std::memory_order_relaxed)), split.PublishedNow(),
                split.EngineConsumedNow(), split.PlayedNow(std::memory_order_acquire),
                sent_packets, played, sent, freed,
                e_wait_names[(std::min)(static_cast<size_t>(e_wait), e_wait_names.size() - 1)],
                e_op < lent_op_names.size() ? lent_op_names[e_op] : "?",
                g.e_wait_number.load(std::memory_order_relaxed),
                g.e_waiting.load(std::memory_order_relaxed),
                back_site != nullptr ? back_site : "none tracked",
                r_packet != 0 ? "in a packet"
                : g.r_sleeping.load(std::memory_order_relaxed) ? "asleep"
                                                               : "between packets",
                r_packet,
                r_kind == static_cast<u8>(PacketKind::Lent)    ? "lent"
                : r_kind == static_cast<u8>(PacketKind::Draw)  ? "draw"
                : r_kind == static_cast<u8>(PacketKind::Delta) ? "delta"
                                                               : "?",
                r_kind == static_cast<u8>(PacketKind::Lent) && r_op < lent_op_names.size()
                    ? lent_op_names[r_op]
                    : "-",
                g.r_sleeping.load(std::memory_order_relaxed),
                locks.empty() ? std::string{"none registered"} : locks);
        }
        // A wake to both sides: harmless if they are not asleep (each rechecks its condition).
        g.rescued.fetch_add(1, std::memory_order_relaxed);
        g.r_wake.fetch_add(1, std::memory_order_seq_cst);
        g.r_wake.notify_one();
        g.e_wake.fetch_add(1, std::memory_order_seq_cst);
        g.e_wake.notify_one();
    }
}

/// E: the header's draw fields from SetNextDraw (consumed), for Stats and the draw span.
u32 TakeDrawKind(PacketHeader& header) {
    if (!t_next_draw_set) {
        return 0;
    }
    header.reserved[0] = static_cast<u8>(t_next_draw.kind);
    return t_next_draw.kind;
}

} // Anonymous namespace

namespace Detail {
const DrawValues* NextDraw() noexcept {
    return t_next_draw_set ? &t_next_draw : nullptr;
}
} // namespace Detail

void ReadSwitches() {
    if (g.read) {
        return; // boot switches: the first session's values stick
    }
    g.read = true;
    using namespace VideoCore::Nx2Debug;
    g.lean = SwitchValue(g_onef_lean, "FRITCHII_ONEF_LEAN") == 1;
    // Before E starts (ReadSwitches runs ahead of the back thread): LEAN is always at step 2 or
    // more once it runs, and tracking from boot only marks words the first seed then covers.
    g_track_writes_always = g.lean;
    g.on = g.lean || SwitchValue(g_onef, "FRITCHII_ONEF") == 1;
    g.pin = SwitchValue(g_onef_pin, "FRITCHII_ONEF_PIN", 1) != 0;
    g_trace = g.on && SwitchValue(g_onef_trace, "FRITCHII_ONEF_TRACE") == 1;
    if (g.lean) {
        // The verified configuration as defaults; a switch set in the environment (the test
        // block) or read before this keeps its value.
        const auto preset = [](std::atomic<int>& value, const char* name, int lean) {
            int unset = -1;
            if (std::getenv(name) == nullptr && value.compare_exchange_strong(unset, lean)) {
                BumpSwitchGeneration();
            }
            return SwitchValue(value, name);
        };
        const int unlend = preset(g_onef_unlend, "FRITCHII_ONEF_UNLEND", 3);
        const int e_pipeline = preset(g_onef_e_pipeline, "FRITCHII_ONEF_E_PIPELINE", 1);
        const int query_ops = preset(g_onef_query_ops, "FRITCHII_ONEF_QUERY_OPS", 1);
        const int waits_e = preset(g_onef_waits_e, "FRITCHII_ONEF_WAITS_E", 1);
        const int inval_e = preset(g_onef_inval_e, "FRITCHII_ONEF_INVAL_E", 1);
        const int verify_every =
            preset(g_onef_verify_every, "FRITCHII_ONEF_VERIFY_EVERY", 0);
        // EXP12=5, the lagged acquire check (fable_b 11.3), from boot. Loaded first, so the
        // experiments' own load does not replace it later.
        (void)Experiment(12);
        if (std::getenv("EDEN_NX2_EXP12") == nullptr) {
            g_experiments[12].store(5, std::memory_order_relaxed);
            BumpSwitchGeneration();
        }
        LOG_INFO(HW_GPU,
                 "One follower LEAN (FRITCHII_ONEF_LEAN=1): step 4 from boot, view and build "
                 "compares off, no forced verify; UNLEND={:#x} E_PIPELINE={} QUERY_OPS={} "
                 "WAITS_E={} INVAL_E={} VERIFY_EVERY={} EXP12={}",
                 unlend, e_pipeline, query_ops, waits_e, inval_e, verify_every, Experiment(12));
    }
    // Default 1 (the step 4 probes ran with it; 0 keeps the facts table path).
    g_e_pipeline = g.on && SwitchValue(g_onef_e_pipeline, "FRITCHII_ONEF_E_PIPELINE", 1) == 1;
    if (g_e_pipeline) {
        LOG_INFO(HW_GPU, "One follower: E owns the pipeline lookup (FRITCHII_ONEF_E_PIPELINE=1)");
    }
    if (g.on) {
        LOG_INFO(HW_GPU,
                 "One follower (FRITCHII_ONEF=1): the render thread R runs the renderer, every "
                 "draw and render operation lent; R {}pinned; the memory stage stays off",
                 g.pin ? "" : "not ");
    }
}

bool On() noexcept {
    return g.on;
}

StepLevel Step() noexcept {
    if (!g.on) {
        return StepLevel::Off;
    }
    const u32 generation =
        VideoCore::Nx2Debug::g_switch_generation.load(std::memory_order_acquire);
    if (generation != g.seen_generation.load(std::memory_order_acquire)) [[unlikely]] {
        Refresh(generation);
    }
    return static_cast<StepLevel>(g.step.load(std::memory_order_relaxed));
}

bool OnRenderThread() noexcept {
    return t_render;
}

bool InLent() noexcept {
    return t_render && t_lent_depth != 0;
}

bool PlayingUnlentDraw() noexcept {
    return t_playing_draw;
}

void DisableUnlend(const char* why) noexcept {
    if (g.unlend_disabled.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    LOG_CRITICAL(HW_GPU,
                 "One follower: step 4 stopped for this session ({}); FRITCHII_ONEF_UNLEND runs 0, "
                 "every draw is lent again",
                 why != nullptr ? why : "?");
    g.seen_generation.store(~u32{0}, std::memory_order_release); // Refresh at the next Step()
}

void MarkEngineThread() noexcept {
    if (g.on) {
        Detail::t_engine = true;
        g.e_thread.store(std::this_thread::get_id(), std::memory_order_relaxed);
    }
}

void Start(Core::System& system) {
    if (!g.on || g.thread.joinable()) {
        return;
    }
    g.system = &system;
    if (!g.ring) {
        g.ring = std::make_unique<u8[]>(RingBytes);
        g.positions = std::make_unique<u64[]>(PositionSlots);
        g.push_ns = std::make_unique<u64[]>(PositionSlots);
    }
    // A session starts with empty positions (GpuSplit's ring starts at 0 again).
    g.write = 0;
    g.pushed = 0;
    g.draw_seq = 0;
    g.sent.store(0, std::memory_order_relaxed);
    g.sent_packets.store(0, std::memory_order_relaxed);
    g.intended.store(0, std::memory_order_relaxed);
    g.read_at = 0;
    g.freed.store(0, std::memory_order_relaxed);
    g.played.store(0, std::memory_order_relaxed);
    g.engine.store(nullptr, std::memory_order_relaxed);
    View() = RegisterView{};
    Detail::ViewTrackingStopped();
    g.seen_generation.store(~u32{0}, std::memory_order_relaxed);
    (void)Step();
    if (g_trace && !g_trace_ring) {
        g_trace_ring = std::make_unique<TraceRecord[]>(TraceSlots);
        const auto path =
            Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) / "onef_lend_trace.bin";
        g_trace_file.open(path, std::ios::binary | std::ios::trunc);
        LOG_INFO(HW_GPU, "One follower lend trace (FRITCHII_ONEF_TRACE=1): {} ({} MiB ring)",
                 path.string(), TraceSlots * sizeof(TraceRecord) >> 20);
    }
    g.thread = std::jthread([](std::stop_token stop) { RenderMain(stop); });
    g.watchdog = std::jthread([](std::stop_token stop) { WatchMain(stop); });
}

void Stop() {
    if (!g.thread.joinable()) {
        return;
    }
    if (g.watchdog.joinable()) {
        g.watchdog.request_stop();
        g.watchdog.join();
    }
    g.thread.request_stop();
    g.r_wake.fetch_add(1, std::memory_order_seq_cst);
    g.r_wake.notify_one();
    g.thread.join();
    TraceFlush(true);
    if (g_trace_file.is_open()) {
        g_trace_file.close();
    }
}

void WatchLock(const char* name, const Common::SpinRecursiveMutex* mutex) noexcept {
    const u32 count = g.lock_count.load(std::memory_order_relaxed);
    for (u32 i = 0; i < count; ++i) {
        if (std::strcmp(g.locks[i].first, name) == 0) {
            g.locks[i].second = mutex; // a new rasterizer (session): its lock replaces the old
            return;
        }
    }
    const u32 index = count;
    if (index >= g.locks.size()) {
        return;
    }
    g.locks[index] = {name, mutex};
    g.lock_count.store(index + 1, std::memory_order_release);
}

void SetNextDraw(u32 kind, bool indexed, bool indirect, u32 instance_arg) noexcept {
    t_next_draw = DrawValues{};
    t_next_draw.kind = kind;
    t_next_draw.indexed = indexed ? 1 : 0;
    t_next_draw.indirect = indirect ? 1 : 0;
    t_next_draw.instance_arg = instance_arg;
    t_next_draw_set = true;
}

namespace {
void PushPacked(PacketHeader& header, u8* buffer, std::size_t body_bytes);
} // Anonymous namespace

void Lend(LentOp op, LentReason reason, void (*function)(void*), void* argument) {
    if (!Detail::t_engine || !g.thread.joinable()) {
        if (op == LentOp::Draw) {
            t_next_draw_set = false;
        }
        t_lend_site = nullptr;
        function(argument); // R, another thread, or R not running: in place, as today
        return;
    }
    const StepLevel step = Step();
    const bool timing = Stats::Timing();
    const u64 begin = timing ? Stats::Now() : 0;
    const bool counting = CountersOn();
    const u64 begin_clock = counting ? Common::Nx2Hitch::NowNs() : 0;
    const u64 behind = counting ? Behind() : 0;
    const u64 number = ++g.pushed;
    if (g_trace) [[unlikely]] {
        TraceArrive(number, t_lend_site, op, false, Common::Nx2Hitch::NowNs());
    }
    u8* const buffer = PackBuffer();
    std::size_t spans_bytes = 0;
    PacketHeader header{};
    header.kind = PacketKind::Lent;
    header.op = static_cast<u8>(op);
    header.reason = static_cast<u8>(reason);
    header.number = number;
    header.position = GpuSplit::Get().BackEntryPosition();
    header.draw_seq = op == LentOp::Draw ? ++g.draw_seq : g.draw_seq;
    header.call = LentCall{function, argument};
    const u32 kind = op == LentOp::Draw ? TakeDrawKind(header) : 0;
    if (step >= StepLevel::View) {
        if (Maxwell3D* const engine = g.engine.load(std::memory_order_acquire)) {
            // The spans: everything E changed since the last packet, after the header.
            spans_bytes = PackSpans(*engine, header,
                                    std::span<u8>{buffer + sizeof(PacketHeader), MaxSpanBytes});
            if (op == LentOp::Draw && !g.lean) {
                header.flags |= PacketVerify; // FRITCHII_ONEF_LEAN: no view compare
            }
        }
    }
    if (op == LentOp::Draw) {
        t_next_draw_set = false;
    }
    header.bytes = static_cast<u32>(sizeof(PacketHeader) + spans_bytes);
    std::memcpy(buffer, &header, sizeof(header));
    Commit(buffer, header.bytes, number, header.position, false);
    if (counting) {
        Stats::Pushed(header, kind, behind);
    }
    // E never parks holding a lock R may take (the deadlock of 12 Nov: E parked in a seam play's
    // lend holding the buffer cache's lock, R in an earlier draw packet waiting for it): a
    // watched lock E holds goes to R for the park (R takes it as its own, recursively; guest
    // threads stay out) and comes back after. R reaching into the buffer cache outside the lent
    // call still counts as "owner on R" (its entry points' asserts name where).
    std::array<Common::SpinRecursiveMutex*, 4> handed{};
    u32 handed_count = 0;
    {
        const std::thread::id self = std::this_thread::get_id();
        const std::thread::id render = g.r_thread.load(std::memory_order_relaxed);
        const u32 lock_count = g.lock_count.load(std::memory_order_acquire);
        for (u32 i = 0; i < lock_count && handed_count < handed.size(); ++i) {
            auto* const mutex = const_cast<Common::SpinRecursiveMutex*>(g.locks[i].second);
            if (mutex != nullptr && mutex->Owner() == self) {
                mutex->HandOver(render);
                handed[handed_count++] = mutex;
            }
        }
        if (handed_count != 0) {
            g.lends_holding_lock.fetch_add(1, std::memory_order_relaxed);
        }
    }
    SCOPE_EXIT {
        const std::thread::id self = std::this_thread::get_id();
        for (u32 i = 0; i < handed_count; ++i) {
            handed[i]->HandOver(self);
        }
    };
    // The park: R plays everything before, runs the call, releases E. Its own profile zone
    // (fable_b 14.1): without it the park landed in whatever zone the lend was made in ("other"
    // for a call entry, "method" for a method's).
    g.e_wait_op.store(static_cast<u8>(op), std::memory_order_relaxed);
    {
        const Nx2Profile::Scope parked_zone{Nx2Profile::Zone::OnefParked};
        EngineWait([&] { return g.played.load(std::memory_order_acquire) >= number; },
                   EWait::Park, number);
    }
    if (g_trace) [[unlikely]] {
        const u64 now = Common::Nx2Hitch::NowNs();
        g_trace_ring[number % TraceSlots].e_resume = now;
        t_e_last = now;
    }
    const u64 parked_ns = counting ? Common::Nx2Hitch::NowNs() - begin_clock : 0;
    if (counting) {
        NoteLendSite(op, parked_ns);
    } else {
        t_lend_site = nullptr;
        t_query_kind = QueryKind::Count; // consumed uncounted
    }
    if (counting && op == LentOp::Query && t_query_kind != QueryKind::Count) {
        const size_t kind_index = static_cast<size_t>(t_query_kind);
        g.query_count[kind_index].fetch_add(1, std::memory_order_relaxed);
        g.query_parked_ns[kind_index].fetch_add(parked_ns, std::memory_order_relaxed);
        t_query_kind = QueryKind::Count;
    }
    if (timing) {
        Stats::Parked(op, Stats::Now() - begin);
    }
    if (g.lent_error) [[unlikely]] {
        std::rethrow_exception(std::exchange(g.lent_error, nullptr));
    }
    if (g.op_error) [[unlikely]] {
        std::rethrow_exception(std::exchange(g.op_error, nullptr));
    }
}

void PushOp(LentOp op, void (*function)(void*), const void* payload, u32 size,
            bool lands_in_memory, u8 packet_flags) {
    if (!Detail::t_engine || !g.thread.joinable() || size > MaxPayloadBytes) {
        t_lend_site = nullptr;
        function(const_cast<void*>(payload));
        return;
    }
    if (op == LentOp::Query && t_query_kind != QueryKind::Count) {
        g.query_count[static_cast<size_t>(t_query_kind)].fetch_add(1, std::memory_order_relaxed);
        t_query_kind = QueryKind::Count;
    }
    u8* const buffer = PackBuffer();
    PacketHeader header{};
    header.kind = PacketKind::Op;
    header.flags = packet_flags;
    header.op = static_cast<u8>(op);
    header.call = LentCall{function, nullptr};
    std::size_t spans_bytes = 0;
    if (Step() >= StepLevel::View) {
        if (Maxwell3D* const engine = g.engine.load(std::memory_order_acquire)) {
            // E's changes before the op, so R's view is at the op's place when it runs.
            spans_bytes = PackSpans(*engine, header,
                                    std::span<u8>{buffer + sizeof(PacketHeader), MaxSpanBytes});
        }
    }
    header.reserved2 = static_cast<u32>(spans_bytes);
    std::memcpy(buffer + sizeof(PacketHeader) + spans_bytes, payload, size);
    const char* const trace_site = std::exchange(t_lend_site, nullptr);
    PushPacked(header, buffer, spans_bytes + ((size + 7) & ~u32{7}));
    if (g_trace) [[unlikely]] {
        TraceArrive(header.number, trace_site, op, true, Common::Nx2Hitch::NowNs());
    }
    if (lands_in_memory) {
        // Before E's next `consumed` store (seq_cst there): a front that sees E past this entry
        // sees this mark too.
        g.guest_write_op.store(header.position + 1, std::memory_order_release);
    }
}

bool WaitsE() noexcept {
    return g.on && VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_waits_e,
                                                     "FRITCHII_ONEF_WAITS_E") != 0;
}

u64 GuestWriteOpMark() noexcept {
    return g.guest_write_op.load(std::memory_order_acquire);
}

bool QueryOps() noexcept {
    using namespace VideoCore::Nx2Debug;
    // From step 3 only: below it R's objects read E's engine, which a render op (E not parked)
    // must not.
    const bool on = g.on &&
                    (SwitchValue(g_onef_query_ops, "FRITCHII_ONEF_QUERY_OPS") != 0 ||
                     SwitchValue(g_onef_posted, "FRITCHII_ONEF_POSTED") != 0) &&
                    Step() >= StepLevel::Build;
    if (on && !g.query_ops_logged.load(std::memory_order_relaxed) &&
        !g.query_ops_logged.exchange(true)) [[unlikely]] {
        // fable_b 11.3: with reports landing on R the full-drain acquire poll (EXP12 = 4) drains
        // to R; the lagged check (EXP12 = 5) is the mode to run with, its target R's position
        // (`played`). Not switched here: the wait mode of acquires already suspended does not
        // change under them (the 4 -> 5 flip went with the first QUERY_OPS hang). Set it from
        // boot.
        const int exp12 = Experiment(12);
        if (exp12 != 5 && exp12 != 6) {
            LOG_WARNING(HW_GPU,
                        "One follower: FRITCHII_ONEF_QUERY_OPS=1 with EXP12={}: set EXP12=5 from "
                        "boot (the lagged acquire check, fable_b 11.3)",
                        exp12);
        }
        LOG_INFO(HW_GPU,
                 "One follower: FRITCHII_ONEF_QUERY_OPS=1 (render enables that cannot set host "
                 "conditional rendering not sent, resets and plain reports as render ops); EXP12 "
                 "{}",
                 exp12);
    }
    return on;
}

bool InvalOnEngine() noexcept {
    using namespace VideoCore::Nx2Debug;
    if (!g.on || SwitchValue(g_onef_inval_e, "FRITCHII_ONEF_INVAL_E") == 0) {
        return false;
    }
    if (!g_e_pipeline) {
        // Exact only while E owns the shader registrations (rule 3): R's own lookups would mark
        // the shader filter without E seeing it in order.
        if (!g.inval_refused_logged.exchange(true, std::memory_order_relaxed)) {
            LOG_WARNING(HW_GPU, "One follower: FRITCHII_ONEF_INVAL_E=1 needs "
                                "FRITCHII_ONEF_E_PIPELINE=1 (boot); running 0");
        }
        return false;
    }
    if (Step() != StepLevel::Unlend) {
        return false;
    }
    if (!g.stamps_on.load(std::memory_order_acquire)) [[unlikely]] {
        // The first call with the switch on: R starts reading the table once `stamps_on` shows
        // it; never freed while the process runs (R may be reading it).
        static std::once_flag once;
        std::call_once(once, [] {
            g.stamps = std::make_unique<std::atomic<u32>[]>(StampSlots);
            g.stamps_on.store(true, std::memory_order_release);
            LOG_INFO(HW_GPU,
                     "One follower: page write stamps on (FRITCHII_ONEF_INVAL_E=1, {} MiB)",
                     StampSlots * sizeof(u32) >> 20);
        });
    }
    return true;
}

bool PostClears() noexcept {
    return g.on && Step() == StepLevel::Unlend &&
           VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_op_clear,
                                            "FRITCHII_ONEF_OP_CLEAR") != 0 &&
           ViewReady();
}

bool PostOpSite(u8 site) noexcept {
    if (!g.on || Step() != StepLevel::Unlend) {
        return false;
    }
    using namespace VideoCore::Nx2Debug;
    switch (site) {
    case 15: // MemStage::Site::InlineUpload
        return SwitchValue(g_onef_op_inline, "FRITCHII_ONEF_OP_INLINE") != 0;
    default:
        return false;
    }
}

void CountOpPacket(OpCount kind, u64 bytes) noexcept {
    g.op_count[static_cast<size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
    if (bytes != 0) {
        g.op_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
}

bool GatherOnEngine() noexcept {
    const int value = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_gather_e,
                                                       "FRITCHII_ONEF_GATHER_E");
    const bool inval = value != 0 && InvalOnEngine();
    // The guard's parts, logged once per change (the GATHER_E probe found it never true).
    static std::atomic<int> logged{-1};
    const int state = (value != 0 ? 1 : 0) | (inval ? 2 : 0) |
                      (static_cast<int>(Step()) << 2) | (g_e_pipeline ? 32 : 0);
    if (logged.exchange(state, std::memory_order_relaxed) != state) [[unlikely]] {
        LOG_INFO(HW_GPU,
                 "One follower: FRITCHII_ONEF_GATHER_E={} (INVAL_E's conditions {}, step {}, "
                 "E_PIPELINE {}, INVAL_E switch {})",
                 value, inval ? "met" : "not met", static_cast<int>(Step()), g_e_pipeline ? 1 : 0,
                 VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_inval_e,
                                                  "FRITCHII_ONEF_INVAL_E"));
    }
    return inval;
}

bool CopyOnEngine() noexcept {
    return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_copy_e,
                                            "FRITCHII_ONEF_COPY_E") != 0 &&
           InvalOnEngine();
}

void StampPageWrites(u64 address, u64 size) noexcept {
    if (size == 0 || !g.stamps_on.load(std::memory_order_acquire)) {
        return;
    }
    const u32 stamp = static_cast<u32>(g.pushed + 1);
    for (u64 page = address >> 12; page <= (address + size - 1) >> 12; ++page) {
        g.stamps[page & (StampSlots - 1)].store(stamp, std::memory_order_relaxed);
    }
}

void CountInvalidation(InvalKind kind) noexcept {
    if (!CountersOn()) {
        return;
    }
    g.inval_count[static_cast<size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
}

bool WrittenAheadOfRender(u64 address, u64 size) noexcept {
    if (!Detail::t_render || size == 0 || !g.stamps_on.load(std::memory_order_acquire)) {
        return false;
    }
    // The other half of E's pair: R marked the texture filter (registering the image) before
    // this; E stamps, fences, then looks at the filter again. Either E saw the mark (the
    // invalidation went to R in order) or this sees E's stamp.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    g.ahead_checks.fetch_add(1, std::memory_order_relaxed);
    const u32 place = t_play_number;
    for (u64 page = address >> 12; page <= (address + size - 1) >> 12; ++page) {
        const u32 stamp = g.stamps[page & (StampSlots - 1)].load(std::memory_order_relaxed);
        if (stamp != 0 && static_cast<s32>(stamp - place) > 0) {
            thread_local std::string note;
            note = fmt::format("image {:#x}+{:#x}: page {:#x} written by E before packet {}, R "
                               "at packet {}",
                               address, size, page << 12, stamp, place);
            Note(Check::TextureReadAhead, note.c_str());
            return true;
        }
    }
    return false;
}

bool QueryOpsOnBufferPages() noexcept {
    return QueryOps() && (Posted() || VideoCore::Nx2Debug::SwitchValue(
                                          VideoCore::Nx2Debug::g_onef_query_ops,
                                          "FRITCHII_ONEF_QUERY_OPS") >= 2);
}

bool Posted() noexcept {
    return g.on && VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_posted,
                                                    "FRITCHII_ONEF_POSTED") != 0;
}

void CountLentFence(bool buffer_flush) noexcept {
    g.fences_lent.fetch_add(1, std::memory_order_relaxed);
    if (buffer_flush) {
        g.fences_with_flush.fetch_add(1, std::memory_order_relaxed);
    }
}

void CountLentGatherRanges(u64 ranges) noexcept {
    g.gathers_lent.fetch_add(1, std::memory_order_relaxed);
    g.gather_ranges.fetch_add(ranges, std::memory_order_relaxed);
}

void SetLendSite(const char* site) noexcept {
    t_lend_site = site;
}

void SetQueryKind(QueryKind kind) noexcept {
    t_query_kind = kind;
}

void CountQueryKind(QueryKind kind) noexcept {
    if (!CountersOn()) {
        return;
    }
    g.query_count[static_cast<size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
}

namespace {
/// E: commits a packet whose body (`body_bytes` after the header) is already in PackBuffer().
void PushPacked(PacketHeader& header, u8* buffer, std::size_t body_bytes) {
    const bool counting = CountersOn();
    const u64 behind = counting ? Behind() : 0;
    header.number = ++g.pushed;
    header.position = GpuSplit::Get().BackEntryPosition();
    if (header.draw_seq == 0) {
        header.draw_seq = header.kind == PacketKind::Draw ? ++g.draw_seq : g.draw_seq;
    }
    header.bytes = static_cast<u32>(sizeof(PacketHeader) + body_bytes);
    const u32 kind = header.kind == PacketKind::Draw ? TakeDrawKind(header) : 0;
    t_next_draw_set = false;
    std::memcpy(buffer, &header, sizeof(header));
    Commit(buffer, header.bytes, header.number, header.position, true);
    if (counting) {
        Stats::Pushed(header, kind, behind);
    }
}
} // Anonymous namespace

void Push(PacketHeader& header, std::span<const u8> spans) {
    if (!Detail::t_engine || !g.thread.joinable()) {
        t_next_draw_set = false;
        return;
    }
    u8* const buffer = PackBuffer();
    if (!spans.empty()) {
        std::memcpy(buffer + sizeof(PacketHeader), spans.data(), spans.size());
    }
    PushPacked(header, buffer, spans.size());
}

void Drain() {
    if (!Detail::t_engine) {
        return;
    }
    const u64 target = g.pushed;
    const u64 waited =
        EngineWait([&] { return g.played.load(std::memory_order_acquire) >= target; },
                   EWait::Drain, target);
    if (waited != 0) {
        Stats::Drained(Stats::Timing() ? waited : 0);
    }
}

u64 Behind() noexcept {
    return g.pushed - g.played.load(std::memory_order_acquire);
}

void EngineConsumed(u64 position) noexcept {
    g.intended.store(position, std::memory_order_seq_cst); // RaisePlayed's fence pairs with R's
    RaisePlayed();
}

void PushDraw(const Maxwell3D& engine, u64 draw_seq, u64 buffer_number, u8 flags,
              const void* pipeline) {
    if (!Detail::t_engine || !g.thread.joinable()) {
        t_next_draw_set = false;
        return;
    }
    // Part (c)'s unlent draw: its ring B packet is filled; this packet carries the spans (the
    // draw span always) and names it. E goes on within FRITCHII_ONEF_AHEAD.
    PacketHeader header{};
    header.kind = PacketKind::Draw;
    header.flags = flags;
    header.draw_seq = draw_seq;
    header.buffer_number = buffer_number;
    header.pipeline = pipeline;
    // Packed straight into the packet buffer, after the header: no copy, no allocation.
    u8* const buffer = PackBuffer();
    const std::size_t bytes =
        PackSpans(engine, header, std::span<u8>{buffer + sizeof(PacketHeader), MaxSpanBytes});
    PushPacked(header, buffer, bytes);
}

void SetEngine(Maxwell3D* engine) noexcept {
    if (g.engine.exchange(engine, std::memory_order_acq_rel) != engine) {
        // A new channel's engine: the view takes it in full at the next lent packet, and E's
        // tracking starts over (the channel bind is lent, so E is parked here).
        Detail::ViewTrackingStopped();
    }
}

Maxwell3D* Engine() noexcept {
    return g.engine.load(std::memory_order_acquire);
}

void CheckRenderOwned(const char* where) noexcept {
    if (!Detail::t_engine) {
        return; // R itself, or a thread the caches' locks already order
    }
    // E with every packet played is R parked or idle: today's serial order.
    if (g.played.load(std::memory_order_acquire) >= g.pushed) {
        return;
    }
    Note(Check::OwnerOnE, where);
}

void CheckEngineOwned(const char* where) noexcept {
    if (t_render && t_lent_depth == 0) {
        Note(Check::OwnerOnR, where);
    }
}

u32 UnlendMask() noexcept {
    (void)Step();
    return g.unlend_mask.load(std::memory_order_relaxed);
}

int VerifyEvery() noexcept {
    (void)Step();
    return g.verify_every.load(std::memory_order_relaxed);
}

int Ahead() noexcept {
    (void)Step();
    return g.ahead.load(std::memory_order_relaxed);
}

} // namespace VideoCommon::OneFollower
