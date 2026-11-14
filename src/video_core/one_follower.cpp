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
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "common/dkb_cpu_trace.h"
#include "common/logging.h"
#include "common/thread.h"
#include "core/core.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/one_follower.h"
#include "video_core/one_follower_stats.h"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
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
    bool on{};   ///< FRITCHII_ONEF=1 at boot (sticks for the process)
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

    std::atomic<Maxwell3D*> engine{};
};

Impl g;

using Detail::t_render;
thread_local u32 t_lent_depth{};
/// E's packet scratch: header and spans before they go into the ring.
thread_local std::vector<u8> t_scratch;
/// E: the draw the next Lend or PackSpans is for (SetNextDraw), consumed by them.
thread_local DrawValues t_next_draw{};
thread_local bool t_next_draw_set{};

/// Applies the nesting: each step needs the one before; a refused switch is logged once per value.
void Refresh(u32 generation) {
    std::scoped_lock lock{g.refresh_mutex};
    using namespace VideoCore::Nx2Debug;
    const int view = SwitchValue(g_onef_view, "FRITCHII_ONEF_VIEW");
    const int build = SwitchValue(g_onef_build, "FRITCHII_ONEF_BUILD");
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
        if (level != StepLevel::Build) {
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
    g.seen_generation.store(generation, std::memory_order_release);
}

/// E waits until `ready` holds; returns the nanoseconds waited (0 at once, or without the profile).
template <typename Ready>
u64 EngineWait(Ready&& ready) {
    if (ready()) {
        return 0;
    }
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
        waited += EngineWait([&] {
            return g.write + pad - g.freed.load(std::memory_order_acquire) <= RingBytes;
        });
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
    });
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
    // `intended` before `sent_packets` (acquire both): an `intended` E stored after committing a
    // packet comes with that packet's count.
    const u64 intended = g.intended.load(std::memory_order_acquire);
    const u64 played = g.played.load(std::memory_order_acquire);
    const u64 sent_packets = g.sent_packets.load(std::memory_order_acquire);
    GpuSplit::Get().RaisePlayed((std::min)(intended, Floor(played, sent_packets)));
}

void LogMaybe() {
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
    }
    switch (header.kind) {
    case PacketKind::Lent: {
        const bool around = view_on && view.seeded && engine != nullptr;
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
            draws->PlayDraw(header);
        } else {
            Note(Check::RenderPacketMissing, "a draw packet with no player");
        }
        return Stats::RenderTime::DrawPackets;
    case PacketKind::Delta:
        break;
    }
    return Stats::RenderTime::OtherPackets;
}

void RenderMain(std::stop_token stop) {
    const Common::DkbCpuTrace::Registration cpu_trace{Common::DkbCpuTrace::Role::GpuMem};
    Common::SetCurrentThreadName("GPURender");
    t_render = true;
    GpuSplit::MarkRenderThread();
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();
    if (g.pin) {
        // FRITCHII_ONEF_PIN (default 1): the core ThreadPinning keeps for GPUMem (thread.cpp
        // reserves it when FRITCHII_ONEF=1).
        Common::PinCurrentThread(Common::PinnedRole::GpuMem);
    }
    g.system->RegisterHostThread();
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
            WakeEngine();
            continue;
        }
        const bool timing = Stats::Timing();
        const u64 begin = timing ? Stats::Now() : 0;
        const Stats::RenderTime where = Play(header, at + sizeof(PacketHeader));
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
    g.on = SwitchValue(g_onef, "FRITCHII_ONEF") == 1;
    g.pin = SwitchValue(g_onef_pin, "FRITCHII_ONEF_PIN", 1) != 0;
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

void MarkEngineThread() noexcept {
    if (g.on) {
        Detail::t_engine = true;
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
    g.thread = std::jthread([](std::stop_token stop) { RenderMain(stop); });
}

void Stop() {
    if (!g.thread.joinable()) {
        return;
    }
    g.thread.request_stop();
    g.r_wake.fetch_add(1, std::memory_order_seq_cst);
    g.r_wake.notify_one();
    g.thread.join();
}

void SetNextDraw(u32 kind, bool indexed, bool indirect, u32 instance_arg) noexcept {
    t_next_draw = DrawValues{};
    t_next_draw.kind = kind;
    t_next_draw.indexed = indexed ? 1 : 0;
    t_next_draw.indirect = indirect ? 1 : 0;
    t_next_draw.instance_arg = instance_arg;
    t_next_draw_set = true;
}

void Lend(LentOp op, LentReason reason, void (*function)(void*), void* argument) {
    if (!Detail::t_engine || !g.thread.joinable()) {
        if (op == LentOp::Draw) {
            t_next_draw_set = false;
        }
        function(argument); // R, another thread, or R not running: in place, as today
        return;
    }
    const StepLevel step = Step();
    const bool timing = Stats::Timing();
    const u64 begin = timing ? Stats::Now() : 0;
    const u64 behind = Behind();
    const u64 number = ++g.pushed;
    std::vector<u8>& scratch = t_scratch;
    scratch.resize(sizeof(PacketHeader));
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
            // The spans: everything E changed since the last packet. Room for every word.
            scratch.resize(sizeof(PacketHeader) + sizeof(RegPair) * Maxwell3D::Regs::NUM_REGS +
                           sizeof(FlagBits) + sizeof(DrawValues) + sizeof(Programs) +
                           sizeof(Pools) + 64);
            const std::size_t bytes = PackSpans(
                *engine, header, std::span<u8>{scratch}.subspan(sizeof(PacketHeader)));
            scratch.resize(sizeof(PacketHeader) + bytes);
            if (op == LentOp::Draw) {
                header.flags |= PacketVerify;
            }
        }
    }
    if (op == LentOp::Draw) {
        t_next_draw_set = false;
    }
    header.bytes = static_cast<u32>(scratch.size());
    std::memcpy(scratch.data(), &header, sizeof(header));
    Commit(scratch.data(), header.bytes, number, header.position, false);
    Stats::Pushed(header, kind, behind);
    // The park: R plays everything before, runs the call, releases E.
    EngineWait([&] { return g.played.load(std::memory_order_acquire) >= number; });
    if (timing) {
        Stats::Parked(op, Stats::Now() - begin);
    }
    if (g.lent_error) [[unlikely]] {
        std::rethrow_exception(std::exchange(g.lent_error, nullptr));
    }
}

void Push(PacketHeader& header, std::span<const u8> spans) {
    if (!Detail::t_engine || !g.thread.joinable()) {
        t_next_draw_set = false;
        return;
    }
    const u64 behind = Behind();
    header.number = ++g.pushed;
    header.position = GpuSplit::Get().BackEntryPosition();
    if (header.draw_seq == 0) {
        header.draw_seq = header.kind == PacketKind::Draw ? ++g.draw_seq : g.draw_seq;
    }
    header.bytes = static_cast<u32>(sizeof(PacketHeader) + spans.size());
    const u32 kind = header.kind == PacketKind::Draw ? TakeDrawKind(header) : 0;
    t_next_draw_set = false;
    std::vector<u8>& scratch = t_scratch;
    scratch.resize(header.bytes);
    std::memcpy(scratch.data(), &header, sizeof(header));
    if (!spans.empty()) {
        std::memcpy(scratch.data() + sizeof(header), spans.data(), spans.size());
    }
    Commit(scratch.data(), header.bytes, header.number, header.position, true);
    Stats::Pushed(header, kind, behind);
}

void Drain() {
    if (!Detail::t_engine) {
        return;
    }
    const u64 target = g.pushed;
    const u64 waited =
        EngineWait([&] { return g.played.load(std::memory_order_acquire) >= target; });
    if (waited != 0) {
        Stats::Drained(Stats::Timing() ? waited : 0);
    }
}

u64 Behind() noexcept {
    return g.pushed - g.played.load(std::memory_order_acquire);
}

void EngineConsumed(u64 position) noexcept {
    g.intended.store(position, std::memory_order_release);
    RaisePlayed();
}

void PushDraw(const Maxwell3D& engine, u64 draw_seq, u64 buffer_number, u8 flags) {
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
    std::vector<u8> spans(sizeof(RegPair) * Maxwell3D::Regs::NUM_REGS + sizeof(FlagBits) +
                          sizeof(DrawValues) + sizeof(Programs) + sizeof(Pools) + 64);
    const std::size_t bytes = PackSpans(engine, header, spans);
    Push(header, std::span<const u8>{spans.data(), bytes});
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
