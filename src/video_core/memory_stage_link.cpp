// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// The memory stage's thread, GPUMem (fable_memstage_step6_spec.md, piece 6a: "the thread, by
// hand-off").
//
// The thread exists while the GPU thread split runs (started and stopped with the back,
// GpuSplit::MaybeStart and Stop) and sleeps until the back hands it work. FRITCHII_MEMSTAGE_THREAD
// (live, read at a frame end by MemStage::TickFrame) picks the mode:
//   0: nothing is handed off; the fills run on the back (2b).
//   1: hand-off. The back's two big fills, the draw's (MemStage::FillDraw) and the dispatch's
//      (FillDispatch), and the buffer recycler's idle destroys, are run here: the back posts the
//      function (Call), GPUMem runs it, the back waits. Same order, same timing, another thread.
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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>
#include <utility>

#include <fmt/format.h>

#include "common/dkb_cpu_trace.h"
#include "common/logging.h"
#include "common/thread.h"
#include "core/core.h"
#include "video_core/gpu_split.h"
#include "video_core/gpu_wait_site.h"
#include "video_core/memory_stage_link.h"
#include "video_core/nx2_profile.h"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

namespace VideoCommon {
namespace {

/// About 2,000 pauses (spec 6.1) before a wait sleeps.
constexpr u32 LinkSpinPauses = 2048;

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
};
constexpr std::array<const char*, static_cast<size_t>(MemoryStageLink::HandOff::Count)>
    LINK_HANDOFF_NAMES{"draw fills", "dispatch fills", "idle destroys"};

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
    request_sent = 0;
    answered_local = 0;
    thread = std::jthread([this](std::stop_token stop) { StageMain(stop); });
    LOG_INFO(HW_GPU, "Memory stage thread (GPUMem) started, sleeping until "
                     "FRITCHII_MEMSTAGE_THREAD=1 hands it the fills");
}

void MemoryStageLink::Stop() {
    if (!thread.joinable()) {
        return;
    }
    // The back has stopped: no Call is in flight.
    thread.request_stop();
    WakeStage();
    thread.join();
    thread = {};
    mode = 0;
    g_checking.store(false, std::memory_order_relaxed);
}

u32 MemoryStageLink::SetMode(u32 wanted) {
    u32 next = (std::min)(wanted, 1U);
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
        LOG_INFO(HW_GPU, "Memory stage thread (FRITCHII_MEMSTAGE_THREAD={}): {}", next,
                 next != 0 ? "the draw and dispatch fills run on GPUMem by hand-off"
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
}

void MemoryStageLink::Call(HandOff kind, Function function, void* argument) {
    // Inline: hand-off off; on GPUMem itself; while the back serves one of GPUMem's requests
    // (GPUMem is waiting then, so this is still one thread at a time).
    if (mode == 0 || t_on_stage || t_back_serving || !GpuSplit::OnBackThread()) {
        function(argument);
        return;
    }
    const u64 start = LinkNowNs();
    call_function = function;
    call_argument = argument;
    call_recording = GpuSplit::RecordingDispatch();
    call_error = nullptr;
    const u32 seq = ++call_sent;
    t_back_lent = true;
    calls.store(seq, std::memory_order_seq_cst);
    WakeStage();
    {
        const Nx2Profile::Scope zone{Nx2Profile::Zone::MemoryHandOff};
        const BackWaitScope site{"memory stage hand-off (a fill on GPUMem)"};
        WaitForStage(seq);
    }
    t_back_lent = false;
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
    t_back_lent = false;
    t_back_serving = true;
    try {
        request_function(request_argument);
    } catch (...) {
        request_error = std::current_exception();
    }
    t_back_serving = false;
    t_back_lent = true;
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
    request_function = function;
    request_argument = argument;
    request_error = nullptr;
    const u32 seq = ++request_sent;
    // Publish before waiting (spec 4.3): the back is in the Call's wait and serves it there.
    t_stage_user = false;
    requests.store(seq, std::memory_order_seq_cst);
    WakeBack();
    {
        const Nx2Profile::Scope zone{Nx2Profile::Zone::MemorySyncWait};
        const StageWaitScope site{"a sync the back serves"};
        for (u32 spin = 0; answered.load(std::memory_order_seq_cst) != seq;) {
            if (spin < LinkSpinPauses) {
                ++spin;
                LinkPause();
                continue;
            }
            stage_sleeping.store(true, std::memory_order_seq_cst);
            const u32 word = stage_word.load(std::memory_order_seq_cst);
            if (answered.load(std::memory_order_seq_cst) != seq) {
                stage_word.wait(word, std::memory_order_seq_cst);
                LinkBump(counters.stage_sleeps);
            }
            stage_sleeping.store(false, std::memory_order_relaxed);
        }
    }
    t_stage_user = true;
    if (request_error) {
        std::rethrow_exception(std::exchange(request_error, nullptr));
    }
}

void MemoryStageLink::StageMain(std::stop_token stop) {
    const Common::DkbCpuTrace::Registration cpu_trace{Common::DkbCpuTrace::Role::GpuMem};
    Common::SetCurrentThreadName("GPUMem");
    t_on_stage = true;
    // As the back (spec 5.8); not pinned in 6a and 6b.
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();
    system->RegisterHostThread();
    // Start reset `calls` to 0 before this thread ran: a Call posted before here is seen.
    u32 seen = 0;
    for (;;) {
        u32 now = seen;
        for (u32 spin = 0;;) {
            now = calls.load(std::memory_order_seq_cst);
            if (now != seen || stop.stop_requested()) {
                break;
            }
            if (spin < LinkSpinPauses) {
                ++spin;
                LinkPause();
                continue;
            }
            stage_sleeping.store(true, std::memory_order_seq_cst);
            const u32 word = stage_word.load(std::memory_order_seq_cst);
            if (calls.load(std::memory_order_seq_cst) == seen && !stop.stop_requested()) {
                stage_word.wait(word, std::memory_order_seq_cst);
                LinkBump(counters.stage_sleeps);
            }
            stage_sleeping.store(false, std::memory_order_relaxed);
        }
        if (now == seen) {
            break; // Stopped.
        }
        seen = now;
        RunCall(now);
        Nx2Profile::Report("GPU memory thread");
    }
}

void MemoryStageLink::RunCall(u32 seq) {
    const u64 start = LinkNowNs();
    t_in_call = true;
    t_stage_user = true;
    // EXP10: the back records the writes of the dispatches it runs; this fill is one of them.
    GpuSplit::AdoptRecordingDispatch(call_recording);
    try {
        call_function(call_argument);
    } catch (...) {
        call_error = std::current_exception();
    }
    GpuSplit::AdoptRecordingDispatch(false);
    t_stage_user = false;
    t_in_call = false;
    LinkBump(counters.stage_ns, LinkNowNs() - start);
    done.store(seq, std::memory_order_seq_cst);
    WakeBack();
}

bool MemoryStageLink::IsCacheUserSlow() noexcept {
    if (t_on_stage) {
        return t_stage_user;
    }
    return GpuSplit::OnBackThread() && !t_back_lent;
}

void MemoryStageLink::CheckCacheUserSlow(Check check) noexcept {
    if (!IsCacheUserSlow()) {
        Note(check, nullptr);
    }
}

void MemoryStageLink::OnBufferAllocator() noexcept {
    g_buffer_allocator_ops.fetch_add(1, std::memory_order_relaxed);
    CheckCacheUser(Check::AllocatorUser);
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
    for (size_t i = 0; i < out.checks.size(); ++i) {
        out.checks[i] = load(counters.checks[i]);
    }
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
             "{:.1f} ms/s and slept {:.0f}/s; served syncs {:.0f}/s ({:.2f} ms/s on the back); "
             "must be 0: {}{}",
             mode, seconds, rate(total_calls), calls_text, ms(call_ns),
             total_calls != 0 ? static_cast<double>(call_ns) / 1e3 / static_cast<double>(total_calls)
                              : 0.0,
             rate(current.back_sleeps - reported.back_sleeps),
             ms(current.stage_ns - reported.stage_ns),
             rate(current.stage_sleeps - reported.stage_sleeps),
             rate(current.served - reported.served), ms(current.served_ns - reported.served_ns),
             total_checks, checks_text);
    reported = current;
}

} // namespace VideoCommon
