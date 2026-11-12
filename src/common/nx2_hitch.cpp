// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "common/cpu_features.h"
#include "common/host_memory.h"
#include "common/logging.h"
#include "common/nx2_hitch.h"
#include "common/nx2_timeline.h"

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace Common::Nx2Hitch {

namespace {

// EDEN_NX2_HITCH_MS=<ms> (boot, default 50): the present gap that logs a Hitch line. Under 50
// (e.g. 25 for the single long frames of NPC dips) the cap rises from 60 to 600 lines a minute.
long long ThresholdUs() {
    static const long long threshold = [] {
        const char* const value = std::getenv("EDEN_NX2_HITCH_MS");
        const long long ms = value != nullptr ? std::strtoll(value, nullptr, 10) : 0;
        return ms > 0 ? ms * 1000 : 50'000LL;
    }();
    return threshold;
}
constexpr size_t Causes = static_cast<size_t>(Cause::Count);
constexpr size_t MaxThreads = 48;

struct ThreadEntry {
    char name[32]{};
#ifdef _WIN32
    HANDLE handle{};
#endif
};

std::array<ThreadEntry, MaxThreads> g_threads;
std::atomic<size_t> g_thread_count{};
std::mutex g_thread_mutex;

struct Snapshot {
    std::array<u64, Causes> ns{};
    std::array<u64, Causes> count{};
    std::array<u64, Causes> amount{};
    std::array<u64, Causes> longest{};
    u64 page_faults{};
    u64 zero_filled{};
    u64 user_100ns{};   ///< the process's CPU time in user mode, all threads
    u64 kernel_100ns{}; ///< and in kernel mode (page faults, VM calls)
    size_t threads{};
    std::array<u64, MaxThreads> thread_user{};
    std::array<u64, MaxThreads> thread_kernel{};
};

#ifdef _WIN32
u64 FileTimeValue(const FILETIME& time) {
    return (u64{time.dwHighDateTime} << 32) | time.dwLowDateTime;
}
#endif

Snapshot Take() {
    Snapshot snapshot;
    for (size_t i = 0; i < Causes; ++i) {
        snapshot.ns[i] = g_counters.ns[i].load(std::memory_order_relaxed);
        snapshot.count[i] = g_counters.count[i].load(std::memory_order_relaxed);
        snapshot.amount[i] = g_counters.amount[i].load(std::memory_order_relaxed);
        snapshot.longest[i] = g_counters.longest[i].exchange(0, std::memory_order_relaxed);
    }
    snapshot.zero_filled = HostMemory::ZeroFilledBytes();
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        snapshot.page_faults = counters.PageFaultCount;
    }
    FILETIME created{}, exited{}, kernel{}, user{};
    if (GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) {
        snapshot.user_100ns = FileTimeValue(user);
        snapshot.kernel_100ns = FileTimeValue(kernel);
    }
    snapshot.threads = g_thread_count.load(std::memory_order_acquire);
    for (size_t i = 0; i < snapshot.threads; ++i) {
        if (GetThreadTimes(g_threads[i].handle, &created, &exited, &kernel, &user)) {
            snapshot.thread_user[i] = FileTimeValue(user);
            snapshot.thread_kernel[i] = FileTimeValue(kernel);
        }
    }
#endif
    return snapshot;
}

bool Enabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_HITCH_LOG");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

// The busiest named threads in the gap, CPU time summed per name (worker pools share one).
std::string BusiestThreads(const Snapshot& last, const Snapshot& now) {
    struct Busy {
        std::string name;
        u64 user{};
        u64 kernel{};
    };
    std::vector<Busy> busy;
    for (size_t i = 0; i < now.threads; ++i) {
        const u64 user = now.thread_user[i] - (i < last.threads ? last.thread_user[i] : 0);
        const u64 kernel = now.thread_kernel[i] - (i < last.threads ? last.thread_kernel[i] : 0);
        if (user + kernel == 0) {
            continue;
        }
        const std::string name{g_threads[i].name};
        const auto it = std::ranges::find(busy, name, &Busy::name);
        if (it != busy.end()) {
            it->user += user;
            it->kernel += kernel;
        } else {
            busy.push_back({name, user, kernel});
        }
    }
    std::ranges::sort(busy, [](const Busy& a, const Busy& b) {
        return a.user + a.kernel > b.user + b.kernel;
    });
    std::string text;
    for (size_t i = 0; i < (std::min)(busy.size(), size_t{6}); ++i) {
        text += fmt::format("{}{} {:.0f}+{:.0f}", i == 0 ? "" : ", ", busy[i].name,
                            static_cast<double>(busy[i].user) / 1e4,
                            static_cast<double>(busy[i].kernel) / 1e4);
    }
    return text.empty() ? "none measured" : text;
}

} // Anonymous namespace

namespace {
std::mutex g_compile_mutex;
std::unordered_map<u64, u8> g_compiled_on; ///< key -> mask of the cores that compiled it
u8 g_cores_seen{};
std::array<std::vector<u64>, 8> g_warm_queues;
} // Anonymous namespace

bool JitWarmEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_JIT_WARM");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

Cause CompileCause(u64 key, u32 core) {
    const u8 bit = static_cast<u8>(1U << (core & 7));
    std::scoped_lock lock{g_compile_mutex};
    g_cores_seen |= bit;
    u8& mask = g_compiled_on[key];
    const bool repeat = (mask & ~bit) != 0;
    mask |= bit;
    if (t_warming) {
        return Cause::WarmCompile;
    }
    // Single-stepping blocks (bit 57 of an A64 LocationDescriptor) are never worth warming.
    if (JitWarmEnabled() && ((key >> 57) & 1) == 0) {
        for (u32 other = 0; other < 8; ++other) {
            const u8 other_bit = static_cast<u8>(1U << other);
            if ((g_cores_seen & other_bit) != 0 && (mask & other_bit) == 0 &&
                g_warm_queues[other].size() < 400'000) {
                g_warm_queues[other].push_back(key);
            }
        }
    }
    return repeat ? Cause::RecompileRepeat : Cause::Recompile;
}

size_t TakeWarmKeys(u32 core, std::vector<u64>& out, size_t max) {
    out.clear();
    std::scoped_lock lock{g_compile_mutex};
    auto& queue = g_warm_queues[core & 7];
    const u8 bit = static_cast<u8>(1U << (core & 7));
    // Oldest first; skip keys this core compiled on demand meanwhile.
    size_t taken = 0;
    while (taken < queue.size() && out.size() < max) {
        const u64 key = queue[taken++];
        if ((g_compiled_on[key] & bit) == 0) {
            out.push_back(key);
        }
    }
    queue.erase(queue.begin(), queue.begin() + static_cast<std::ptrdiff_t>(taken));
    return out.size();
}

namespace {

constexpr std::array<const char*, static_cast<size_t>(StallThread::Count)> StallThreadNames{
    "back", "front", "worker", "fencing", "present"};

std::mutex g_stall_reporter_mutex;
std::string (*g_stall_reporter)(const void*){};
const void* g_stall_reporter_owner{};

u64 StallNow() {
    return static_cast<u64>(Common::g_wall_clock.GetTimeNS().count());
}

// Only the slot's own thread writes it; a span within 0.5 ms of the recorded one does not replace
// it, so a wait keeps its inner name ("macro dma read") over the zone around it ("split drain").
void NoteLongest(StallSlot* slot, const char* zone, u64 ns) {
    if (ns > slot->longest_ns.load(std::memory_order_relaxed) + 500'000) {
        slot->longest_zone.store(zone, std::memory_order_relaxed);
        slot->longest_ns.store(ns, std::memory_order_release);
    }
}

// EDEN_NX2_STALL_SNAPSHOT=1, read at every thread name (not cached: threads are named before
// main() applies nx2_switches.txt).
bool StallSwitch() {
    const char* const value = std::getenv("EDEN_NX2_STALL_SNAPSHOT");
    return value != nullptr && value[0] == '1';
}

} // Anonymous namespace

StallMark StallEnterSlow(StallSlot* slot, const char* zone) {
    const u64 now = StallNow();
    StallMark mark{slot, slot->zone.load(std::memory_order_relaxed),
                   slot->since.load(std::memory_order_relaxed), now};
    if (mark.prev_zone == nullptr && mark.prev_since != 0) {
        NoteLongest(slot, "(no span)", now - mark.prev_since);
    }
    slot->zone.store(zone, std::memory_order_relaxed);
    slot->since.store(now, std::memory_order_relaxed);
    return mark;
}

void StallLeaveSlow(const StallMark& mark) {
    StallSlot* const slot = mark.slot;
    const u64 now = StallNow();
    NoteLongest(slot, slot->zone.load(std::memory_order_relaxed), now - mark.since);
    // Back in the span around this one, with its own start; outside every span from now.
    slot->zone.store(mark.prev_zone, std::memory_order_relaxed);
    slot->since.store(mark.prev_zone != nullptr ? mark.prev_since : now,
                      std::memory_order_relaxed);
}

void StallAcquired() {
    if (StallOn()) {
        g_stall_acquired_since.store(StallNow(), std::memory_order_relaxed);
    }
}

void StallPresented() {
    if (!StallOn()) {
        return;
    }
    const u64 since = g_stall_acquired_since.exchange(0, std::memory_order_relaxed);
    if (since == 0) {
        return;
    }
    const u64 ns = StallNow() - since;
    u64 longest = g_stall_acquire_longest.load(std::memory_order_relaxed);
    while (ns > longest && !g_stall_acquire_longest.compare_exchange_weak(
                               longest, ns, std::memory_order_relaxed)) {
    }
}

void SetStallTickReporter(std::string (*reporter)(const void*), const void* owner) {
    std::scoped_lock lock{g_stall_reporter_mutex};
    g_stall_reporter = reporter;
    g_stall_reporter_owner = owner;
}

void ClearStallTickReporter(const void* owner) {
    std::scoped_lock lock{g_stall_reporter_mutex};
    if (g_stall_reporter_owner == owner) {
        g_stall_reporter = nullptr;
        g_stall_reporter_owner = nullptr;
    }
}

namespace {

std::string StallSnapshotLine(long long gap_us) {
    const u64 now = StallNow();
    std::string current;
    std::string longest;
    for (size_t i = 0; i < g_stall_slots.size(); ++i) {
        StallSlot& slot = g_stall_slots[i];
        const char* const sep = i == 0 ? "" : ", ";
        const u64 since = slot.since.load(std::memory_order_relaxed);
        const char* const zone = slot.zone.load(std::memory_order_relaxed);
        if (since == 0) {
            current += fmt::format("{}{} not seen", sep, StallThreadNames[i]);
        } else {
            current += fmt::format("{}{} {} {:.1f} ms", sep, StallThreadNames[i],
                                   zone != nullptr ? zone : "(no span)",
                                   now > since ? static_cast<double>(now - since) / 1e6 : 0.0);
        }
        // Reset for the next gap (OnPresent resets every present too).
        const u64 longest_ns = slot.longest_ns.exchange(0, std::memory_order_acquire);
        const char* const longest_zone = slot.longest_zone.load(std::memory_order_relaxed);
        longest += fmt::format("{}{} {} {:.1f} ms", sep, StallThreadNames[i],
                               longest_ns != 0 && longest_zone != nullptr ? longest_zone : "-",
                               static_cast<double>(longest_ns) / 1e6);
    }
    const u64 acquired = g_stall_acquired_since.load(std::memory_order_relaxed);
    const u64 acquire_longest = g_stall_acquire_longest.exchange(0, std::memory_order_relaxed);
    const std::string acquire =
        acquired != 0 ? fmt::format("between acquire and present for {:.1f} ms",
                                    now > acquired ? static_cast<double>(now - acquired) / 1e6
                                                   : 0.0)
                      : std::string{"not between acquire and present"};
    std::string ticks;
    {
        std::scoped_lock lock{g_stall_reporter_mutex};
        ticks = g_stall_reporter != nullptr ? g_stall_reporter(g_stall_reporter_owner)
                                            : std::string{"no renderer"};
    }
    return fmt::format("Stall snapshot: {:.1f} ms between presents; now: {}; longest in the gap: "
                       "{}; present thread {} (longest acquire to present in the gap {:.1f} ms); "
                       "ticks: {}",
                       static_cast<double>(gap_us) / 1e3, current, longest, acquire,
                       static_cast<double>(acquire_longest) / 1e6, ticks);
}

void StallResetGap() {
    for (StallSlot& slot : g_stall_slots) {
        slot.longest_ns.store(0, std::memory_order_relaxed);
    }
    g_stall_acquire_longest.store(0, std::memory_order_relaxed);
}

} // Anonymous namespace

void RegisterThread(const char* name) {
    const std::string_view view{name};
    t_role = view.starts_with("CPUCore") ? Role::GuestCore
             : view == "GPUBack"         ? Role::GpuBack
             : view == "GPU"             ? Role::GpuFront
                                         : Role::Other;
    {
        const StallThread stall = view == "GPUBack"            ? StallThread::GpuBack
                                  : view == "GPU"              ? StallThread::GpuFront
                                  : view == "VulkanWorker"     ? StallThread::VulkanWorker
                                  : view == "GPUFencingThread" ? StallThread::Fencing
                                  : view == "VulkanPresent"    ? StallThread::Present
                                                               : StallThread::Count;
        if (stall != StallThread::Count && StallSwitch()) {
            StallSlot& slot = g_stall_slots[static_cast<size_t>(stall)];
            slot.zone.store(nullptr, std::memory_order_relaxed);
            slot.since.store(StallNow(), std::memory_order_relaxed);
            t_stall_slot = &slot;
            if (!g_stall_on.exchange(true, std::memory_order_relaxed)) {
                // The profiler's zones call out of line for the slots too (coarse zones only).
                Nx2Timeline::g_zone_gate.fetch_or(Nx2Timeline::ZoneGateStall,
                                                  std::memory_order_release);
                LOG_INFO(Render, "Stall snapshot on (EDEN_NX2_STALL_SNAPSHOT=1): a line after "
                                 "each Hitch line of 100 ms or more");
            }
        } else {
            t_stall_slot = nullptr;
        }
    }
#ifdef _WIN32
    std::scoped_lock lock{g_thread_mutex};
    const size_t index = g_thread_count.load(std::memory_order_relaxed);
    if (index >= MaxThreads) {
        return;
    }
    HANDLE handle{};
    if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
                         THREAD_QUERY_LIMITED_INFORMATION, FALSE, 0)) {
        return;
    }
    ThreadEntry& entry = g_threads[index];
    std::strncpy(entry.name, name, sizeof(entry.name) - 1);
    entry.handle = handle;
    g_thread_count.store(index + 1, std::memory_order_release);
#else
    static_cast<void>(name);
#endif
}

void OnPresent(long long gap_us) {
    static Snapshot last = Take();
    static u64 window_start_ns = NowNs();
    static int lines_in_window = 0;
    if (!Enabled()) {
        return;
    }
    const Snapshot now = Take();
    if (gap_us >= ThresholdUs()) {
        // At most 60 lines a minute (a loading screen can stall many presents in a row; 20 hid a
        // 740 ms first load behind the title screen's gaps, 2026-11-10).
        const u64 now_ns = NowNs();
        if (now_ns - window_start_ns >= 60'000'000'000ULL) {
            window_start_ns = now_ns;
            lines_in_window = 0;
        }
        if (lines_in_window++ < (ThresholdUs() < 50'000 ? 600 : 60)) {
            std::string causes;
            for (size_t i = 0; i < Causes; ++i) {
                const u64 ns = now.ns[i] - last.ns[i];
                const u64 count = now.count[i] - last.count[i];
                if (count == 0) {
                    continue;
                }
                const u64 amount = now.amount[i] - last.amount[i];
                causes += fmt::format("{}{} {:.1f} ms ({}{}, longest {:.1f})",
                                      causes.empty() ? "" : ", ", CauseNames[i],
                                      static_cast<double>(ns) / 1e6, count,
                                      amount != 0 ? fmt::format(", {} MiB", amount >> 20) : "",
                                      static_cast<double>(now.longest[i]) / 1e6);
            }
            const u64 trim_since = g_counters.zero_trim_since.load(std::memory_order_relaxed);
            if (trim_since != 0) {
                causes += fmt::format("{}a zero-page pass running for {:.0f} ms",
                                      causes.empty() ? "" : ", ",
                                      static_cast<double>(now_ns - trim_since) / 1e6);
            }
            LOG_INFO(Render, // with the frame lines it lines up against
                     "Hitch: {:.1f} ms between presents; in it: {}; page faults {}, kernel "
                     "zero-filled {} MiB, CPU {:.0f} ms user + {:.0f} ms kernel; busiest threads "
                     "(user+kernel ms): {}",
                     static_cast<double>(gap_us) / 1e3, causes.empty() ? "nothing counted" : causes,
                     now.page_faults - last.page_faults, (now.zero_filled - last.zero_filled) >> 20,
                     static_cast<double>(now.user_100ns - last.user_100ns) / 1e4,
                     static_cast<double>(now.kernel_100ns - last.kernel_100ns) / 1e4,
                     BusiestThreads(last, now));
            if (gap_us >= 100'000 && StallOn()) {
                LOG_INFO(Render, "{}", StallSnapshotLine(gap_us));
            }
        }
    }
    if (StallOn()) {
        StallResetGap();
    }
    last = now;
}

} // namespace Common::Nx2Hitch
