// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "common/cpu_features.h"
#include "common/host_memory.h"
#include "common/logging.h"
#include "common/nx2_hitch.h"
#include "common/nx2_timeline.h"
#include "common/thread.h"

#ifdef _WIN32
#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>
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
    u64 system_busy_100ns{}; ///< every logical CPU's busy time, all processes (GetSystemTimes)
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
    FILETIME idle{};
    if (GetSystemTimes(&idle, &kernel, &user)) {
        // The kernel time includes the idle time.
        snapshot.system_busy_100ns =
            FileTimeValue(kernel) - FileTimeValue(idle) + FileTimeValue(user);
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
    "back", "front", "worker", "fencing", "present", "memory"};

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

// EDEN_NX2_STALL_SNAPSHOT, read at every thread name (not cached: threads are named before
// main() applies nx2_switches.txt). On by default since 2026-11-14 (owner): a 750 ms freeze in an
// owner playtest could not be explained without it. EDEN_NX2_STALL_SNAPSHOT=0 turns it off.
bool StallSwitch() {
    const char* const value = std::getenv("EDEN_NX2_STALL_SNAPSHOT");
    return value == nullptr || value[0] != '0';
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

// Machine pulse (fable52_answer_dips section 4c): tells "the machine got slower" (another program,
// a power or heat clock drop, a contended cache or memory bus) apart from the emulator's own work.
// With the hitch log on, every Hitch line says how much CPU the rest of the machine used in the gap
// and over the last second, and a normal-priority "FritchiiPulse" thread samples once a second: the
// clock against nominal (PDH % Processor Performance), a fixed integer loop over 64 KiB timed (the
// speed probe: how fast a core is for us right now, whatever the cause), free memory, and the
// thermal zones when the board exposes them. It logs one "Pulse:" line a second; the Hitch line
// shows its last clock and probe. FRITCHII_PULSE=0 (boot) turns the thread and its line off.
// Cost: one GetSystemTimes a present; one PDH read and about 2 ms of one core a second.
namespace {

std::atomic<u64> g_presents{};
std::atomic<int> g_pulse_clock{-1}; ///< % Processor Performance over the last second; -1 unknown
std::atomic<u64> g_pulse_probe_ns{}; ///< the last speed probe; 0 none yet
std::atomic<u32> g_probe_sink{};     ///< keeps the probe's result alive

#ifdef _WIN32

bool PulseEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("FRITCHII_PULSE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// 9400 -> "9,400"
std::string Grouped(u64 value) {
    std::string digits = std::to_string(value);
    for (size_t i = digits.size(); i > 3; i -= 3) {
        digits.insert(i - 3, 1, ',');
    }
    return digits;
}

/// ", clock 112%, probe 1.96 ms" from the pulse thread's last sample, empty before it has one.
std::string PulseLastText() {
    std::string text;
    const int clock = g_pulse_clock.load(std::memory_order_relaxed);
    if (clock >= 0) {
        text += fmt::format(", clock {}%", clock);
    }
    const u64 probe_ns = g_pulse_probe_ns.load(std::memory_order_relaxed);
    if (probe_ns != 0) {
        text += fmt::format(", probe {:.2f} ms", static_cast<double>(probe_ns) / 1e6);
    }
    return text;
}

#endif // _WIN32

/// The rest of the machine's CPU time (all processes but this one) over about the last second,
/// from the presents' samples. Only the game's present thread touches it.
class RestWindow {
public:
    void Push(u64 ns, s64 rest_100ns) {
        samples[next] = {ns, rest_100ns};
        next = (next + 1) % samples.size();
        count = (std::min)(count + 1, samples.size());
    }

    /// Logical CPUs the rest of the machine kept busy from the newest sample a second or more
    /// old (or the oldest kept) until `ns`; negative without a sample.
    [[nodiscard]] double Cpus(u64 ns, s64 rest_100ns) const {
        const Sample* base = nullptr;
        for (size_t back = 1; back <= count; ++back) {
            base = &samples[(next + samples.size() - back) % samples.size()];
            if (ns - base->ns >= 1'000'000'000ULL) {
                break;
            }
        }
        if (base == nullptr || ns <= base->ns) {
            return -1.0;
        }
        const double span_100ns = static_cast<double>(ns - base->ns) / 100.0;
        return static_cast<double>((std::max)(rest_100ns - base->rest_100ns, s64{0})) /
               span_100ns;
    }

private:
    struct Sample {
        u64 ns{};
        s64 rest_100ns{};
    };
    std::array<Sample, 512> samples{};
    size_t next{};
    size_t count{};
};

#ifdef _WIN32

constexpr size_t ProbeWords = 64 * 1024 / sizeof(u32);
// About 2 ms on a 5 GHz desktop core: each step's load index hangs on the step before, so the
// loop runs at the core's clock and its L1/L2 latency, never vectorized.
constexpr u32 ProbeSteps = 800'000;

/// The same work every call: the array is reset first, so every run walks the same indices.
u32 ProbeLoop(u32* words) {
    for (size_t i = 0; i < ProbeWords; ++i) {
        words[i] = static_cast<u32>(i) * 2654435761U;
    }
    u32 x = 0x9E3779B9U;
    u32 index = 0;
    for (u32 step = 0; step < ProbeSteps; ++step) {
        index = (index + x) & static_cast<u32>(ProbeWords - 1);
        x = x * 0x01000193U + words[index];
        words[index] = x ^ (x >> 13);
    }
    return x;
}

struct MachineTimes {
    u64 ns{};
    u64 system_busy_100ns{};
    u64 process_busy_100ns{};
};

MachineTimes ReadMachineTimes() {
    MachineTimes times{.ns = NowNs()};
    FILETIME idle{}, kernel{}, user{}, created{}, exited{};
    if (GetSystemTimes(&idle, &kernel, &user)) {
        times.system_busy_100ns =
            FileTimeValue(kernel) - FileTimeValue(idle) + FileTimeValue(user);
    }
    if (GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) {
        times.process_busy_100ns = FileTimeValue(kernel) + FileTimeValue(user);
    }
    return times;
}

/// The PDH counters; one that fails to open is left out.
class PdhCounters {
public:
    struct Values {
        double clock = -1.0;   ///< % of nominal, above 100 with boost
        double kelvin = -1.0;  ///< the hottest thermal zone
        double passive = -1.0; ///< the lowest % Passive Limit of the zones (100 = no limit)
    };

    PdhCounters() {
        if (PdhOpenQueryW(nullptr, 0, &query) != ERROR_SUCCESS) {
            query = nullptr;
            return;
        }
        clock = Add(L"\\Processor Information(_Total)\\% Processor Performance");
        temperature = Add(L"\\Thermal Zone Information(*)\\Temperature");
        passive = Add(L"\\Thermal Zone Information(*)\\% Passive Limit");
        // The first sample; the clock is a rate, so the first value comes a second later.
        PdhCollectQueryData(query);
    }

    ~PdhCounters() {
        if (query != nullptr) {
            PdhCloseQuery(query);
        }
    }

    PdhCounters(const PdhCounters&) = delete;
    PdhCounters& operator=(const PdhCounters&) = delete;

    [[nodiscard]] std::string Found() const {
        if (query == nullptr) {
            return "no PDH query";
        }
        return fmt::format("clock {}, thermal zones {}", clock != nullptr ? "yes" : "no",
                           temperature != nullptr || passive != nullptr ? "yes" : "no");
    }

    [[nodiscard]] Values Read() {
        Values values;
        if (query == nullptr || PdhCollectQueryData(query) != ERROR_SUCCESS) {
            return values;
        }
        if (clock != nullptr) {
            PDH_FMT_COUNTERVALUE value{};
            if (PdhGetFormattedCounterValue(clock, Format, nullptr, &value) == ERROR_SUCCESS &&
                Valid(value.CStatus)) {
                values.clock = value.doubleValue;
            }
        }
        values.kelvin = Extreme(temperature, true);
        values.passive = Extreme(passive, false);
        return values;
    }

private:
    static constexpr DWORD Format = PDH_FMT_DOUBLE | PDH_FMT_NOCAP100;

    static bool Valid(DWORD status) {
        return status == PDH_CSTATUS_VALID_DATA || status == PDH_CSTATUS_NEW_DATA;
    }

    PDH_HCOUNTER Add(const wchar_t* path) {
        PDH_HCOUNTER counter{};
        return PdhAddEnglishCounterW(query, path, 0, &counter) == ERROR_SUCCESS ? counter : nullptr;
    }

    /// The highest (or lowest) value over a wildcard counter's instances; negative without one.
    double Extreme(PDH_HCOUNTER counter, bool highest) {
        if (counter == nullptr) {
            return -1.0;
        }
        DWORD bytes = 0;
        DWORD items = 0;
        if (static_cast<DWORD>(PdhGetFormattedCounterArrayW(counter, Format, &bytes, &items,
                                                            nullptr)) != PDH_MORE_DATA ||
            bytes == 0) {
            return -1.0;
        }
        buffer.resize((bytes + sizeof(u64) - 1) / sizeof(u64));
        auto* const entries = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buffer.data());
        if (PdhGetFormattedCounterArrayW(counter, Format, &bytes, &items, entries) !=
            ERROR_SUCCESS) {
            return -1.0;
        }
        double result = -1.0;
        for (DWORD i = 0; i < items; ++i) {
            if (!Valid(entries[i].FmtValue.CStatus)) {
                continue;
            }
            const double value = entries[i].FmtValue.doubleValue;
            if (result < 0.0 || (highest ? value > result : value < result)) {
                result = value;
            }
        }
        return result;
    }

    PDH_HQUERY query{};
    PDH_HCOUNTER clock{};
    PDH_HCOUNTER temperature{};
    PDH_HCOUNTER passive{};
    std::vector<u64> buffer;
};

/// The once-a-second sampler; stops and joins when the process's statics are destroyed.
class PulseThread {
public:
    PulseThread() : thread{[this] { Run(); }} {}

    ~PulseThread() {
        {
            std::scoped_lock lock{mutex};
            stop = true;
        }
        cv.notify_all();
        if (thread.joinable()) {
            thread.join();
        }
    }

    PulseThread(const PulseThread&) = delete;
    PulseThread& operator=(const PulseThread&) = delete;

private:
    void Run() {
        Common::SetCurrentThreadName("FritchiiPulse");
        PdhCounters pdh;
        std::vector<u32> words(ProbeWords);
        LOG_INFO(Render,
                 "Machine pulse on: a Pulse line a second, clock and probe in the Hitch lines "
                 "(FRITCHII_PULSE=0 turns it off); counters: {}",
                 pdh.Found());
        MachineTimes last = ReadMachineTimes();
        u64 last_presents = g_presents.load(std::memory_order_relaxed);
        std::unique_lock lock{mutex};
        while (!cv.wait_for(lock, std::chrono::seconds{1}, [this] { return stop; })) {
            lock.unlock();
            Sample(pdh, words.data(), last, last_presents);
            lock.lock();
        }
    }

    static void Sample(PdhCounters& pdh, u32* words, MachineTimes& last, u64& last_presents) {
        const u64 probe_begin = NowNs();
        const u32 probe_result = ProbeLoop(words);
        const u64 probe_ns = NowNs() - probe_begin;
        g_probe_sink.store(probe_result, std::memory_order_relaxed);
        g_pulse_probe_ns.store((std::max)(probe_ns, u64{1}), std::memory_order_relaxed);

        const PdhCounters::Values values = pdh.Read();
        if (values.clock >= 0.0) {
            g_pulse_clock.store(static_cast<int>(values.clock + 0.5), std::memory_order_relaxed);
        }
        MEMORYSTATUSEX memory{};
        memory.dwLength = sizeof(memory);
        const bool memory_ok = GlobalMemoryStatusEx(&memory) != FALSE;

        const MachineTimes now = ReadMachineTimes();
        const u64 presents = g_presents.load(std::memory_order_relaxed);
        if (now.ns > last.ns) {
            const double span_100ns = static_cast<double>(now.ns - last.ns) / 100.0;
            const s64 process = static_cast<s64>(now.process_busy_100ns - last.process_busy_100ns);
            const s64 system = static_cast<s64>(now.system_busy_100ns - last.system_busy_100ns);
            std::string extra;
            if (values.clock >= 0.0) {
                extra += fmt::format(", clock {:.0f}%", values.clock);
            }
            extra += fmt::format(", probe {:.2f} ms", static_cast<double>(probe_ns) / 1e6);
            if (memory_ok) {
                extra += fmt::format(", {} MiB free", Grouped(memory.ullAvailPhys >> 20));
            }
            if (values.kelvin > 0.0) {
                extra += fmt::format(", heat {:.0f} C", values.kelvin - 273.15);
            }
            if (values.passive >= 0.0) {
                extra += fmt::format(", passive limit {:.0f}%", values.passive);
            }
            LOG_INFO(Render, "Pulse: {} frames, process {:.1f} CPUs, rest of the machine {:.1f} CPUs{}",
                     presents - last_presents, static_cast<double>(process) / span_100ns,
                     static_cast<double>((std::max)(system - process, s64{0})) / span_100ns,
                     extra);
        }
        last = now;
        last_presents = presents;
    }

    std::mutex mutex;
    std::condition_variable cv;
    bool stop{};
    std::thread thread; ///< last: the members above exist before it runs
};

#endif // _WIN32

} // Anonymous namespace

void RegisterThread(const char* name) {
    const std::string_view view{name};
    t_role = view.starts_with("CPUCore") ? Role::GuestCore
             : view == "GPUBack"         ? Role::GpuBack
             : view == "GPU"             ? Role::GpuFront
             : view == "GPUMem"          ? Role::GpuMem
                                         : Role::Other;
    {
        const StallThread stall = view == "GPUBack"            ? StallThread::GpuBack
                                  : view == "GPU"              ? StallThread::GpuFront
                                  : view == "VulkanWorker"     ? StallThread::VulkanWorker
                                  : view == "GPUFencingThread" ? StallThread::Fencing
                                  : view == "VulkanPresent"    ? StallThread::Present
                                  : view == "GPUMem"           ? StallThread::GpuMem
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
                LOG_INFO(Render, "Stall snapshot on (the default; EDEN_NX2_STALL_SNAPSHOT=0 turns "
                                 "it off): a line after each Hitch line of 100 ms or more");
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
    static RestWindow rest_window;
    if (!Enabled()) {
        return;
    }
    g_presents.fetch_add(1, std::memory_order_relaxed);
#ifdef _WIN32
    if (PulseEnabled()) {
        static PulseThread pulse; // started at the first present
    }
#endif
    const Snapshot now = Take();
    const u64 present_ns = NowNs();
    const s64 rest_now = static_cast<s64>(now.system_busy_100ns) -
                         static_cast<s64>(now.user_100ns + now.kernel_100ns);
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
            std::string machine;
#ifdef _WIN32
            if (now.system_busy_100ns != 0 && last.system_busy_100ns != 0) {
                const s64 rest_gap = rest_now - (static_cast<s64>(last.system_busy_100ns) -
                                                 static_cast<s64>(last.user_100ns +
                                                                  last.kernel_100ns));
                const double cpus = rest_window.Cpus(present_ns, rest_now);
                machine = fmt::format(
                    "; rest of the machine {:.0f} ms{}{}",
                    static_cast<double>((std::max)(rest_gap, s64{0})) / 1e4,
                    cpus >= 0.0 ? fmt::format(" ({:.1f} CPUs over the last second)", cpus) : "",
                    PulseLastText());
            }
#endif
            LOG_INFO(Render, // with the frame lines it lines up against
                     "Hitch: {:.1f} ms between presents; in it: {}; page faults {}, kernel "
                     "zero-filled {} MiB, CPU {:.0f} ms user + {:.0f} ms kernel; busiest threads "
                     "(user+kernel ms): {}{}",
                     static_cast<double>(gap_us) / 1e3, causes.empty() ? "nothing counted" : causes,
                     now.page_faults - last.page_faults, (now.zero_filled - last.zero_filled) >> 20,
                     static_cast<double>(now.user_100ns - last.user_100ns) / 1e4,
                     static_cast<double>(now.kernel_100ns - last.kernel_100ns) / 1e4,
                     BusiestThreads(last, now), machine);
            if (gap_us >= 100'000 && StallOn()) {
                LOG_INFO(Render, "{}", StallSnapshotLine(gap_us));
            }
        }
    }
    if (StallOn()) {
        StallResetGap();
    }
    rest_window.Push(present_ns, rest_now);
    last = now;
}

} // namespace Common::Nx2Hitch
