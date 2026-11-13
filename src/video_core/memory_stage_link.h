// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <exception>
#include <thread>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace VideoCommon {

/// The memory stage's thread, GPUMem, and the back's link to it (memory_stage_link.cpp).
class MemoryStageLink {
public:
    using Function = void (*)(void* argument);

    enum class HandOff : u8 { Draw, Dispatch, Idle, Count };

    /// Counters that must read 0 (memory_stage_link.cpp).
    enum class Check : u8 {
        AllocatorUser,
        StagingUser,
        PushUser,
        BudgetUser,
        StageScheduler,
        PushNowhere,
        PushWhilePending,
        PlayerOnStage,
        FrontWaitOnStage,
        Count,
    };

    [[nodiscard]] static MemoryStageLink& Get() noexcept {
        return instance;
    }

    void Start(Core::System& system);
    void Stop();

    /// Back, at a frame end: 0 fills on the back, 1 hand-off. Returns the mode it runs.
    u32 SetMode(u32 wanted);
    [[nodiscard]] u32 Mode() const noexcept {
        return mode;
    }

    void Call(HandOff kind, Function function, void* argument);
    template <typename F>
    void Call(HandOff kind, F& function) {
        Call(kind, [](void* argument) { (*static_cast<F*>(argument))(); }, &function);
    }

    void Serve(Function function, void* argument);
    template <typename F>
    void Serve(F& function) {
        Serve([](void* argument) { (*static_cast<F*>(argument))(); }, &function);
    }

    [[nodiscard]] static bool OnStageThread() noexcept {
        return t_on_stage;
    }
    [[nodiscard]] static bool InCall() noexcept {
        return t_in_call;
    }
    [[nodiscard]] static bool IsCacheUser() noexcept {
        return !g_checking.load(std::memory_order_relaxed) || IsCacheUserSlow();
    }
    static void CheckCacheUser(Check check) noexcept {
        if (g_checking.load(std::memory_order_relaxed)) [[unlikely]] {
            CheckCacheUserSlow(check);
        }
    }
    static void Note(Check check, const char* where) noexcept;
    /// FRITCHII_MEMSTAGE_THREAD >= 1: the thread rules are checked (and the buffer allocator's
    /// heap budgets are published by its user instead of read from VMA by anyone).
    [[nodiscard]] static bool Checking() noexcept {
        return g_checking.load(std::memory_order_relaxed);
    }

    /// Around a call on a VMA allocator (vulkan_memory_allocator.cpp, vulkan_wrapper.cpp). On the
    /// buffer allocator: checks the caller is the cache's user, and counts the operation once it
    /// is done, so a heap budget read during it keeps the old count and is taken again
    /// (Device::PublishBufferHeapBudgets).
    class AllocatorOp {
    public:
        explicit AllocatorOp(const void* allocator) noexcept
            : watched{allocator != nullptr &&
                      allocator == g_buffer_allocator.load(std::memory_order_relaxed)} {
            if (watched) [[unlikely]] {
                CheckCacheUser(Check::AllocatorUser);
            }
        }
        ~AllocatorOp() {
            if (watched) [[unlikely]] {
                g_buffer_allocator_ops.fetch_add(1, std::memory_order_release);
            }
        }
        AllocatorOp(const AllocatorOp&) = delete;
        AllocatorOp& operator=(const AllocatorOp&) = delete;

    private:
        bool watched;
    };
    static void WatchBufferAllocator(const void* allocator) noexcept {
        g_buffer_allocator.store(allocator, std::memory_order_relaxed);
    }
    /// Operations finished on the buffer allocator (AllocatorOp).
    [[nodiscard]] static u64 BufferAllocatorOps() noexcept {
        return g_buffer_allocator_ops.load(std::memory_order_acquire);
    }

    [[nodiscard]] static bool FullVertexBinds() noexcept {
        return g_full_vertex_binds.load(std::memory_order_relaxed);
    }
    static void SetFullVertexBinds(bool on) noexcept {
        g_full_vertex_binds.store(on, std::memory_order_relaxed);
    }

    void LogStats();

private:
    static bool IsCacheUserSlow() noexcept;
    static void CheckCacheUserSlow(Check check) noexcept;

    void StageMain(std::stop_token stop);
    void RunCall(u32 seq);
    void WaitForStage(u32 seq);
    void ServeOne();
    void WakeStage();
    void WakeBack();

    static MemoryStageLink instance;

    static inline std::atomic<bool> g_checking{false};
    static inline std::atomic<const void*> g_buffer_allocator{nullptr};
    static inline std::atomic<u64> g_buffer_allocator_ops{0};
    static inline std::atomic<bool> g_full_vertex_binds{false};

    static inline thread_local bool t_on_stage{};
    static inline thread_local bool t_in_call{};
    static inline thread_local bool t_stage_user{};
    static inline thread_local bool t_back_lent{};
    static inline thread_local bool t_back_serving{};

    Core::System* system{};
    u32 mode{};
    bool refused_logged{};

    // Written by the back before `calls`, read by the stage after it.
    Function call_function{};
    void* call_argument{};
    bool call_recording{};
    std::exception_ptr call_error;
    u32 call_sent{};
    // Written by the stage before `requests`, read by the back after it.
    Function request_function{};
    void* request_argument{};
    std::exception_ptr request_error;
    u32 request_sent{};
    u32 answered_local{};

    alignas(64) std::atomic<u32> calls{0};
    alignas(64) std::atomic<u32> done{0};
    alignas(64) std::atomic<u32> requests{0};
    alignas(64) std::atomic<u32> answered{0};
    alignas(64) std::atomic<u32> stage_word{0};
    std::atomic<bool> stage_sleeping{false};
    alignas(64) std::atomic<u32> back_word{0};
    std::atomic<bool> back_sleeping{false};

    struct Counters {
        std::array<std::atomic<u64>, static_cast<size_t>(HandOff::Count)> calls{};
        std::atomic<u64> call_ns{};
        std::atomic<u64> stage_ns{};
        std::atomic<u64> served{};
        std::atomic<u64> served_ns{};
        std::atomic<u64> back_sleeps{};
        std::atomic<u64> stage_sleeps{};
        std::array<std::atomic<u64>, static_cast<size_t>(Check::Count)> checks{};
    };
    Counters counters;
    struct Snapshot {
        std::array<u64, static_cast<size_t>(HandOff::Count)> calls{};
        u64 call_ns{};
        u64 stage_ns{};
        u64 served{};
        u64 served_ns{};
        u64 back_sleeps{};
        u64 stage_sleeps{};
        std::array<u64, static_cast<size_t>(Check::Count)> checks{};
    };
    void Take(Snapshot& out) const noexcept;
    Snapshot reported{};
    u64 last_log_ns{};

    std::jthread thread;
};

} // namespace VideoCommon
