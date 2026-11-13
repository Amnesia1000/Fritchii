// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdint>
#include <thread>

#include "common/nx2_hitch.h"

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace Common {

/// A recursive mutex that spins briefly before sleeping. The GPU caches' locks are taken tens of
/// thousands of times a second by the GPU thread, the fencing thread and CPU write traps, and
/// held for microseconds: std::recursive_mutex put a waiting thread to sleep at once, and its
/// wake-up took longer than the critical section (title 1's GPU thread spent ~8% asleep there).
class SpinRecursiveMutex {
public:
    void lock() {
        const std::thread::id self = std::this_thread::get_id();
        if (owner.load(std::memory_order_relaxed) == self) {
            ++depth;
            return;
        }
        std::uint32_t expected = 0;
        if (!state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                           std::memory_order_relaxed)) {
            LockContended();
        }
        owner.store(self, std::memory_order_relaxed);
        depth = 1;
    }

    bool try_lock() {
        const std::thread::id self = std::this_thread::get_id();
        if (owner.load(std::memory_order_relaxed) == self) {
            ++depth;
            return true;
        }
        std::uint32_t expected = 0;
        if (!state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                           std::memory_order_relaxed)) {
            return false;
        }
        owner.store(self, std::memory_order_relaxed);
        depth = 1;
        return true;
    }

    void unlock() {
        if (--depth != 0) {
            return;
        }
        // Still held here: the hook runs as the holder's last act (the buffer cache's page
        // mirrors close their sequence counter, EXP86). Null on every other mutex.
        if (on_release != nullptr) [[unlikely]] {
            on_release(on_release_context);
        }
        owner.store(std::thread::id{}, std::memory_order_relaxed);
        if (state.exchange(0, std::memory_order_release) == 2) {
            state.notify_one();
        }
    }

    /// Called by the holder at each final unlock (depth back to 0), before the mutex is given up.
    /// Set once, before the mutex is shared between threads; the hook must not lock this mutex.
    void SetReleaseHook(void (*hook)(void*), void* context) {
        on_release = hook;
        on_release_context = context;
    }

    /// Whether the calling thread holds the mutex.
    [[nodiscard]] bool IsHeldByCurrentThread() const {
        return owner.load(std::memory_order_relaxed) == std::this_thread::get_id();
    }

    /// The holding thread's: the mutex stays held, by `thread` from now on (a thread that works
    /// for the holder while the holder waits). The caller orders the two threads.
    void HandOver(std::thread::id thread) {
        owner.store(thread, std::memory_order_relaxed);
    }

    /// A thread that may have to work for the holder while it waits for the lock (the memory
    /// stage's back serves GPUMem's syncs, and GPUMem can hold the buffer cache's lock while it
    /// waits for one: FRITCHII_MEMSTAGE_AHEAD). Set per thread; while set, a contended lock calls
    /// it between tries with the holder's id and never sleeps. Null everywhere else.
    using ContendedHook = void (*)(std::thread::id holder);
    static inline thread_local ContendedHook t_contended_hook = nullptr;

private:
    // About 10 microseconds of spinning before sleeping.
    static constexpr int SpinCount = 1024;

    static void Pause() {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
        _mm_pause();
#elif defined(__aarch64__)
        asm volatile("yield");
#endif
    }

    // Timed per thread role for the drain sites line (fable52_answer_13 section 3).
    void LockContended() {
        const std::uint64_t begin = Nx2Hitch::NowNs();
        LockContendedUntimed();
        Nx2Hitch::AddLockWait(Nx2Hitch::NowNs() - begin);
    }

    void LockContendedUntimed() {
        if (t_contended_hook != nullptr) [[unlikely]] {
            for (int spin = 0;; ++spin) {
                if (state.load(std::memory_order_relaxed) == 0) {
                    std::uint32_t expected = 0;
                    if (state.compare_exchange_weak(expected, 1, std::memory_order_acquire,
                                                    std::memory_order_relaxed)) {
                        return;
                    }
                }
                t_contended_hook(owner.load(std::memory_order_relaxed));
                if (spin < SpinCount) {
                    Pause();
                } else {
                    std::this_thread::yield();
                }
            }
        }
        for (int spin = 0; spin < SpinCount; ++spin) {
            Pause();
            if (state.load(std::memory_order_relaxed) == 0) {
                std::uint32_t expected = 0;
                if (state.compare_exchange_weak(expected, 1, std::memory_order_acquire,
                                                std::memory_order_relaxed)) {
                    return;
                }
            }
        }
        // 2: held with (possible) sleepers; taken this way it stays marked, so its unlock wakes
        // the next sleeper.
        while (state.exchange(2, std::memory_order_acquire) != 0) {
            state.wait(2, std::memory_order_relaxed);
        }
    }

    std::atomic<std::uint32_t> state{0}; ///< 0 free, 1 held, 2 held with sleepers
    std::atomic<std::thread::id> owner{};
    std::uint32_t depth{}; ///< Recursion depth, touched only by the owner
    void (*on_release)(void*){};
    void* on_release_context{};
};

} // namespace Common
