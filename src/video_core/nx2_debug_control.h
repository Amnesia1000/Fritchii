// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"
#include "common/settings.h"

/// Requests from the debug server (EDEN_NX2_DEBUG_SERVER, docs/nx2/debug-mcp.md) to the thread
/// that owns the rasterizer. The server posts them from its own thread; RasterizerVulkan takes
/// them at the end of each presented frame, so every cache they touch is used from its owner.
/// Without the server nothing is posted and the per-frame cost is one relaxed atomic load.
namespace VideoCore::Nx2Debug {

/// Bumped by every store a live switch or experiment can take (the debug server's `switch`,
/// LoadExperiments, a switch's first read in SwitchValue). Hot paths keep what they derive from
/// the switches and look again only when this moved (Maxwell3D::FrontRunMacro's feature byte).
inline std::atomic<u32> g_switch_generation{0};
inline void BumpSwitchGeneration() {
    g_switch_generation.fetch_add(1, std::memory_order_release);
}

[[nodiscard]] inline int EnvSwitch(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? 0 : std::atoi(value); // whole number: EDEN_NX2_DMA_SNAPSHOT=10
}
/// The switch's value, read from the environment on first use (-1: not read yet). Static
/// initialization ran before the frontend applied nx2_switches.txt, so those settings were lost.
[[nodiscard]] inline int SwitchValue(std::atomic<int>& value, const char* name,
                                    int fallback = 0) {
    int current = value.load(std::memory_order_relaxed);
    if (current < 0) [[unlikely]] {
        int unset = -1;
        const int initial = std::getenv(name) != nullptr ? EnvSwitch(name) : fallback;
        if (value.compare_exchange_strong(unset, initial, std::memory_order_relaxed)) {
            BumpSwitchGeneration();
        }
        current = value.load(std::memory_order_relaxed);
    }
    return current;
}
/// EDEN_NX2_DMA_WAIT_WRITERS: docs/fritchii/switches.md.
inline std::atomic<int> g_dma_wait_writers{-1};
/// EDEN_NX2_SEM_FAST: docs/fritchii/switches.md.
inline std::atomic<int> g_sem_fast{-1};
/// EDEN_NX2_LIST_VERIFY: docs/fritchii/switches.md.
inline std::atomic<int> g_list_verify{-1};
/// EDEN_NX2_LOOP_SKIP: docs/fritchii/switches.md.
inline std::atomic<int> g_loop_skip{-1};

/// EDEN_NX2_DMA_SNAPSHOT: docs/fritchii/switches.md.
inline std::atomic<int> g_dma_snapshot{-1};

/// EDEN_NX2_ACCURATE_PARTS: docs/fritchii/switches.md.
/// Whether GPU accuracy High is on, or the EDEN_NX2_ACCURATE_PARTS bit `part` asks for it.
[[nodiscard]] inline bool AccuratePart(int part) {
    static const int parts = EnvSwitch("EDEN_NX2_ACCURATE_PARTS");
    return Settings::IsGPULevelHigh() || (parts & part) != 0;
}

/// Whether fences run their callbacks (fence-timed releases among them) once the host GPU passed
/// the work before them, not when made (FenceManager::DelayFences).
[[nodiscard]] inline bool DelayedFences() {
    return Settings::IsGPUFenceBehaviorDefault()
               ? AccuratePart(16)
               : Settings::IsGPUFenceBehaviorBalanced() || Settings::IsGPUFenceBehaviorAccurate();
}

/// EDEN_NX2_HOST_PACE: docs/fritchii/switches.md.
inline std::atomic<int> g_host_pace{-1};

/// EDEN_NX2_ARB_SPIN: docs/fritchii/switches.md.
inline std::atomic<int> g_arb_spin{-1};
/// EDEN_NX2_IDLE_PAUSE: docs/fritchii/switches.md.
inline std::atomic<int> g_idle_pause{-1};
/// EDEN_NX2_GPU_WAKE: docs/fritchii/switches.md.
inline std::atomic<int> g_gpu_wake{-1};
[[nodiscard]] inline bool GpuWake() {
    return SwitchValue(g_gpu_wake, "EDEN_NX2_GPU_WAKE", 0) > 0;
}
/// EDEN_NX2_BACK_FENCE_STATS: docs/fritchii/switches.md.
inline std::atomic<int> g_back_fence_stats{-1};
[[nodiscard]] inline bool BackFenceStats() {
    return SwitchValue(g_back_fence_stats, "EDEN_NX2_BACK_FENCE_STATS", 0) > 0;
}
/// EDEN_NX2_RUNAHEAD: docs/fritchii/switches.md.
inline std::atomic<int> g_runahead{-1};
/// EDEN_NX2_RUNAHEAD_MS: docs/fritchii/switches.md.
inline std::atomic<int> g_runahead_ms{-1};
/// EDEN_NX2_BUFFER_CHURN_STATS: docs/fritchii/switches.md.
inline std::atomic<int> g_buffer_churn_stats{-1};

/// EDEN_NX2_CB_DIRECT: docs/fritchii/switches.md.
inline std::atomic<int> g_cb_direct{-1};
/// EDEN_NX2_RT_DEPTH_NOUPLOAD: docs/fritchii/switches.md.
inline std::atomic<int> g_rt_depth_noupload{-1};
/// EDEN_NX2_FOLD_INSTANCE: docs/fritchii/switches.md.
inline std::atomic<int> g_fold_instance{-1};
/// EDEN_NX2_OOT_MACROS: docs/fritchii/switches.md.
inline std::atomic<int> g_oot_macros{-1};
/// EDEN_NX2_DRAW_PACKET: docs/fritchii/switches.md.
inline std::atomic<int> g_draw_packet{-1};
/// EDEN_NX2_BANK_SLOT_STATS: docs/fritchii/switches.md.
inline std::atomic<int> g_bank_slot_stats{-1};
/// EDEN_NX2_BANK_PAGE_TABLE: docs/fritchii/switches.md.
inline std::atomic<int> g_bank_page_table{-1};
/// EDEN_NX2_BANK_SLOTS: docs/fritchii/switches.md.
inline std::atomic<int> g_bank_slots{-1};
/// EDEN_NX2_BANK_SKIP_VERIFY: docs/fritchii/switches.md.
inline std::atomic<int> g_bank_skip_verify{-1};
/// EDEN_NX2_TEX_BANK_GEN: docs/fritchii/switches.md.
inline std::atomic<int> g_tex_bank_gen{-1};
/// EDEN_NX2_STORAGE_SYNC_SCAN: docs/fritchii/switches.md.
inline std::atomic<int> g_storage_sync_scan{-1};
/// EDEN_NX2_STORAGE_SYNC_MEMO: docs/fritchii/switches.md.
inline std::atomic<int> g_storage_sync_memo{-1};
/// EDEN_NX2_FEED_MEMORY_OVERLAY: docs/fritchii/switches.md.
inline std::atomic<int> g_feed_memory_overlay{-1};
/// EDEN_NX2_QMD_3D_PATCH: docs/fritchii/switches.md.
inline std::atomic<int> g_qmd_3d_patch{-1};
/// FRITCHII_MEMSTAGE_THREAD: docs/fritchii/switches.md.
inline std::atomic<int> g_memstage_thread{-1};
/// FRITCHII_MEMSTAGE_FULL_BINDS: docs/fritchii/switches.md.
inline std::atomic<int> g_memstage_full_binds{-1};
/// FRITCHII_MEMSTAGE_AHEAD: docs/fritchii/switches.md.
inline std::atomic<int> g_memstage_ahead{-1};
/// FRITCHII_MEMSTAGE_LIST_DRAWS: docs/fritchii/switches.md.
inline std::atomic<int> g_memstage_list_draws{-1};
/// FRITCHII_MEMSTAGE_FAST_LOOP: docs/fritchii/switches.md.
inline std::atomic<int> g_memstage_fast_loop{-1};
/// FRITCHII_MEMSTAGE_GATHER: docs/fritchii/switches.md.
inline std::atomic<int> g_memstage_gather{-1};
/// FRITCHII_MEMSTAGE_RELEASE: docs/fritchii/switches.md.
inline std::atomic<int> g_memstage_release{-1};
/// FRITCHII_MEMSTAGE_PASS: docs/fritchii/switches.md.
inline std::atomic<int> g_memstage_pass{-1};
/// FRITCHII_MEMSTAGE_MARKS_AHEAD: docs/fritchii/switches.md.
inline std::atomic<int> g_memstage_marks_ahead{-1};
/// FRITCHII_DELAY_MACRO_DRAIN_US: docs/fritchii/switches.md.
inline std::atomic<int> g_delay_macro_drain_us{-1};
/// FRITCHII_DELAY_POLL_DRAIN_US: docs/fritchii/switches.md.
inline std::atomic<int> g_delay_poll_drain_us{-1};
/// FRITCHII_DELAY_BEFORE: docs/fritchii/switches.md.
inline std::atomic<int> g_delay_before{-1};
/// FRITCHII_DELAY_E2C_US: docs/fritchii/switches.md.
inline std::atomic<int> g_delay_e2c_us{-1};
/// FRITCHII_DELAY_DISPATCH_US: docs/fritchii/switches.md.
inline std::atomic<int> g_delay_dispatch_us{-1};
/// FRITCHII_DELAY_DRAW_US: docs/fritchii/switches.md.
inline std::atomic<int> g_delay_draw_us{-1};
/// FRITCHII_DELAY_RUN_US: docs/fritchii/switches.md.
inline std::atomic<int> g_delay_run_us{-1};
/// FRITCHII_DELAY_RUN_KIND: docs/fritchii/switches.md.
inline std::atomic<int> g_delay_run_kind{-1};
/// FRITCHII_DELAY_HANDOVER_US: docs/fritchii/switches.md.
inline std::atomic<int> g_delay_handover_us{-1};
/// FRITCHII_DELAY_HANDOVER_KIND: docs/fritchii/switches.md.
inline std::atomic<int> g_delay_handover_kind{-1};
/// FRITCHII_FRONT_RAN_READS: docs/fritchii/switches.md.
inline std::atomic<int> g_front_ran_reads{-1};
/// FRITCHII_ACQUIRE_TABLE_FIRST: docs/fritchii/switches.md.
inline std::atomic<int> g_acquire_table_first{-1};
/// FRITCHII_E2C_CENSUS: docs/fritchii/switches.md.
inline std::atomic<int> g_e2c_census{-1};
/// FRITCHII_ACQUIRE_MET_BY: docs/fritchii/switches.md.
inline std::atomic<int> g_acquire_met_by{-1};
/// FRITCHII_GATHER_ONE_LOCK: docs/fritchii/switches.md.
inline std::atomic<int> g_gather_one_lock{-1};
/// FRITCHII_GATHER_EMPTY_TEST: docs/fritchii/switches.md.
inline std::atomic<int> g_gather_empty_test{-1};
/// FRITCHII_FENCE_COMMIT_FAST: docs/fritchii/switches.md.
inline std::atomic<int> g_fence_commit_fast{-1};
/// FRITCHII_PIPELINE_KEY_HASH: docs/fritchii/switches.md.
inline std::atomic<int> g_pipeline_key_hash{-1};
/// FRITCHII_CB_WORDS_BLOCK: docs/fritchii/switches.md.
inline std::atomic<int> g_cb_words_block{-1};
/// FRITCHII_BANK_CENSUS: docs/fritchii/switches.md.
inline std::atomic<int> g_bank_census{-1};
/// FRITCHII_BANK_FLAGGED: docs/fritchii/switches.md.
inline std::atomic<int> g_bank_flagged{-1};
/// FRITCHII_BANK_FLAGGED_SLOTS: docs/fritchii/switches.md.
inline std::atomic<int> g_bank_flagged_slots{-1};
/// FRITCHII_VERTEX_INPUT_COMPACT: docs/fritchii/switches.md.
inline std::atomic<int> g_vertex_input_compact{-1};
/// FRITCHII_WORKER_SPIN: docs/fritchii/switches.md.
inline std::atomic<int> g_worker_spin{-1};
/// FRITCHII_BIND_ADDR_MEMO: docs/fritchii/switches.md.
inline std::atomic<int> g_bind_addr_memo{-1};
/// EDEN_NX2_LIGHT_TABLE_ARRAYS: docs/fritchii/switches.md.
[[nodiscard]] inline bool LightTableArrays() {
    static const bool on = EnvSwitch("EDEN_NX2_LIGHT_TABLE_ARRAYS") > 0;
    return on;
}

/// Numbered experiment switches EDEN_NX2_EXP0 .. EDEN_NX2_EXP<ExperimentCount - 1>, live
/// through the debug server. What each one does: docs/fritchii/switches.md; the defaults:
/// src/common/fritchii_switch_defaults.cpp (LoadExperiments).
inline constexpr int ExperimentCount = 128;
inline std::atomic<int> g_experiments[ExperimentCount]{};
/// Set once LoadExperiments has run: Experiment() tests it inline instead of calling
/// LoadExperiments (a function-local static guard) on every check (0.5% of the back thread in
/// the 5.7 -> 5.8.4 sampler comparison, 2026-11-11).
inline std::atomic<bool> g_experiments_loaded{false};

/// Sets the experiments' defaults and reads EDEN_NX2_EXP<n> from the environment, once (a
/// function-local static: thread-safe, and safe from any static initializer). Defined in
/// src/common/fritchii_switch_defaults.cpp, compiled into common, since common (thread.cpp)
/// and tools that link only common read Experiment() too.
void LoadExperiments();

/// The value of experiment switch `n` (EDEN_NX2_EXP<n>), 0 unless set.
[[nodiscard]] inline int Experiment(int n) {
    if (!g_experiments_loaded.load(std::memory_order_acquire)) [[unlikely]] {
        LoadExperiments();
    }
    return g_experiments[n].load(std::memory_order_relaxed);
}

/// The live switch called `name`, or null.
[[nodiscard]] std::atomic<int>* FindSwitch(std::string_view name);

struct CaptureRequest {
    u64 id;
    std::filesystem::path directory;
    bool sampled;
    /// The uniform and storage buffers of every draw and dispatch, and dispatch group counts.
    bool buffers;
};

struct CaptureResult {
    std::string error;
};

struct GpuReadRequest {
    u64 id;
    u64 gpu_address;
    u64 size;
};

struct GpuReadResult {
    std::vector<u8> bytes;
    /// Tegra::MemoryManager::GetID of the address space that maps the whole range.
    u64 address_space;
    std::string error;
};

struct RejectChange {
    u64 generation;
    /// Whether a hash left the list, so programs rejected before must be lowered again.
    bool unrejected;
};

/// Opens a debug server session (clears requests and rejections) or ends it (wakes waiters).
void BeginSession();
void EndSession();

/// Rasterizer thread: whether a capture or a reject change is waiting.
[[nodiscard]] bool HasRequests() noexcept;

/// Server thread: queues a frame capture; the rasterizer takes it at the next frame end.
void RequestCapture(CaptureRequest request);
/// Server thread: withdraws a capture that was not taken yet (it timed out).
void CancelCapture(u64 id);
/// Server thread: waits for the capture's manifest to be complete. Empty on timeout or when the
/// session ends.
[[nodiscard]] std::optional<CaptureResult> WaitCapture(u64 id, std::chrono::milliseconds timeout);
/// Rasterizer thread.
[[nodiscard]] std::optional<CaptureRequest> TakeCapture();
/// Frame dump writer thread, once the manifest summary is written (or the capture failed).
void FinishCapture(u64 id, CaptureResult result);

/// Server thread: queues a read of GPU virtual memory; the rasterizer takes it at the next frame
/// end, where the GPU's own reads of its address spaces run.
void RequestGpuRead(GpuReadRequest request);
/// Server thread: withdraws a read that was not taken yet (it timed out).
void CancelGpuRead(u64 id);
/// Server thread: empty on timeout or when the session ends.
[[nodiscard]] std::optional<GpuReadResult> WaitGpuRead(u64 id, std::chrono::milliseconds timeout);
/// Rasterizer thread.
[[nodiscard]] std::optional<GpuReadRequest> TakeGpuRead();
/// Rasterizer thread.
void FinishGpuRead(u64 id, GpuReadResult result);

/// Server thread: adds or removes native program keys or shader hashes (as EDEN_NX2_REJECT_HASH
/// takes them). Returns the generation to wait for.
u64 SetRejected(std::span<const u64> hashes, bool rejected);
[[nodiscard]] std::vector<u64> RejectedHashes();
/// Lowering: whether the debug server rejects this program. Cheap while the list is empty.
[[nodiscard]] bool IsRejected(u64 hash);
[[nodiscard]] bool AnyRejected() noexcept;
/// Rasterizer thread: the pending reject change, once.
[[nodiscard]] std::optional<RejectChange> TakeRejectChange();
/// Rasterizer thread: the cached programs of this generation were dropped.
void NoteRejectApplied(u64 generation);
/// Server thread: waits until the rasterizer applied the generation.
[[nodiscard]] bool WaitRejectApplied(u64 generation, std::chrono::milliseconds timeout);

} // namespace VideoCore::Nx2Debug
