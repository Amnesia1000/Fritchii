// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <unordered_map>

#include "video_core/nx2_debug_control.h"

namespace VideoCore::Nx2Debug {
namespace {

struct State {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<CaptureRequest> captures;
    std::unordered_map<u64, CaptureResult> finished;
    std::deque<GpuReadRequest> gpu_reads;
    std::unordered_map<u64, GpuReadResult> gpu_read_results;
    std::vector<u64> rejected;
    u64 reject_generation{};
    u64 applied_generation{};
    bool reject_pending{};
    bool unrejected{};
    bool open{};
};

State state;
std::atomic<bool> pending{};
std::atomic<bool> any_rejected{};

void UpdatePending() {
    pending.store(!state.captures.empty() || !state.gpu_reads.empty() || state.reject_pending, std::memory_order_relaxed);
}

} // Anonymous namespace

void BeginSession() {
    std::scoped_lock lock{state.mutex};
    state.captures.clear();
    state.finished.clear();
    state.gpu_reads.clear();
    state.gpu_read_results.clear();
    state.rejected.clear();
    state.reject_pending = false;
    state.unrejected = false;
    state.applied_generation = state.reject_generation;
    state.open = true;
    any_rejected.store(false, std::memory_order_relaxed);
    UpdatePending();
}

void EndSession() {
    {
        std::scoped_lock lock{state.mutex};
        state.open = false;
        state.captures.clear();
        state.gpu_reads.clear();
        UpdatePending();
    }
    state.changed.notify_all();
}

bool HasRequests() noexcept {
    return pending.load(std::memory_order_relaxed);
}

void RequestCapture(CaptureRequest request) {
    std::scoped_lock lock{state.mutex};
    state.captures.push_back(std::move(request));
    UpdatePending();
}

void CancelCapture(u64 id) {
    std::scoped_lock lock{state.mutex};
    std::erase_if(state.captures, [id](const CaptureRequest& request) { return request.id == id; });
    state.finished.erase(id);
    UpdatePending();
}

std::optional<CaptureResult> WaitCapture(u64 id, std::chrono::milliseconds timeout) {
    std::unique_lock lock{state.mutex};
    state.changed.wait_for(lock, timeout, [id] { return !state.open || state.finished.contains(id); });
    const auto it = state.finished.find(id);
    if (it == state.finished.end()) {
        return std::nullopt;
    }
    CaptureResult result = std::move(it->second);
    state.finished.erase(it);
    return result;
}

std::optional<CaptureRequest> TakeCapture() {
    std::scoped_lock lock{state.mutex};
    if (state.captures.empty()) {
        return std::nullopt;
    }
    CaptureRequest request = std::move(state.captures.front());
    state.captures.pop_front();
    UpdatePending();
    return request;
}

void FinishCapture(u64 id, CaptureResult result) {
    {
        std::scoped_lock lock{state.mutex};
        if (!state.open) {
            return;
        }
        state.finished.insert_or_assign(id, std::move(result));
    }
    state.changed.notify_all();
}

void RequestGpuRead(GpuReadRequest request) {
    std::scoped_lock lock{state.mutex};
    state.gpu_reads.push_back(request);
    UpdatePending();
}

void CancelGpuRead(u64 id) {
    std::scoped_lock lock{state.mutex};
    std::erase_if(state.gpu_reads, [id](const GpuReadRequest& request) { return request.id == id; });
    state.gpu_read_results.erase(id);
    UpdatePending();
}

std::optional<GpuReadResult> WaitGpuRead(u64 id, std::chrono::milliseconds timeout) {
    std::unique_lock lock{state.mutex};
    state.changed.wait_for(lock, timeout,
                           [id] { return !state.open || state.gpu_read_results.contains(id); });
    const auto it = state.gpu_read_results.find(id);
    if (it == state.gpu_read_results.end()) {
        return std::nullopt;
    }
    GpuReadResult result = std::move(it->second);
    state.gpu_read_results.erase(it);
    return result;
}

std::optional<GpuReadRequest> TakeGpuRead() {
    std::scoped_lock lock{state.mutex};
    if (state.gpu_reads.empty()) {
        return std::nullopt;
    }
    const GpuReadRequest request = state.gpu_reads.front();
    state.gpu_reads.pop_front();
    UpdatePending();
    return request;
}

void FinishGpuRead(u64 id, GpuReadResult result) {
    {
        std::scoped_lock lock{state.mutex};
        if (!state.open) {
            return;
        }
        state.gpu_read_results.insert_or_assign(id, std::move(result));
    }
    state.changed.notify_all();
}

u64 SetRejected(std::span<const u64> hashes, bool rejected) {
    std::scoped_lock lock{state.mutex};
    bool changed = false;
    for (const u64 hash : hashes) {
        const auto it = std::ranges::find(state.rejected, hash);
        if (rejected && it == state.rejected.end()) {
            state.rejected.push_back(hash);
            changed = true;
        } else if (!rejected && it != state.rejected.end()) {
            state.rejected.erase(it);
            state.unrejected = true;
            changed = true;
        }
    }
    any_rejected.store(!state.rejected.empty(), std::memory_order_relaxed);
    if (changed) {
        ++state.reject_generation;
        state.reject_pending = true;
        UpdatePending();
    }
    return state.reject_generation;
}

std::vector<u64> RejectedHashes() {
    std::scoped_lock lock{state.mutex};
    return state.rejected;
}

bool IsRejected(u64 hash) {
    if (!any_rejected.load(std::memory_order_relaxed)) {
        return false;
    }
    std::scoped_lock lock{state.mutex};
    return std::ranges::find(state.rejected, hash) != state.rejected.end();
}

bool AnyRejected() noexcept {
    return any_rejected.load(std::memory_order_relaxed);
}

std::optional<RejectChange> TakeRejectChange() {
    std::scoped_lock lock{state.mutex};
    if (!state.reject_pending) {
        return std::nullopt;
    }
    const RejectChange change{.generation = state.reject_generation,
                              .unrejected = state.unrejected};
    state.reject_pending = false;
    state.unrejected = false;
    UpdatePending();
    return change;
}

void NoteRejectApplied(u64 generation) {
    {
        std::scoped_lock lock{state.mutex};
        state.applied_generation = (std::max)(state.applied_generation, generation);
    }
    state.changed.notify_all();
}

bool WaitRejectApplied(u64 generation, std::chrono::milliseconds timeout) {
    std::unique_lock lock{state.mutex};
    return state.changed.wait_for(lock, timeout, [generation] {
        return !state.open || state.applied_generation >= generation;
    }) && state.applied_generation >= generation;
}

std::atomic<int>* FindSwitch(std::string_view name) {
    if (constexpr std::string_view prefix = "EDEN_NX2_EXP";
        name.size() > prefix.size() && name.substr(0, prefix.size()) == prefix) {
        int n = 0;
        for (const char c : name.substr(prefix.size())) {
            if (c < '0' || c > '9') {
                return nullptr;
            }
            n = n * 10 + (c - '0');
        }
        if (n >= ExperimentCount) {
            return nullptr;
        }
        LoadExperiments();
        return &g_experiments[n];
    }
    if (name == "EDEN_NX2_LIST_VERIFY") {
        return &g_list_verify;
    }
    if (name == "EDEN_NX2_DMA_WAIT_WRITERS") {
        return &g_dma_wait_writers;
    }
    if (name == "EDEN_NX2_SEM_FAST") {
        return &g_sem_fast;
    }
    if (name == "EDEN_NX2_DMA_SNAPSHOT") {
        return &g_dma_snapshot;
    }
    if (name == "EDEN_NX2_LOOP_SKIP") {
        return &g_loop_skip;
    }
    if (name == "EDEN_NX2_HOST_PACE") {
        return &g_host_pace;
    }
    if (name == "EDEN_NX2_ARB_SPIN") {
        return &g_arb_spin;
    }
    if (name == "EDEN_NX2_IDLE_PAUSE") {
        return &g_idle_pause;
    }
    if (name == "EDEN_NX2_GPU_WAKE") {
        return &g_gpu_wake;
    }
    if (name == "EDEN_NX2_BACK_FENCE_STATS") {
        return &g_back_fence_stats;
    }
    if (name == "EDEN_NX2_BUFFER_CHURN_STATS") {
        return &g_buffer_churn_stats;
    }
    if (name == "EDEN_NX2_RUNAHEAD") {
        return &g_runahead;
    }
    if (name == "EDEN_NX2_RUNAHEAD_MS") {
        return &g_runahead_ms;
    }
    if (name == "EDEN_NX2_CB_DIRECT") {
        return &g_cb_direct;
    }
    if (name == "EDEN_NX2_RT_DEPTH_NOUPLOAD") {
        return &g_rt_depth_noupload;
    }
    if (name == "EDEN_NX2_FOLD_INSTANCE") {
        return &g_fold_instance;
    }
    if (name == "EDEN_NX2_OOT_MACROS") {
        return &g_oot_macros;
    }
    if (name == "EDEN_NX2_DRAW_PACKET") {
        return &g_draw_packet;
    }
    if (name == "EDEN_NX2_BANK_SLOT_STATS") {
        return &g_bank_slot_stats;
    }
    if (name == "EDEN_NX2_BANK_PAGE_TABLE") {
        return &g_bank_page_table;
    }
    if (name == "EDEN_NX2_BANK_SLOTS") {
        return &g_bank_slots;
    }
    if (name == "EDEN_NX2_BANK_SKIP_VERIFY") {
        return &g_bank_skip_verify;
    }
    if (name == "EDEN_NX2_TEX_BANK_GEN") {
        return &g_tex_bank_gen;
    }
    if (name == "EDEN_NX2_STORAGE_SYNC_SCAN") {
        return &g_storage_sync_scan;
    }
    if (name == "EDEN_NX2_STORAGE_SYNC_MEMO") {
        return &g_storage_sync_memo;
    }
    if (name == "EDEN_NX2_FEED_MEMORY_OVERLAY") {
        return &g_feed_memory_overlay;
    }
    if (name == "EDEN_NX2_QMD_3D_PATCH") {
        return &g_qmd_3d_patch;
    }
    if (name == "FRITCHII_MEMSTAGE_THREAD") {
        return &g_memstage_thread;
    }
    if (name == "FRITCHII_MEMSTAGE_FULL_BINDS") {
        return &g_memstage_full_binds;
    }
    if (name == "FRITCHII_MEMSTAGE_AHEAD") {
        return &g_memstage_ahead;
    }
    if (name == "FRITCHII_MEMSTAGE_LIST_DRAWS") {
        return &g_memstage_list_draws;
    }
    if (name == "FRITCHII_MEMSTAGE_FAST_LOOP") {
        return &g_memstage_fast_loop;
    }
    if (name == "FRITCHII_MEMSTAGE_ENTRY_FAST") {
        return &g_memstage_entry_fast;
    }
    if (name == "FRITCHII_MEMSTAGE_GATHER") {
        return &g_memstage_gather;
    }
    if (name == "FRITCHII_MEMSTAGE_RELEASE") {
        return &g_memstage_release;
    }
    if (name == "FRITCHII_MEMSTAGE_PASS") {
        return &g_memstage_pass;
    }
    if (name == "FRITCHII_MEMSTAGE_MARKS_AHEAD") {
        return &g_memstage_marks_ahead;
    }
    if (name == "FRITCHII_DELAY_MACRO_DRAIN_US") {
        return &g_delay_macro_drain_us;
    }
    if (name == "FRITCHII_DELAY_POLL_DRAIN_US") {
        return &g_delay_poll_drain_us;
    }
    if (name == "FRITCHII_DELAY_BEFORE") {
        return &g_delay_before;
    }
    if (name == "FRITCHII_DELAY_E2C_US") {
        return &g_delay_e2c_us;
    }
    if (name == "FRITCHII_DELAY_DISPATCH_US") {
        return &g_delay_dispatch_us;
    }
    if (name == "FRITCHII_DELAY_DRAW_US") {
        return &g_delay_draw_us;
    }
    if (name == "FRITCHII_DELAY_RUN_US") {
        return &g_delay_run_us;
    }
    if (name == "FRITCHII_DELAY_RUN_KIND") {
        return &g_delay_run_kind;
    }
    if (name == "FRITCHII_DELAY_HANDOVER_US") {
        return &g_delay_handover_us;
    }
    if (name == "FRITCHII_DELAY_HANDOVER_KIND") {
        return &g_delay_handover_kind;
    }
    if (name == "FRITCHII_MEMSTAGE_RUN_SPIN_US") {
        return &g_memstage_run_spin_us;
    }
    if (name == "FRITCHII_FRONT_RAN_READS") {
        return &g_front_ran_reads;
    }
    if (name == "FRITCHII_ACQUIRE_TABLE_FIRST") {
        return &g_acquire_table_first;
    }
    if (name == "FRITCHII_E2C_CENSUS") {
        return &g_e2c_census;
    }
    if (name == "FRITCHII_ACQUIRE_MET_BY") {
        return &g_acquire_met_by;
    }
    if (name == "FRITCHII_GATHER_ONE_LOCK") {
        return &g_gather_one_lock;
    }
    if (name == "FRITCHII_GATHER_EMPTY_TEST") {
        return &g_gather_empty_test;
    }
    if (name == "FRITCHII_FENCE_COMMIT_FAST") {
        return &g_fence_commit_fast;
    }
    if (name == "FRITCHII_PIPELINE_KEY_HASH") {
        return &g_pipeline_key_hash;
    }
    if (name == "FRITCHII_CB_WORDS_BLOCK") {
        return &g_cb_words_block;
    }
    if (name == "FRITCHII_BANK_CENSUS") {
        return &g_bank_census;
    }
    if (name == "FRITCHII_BANK_FLAGGED") {
        return &g_bank_flagged;
    }
    if (name == "FRITCHII_BANK_FLAGGED_SLOTS") {
        return &g_bank_flagged_slots;
    }
    if (name == "FRITCHII_VERTEX_INPUT_COMPACT") {
        return &g_vertex_input_compact;
    }
    if (name == "FRITCHII_WORKER_SPIN") {
        return &g_worker_spin;
    }
    if (name == "FRITCHII_BIND_ADDR_MEMO") {
        return &g_bind_addr_memo;
    }
    return nullptr;
}

} // namespace VideoCore::Nx2Debug
