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

/// EXPERIMENT (not for nx2): EDEN_NX2_DMA_WAIT_WRITERS, read at startup and settable live by the
/// debug server's `switch` method, so one session can A/B it.
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
inline std::atomic<int> g_dma_wait_writers{-1};
/// EXPERIMENT, off by default: Ocarina of Time flickers white on the bottom half of Link's house
/// ladder with it (confirmed toggling it live, 2026-11-06), for +1.4 fps. EDEN_NX2_SEM_FAST=1,
/// semaphore acquires without draining the back
/// when the value is already met and no queued write overlaps it (puller.cpp).
inline std::atomic<int> g_sem_fast{-1};
/// TEST: EDEN_NX2_LIST_VERIFY=1 (live), each command list the split's front reads on GPU-written
/// pages is read again as before EXP40 and compared (video_core/nx2_list_verify.h; "list
/// verify" every 10 s). Changes timing: the front waits for every writer again for each list.
inline std::atomic<int> g_list_verify{-1};
/// EXPERIMENT (not for nx2): EDEN_NX2_LOOP_SKIP=1, the split's front does not send the instance
/// loop writes the back would ignore (draw.end in an instance loop, unchanged vertex and index
/// buffer firsts and counts) (maxwell_3d.cpp).
inline std::atomic<int> g_loop_skip{-1};

/// EDEN_NX2_DMA_SNAPSHOT (from EvilOOT's dma-snapshot fix): macro DMA reads of pages a drained
/// read flushed whole skip the drain while nothing sent since may write them (Maxwell3D::
/// FrontDmaRead). 9 = on (default: Lake Hylia 46.5 -> 52.3 fps; verify 0 differences in 235k
/// reads), 10 = on and verify every skipped read against the drained path, 0 = off.
inline std::atomic<int> g_dma_snapshot{-1};

/// EDEN_NX2_ACCURATE_PARTS (bit mask, 0 = none, read once at boot): parts of GPU accuracy High in
/// Fast mode, the split GPU thread kept, to find which one stops a flicker that Accurate mode
/// stops. Not live: switching the counter query mode mid-game crashed (queries opened in one
/// mode were closed in the other).
/// 1 = counter reports written once their host query completes (synced), 2 = precise occlusion
/// queries, 4 = CPU reads of GPU-written images download them, 8 = CPU writes flush GPU data
/// before invalidating it, 16 = fences wait for the GPU work before them.
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

/// EDEN_NX2_HOST_PACE: the host frame pacer (vk_host_pacer.h). 1 = on (default), 0 = only
/// measure how long each frame was shown.
inline std::atomic<int> g_host_pace{-1};

/// EDEN_NX2_ARB_SPIN (zortos293/Lichii#641; off by default here, #641 uses 8): microseconds a
/// guest thread about to sleep in WaitForAddress first watches the word (KAddressArbiter), 0 =
/// off. Live: read on every wait, so profile_toggle.py can A/B it in one session.
inline std::atomic<int> g_arb_spin{-1};
/// EDEN_NX2_IDLE_PAUSE=1 (zortos293/Lichii#641; off by default here, on in #641): the guest
/// core's idle poll (PhysicalCore::Idle, EDEN_NX2_IDLE_SPIN_US) runs PAUSE and reads the clock
/// every 16 pauses, instead of a SwitchToThread and a clock read per iteration. Live: read on
/// every Idle.
inline std::atomic<int> g_idle_pause{-1};
/// EDEN_NX2_GPU_WAKE=1 (live, off by default; port of zortos293/Lichii#1376 9f932fe3c5): the GPU
/// thread (the split's front) waits for suspended channels in a wait that ends on a wake, instead
/// of polling them (ResumePending, then yield or sleep per EXP6, ~120-150 ms a second of front
/// time in Ocarina of Time). A new command (PushCommand) or a GPU write that latches a waiter
/// (Nx2SemaphoreWaiters::OnWrite) wakes it at once (Tegra::Nx2PendingWake); a guest CPU store to
/// the word a channel waits on ends the wait within ~2 us (Nx2SemaphoreWaiters::WordStored). For
/// 500 us after the last progress it waits in hardware (MWAITX/UMWAIT, Common::WaitForFlag) with
/// a full poll every 20 us, then blocks in the OS 500 us at a time (Common::Event::SleepFor).
/// 0 = the polling loop as before (EXP6 applies), and the wake sources return at once. Read on
/// every wait, so profile_toggle.py can A/B it in one session.
inline std::atomic<int> g_gpu_wake{-1};
[[nodiscard]] inline bool GpuWake() {
    return SwitchValue(g_gpu_wake, "EDEN_NX2_GPU_WAKE", 0) > 0;
}
/// EDEN_NX2_BACK_FENCE_STATS=1 (live, diagnostic, off by default; fable52_answer_fences2 4): the
/// back's SignalFence timed in five parts (CommitCacheFlushes with DownloadStagingBuffer, guard,
/// the fence's operation, FlushCommands, InvalidateGPUCache), summed per 5 s ("Back fence split
/// (5 s"), and every fence over 1 ms with its parts ("Back fence over 1 ms", 30 lines per 5 s at
/// most). Also counts the fences that committed no download (FenceReleaseStats::BackFence).
inline std::atomic<int> g_back_fence_stats{-1};
[[nodiscard]] inline bool BackFenceStats() {
    return SwitchValue(g_back_fence_stats, "EDEN_NX2_BACK_FENCE_STATS", 0) > 0;
}
/// EDEN_NX2_RUNAHEAD (live, Ocarina of Time only, default 0; video_core/nx2_runahead.h): 1 = the
/// split's front answers acquires a release it sent meets without draining and writes the word
/// early, with the guest's CPU writes snapshotted per boundary for the back; 2 = verify (old path,
/// chunks compared with guest memory at their stamp); 3 = early answers without writing the word.
inline std::atomic<int> g_runahead{-1};
/// EDEN_NX2_RUNAHEAD_MS (live, default 16): how far the back may lag the oldest early answer.
inline std::atomic<int> g_runahead_ms{-1};
/// EDEN_NX2_BUFFER_CHURN_STATS=1 (live, diagnostic, off by default; Fable's ideal pipeline design
/// 8.3): buffer cache creates and deletes per second by cause, and staging buffer creates by
/// caller (video_core/buffer_churn_stats.h). "Buffer churn (5 s" every 5 s while on.
inline std::atomic<int> g_buffer_churn_stats{-1};

/// EDEN_NX2_CB_DIRECT (live, default 1; from Skitchii (skolvolt/Lichii)): a run of constant
/// buffer words within one device page outside Switch 2 sparse memory is written through one
/// translation and a memcpy instead of WriteWordsCached's page walk (Maxwell3D::
/// CallConstBufferWords). Exact: the same bytes and the same accumulated invalidations; 0 = the
/// page walk.
inline std::atomic<int> g_cb_direct{-1};
/// EDEN_NX2_RT_DEPTH_NOUPLOAD (live, bits, default 0; from Skitchii (skolvolt/Lichii)): new
/// depth/stencil images skip the upload from guest memory when the exact rules (EXP56's clear,
/// EXP8 bit 2's copies) did not already (TextureCache::JoinImages). 1 = render targets with
/// nothing to join, 2 = images whose every join copy is GPU-written, 3 = both (Skitchii).
inline std::atomic<int> g_rt_depth_noupload{-1};
/// EDEN_NX2_FOLD_INSTANCE (live, default 0; from Skitchii (skolvolt/Lichii)): an instance loop
/// of Ocarina of Time's instanced draw macros (0xe52, 0xea0 by code hash) does not draw its
/// first instance alone; the loop's deferred draw covers it (Maxwell3D::FrontRunMacro,
/// DrawManager::FoldFirstInstance). 1 = on.
inline std::atomic<int> g_fold_instance{-1};
/// EDEN_NX2_OOT_MACROS (live, default 1 since 2026-11-12; from Skitchii (skolvolt/Lichii) macro_hle_oot.h): Ocarina
/// of Time's hot 3D macros, found by code hash, run through native C++ handlers on the split
/// front instead of the MME JIT (Maxwell3D::FrontRunMacro). 1 = handlers, 2 = verify (the JIT's
/// writes count; the handler runs on copies for the first 4096 calls of each macro and then one
/// call in 64, and differences are logged: "OoT macro verify"), 3 = verify every call.
inline std::atomic<int> g_oot_macros{-1};
/// EDEN_NX2_DRAW_PACKET (live, default 0; Fable's back-thread plan, Build C step 1): the
/// register writes of an OoT draw macro call (0xe04, 0xe52, 0xea0, identified by code hash) that
/// the front would send as single Direct entries go as one ring entry the back applies in one
/// call, in the same order through the same CallMethod (Maxwell3D::ApplyDrawPacket). Constant
/// buffer words and anything else still go one by one, splitting the packet. 1 = on, 2 = on and
/// the back checks each packet's registers against the values the front wrote ("Draw packet
/// verify").
inline std::atomic<int> g_draw_packet{-1};
/// EDEN_NX2_BANK_SLOT_STATS (live, default 0; Hookshot (NotRexed/Hookshot) research idea 1, step
/// 1): with the tracked native bank check (EXP9 != 0), counts per constant buffer index (c0, c1,
/// c2, ... as the shader numbers them; NVN2: c0 driver storage table, c1 program constants, c2
/// texture handles, c3+ the game's uniform buffers) the bank binds, the pages they cover and
/// the pages compared and found changed for each reason, logged every 5 s next to the "Native
/// bank check" line ("Native bank slots"). Counting only (BufferCache::
/// CheckNativeBankPagesTracked). 1 = on.
inline std::atomic<int> g_bank_slot_stats{-1};
/// EDEN_NX2_BANK_PAGE_TABLE (live, default 0; Hookshot (NotRexed/Hookshot) bf13340's page table,
/// research idea 1 step 3): the native bank check's page snapshots live in a page-indexed table
/// (2 MiB regions of 512 page slots, the last region kept) instead of a node map hashed per page
/// (~1.9% self of the back in the 8 Nov sampler, 720k lookups/s at Castle Town). Exact: the
/// same snapshots, the same compares; a change of the switch empties both stores, so the next
/// bind of each page takes a new snapshot and marks it changed, as after the 16384-page reset.
/// 1 = on.
inline std::atomic<int> g_bank_page_table{-1};
/// EDEN_NX2_BANK_SLOTS (live, default 0; Hookshot (NotRexed/Hookshot) research idea 1): a bit
/// mask of constant buffer indices (bit n = cn) whose 64 KiB native bank binds rely on the write
/// flags alone, for the banks NVN2's driver fills only through the command stream (c0 storage
/// table, c2 texture handles) or never rewrites (c1 program constants). For those binds, with
/// EXP9=3 (the default): (a) the "hot" rule (pages that changed lately are compared on every
/// bind even with no flag set) is dropped, and (b) with EDEN_NX2_BANK_PAGE_TABLE=1 a bind whose
/// whole range has no collected CPU write, no buffer cache write (our own constant buffer words
/// among them, at FlushCaching), no recheck pending and no CPU-modified page, with every page
/// already snapshotted, skips the page loop at once (BufferCache::CheckNativeBankPagesTracked).
/// The game's own banks (c3+) keep the full check: never set their bits (the white flashes,
/// d79572d28a). Pick the mask from EDEN_NX2_BANK_SLOT_STATS: indices whose "hot" changes stay 0.
/// 0 = off. Suggested after that check: 0x7 (c0, c1, c2).
inline std::atomic<int> g_bank_slots{-1};
/// EDEN_NX2_BANK_SKIP_VERIFY (live, default 0): with EDEN_NX2_BANK_SLOTS, every bank the skip
/// passes over and every hot page the slot rule leaves alone is still compared; changes are taken
/// as before (the page is marked) and counted as "missed", which must stay 0, and pages of a
/// skipped bank that the page loop finds a reason for are counted as "disagreed" (must stay 0).
/// Logged every 5 s ("Native bank slot skip"). Costs what the skip saves. 1 = on.
inline std::atomic<int> g_bank_skip_verify{-1};
/// EDEN_NX2_TEX_BANK_GEN (live, default 0; Hookshot (NotRexed/Hookshot) 11d2c1a's per-stage
/// texture bank generations, research idea 2): a graphics pipeline whose bindings are all sampled
/// textures with handles read straight from constant buffers (no handle programs) keeps the view
/// and sampler ids its last draw resolved, with the generations of the pages that draw read
/// handle words and TIC/TSC entries from (TextureCache::TexBankPage: watched pages trap CPU
/// writes; our constant buffer words, buffer cache writes and InvalidateRegion mark them too).
/// A draw with the same handle constant buffer bindings, the same TextureMemoState (pools,
/// limits, channel, mappings, no view removed), the same sampler binding mode and every page at
/// its generation takes those ids: no handle reads, no table reads, no GetSamplerId; the views
/// are still prepared (PrepareImageView). A miss costs the compares. Vulkan GraphicsPipeline::
/// ConfigureImpl; not with EXP32. 1 = on, 2 = on and every 16th hit resolves the full way and
/// compares ("Texture bank memo mismatch", must stay 0). Logged every 10 s ("Texture bank memo").
inline std::atomic<int> g_tex_bank_gen{-1};
/// EDEN_NX2_STORAGE_SYNC_SCAN (live, default 0): storage buffer synchronizes (graphics, and
/// compute while EXP59 is off) walk the memory tracker only from the first to the last word of
/// each 4 MiB manager that holds a CPU-modified or unprotected page, found by reading the words
/// (MemoryTrackerBase::ForEachScannedUploadRange). The words left out hold neither, so the full
/// walk would upload nothing there, re-protect nothing and store back what they hold: the same
/// result without the stores. Not the CleanWords walk (that one trusts clean_words). The sample
/// capture of 10 Nov had 72% of the graphics storage binds in that walk over clean words.
/// 1 = on, 2 = the full walk as before, after counting the CPU and unprotected pages the scan
/// would have left out ("Nx2 storage sync scan verify" every 10 s, must stay 0).
inline std::atomic<int> g_storage_sync_scan{-1};
/// EDEN_NX2_STORAGE_SYNC_MEMO (live, default 0): a graphics storage binding synchronized while
/// the dirty generations of the 4 MiB tracker managers it covers stayed put (WordManager::
/// dirty_gen: bumped whenever a page gains a CPU-modified or unprotected bit) is clean after it,
/// and stays clean while that sum holds: the next bind of the same range skips SynchronizeBuffer,
/// whose walk would find nothing (BufferCache::SynchronizeStorageMemo; 512 entries keyed by
/// range). Not with EXP48's hot pages (they stay dirty) nor with the CleanWords walk unless
/// EDEN_NX2_STORAGE_SYNC_SCAN=1 picks the walk. A miss costs a sum over the managers twice.
/// 1 = on, 2 = every hit synchronizes anyway after checking that no word of the range holds a
/// CPU-modified or unprotected page ("Nx2 storage sync memo verify" every 10 s, must stay 0).
inline std::atomic<int> g_storage_sync_memo{-1};
/// EDEN_NX2_FEED_MEMORY_OVERLAY (live, default 1; fable_answer_pair_2 section 2): the EXP13=3/4
/// feed takes records whose QMD the copy read from an overlay built from guest memory with a
/// small upload patched over it (QmdPrediction::Source::OverlayFromMemory; OoT's per-dispatch
/// 12-byte release patch at QMD word 52). Only overlays built while the feed and this switch were
/// on and no unbounded writer was queued over the QMD (GpuSplit::QueuedGuestWriters) count; the
/// back verifies every release either way (QmdPrediction::Verify). 0 = refused as before.
inline std::atomic<int> g_feed_memory_overlay{-1};
/// EDEN_NX2_QMD_3D_PATCH (live, default 0; fable_answer_pair_2 3.1): a 3D inline upload over a
/// QMD the EXP13=2-4 copy holds retires its own 3D method-write mark (GpuSplit::
/// RangedMethodWrite, which runs right after anyway) before the copy patches the holders
/// (Nx2PatchQmdCopy). Without it the holder's write-stamp check counts the upload's own method
/// write as a later writer, so no holder ever takes a 3D patch and the QMD read waits. 1 = on.
inline std::atomic<int> g_qmd_3d_patch{-1};
/// EDEN_NX2_LIGHT_TABLE_ARRAYS=1 (boot, default 0; port of zortos293/Lichii#1485, jpkto's #354):
/// a texture handle the shader reads from a storage buffer table at an index each invocation
/// reads for itself (OoT's clustered lights: each light's shadow map handle, read per pixel from
/// the light table) becomes a descriptor array of the table's first 64 entries, indexed
/// NonUniform, instead of binding the null image (zortos #778), whose depth-compare sample is 0:
/// fully shadowed, the dark grid that follows Link over water. With it on also: depth-compare
/// samples of uncomparable images (garbage table entries) bind the null image, view lookups skip
/// repeated table elements, a pipeline's texture arrays fit the sampler limits too, and dual
/// vertex programs join their handle programs. Read once, on first use (PipelineCache's
/// constructor, after nx2_switches.txt is applied): it changes what shaders translate to, so it
/// is not live. Off, translation and binding are exactly what they were before the port.
[[nodiscard]] inline bool LightTableArrays() {
    static const bool on = EnvSwitch("EDEN_NX2_LIGHT_TABLE_ARRAYS") > 0;
    return on;
}

/// Numbered experiment switches EDEN_NX2_EXP0 .. EDEN_NX2_EXP79, live through the debug server:
/// slots for batches of fixes built together and A/B-tested in one session
/// (E:\OotRemakeEmulation\work\EXPERIMENTS.md lists which fix uses which slot). 0 = off.
/// EXP8 (texture upload hitches), bits: 1 = keep idle upload staging buffers up to 32 MiB
/// (StagingBufferPool::ReleaseLevel), 2 = no guest upload for a new image that GPU copies
/// overwrite whole (TextureCache::JoinImages), 4 = large 2D levels untiled on the transcode
/// workers too (UnswizzleImage), 8 = guest memory of uploaded textures not hinted cold
/// (TextureCache::TickFrame).
/// EXP10 = 1: macro DMA read snapshots (EDEN_NX2_DMA_SNAPSHOT) skip the drain for compute and
/// copies sent since whose write ranges miss the pages read (Maxwell3D::FrontDmaRead): copies,
/// inline uploads and releases as the front notes them, native dispatches as the back records
/// them once it ran them (the read waits for that much); 2 = the same, but a dispatch still
/// queued drains as before. +4 (5, 6): compute report semaphores noted as before, 16 bytes or
/// anything when unmapped (GpuSplit::NoteReleaseWriter).
/// EXP12 (EDEN_NX2_SEM_FAST acquires read current but unmet, puller.cpp ReleaseFencesForReadable):
/// 1 = their fence release waits only for the queued writers the read can't account for, full
/// drain only if still unmet; 2 = suspend then too, without the full drain; 3 = measure only.
/// EXP17 = 1: a small streamed uniform buffer binds again any stage or slot's copy of the same
/// guest range made in the current command buffer, when the bytes still match
/// (BindSmallUniformReusing; zortos293/Lichii#641 EDEN_NX2_UBO_RANGE_REUSE).
/// EXP18 (buffer binding memos, #641), bits: 1 = a constant buffer bind of the slot's current
/// GPU range keeps its translation and resolved buffer (UBO_REBIND), 2 = storage bindings reuse
/// their last FindBuffer result (STORAGE_MEMO), 4 = buffers touch the LRU once per frame
/// (default 4: exact, see LoadExperiments). All while the mappings and the set of buffers are
/// unchanged (BufferCache::binding_epoch).
/// EXP19 (graphics pipeline lookup without Refresh, PipelineCache::CurrentGraphicsPipelineClean):
/// 1 = as EDEN_NX2_PIPELINE_CLEAN=1, 2 = also #641's patches (blend registers left to the
/// dual-source bit and Dirty::Blending, attribute types redone alone under Dirty::VertexInput).
/// EXP50 (live; with EXP19 / EDEN_NX2_PIPELINE_CLEAN on; after the-cashman's fix in
/// zortos293/Lichii#717): 1 = the blend registers leave the clean lookup's word compare and
/// Dirty::Blending is not looked at; when they moved since the key was made, the attachments
/// RefreshBlending would build from them (FixedPipelineState::BlendingMatches) and the
/// dual-source bit are compared with the key's: equal keeps the pipeline without a Refresh,
/// different patches only the blend part of the key. Reads the registers, so #679's writes
/// without the flag are still caught. Logs "saved by the blend value check" and why the other
/// lookups refreshed in "EDEN_NX2_PIPELINE_CLEAN per second". Off by default, not measured yet.
/// EXP22 (split ring traffic, zortos293/Lichii#641; gpu_split.h DeltaWrite and DrawBatch), bits:
/// 1 = 3D state writes that went as Direct macro writes or (EDEN_NX2_RING_PAIRS) as pairs go in
/// StateDelta blocks, applied in one loop (FRONT_STATE, without its per-register merge),
/// 2 = repeated instance-loop groups go as one DrawBatch entry that counts the repeats
/// (DRAW_BATCH), 4 = same-value index_buffer.first writes stay on the front
/// (SKIP_SAME_DRAWREG, as EXP2), 8 = the back checks each delta and batch against the single
/// writes and keeps theirs (logs "EXP22 check"). 7 = all three. #641's FRONT_CB_ENCODE is not
/// ported (it holds constant buffer writes off the ring).
/// EXP23 = 1: trim unused pipeline libraries every 600 frames past 4096 (#579 stage B;
/// PipelineCache::TickFrame, GraphicsPipelineLibraryCache::TrimUnused). Off by default, not
/// measured yet.
/// EXP24 = 1: Ocarina of Time's present poll (`while (nvnEventGetValue(e) != 1) SleepThread(1 ms)`,
/// from Hookshot) sleeps until the GPU's next write of a 1 instead, at most 1 ms
/// (core/hle/kernel/svc/svc_nx2_oot_present.cpp; the site per build id is in its PresentSites
/// table, or EDEN_NX2_OOT_PRESENT_SITE=<ret>[:<blr>] in hex main offsets). Logs "OoT present
/// poll (EXP24)" counts every 10 s. EXP24 = 2: locate only, sleeps unchanged: short sleeps
/// (<= 2 ms) are counted by call chain and the top 8 logged every 10 s ("OoT present poll
/// locate"). EXP24 = 3: no GPU wake, the poll's 1 ms sleep is EDEN_NX2_OOT_PRESENT_SLEEP_US
/// instead (default 250 us, at most the time asked for). EXP24 = 4 (the default, LoadExperiments)
/// (fable52_answer_pipeline section 2): as 3, but returns at once when the event's value word
/// (event object field EDEN_NX2_OOT_PRESENT_FIELD, default 0) already reads 1, and the sleep ends
/// as soon as it does (Nx2OotPresentPollCheck from KThread::BeginWait: the game's MainThread
/// writes it, not the GPU). Logs "OoT present poll (EXP24=4" every 10 s.
/// EXP25 (Ocarina of Time resource poll backoff, zortos293/Lichii#641; needs the built-in patch
/// placed at launch by EDEN_NX2_OOT_POLL_BACKOFF=1, core/hle/kernel/svc/svc_nx2_oot_poll.cpp):
/// the resource request thread sleeps before rescanning requests that are not ready.
/// 0 = no sleep (as without the patch, plus one SVC per rescan), 1 = 500 us (#641's default),
/// above 1 = that many us, at most 16000. Off by default, not measured yet.
/// EXP26 (native uniform bank check trims, zortos293/Lichii#717; BufferCache::
/// BindHostGraphicsUniformBuffer and CheckNativeBankPagesTracked), bits: 1 = a bind that streams
/// the buffer (at most 4 KiB, not GPU-modified) skips the check, the stream copies guest memory
/// itself (BUFCACHE_BANK), 2 = pages compared with an inline SSE2 loop instead of memcmp
/// (PAGE_CMP), 4 = changed pages copied to the host buffer from their snapshots instead of being
/// marked CPU-modified (two VirtualProtect calls each), for pages whole in the bound buffer with
/// no GPU-written data (UploadNativeBank's upload, without its once-a-frame bank shadows: the
/// check stays per bind, d79572d28a). Off by default; measure 1 and 2 first, then 4. Verify
/// with EXP9=2 ("missed" stays 0) and the flash watch.
/// EXP35 (guest writes of never-GPU-written cached pages, Core::Memory::HandleRasterizerWrite):
/// 1 = collected for the split's back (applied at its next FlushCaching) instead of OnCPUWrite
/// under both caches' mutexes. On by default (2026-11-08): Castle Town guest-core lock spinning
/// 11.6% -> 7.6% of busy (writes 6.1% -> 0.7%), standing 43.0 -> 45.3 fps, route 45.0 fps with
/// 175 flash reports (camera-motion noise ~270 either way).
/// EXP36 (live): 1 = guest reads of cached memory always take GetFlushArea's locked path (turns
/// off its lock-free answer for image-free, never-GPU-written pages). For the washed-out hunt.
/// EXP37 (live): 1 = nx2:gsv answers the driver's armed syncpoint waits right after each guest
/// syncpoint increment (SyncpointManager guest increment hook), not only at its 500 us poll.
/// EXP35=2 (live): every guest write of cached memory is collected for the back, not only writes to
/// never-GPU-written pages. EXP60 (live): 1 = a guest read whose cache mutexes stay taken after a
/// short spin is answered with a flush request served by the back (GetFlushArea).
/// EXP49 (live; async read-back, the core of zortos293/Lichii#1028 0b83798bf6): 1 = a guest read
/// of GPU-written buffer memory (GPU::OnCPURead) has the GPU thread record and submit the
/// downloads without waiting for the host GPU (RasterizerVulkan::FlushRegionForRead). The reading
/// thread waits for them and writes them to guest memory (BufferCache::FinishReadDownloads); a
/// read whose bytes are already on their way only waits. Until they land the bytes stay
/// GPU-modified; guest CPU writes keep their bytes (ClearReplacedDownload, pending writes at
/// landing); writes from the GPU's side (MemoryManager, query results, DMA clears and copies) land
/// them first. 0 = the synchronous flush (Scheduler::Finish on the GPU thread under the buffer
/// cache's lock). Both log "Read-back (EXP49) per second" every 10 s (reads, GPU thread time in
/// read flushes, reader waits). Not ported: #1028's early downloads and EDEN_NX2_READ_CHECK.
/// EXP27 (keep the split's back fed, fewer submits; zortos293/Lichii#717), bits: 1 = the front
/// publishes before it yields or sleeps polling suspended channels (PUBLISH_WAITS,
/// GpuSplit::PublishBeforeWait), 2 = retired (was PUBLISH_IDLE, each entry published at once
/// while the back found the ring empty; EXP39 & 1 replaces it), 4 = the back prefetches the ring 128 words
/// ahead (RING_PREFETCH), 8 = command list copies after a dispatch are submitted when the front
/// waits for them, when the back goes idle or with the next submission, instead of at once
/// (DISPATCH_BATCH, RasterizerVulkan::SubmitNx2ListCopies). Off by default; 1, 2 and 4 change
/// only when work becomes visible, 8 changes submission timing (the #161 family: flash watch).
/// EXP28 (fewer fences, zortos293/Lichii#717; FenceManager::SignalFence), bits: 1 = a fence whose
/// only work is syncing payload releases into host buffers is skipped like an empty one (the
/// guest word was written at once; the copy goes with the next fence made or wait for idle,
/// values kept per release since the query slot may be reused: GuestStreamer::
/// SyncDeferredWrites) (FENCE_TRIM), 2 = a made fence with nothing to wait for on the host GPU
/// joins the newest queued fence, or with none queued and the fencing thread idle runs its work
/// on the split's back at once (FENCE_FOLD, FoldStubbedFence). Off by default; timing family
/// (#161, boulder flashes): flash watch after each. "Fence decisions/s" (GPU profile) counts
/// both.
/// EXP29 (zortos293/Lichii#717's micro bundle), bits: 1 = Device::GetDeviceMemoryUsage reuses an
/// answer younger than 50 ms (MEMSTAT_MS; the buffer and texture caches ask twice per frame on
/// the back; zone "memory query" times the driver query), 2 = GPU::TickWork checks an atomic
/// before taking the sync request mutex (OPT_TICKWORK), 4 = native compute engines no longer
/// record inline upload execs for the old launch method they never run (DROP_NATIVE_UPLOADS).
/// Default 6: 2 and 4 are exact (2 only allows interleavings the mutex already allowed, 4 drops
/// a list nothing in native mode reads); 1 is off, since the caches' GC then sees usage up to
/// 50 ms old. EDEN_NX2_EXP29=0 is the behaviour before.
/// EXP30 = 1 (zortos293/Lichii#559, no env var of its own): a macro DMA read of report memory
/// with pending query results (#215's white flash fix, RasterizerVulkan::FlushQueryResults) waits
/// only until the counter reports in its range are written (Tegra::Nx2WaitAsyncReportsSettled,
/// slots from 12 bytes before the read), not for an operation queued behind every fence, which
/// also waited for all GPU work sent after those reports (8-15% of the back while walking in
/// Ocarina, ~11 waits/s of ~3.7 ms, and the front's FrontDmaRead drain with it). Vulkan only
/// (OpenGL keeps its FlushRegion). 0 = the operation wait as before. Off by default, not measured
/// yet: compare the back's "query DMA wait" profile zone (ms and waits per second) and the white
/// flash watch.
/// EXP31 (render target memo, RT_FAST from zortos293/Lichii#717; TextureCache::
/// UpdateRenderTargets), bits: 1 = a draw whose render targets are clean (no Dirty::RenderTargets)
/// skips preparing them again while the same views were prepared for a draw in the same frame,
/// no image was marked modified, created, deleted, rescaled or CPU written since (modification
/// tick and bindings_epoch unchanged), and every target image is GPU-modified with no CPU
/// modification, alias or sparse state, 2 = with 1: such draws run the loop anyway and check it
/// changed nothing but the targets' own modification ticks (flags, bindings_epoch; logs "RT_FAST
/// (EXP31) mismatch" and drops the memo on a difference). Logs skips, verified and mismatches
/// every 600 frames while on. Off by default; measure the back's "render targets" zone (3 = verify
/// first, then 1).
/// EXP32 (texture binding memo, the core of zortos293/Lichii#717's BINDMEMO for "the same textures
/// as the pipeline's previous draw"; Vulkan GraphicsPipeline::TextureMemo, graphics pipelines whose
/// image bindings are all sampled textures), bits: 1 = a draw that reads the same TIC and TSC
/// handles as its pipeline's previous draw, with the same pools, channel, GPU mappings and texture
/// cache epoch (no view removed: image deleted, joined or rescaled), the same guest TIC/TSC
/// entries and images without CPU writes to upload or aliases, takes that draw's view and sampler
/// ids and descriptor entries instead of GetSamplerId, FillImageViews and PushImageDescriptors
/// (handles are still read every draw: the EXP21 storage pointer rule and the per-bind uniform bank
/// check of d79572d28a are untouched), 2 = verify: every hit for 120 frames after the switch
/// changes, then one hit in 64, runs the full path instead and compares view ids, sampler ids,
/// descriptor entries and rescaling bits ("Texture memo mismatch (EXP32)" log, first 32), 4 = a
/// mismatch turns the memo off until the switch changes. "Texture memo (EXP32=n)" logs hits, misses
/// by cause, records and verify mismatches every 10 s while set. Off by default; 7 = memo, verify
/// and auto-off. Measure the back's "read handles", "fill image views" and "push descriptors"
/// zones (the sampler lookups move from the first to the second with the memo on).
/// EXP33 = 1 (zortos293/Lichii#337 by Inc0rporate; QueryCacheBase::CounterReport and
/// FenceManager::SignalFence): at fence behaviour Default without GPU accuracy High (or
/// EDEN_NX2_ACCURATE_PARTS bit 1), a fence-flagged query report is no longer treated as
/// synchronized: the query stays registered until its fence runs, and that fence is made even
/// when empty and runs the report's operation once the host GPU passed the work before it
/// (VideoCore::g_signaling_query_report_fence), as with delayed fences. Other fences are skipped
/// as before. Read once per report, live. #337's fix for Ocarina's white flashes on leaves and
/// walls at low accuracy; ours already has #215's FlushQueryResults wait (EXP30 trims it), so
/// off by default, not measured yet: white flash watch, and the front's semaphore waits (the
/// reports land later than with the fence skipped).
/// EXP34 (EDEN_NX2_ACQUIRE_PRECHECK=2's reset hole, fable_answer_6 6i (b); puller.cpp
/// PrecheckAudit): where level 2 would answer a GEQ acquire met from guest memory alone,
/// 1 = it takes the normal path instead when the front knows a write to the word is still queued
/// for the back: a plain release (puller, SEM_EXECUTE, 3D or compute 0x6c3) with any payload, a
/// noted write over the word (puller semaphores, 3D reports, compute 0x6c3 releases and
/// reductions, copies, uploads) or, with EXP13 on, a QMD prediction record, and when
/// SemaphoreReadableNow read the word current but below the value (the Castle Town route's only
/// cases, 5-13 per 5 s; most likely work it doesn't track ran between that read and level 2's,
/// the log's first 8 say); 2 = only when that
/// queued plain release writes less than the word read (a reset); 0 = count only. The counts
/// are logged either way ("Acquire precheck (level 2)", every 5 s). Default 1: it only turns met
/// answers into the path level 2 takes for a word below the value, never the other way, so it is
/// a strict tightening; its cost is the drains it puts back (compare 0, 1 and 2 with
/// profile_toggle and the AcquiresFast count).
/// EXP39 (live; keep the split's back fed, zortos293/Lichii#805 62db242b34), bits: 1 = the back
/// stores its position when it runs out of work (GpuSplit::back_idle_at) and, while that is the
/// position last published, the front publishes at its next Commit once it holds 8 words, not
/// after 256 (outside open Pairs and StateDelta blocks; replaces EXP27 & 2), 2 = when an acquire
/// resumes, the snapshot refresh (DmaPusher::RefreshSnapshot) skips the current list's entries
/// that ran already and the entry the acquire suspended inside, which reads memory anyway
/// (CanUseSubmittedCommands); the lists behind it are copied as before. Default 3: both only
/// change when the back sees work or which words nobody parses again are copied. Measure with
/// profile_toggle (EXP39 0 vs 3): back "split wait", front "resume pending".
/// EXP40 (live; command lists on GPU-written pages, zortos293/Lichii#805 34656e8225), bits:
/// 1 = each command list page keeps the 4-byte words GPU work wrote that guest memory lacks
/// (BufferCache::Nx2ListPage::gpu_words: set by every GPU write noted to the page, the whole range
/// a dispatch or copy may write, cleared for words the host writes whole and for all of them when
/// the front flushes the page); once the page was flushed, a list on none of those words reads
/// guest memory and waits for no copy, and a list on some reads those from the copy and the rest
/// from guest memory (the "list GPU wait" zone falls; a storage write mark that
/// EDEN_NX2_WRITTEN_SSBO_MARK_KIB capped counts as written to the end of its last page, since
/// the shader may write past the cap); 2 = a list on GPU-written pages waits
/// only for the compute and copy work sent before the last host sync ("list copy writers"
/// site), not for every one sent so far ("list writers"). Off by default: both read words
/// earlier than before. Verify with EDEN_NX2_LIST_VERIFY=1 ("list verify ... 0 differ"). Always
/// on (bug fixes): copy engine copies and clears tell the list pages what they write to guest
/// memory (their copies went stale), DMAClear checks its whole range (it passed words as bytes),
/// and a front waiting for a list copy publishes first.
/// EXP41 (live; command entries on pages no GPU work wrote yet, DmaPusher::Step): 1 = the writer
/// waits before the port of zortos293/Lichii#805 6229487c8c + d4e8d85d4d: EDEN_NX2_LIST_WAIT_SKIP=1
/// (built in) skips every entry's wait, =0 waits for every queued dispatch and copy. 0 (default)
/// = the new shape: an entry waits only for the compute and copy work sent before the last host
/// sync (WaitForSyncedWriters), and with LIST_WAIT_SKIP=1 an entry that begins between methods
/// (a command buffer the CPU writes) doesn't wait unless copy engine work is still queued;
/// entries inside a method's data (the parameters GPU work writes) still wait. Against the
/// built-in skip-everything this only adds waits. Entries that didn't wait for every writer are
/// checked on the back ("EDEN_NX2_LIST_WAIT_SKIP: command list at ... was GPU-written": stays 0).
/// EXP42 (live; VkImage pool, fable_answer_12 2B; vk_texture_cache.cpp): 1 = a deleted
/// single-level, single-layer render target's VkImage (and its rescaled copy), once the GPU is done
/// with it, is kept with its memory for the next image made with the same create info and view
/// formats, instead of vkDestroyImage + vkCreateImage. Only device-local, non-dedicated images of
/// 256 KiB or more; the pool holds at most 512 MiB, 180 frames each, and empties when the switch
/// goes off or an image does not fit in device memory. A taken image starts uninitialized
/// (UNDEFINED layout, contents discarded) as a new one does. Its bytes are left out of the
/// collector's device memory figure, as deleted images waiting for the GPU are. Logs "Image pool
/// (EXP42) last 600 frames" and counts pooled creates in "Slow clear texture cache". 1 is the default
/// (LoadExperiments); 0 = off.
/// EXP79 (live, 0 = off; port of zortos293/Lichii#1373 cedfc26f47; Kernel::PhysicalCore::Idle):
/// a guest core going idle polls its interrupt flag in a hardware wait (MONITORX/MWAITX on AMD,
/// UMONITOR/UMWAIT on Intel, Common::WaitForFlag; yields elsewhere) instead of a spin of yields
/// (or EDEN_NX2_IDLE_PAUSE's pauses), so Interrupt's store ends the poll at once and the core
/// issues nothing meanwhile. The poll length adapts per core (Common::NextPollWindow, the
/// cpuidle-haltpoll rule): a period a longer poll would have caught doubles it, one past
/// EDEN_NX2_IDLE_SPIN_US (the limit, 100 us for Switch 2) halves it, down to an eighth of the
/// limit. EDEN_NX2_IDLE_STATS=1 logs each core's idle periods every 10 s (polled, blocked, wake
/// latency). Blocking still uses EDEN_NX2_CORE_LOCK's address wait when that is on. In OoT the
/// guest threads' handoffs sit on the frame's critical path. Off by default.
/// EXP87 (live, read per draw, 0 = off; port of zortos293/Lichii#1447; GraphicsPipeline::
/// ConfigureImpl): draw templates. A pipeline whose bindings are all sampled textures with handles
/// read from constant buffers keeps each draw's views, samplers, texture descriptors and rescaling
/// words in a shared table (8,192 slots, 4-way, keyed on the pipeline, the address space, the
/// texture and sampler pools and the handle words). A later draw with the same key, also in a
/// later frame, takes them instead of resolving its views and samplers and pushing descriptors,
/// while the template's TIC and TSC bytes are still in memory, no mapping changed
/// (HostPointerSerial), no image changed its scale (ScaleEpoch) and none of its views was deleted
/// or moved (ImageViewGeneration). Not with EXP32 or EDEN_NX2_TEX_BANK_GEN on, for mesh pipelines,
/// or with the OoT FSR LOD bias on a sampler. 1 = on; 2 = verify: draws a template holds for are
/// resolved in full and compared with it, nothing taken from templates ("Draw template verify in
/// 10 s: ... mismatched" must stay 0). With EXP87 != 0, "Draw templates in 10 s" logs hits,
/// misses, kept templates and rejections every 10 s.
/// EXP86 (live, 0 = off; fable52_answer_gpuwritten; RasterizerVulkan::GetFlushArea): a guest
/// read of pages the GPU ever wrote (no image there, EXP36 off) skips the caches' locks
/// when the buffer cache's page mirrors (gpu_written_mirror.h) say the locked path would answer
/// "preemptive" and change nothing: no GPU-modified byte on the pages now, and a Preflushable
/// mark or no armed read trap there. The mirrors change only under the buffer cache's mutex; a
/// sequence counter is odd from a hold's first change to its release, and a busy counter sends
/// the read to the locks. 1 = the fast path; 2 = verify: locked reads compare the mirrors' answer
/// with what the locked path decides ("mirror clean but cache not" must stay 0), no fast answers.
/// "GPU-written page mirror" logs the answers every 5 s, also while off.
/// EXP88 (live, bits, 0 = off; port of zortos293/Lichii#1441 5e186d4f13 + d18b636a69 parts a to
/// c, fable52_answer_next70 3; nx2_cb_path.h): 1 = part c: RasterizerVulkan::InnerInvalidation
/// (FlushCaching's ranges) and, on the back, OnCacheInvalidations (collected guest writes) skip
/// BufferCache::WriteMemory and its lock where it would change nothing: no page ever left the
/// CPU-modified state (MemoryTrackerBase::IsRegionEverSynchronized), no GPU write ever marked one
/// and no enabled vertex binding overlaps (BufferCache::WriteReachesBuffers); its three page marks
/// stay (NoteWriteUnlocked: the bank check reads g_buffer_write_pages). The collected writes also
/// skip the texture lock and the pipeline cache where their page filters are clear, and the
/// ClaimCpuWrites lock where no GPU-modified mark ever reached. 2 = parts a and b: constant buffer
/// data (CallConstBufferWords in place of CB_DIRECT, ProcessCBMultiData in place of
/// WriteBlockCached) goes to the bound buffer's device address, translated once per CB_ADDRESS,
/// CB_SIZE and memory map (Maxwell3D::WriteBoundConstantBuffer), its device ranges logged for
/// FlushCaching (MemoryManager::WriteConstantBufferData); buffers that aren't one mapped device
/// range and Switch 2 sparse memory take the old path. 4 = verify: skipped ranges take the locked
/// WriteMemory anyway after checking it changes nothing; direct writes are read back and their
/// translation redone at FlushCaching. Both mismatch counts must stay 0. Same family as EXP85
/// (its inline writes reach FlushCaching with & 1) and CB_DIRECT (& 2 replaces it). "Constant
/// buffer path (EXP88" logs the counters every 5 s while not 0.
/// EXP90 (live, 0 = off; Fable, fable52_answer_after88 2; TextureCache::TouchImageLru): an image's
/// LRU node is touched once a frame, decided from a frame tick kept on the image
/// (ImageBase::lru_touched_tick) instead of reading the node (a cache miss) on every
/// PrepareImage, the way EXP18 & 4 does it for buffers. Exact: the node already ignores a second
/// touch with the same tick, so the LRU order and the ticks eviction and the sweeps read stay the
/// same. "Image LRU touch (EXP90" logs touches made and skipped every 5 s while not 0.
/// EXP89 (live, 0 = off; N > 0 = frames; Fable's lingering protection, fable52_answer_after88
/// section 1; core DeviceMemoryManager::UpdatePagesCachedCountNoLock): when a page's cached
/// count falls to 0 it stays protected and is noted with the frame; when a cache counts it again
/// within N frames nothing is called (no unprotect, no protect). RasterizerVulkan::TickFrame's
/// sweep (DeviceMemoryManager::ReleaseLingering) really uncaches the pages noted more than N
/// frames ago; Map, Unmap and UnregisterProcess uncache their pages at once. Exact: a lingering
/// page is a protected page no cache tracks, whose collected writes reach the caches as today.
/// Switching to 0 live releases every noted page at the next sweep. Start with 4. Measure first
/// with EDEN_NX2_PROTECT_STATS=1 and EXP89=0: "Page protection frames since the last unprotect"
/// gives the share of protects each N would cancel; "Lingering protection (EXP89" logs the
/// cancelled calls, releases, pages held and, as the verify, pages left lingering after an
/// Unmap (must stay 0). Watch "writes collected" in the Guest access line: if it rises much,
/// lower N.
/// EXP91 (live, 0 = off; fable52_answer_uniformkeeps stage 0; counting only, nothing bound
/// changes): RasterizerVulkan::InnerInvalidation notes every cached write range it gets
/// (FlushCaching's, EXP88's constant buffer data among them) in a ring of 128 with a sequence
/// number, before any early return (BufferCache::NoteCachedWrites). Each of today's slot rebinds
/// in BindSmallUniformReusing (same bytes, same tick) is then judged as a keep would be: K when
/// the slot's last bind was the same range and no range noted since overlaps it, with nothing
/// pending in MemoryManager (accumulator, EXP88's ranges); else not kept, by reason (range
/// written, ring overflow, pending cached writes, other range, no keep yet). "Uniform keeps
/// count (EXP91" logs small binds, rebinds and those counts per second every 5 s while not 0.
/// Gain of the later stages is about K x 55 ns; under 100,000 a second, stop there.
/// EXP102 (live, pauses, 0 = none; fable52_answer_yield 7): with EXP98=1, an early-returned
/// yield pauses this many times, keeping the old cadence (try 6, 16, 64).
/// EXP103 (live, pauses, 0 = off): common WaitForFlag's fallback (no MONITORX/WAITPKG, as on
/// this owner's PC; EXP79's idle wait uses it) polls with N pauses between checks instead of
/// SwitchToThread (try 64).
/// EXP104 (live, 0 = off, 1 = on with entries kept 180 frames, N > 1 = kept N frames; Fable's
/// ideal pipeline design 8.3, step 3b; renderer_vulkan/vk_buffer_recycler.h): a pool of VkBuffers
/// by size class for the buffer cache's buffers and the staging buffers. Buffer cache buffers
/// are made at their size class (powers of two from 64 KiB, up to 32 MiB; sparse-compatible ones
/// never) and, when the delayed destruction ring lets a Buffer go, its VkBuffer is pooled instead
/// of destroyed; staging buffers the staging pool releases are pooled by memory usage and size.
/// Creates take a pooled one of their class first. Exact: an entry carries the scheduler tick
/// current when it was returned and is handed out only once the GPU passed it (IsFree); the
/// cache clears a new Buffer's range as before. 256 MiB held at most (oldest go first); entries
/// past their age, over the cap or left when the switch goes off are destroyed by the back in its
/// split idle hook (EXP99's way), at the frame's end without the split or past 256 MiB waiting.
/// Costs up to 2x memory per buffer cache buffer (the class rounding). Gate (Fable): real creates
/// ("missed") near 0 a second in steady play, device memory flat after 10 minutes. With
/// EDEN_NX2_BUFFER_CHURN_STATS=1 the "Buffer churn (5 s" line ends with the pool's counters.
/// EXP106 (live, 0 = off; Fable's delta draw port plan, fable_answer_backthread_threads.md 10,
/// step D2): every graphics pipeline keeps its last draw (GraphicsPipeline::KeptLastDraw: raw
/// handle words, computed handle values, views, samplers, view generations, image descriptor
/// entries per stage, rescaling words, ScaleEpoch), through a second instantiation of
/// ConfigureImpl. Records only, no draw changes; for the delta draw steps after D2.
/// EDEN_NX2_FASTDRAW_COUNT=1 keeps them too. Off, PrepareDraw reads this switch once a draw.
/// EXP107 (live, 0 = off; port of zortos293/Lichii#1161 13d9320cfb, lucastaonline): inline
/// constant buffer data (Maxwell3D::CallConstBufferWords, ProcessCBMultiData) is compared with
/// the words memory holds, on the back, in order (Maxwell3D::CompareConstantData; one GPU page,
/// not sparse). Equal and safe: not written at all, so no invalidation, no buffer cache mark and
/// no delta-draw break (OoT rewrites ~258k blocks a second with the same words, often over
/// texture handles). Safe means #1161's conditions: no GPU-written data may be on the pages
/// (VideoCore::MayHoldGpuWrites, always true outside Switch 2 titles) and no guest read download
/// may land there; plus ours: not the texture header or sampler pools (also through an alias)
/// and no run-ahead used this session. Equal but not safe: written as before. With
/// EDEN_NX2_FASTDRAW_COUNT=1 equal writes are neutral for the count, and the 5 s line ends with
/// how many changed nothing and how many were not written.
/// EXP108 (live, 0 = off; port of zortos293/Lichii#1436 = #1161 4c44094e30, lucastaonline):
/// the GPU's own cached writes (MemoryManager::FlushCaching: inline constant buffer data, its
/// accumulated ranges) mark their pages CPU-modified but leave them tracked
/// (MemoryTracker::MarkRegionAsWrittenTracked) instead of untracking them as a guest CPU write
/// does; the upload clears the bits and has nothing to track again. Saves a VirtualProtect pair
/// per page per write cycle (~4% of the GPU thread at OoT's Kokiri water, +1.4 fps on the
/// author's PC). Only on pages with no GPU-modified range and no GPU-modified bit (a tracked
/// page's GPU bits count); never in Nx2Perf verify. A guest CPU access to such a page takes the
/// slow path, as for any tracked page; a guest write there untracks it as before.
/// EXP109 (live, 0 = off; port of zortos293/Lichii#1161 f5902c6c68, lucastaonline): a guest write
/// that untracks buffer cache pages (RasterizerVulkan::OnCPUWrite) changes their host protection
/// after dropping buffer_cache.mutex instead of under it (the back waited for those system calls
/// before draws). The word state changes under the lock as before; the page count decrements are
/// collected per thread (VideoCommon::DeferredPageCaching) and applied in order before OnCPUWrite
/// returns, so the guest's write still finds the page unprotected. Ours defers only decrements
/// (increments, e.g. an upload inside the write's slow path, apply at once): late decrements only
/// keep a page protected a little longer; a late increment could let a page the texture cache
/// tracks lose its protection in between. Never in Nx2Perf verify.
/// EXP110 (live, 0 = off, 1 = on at 512 MiB, N > 1 = on at N MiB; fable52_answer_stall3 3, the
/// ghost pools' "unproven first + takes"): the EXP42 pool's eviction rule. Over the cap, oldest
/// first, the first entry whose shape (exact create info) was never taken from the pool goes;
/// failing that the one whose shape has the fewest takes, then the least recent take, then the
/// oldest. Takes and the last take are kept per shape in a map that outlives entries, evictions
/// and the emptying on memory pressure (bumped on a take only, never aged; at 4,096 shapes the
/// 3,072 most recently taken stay). Ages as before (180 frames, EXP82 for proven shapes, proof
/// looked up per shape when tested); EXP65 is not applied. The cap replaces EXP66's and is shared
/// with EXP99's idle backlog: the pool holds the cap less the backlog, the backlog is bounded at
/// 128 MiB instead of 512 and, past 64 MiB, loses two images a frame end too. Memory pressure,
/// EXP42 off and a lowered cap still trim by the plain oldest-first loop. The "Image pool (EXP42)"
/// line ends with the rule, the backlog in MiB and the frame-end destroys. Ghost run (orbit,
/// 182-212 s): 66-68% of lookups taken at 512 MiB against 60-64% for today's rule at 1,024 MiB.
/// EXP111 (live; 0 = Windows' default, 1 = above normal, 2 = high; vk_rasterizer.cpp TickFrame):
/// the process priority class, applied when the value changes. The owner gets more fps with the
/// window focused (Windows' foreground boost); this keeps the game's threads ahead of other
/// normal-priority programs when it isn't. Watch for a stream encoder starving at 2.
/// EXP101 (live, 0 = off, 1 = runs, 2 = cached page pointer + prefetch; fable52_answer_presenter): nvhost_gpu::SubmitGPFIFOImpl's pushbuffer
/// snapshot reads runs of entries whose GPU addresses follow each other with one ReadBlockUnsafe
/// instead of one per entry (OoT's presenting thread submits ~6,500 entries a frame; the loop is
/// on the frame's critical path). Exact: same words at the same offsets. "Pushbuffer snapshot
/// (EXP101" logs submits, entries, reads and loop time every 5 s while on.
/// EXP105 (live, 0 = every entry; N > 1 = 1 in N): DmaPusher::Step's check that a command list
/// the front read early was not GPU-written by queued work (EDEN_NX2_LIST_WAIT_SKIP log). It only
/// reads and logs, never fired in OoT on 2026-11-12, and cost ~8 ms/s of the back (try 64).
/// EXP99 (live, 0 = off; fable52_answer_chicken 1): evicted EXP42 pool images (OoT's per-NPC
/// depth targets, ~3 a frame while a chicken passes or an enemy dies) are destroyed on the image
/// free thread (2; needs EDEN_NX2_ASYNC_IMAGE_FREE=1) or, with 1, by the back itself in its idle hook
/// (up to 2 each time it runs out of work), instead of at the frame's end. The pool line every 600
/// frames counts where they went; the timeline shows "back: image pool tick".
/// EXP100 (live, 0 = off; fable52_answer_imagereads, Fable's F1; RasterizerVulkan::GetFlushArea):
/// a guest read of pages that hold an image skips the caches' locks when no image there was
/// GPU-modified. TextureCache::gpu_image_filter counts per 64 KiB page every registered,
/// non-sparse image that was GPU-modified since it was registered (CountGpuModified at
/// MarkModification, SynchronizeAliases, JoinImages and RegisterImage; removed at
/// UnregisterImage; clears of the flag are ignored, so it is conservative), under the texture
/// cache's mutex, with a sequence counter (image_seq, odd while a hold changes the count, closed
/// by the mutex's release hook). Nothing counted on the pages, no sparse image ever there, the
/// counter even and unmoved, and the buffer side clean as today (no GPU write ever marked the
/// pages, or EXP86 = 1's mirrors say clean): the locked path would answer the same preemptive
/// area and change nothing. Not with EXP36 or EXP81 on or EXP43 = 4. 1 = the fast path; 2 =
/// verify: locked reads compare the count with the texture cache's locked look, no fast answers
/// (mismatches must stay 0, the first 16 are logged with the image's range, flags and
/// nx2_gpu_counted). "Guest reads locked because" logs the answers and why reads stayed locked.
/// EXP96 (live; fable52_answer_guest 4): the HLE reply hold (k_server_session.cpp HoldHleReply,
/// written for Mario Tennis Fever, active in every title). 0 = EDEN_NX2_IPC_MIN_US or 50 us;
/// 1 = off; N > 1 = N us. In OoT the service thread that presents waits in it ~14 ms/s.
/// EXP97 (live; fable52_answer_guest 3a): spins of a contended KSpinLock (the scheduler lock)
/// before it sleeps. 0 = 128 as before; N > 0 = N. EDEN_NX2_KLOCK_STATS=1 logs its locks every
/// 5 s ("Kernel spin lock (5 s").
/// EXP98 (live; fable52_answer_yield): guest yields (svcSleepThread 0 and -1) skip the scheduler
/// lock when it would find nothing to do (k_scheduler.cpp YieldIsKnownNoOp). A change counter
/// per core, raised under the lock for the cores of every thread whose scheduling changes; a
/// yield that did nothing stamps its core and that core's counter and raises nothing; the early
/// return reads the lock word, then the counter. 0 = the code as before (per-process count);
/// 1 = early return; 2 = verify: always take the lock, check the stamp's prediction under it
/// and count mismatches (must be 0, the first 16 are logged). While not 0, "Guest yields
/// (EXP98" logs yields by form, early returns, lock-held checks and locked outcomes every 5 s.
/// -2 (YieldToAnyThread) is only counted.
/// EXP94 (live, 0 = off; fable52_answer_front 1): RasterizerVulkan::HasCachedReports answers
/// "none" from the query cache's page filter before taking its mutex (the front asks for every
/// macro DMA read, ~5 ms/s of the front in the market heavy phase). Exact: the filter marks every
/// page a query was ever registered on.
/// EXP95 (live, 0 = EDEN_NX2_OOT_PRESENT_SLEEP_US or 250; N > 0 = N us; fable52_answer_front 2):
/// the short sleep of OoT's present poll with EXP24 on. The event word is written by the game's
/// main thread, so a sleep only ends at its timeout; shorter sleeps end the poll sooner after
/// the write (try 100).
/// EXP93 (live, 0 = as EDEN_NX2_UBO_INPLACE says, 1 = on; fable52_answer_writer 3): small
/// uniform buffers are compared with their slot straight in guest memory instead of through a
/// scratch copy (VideoCommon::UniformInPlaceCompareEnabled). Judge on "bind stage buffers".
/// EXP92 (live, 0 = off; fable52_answer_uniformkeeps 3 and 4, idea zortos293/Lichii#1441 part d;
/// BufferCache::BindSmallUniformKeeping): a small uniform slot bound again with the same range,
/// in the same command buffer, with nothing written to that range since, binds the slot's stream
/// copy again without reading or comparing guest memory (a "keep"). Written-since is known two
/// ways: tier 1, the back's cached writes by exact range (EXP91's ring, filled while EXP91 or
/// EXP92 is on; nothing pending in MemoryManager); tier 2, a generation per device page
/// (Common::UniformKeepPages) raised after the bytes land by every other writer (MemoryManager
/// writes, device memory writes, the GPU write listener, in-place spans, run-ahead words, and
/// guest CPU stores, which trap because a kept range's pages are watched: one
/// UpdatePagesCachedCount +1 each, given back at RasterizerVulkan::UnmapMemory and at
/// DeviceMemoryManager Map/Unmap; a page stored to more than 64 times a frame is denied). Plus
/// the device translation generation. Not on native bank binds or behind a run-ahead frontier.
/// 1 = keeps on, every 256th kept bind verified (the tripwire): one stale bind logs an error and
/// turns keeps off for the session. 2 = verify: a keepable bind runs the normal path anyway, which
/// must rebind the same copy ("same"); stale ones are counted, the first 32 logged, and bound
/// from their fresh bytes. "Uniform keeps (EXP92" logs binds, kept, not kept by reason, watched
/// pages, guest stores on them, denied pages, verified and stale every 5 s while not 0. Stale
/// must stay 0. 1 -> 0 at any time drops every keep (watched pages stay watched until unmapped).
/// EXP85 (live, bits, 0 = off; fable52_answer_microcuts 1 and 2b): 1 = a small inline upload to
/// pages no GPU write ever marked skips the buffer cache's lock (InlineMemory would find nothing
/// there; the list note and the query cache as before), 2 = BufferCache::WriteMemory answers
/// "no GPU-written data on these pages" from the GPU-written bit before the range set. 3 = both.
/// EXP84 (live, 0 = off; fable52_answer_microcuts 3.4; RasterizerVulkan::GetFlushArea): a guest
/// read skips the caches' locks when no image is registered on its pages now (TextureCache::
/// image_count_filter, counted at register and unregister, plus the pages sparse images ever
/// covered) instead of never (page_filter, whose bits never clear while the game's targets move
/// every frame) and no GPU write ever marked them. Daytime market: guest cores waited 22-54 ms/s
/// for those locks, ~3,500 locked reads a second, 65 of them needing a flush. "Guest reads
/// locked because" logs the reasons every 5 s, also while off.
/// EXP83 (live, 0 = off; fable52_answer_microcuts 3.2; RasterizerVulkan::UpdateVertexInput): a
/// vertex input state identical to the last one recorded in this command buffer is not sent
/// again (any vertex buffer address or stride write resent all 32 bindings). The memo drops on
/// StateTracker invalidation (new command buffer, input or whole state; helper passes call
/// InvalidateState) and when a pipeline without dynamic vertex input is drawn. "Vertex input
/// (EXP83" logs sends and repeats every 10 s, also while off.
/// EXP82 (live, frames, 0 = off; fable52_answer_after7778 1; TextureCacheRuntime::TickFrame):
/// proven EXP42 pool entries (a shape taken from the pool before) stay this many frames instead
/// of 180, so per-NPC depth targets whose size comes back later find their image. Unproven
/// entries and the byte cap (EXP66) are unchanged. Try 1800. Judge on the pool's taken/missed
/// line and the clear zone.
/// EXP81 (live, frames, 0 = off; fable52_answer_readback_card 3; BufferCache TickFrame and
/// GetFlushArea): with EXP69's Preflushable reset, unmark only the pages the CPU has not read for
/// this many frames. Every CPU read, locked (GetFlushArea) or EXP86's lock-free clean answer,
/// sets its pages in a page bitmap that TickFrame folds into per-page stamps every frame
/// (fable52_answer_fences2 2), so it works with EXP86 on. Unmarking every page made each page
/// the game still reads need one async read (~40 a second at the market) to be marked again; a
/// mark only decides whether a GPU write is copied at the fence or at the read, so it is exact.
/// "Preflush reset" logs the KiB kept. Try EXP69=4 with EXP81=4, then 2 and 8.
/// EXP78 (live, 0 = off; fable52_answer_killuploads 3-4; TextureCache::TcSkipStale): what
/// EDEN_NX2_TC_SKIP_STALE=1 does at boot, switchable: a deleted GPU-written image marks its whole
/// pages stale (CPU writes clear them) and a new image whose range is all stale (and not
/// buffer-written) skips its upload from guest memory nobody wrote. Fight kill bursts: 30-105
/// such uploads inside draws, 3-10 ms. Marks start when it turns on. "stale uploads skipped" in
/// the texture cache's 600-frame line.
/// EXP77 (live, 0 = off; fable52_answer_keeptex_2 2b; TextureCache): a GPU-written chain target
/// the cold sweep deletes leaves its shape at its address; a texture header that asks for that
/// address again marks the shape (and address) GuardSampled, so the sweep keeps that shape from
/// then on instead of deleting it and remaking it from stale guest bytes. Fight: 197 GPU-written
/// images a window remade after a sweep. Only keeps longer. Counted in the "Cold sweep" line.
/// EXP76 (live, bits; fable52_answer_keeptex 2a; TextureCache): kill frames remade textures the
/// game already had (fight: 767 of 833 new sampled images in 10 s had bytes made before; a BC5
/// 2448x3056 one 129 times). 1 = FindImage keeps an image the game's remap flagged when the same
/// texture is asked for at the same GPU and device address with the same shape (never
/// GPU-written, no alias or sparse state): no delete, create or new views; it is uploaded as a new
/// one would be. 2 = whole uploads of 64 KiB or more store a CityHash128 of the guest bytes (read
/// after the upload, dropped if the image was CPU-written meanwhile); the next whole upload whose
/// bytes hash the same is skipped. Any other refresh path clears the hash. The
/// EDEN_NX2_IMAGE_HASH_LOG=1 summary logs "Remaps (EXP76=n)": images flagged, kept, hash checks,
/// uploads skipped. 4 (with 2; measure only, fable52_answer_keeptex_2 step 1): hashed images are
/// indexed by (shape, content hash); each new sampled image hashes its guest bytes and the
/// "Content reuse" line counts which reuse would serve it (live idle twin, live busy twin,
/// kept deleted image, nothing) and the hash cost in GB/s. GPU writes clear the hash
/// (MarkModification, SynchronizeAliases). Off by default.
/// EXP75 (live; us, 0 = off; fable52_answer_pipeline section 3; Core::Timing::CoreTiming's timer
/// thread): precise guest timers. The thread spins the last this-many us before a precise event
/// (the kernel's hardware timer: guest sleeps and timeouts, EventType::precise) instead of waking
/// on the OS timer, whose 0.5 ms tick made the game's 100 us sleeps last ~420 us and its 1 ms
/// present poll 1.45 ms. Other events stay on the timer. Costs a share of one logical CPU (kept
/// off the GPU threads' cores). Default 600 (LoadExperiments).
/// EXP74 (live; us, 0 = off; fable52_answer_perdraw 1; FenceManager::ReleaseThreadFunc): the fencing
/// thread lets the texture/buffer cache lock go between two pops after holding it this long, so a
/// batch of 20-60 fences no longer keeps the back waiting 1-13 ms.
/// EXP73 (live; 0 = off; fable52_answer_bosswave_2 3b; Maxwell3D macro DMA snapshot miss path): the
/// report flush after a drained read runs only when reports are listed over the bytes read; else it
/// was an empty round trip to the back (~2,850 a second in the Stalchild fight).
/// EXP72 (live; 0 = off; fable52_answer_bosswave 3; BufferCache::GetFlushArea, WaitFenceDownload):
/// a preflushed guest read whose GPU-written bytes are all in a submitted fence download waits (up
/// to 20 ms) for that download to land instead of a new copy behind everything recorded since.
/// EXP70 (live; ms, 0 = off; fable52_answer_acquire 4; PresentManager::GetRenderFrame): after this
/// long without a free render frame, the renderer drops the oldest queued frame that is not at
/// the present thread's front and reuses it. With EDEN_NX2_PRESENT_EXTRA_FRAMES (boot) giving more
/// render frames than swapchain images. Display stalls (acquire blocked 76-140 ms) stalled emulation.
/// EXP69 (live; frames, 0 = off; fable52_answer_stalls 5.1; BufferCache::TickFrame): every N frames
/// the Preflushable marks GetFlushArea set go; a range the CPU still reads is trapped on its next
/// read and marked again. Marks never cleared before: ~100 MB of fence downloads a second.
/// EXP68 (live; bits, 0 = off; fable52_answer_npcdips 6): bit 1 = a depth clear whose scissor is
/// the real width (zeta width rounded up to 16, same 64-byte column) looks up the scissor-wide image
/// the draws then use, so OoT's per-NPC depth targets are one image a frame, not two plus an upload
/// and alias copies. The next depth bind at the address checks the size; a miss turns it off there.
/// Bit 4 = images at an address whose target keeps changing size skip the EXP42 pool (their dims
/// never come back; the pool sat at 512 MiB during NPC passes).
/// EXP67 (live; bucket in px, 0 = off): depth render targets at an address whose target changes
/// dims get a host image rounded up to the bucket (ImageFlagBits::HostPadded), so the EXP42 pool
/// reuses the VkImage for the next frames' nearby dims instead of a create + 0.2-1.3 ms depth bind
/// (OoT per-NPC depth targets). A texture lookup that finds a padded image stops padding there.
/// EXP66 (live; MiB, 0 = built in): the EXP42 pool's cap (512 MiB; 1,536 with EXP42=2), trimmed
/// at once when lowered. NPC passes held it at its cap (~0.5 of the owner's +0.8 GB).
/// EXP65 (live; 0 = off; vk_texture_cache.h): EXP42 pool entries whose exact create info was never
/// taken back age out after this many frames (proven shapes keep the 180). Per-NPC targets sized by
/// screen bounds filled the pool to 512 MiB with dims that never repeat.
/// EXP64 (live; N = min frames untouched, default 16, 0 = off; fable_answer_npcdips 3, NPC dips): a new render target that a clear
/// overwrites whole retires the images at its exact address that the GPU wrote in an earlier
/// frame (TextureCache::RetireSlotPredecessors) instead of leaving them as bad overlaps for EXP58's
/// sweep 60 frames later. Ocarina draws each passing NPC into a D32 target sized by its screen
/// bounds at a fixed slot, so each frame made a new image and the old ones piled up (80-95 bad
/// overlaps walked per clear, ~2,000 live images, the pool full at 512 MiB, VMA growing: device
/// memory +0.8 GB while the chicken passes). Not for shapes the CPU read or that were sampled in a
/// later frame (the relocation guard's GuardCpuRead / GuardSampled), images bound, queued for
/// download, aliased, scaled or of the excluded kinds. Logs "Slot retire (EXP64) last 600 frames".
/// EXP45 (live; 1 = off): the native compute write extent self-check (BufferCache::ExtentTrust,
/// fable_answer_25): a program is marked in full until a dispatch at its dims showed its proof
/// holds, and for good once one wrote past it (Ocarina of Time's glowing triangles, 2026-11-09).
/// EXP43 (live; render target relocation, fable_answer_12 2A; TextureCache::JoinImages): a render
/// target a clear overwrites whole, with no copies, aliases or remapped images among its overlaps
/// and exactly one predecessor among its bad overlaps (an image of the same single-level,
/// single-layer 2D shape it drifted off by less than its size, not written in this frame: Ocarina
/// of Time's bump-allocated per-frame targets while an NPC moves), takes that image's place;
/// bad overlaps of other shapes stay as for a new image. 2 = relocate: the image moves to the new
/// address (stale at the old one, untracked, unregistered, its overlap links dropped, registered
/// again; its VkImage, memory, views and framebuffers stay). 3 = as 2, and with several
/// predecessors the most recently modified one moves (lets a chain made before the switch went
/// on stop growing and age out). 1 = retire early: the predecessor is deleted at once and a new
/// image made, as with 0. 0 = off: candidates are only counted. 4 = verify (fable_answer_16 1c): as
/// 3 under the render-frame clock (as EDEN_NX2_RENDER_CLOCK=1), and each old range is kept as a
/// tombstone for 2 render frames; reads that land on one are counted ("Relocation tombstones
/// (EXP43=4) last 600 frames: N hits", first 8 logged), which must stay 0. Default 3 (LoadExperiments). Never for a shape
/// whose GPU-written contents were sampled or rendered to without a clear in a later frame, or
/// read by the CPU (sticky per shape for the session: "Render target relocation guard (EXP43)"),
/// nor for rescaled, sparse, converted or CPU-read images, ones with a download queued, bound as
/// another target of the clear or in another channel. Logs each shape's first candidate and
/// relocation and "Render target relocation (EXP43=n) last 600 frames".
/// EXP51 (live; RAM, fable_answer_26 C2c; Vulkan PipelineCache::EvictUnusedComputePipelines):
/// compute pipelines unused for 9000 frames are destroyed, least recently used first, while more
/// than 500 are held (checked every 600 frames). Their next dispatch translates them again on the
/// GPU thread, binding the pointers they bound, and builds them from the driver's pipeline cache
/// (no new disk cache record). 1 = only programs dispatched this session, 2 = also boot builds no
/// dispatch used yet. Never one still building or with a rebuild pending; destroyed through
/// retired_compute_pipelines once the GPU and their worker build are done. Off by default, not
/// measured yet: compare the 10 s memory line's compute count and the working set over a long
/// session, and the "Evicted N compute pipelines" log's "dispatched again" count (each is a
/// GPU-thread rebuild, a possible hitch).
/// EXP52 (live; with EXP19 / EDEN_NX2_PIPELINE_CLEAN on; fable52_answer_1 section 1; Vulkan
/// PipelineCache::CurrentGraphicsPipelineClean): 1 = when only the bound programs changed since
/// the last key (every other Refresh input stands), the key's state is kept and only what the
/// programs feed is redone (with dynamic vertex input, the attribute types masked by what the
/// programs read), then Next / the cache map looks the new key up, without
/// FixedPipelineState::Refresh. 2 = as 1, and each such key is also built by the full Refresh and
/// compared byte for byte; Refresh's is kept, the first 16 differences are logged ("Programs-only
/// key (EXP52=2) differs"). The "EDEN_NX2_PIPELINE_CLEAN per second" line counts "programs only"
/// and, at 2, "verified N, M differ". With EXP52 on its "programs" reason stays 0 (a change of
/// programs together with other state counts under that state). Off by default, not measured yet:
/// prove 2 shows 0 differ first (under EXP19=2 without EXP50, blend writes without
/// Dirty::Blending already leave the clean lookup's attachments stale, and verify reports that
/// too), then compare the lookup zone with profile_toggle (EXP52 0 vs 1).
inline constexpr int ExperimentCount = 112;
inline std::atomic<int> g_experiments[ExperimentCount]{};
/// Set once LoadExperiments has run: Experiment() tests it inline instead of calling
/// LoadExperiments (a function-local static guard) on every check (0.5% of the back thread in
/// the 5.7 -> 5.8.4 sampler comparison, 2026-11-11).
inline std::atomic<bool> g_experiments_loaded{false};

inline void LoadExperiments() {
    static const bool loaded = [] {
        // Kept experiments: on unless their variable says otherwise. EXP8 = 15: Ocarina of Time
        // walking through Hyrule Field, fresh launches: 1% low 30.3 -> 35.1 fps, frames over
        // 25 ms 68 -> 38 of ~1650.
        g_experiments[8].store(15, std::memory_order_relaxed);
        // EXP9 = 1: native uniform bank pages compared only when they may have changed (verified
        // with EXP9=2: 0 missed in 120 s walking). Idle 54.5 -> 57.7 fps, walking 56.6 -> 58.9.
        // EXP9 = 3 (2026-11-08): the same without the once-a-frame compare of every page, which
        // never found a change ("frame check N / 0", "missed without the frame check 0"). Castle
        // Town, one session, 4 x 35 s each: uniform buffers 135.6 -> 97.9 ms/s on the split's back,
        // 40.8 -> 42.3 fps; flash watch on the Castle Town route 267 reports vs 273 with EXP9=1.
        g_experiments[9].store(3, std::memory_order_relaxed);
        // EXP10 = 1: macro DMA reads wait only for the dispatches, uploads, copies and releases
        // that may have written them (EDEN_NX2_DMA_SNAPSHOT=10 verify: 0 differed in 2 x 90 s
        // walking). GPU front idle 81 -> 105 ms/s, waits for the back 368 -> 338 ms/s.
        g_experiments[10].store(1, std::memory_order_relaxed);
        // EXP14: unchanged macro register writes dropped before FrontWrite's other checks.
        // Castle Town, profile toggle: front macro 250 -> 236 ms/s, push 77 -> 70 ms/s.
        g_experiments[14].store(1, std::memory_order_relaxed);
        // EXP18 = 4 (exact, zortos293/Lichii#717 wave 1): a buffer touches its LRU node once
        // per frame; a second touch in the same frame was already a no-op in LruCache::Touch
        // (its tick is not older), so only the node's cache miss goes. Bits 1 and 2 stay off.
        g_experiments[18].store(4, std::memory_order_relaxed);
        // EXP21 = 1: a texture whose handle word is a storage buffer's address gets no texture
        // (VideoCommon::HandleIsStoragePointer). Ocarina of Time's Zora's River waterfall
        // flickered every other frame in bursts without it.
        g_experiments[21].store(1, std::memory_order_relaxed);
        // EXP29 = 6: the exact parts of #717's micro bundle (TickWork's lock-free check, native
        // compute inline uploads not recorded); 1 (memory usage reuse) changes what the caches'
        // GC sees and stays off.
        g_experiments[29].store(6, std::memory_order_relaxed);
        // EXP34 = 1: ACQUIRE_PRECHECK=2 no longer answers met over a write to the word still
        // queued for the back (a reset there let a channel run before its producer). A strict
        // tightening: only met answers become the normal path.
        g_experiments[34].store(1, std::memory_order_relaxed);
        // EXP35 = 1: guest writes of never-GPU-written cached pages collected for the back instead
        // of OnCPUWrite under both caches' mutexes (see the EXP35 notes above).
        // EXP35 = 2 (2026-11-08): every guest write collected (guest-core write lock spinning 0.7% ->
        // 0, standing 45.3 -> 46.5 fps, route flash reports 253 = camera noise).
        g_experiments[35].store(2, std::memory_order_relaxed);
        // EXP42 = 1 and EXP43 = 3 (2026-11-08, moving-NPC dips): pooled VkImages for deleted
        // render targets, and drifted per-frame targets relocated in place (the newest of several
        // predecessors). Castle Town live toggles: EXP43 0/3 46.3 -> 49.3 fps, EXP42 0/1 on top
        // 44.7 -> 47.9 (each won all 4 pairs); 21 loads 0 washed out; owner playtest: no visual
        // bugs, rarely under 40 fps.
        g_experiments[42].store(1, std::memory_order_relaxed);
        g_experiments[43].store(3, std::memory_order_relaxed);
        // EXP56 = 1 (2026-11-09): a cleared depth target larger than its colour targets is not
        // uploaded from guest memory first (TextureCache::JoinImages, overwritten_by_clear).
        // Castle Town live toggle 41.3 -> 43.8 fps, clear zone 54.7 -> 36.3 ms/s; screenshots
        // off/on identical.
        g_experiments[56].store(1, std::memory_order_relaxed);
        // EXP58 = 1 (2026-11-10): guarded render targets that a newer target of another shape
        // superseded are deleted once 60 frames old (TextureCache::SweepColdImages). Castle Town
        // 150 s A/B: device memory 7.8-10.5 -> 5.2-5.7 GB, collector evictions 83-1915 a window
        // -> 0, worst frames 140-180 -> 30-48 ms, moving median 42.2 -> 46.9 fps; skeleton fight
        // 5.1-5.5 GB instead of 10.9 with 0 evictions; two owner playtests without glitches.
        g_experiments[58].store(1, std::memory_order_relaxed);
        // EXP61 = 1 (2026-11-10): FlushCommands submits nothing when nothing was recorded since
        // the last submit. Castle Town live toggle: empty submits 4,600 -> 145 a second, 43.0 ->
        // 45.6 fps (3 of 4 pairs, the 4th a tie), GPU thread split drain 432 -> 396 ms/s.
        g_experiments[61].store(1, std::memory_order_relaxed);
        // EXP64 = 16 (2026-11-11, NPC-pass dips): a cleared target retires its slot's old
        // same-format images untouched for 16 frames. Night market live toggle 0/16: peak texture
        // cache memory 5,972 -> 5,482 MiB, binds over 1 ms 30 -> 6 a 2 min (over 10 ms 1 -> 0),
        // 53.4 -> 54.6 fps; 0 retired images wanted again within 16 frames (at 1: 29, at 8: 2).
        g_experiments[64].store(16, std::memory_order_relaxed);
        // EXP68 = 1 (2026-11-11, NPC-pass dips): a depth clear with a scissor of the real width
        // goes to the exact image (one per NPC a frame, not two plus an upload and alias copies;
        // fable52_answer_npcdips). Night market toggle: images made 483 -> 379, texture upload
        // 18.8 -> 14.1 ms/s, about +1 fps; owner playtest: no visual glitches, 50-55 fps except
        // when a dog enters from behind the camera. Bit 4 (no pool) stays off: binds over 1 ms x3.
        g_experiments[68].store(1, std::memory_order_relaxed);
        // EXP70 = 2 (2026-11-11, display stalls): with no free render frame for 2 ms the renderer
        // drops a queued frame instead of stalling emulation. Night market with a simulated 150 ms
        // display stall every 10 s (EXP71): off, every stall froze emulation 65-82 ms; on, one
        // 50 ms hitch in a minute (21 frames dropped). No drops in a minute of normal play.
        g_experiments[70].store(2, std::memory_order_relaxed);
        // EXP69 = 600 (2026-11-11): Preflushable marks reset every 600 frames. Night market
        // toggle: buffer downloads 81 -> 53 MB/s, back fence 45.9 -> 38.1 ms/s, +0.6 fps. Owner
        // playtest with EXP70 (market + Stalchild fight): constant 60 outside the boss wave
        // (40-45 there), no flashes or glitches.
        g_experiments[69].store(600, std::memory_order_relaxed);
        // EXP69 = 4 and EXP81 = 4 (2026-11-12, fable52_answer_fences2 (a); EXP81 now fed by a
        // lock-free read bitmap, so it works with EXP86): pages nobody read for 4 frames stop being
        // downloaded at fences. Owner skeleton fight A/B: downloads 412-719 -> 162-203 MB per 600
        // frames, unread landed pages 77-83% -> 27-30%, fence thread buffer mutex held 383 -> 83
        // ms / 5 s, worst hitch 788 -> 409 ms, frames over 40 ms 4.5 -> 4.1 per 1000; owner: not
        // less smooth.
        g_experiments[69].store(4, std::memory_order_relaxed);
        // EXP81 16, not 4 (fable52_answer_fight3 5): with 4 the game's read-backs doubled (50 -> 116
        // a second, guest waiting 64 -> 117 ms/s); a longer memory keeps pages it reads every few frames.
        g_experiments[81].store(16, std::memory_order_relaxed);
        // EXP72 = 1 (2026-11-11): a preflushed read whose bytes are all in a submitted fence download
        // waits for it. Night market toggle: reader waits 683 -> 479, wait time 515 -> 299 ms. Owner
        // Stalchild fight playtest: readers waited for the host GPU ~280-350 -> 40-70 ms/s, 0
        // timeouts, no flicker or glitches (the boss wave still dips: front waits, not reads).
        g_experiments[72].store(1, std::memory_order_relaxed);
        // EXP73 = 1 (2026-11-12): no report flush round trip after a drained macro read with no
        // report listed over the bytes. Owner fight toggle with EXP30: macro dma flush 34.5 -> 20.4
        // ms/s, front waits 442 -> 416 ms/s, no flashes; it only drops empty work.
        g_experiments[73].store(1, std::memory_order_relaxed);
        // EXP24 = 4 (2026-11-12): OoT's present poll sleeps at most 250 us and ends when the
        // event's value word reads 1. The guest timeline put that poll on the night market's
        // critical path (1.9 ms a frame, 5.2 at p90); 3 alone gave 57.8 -> 58.4 fps. Only the
        // verified site. With EXP75 = 600 (precise guest timers, +0.9 fps at the night market):
        // together they made the GPU side the frame's critical path, so its cuts now show
        // (OOT_MACROS +2.1 with them on). Owner playtest of all three: dogs no longer hitch,
        // the chicken much better, no glitches; owner asked for them as defaults.
        g_experiments[24].store(4, std::memory_order_relaxed);
        g_experiments[75].store(600, std::memory_order_relaxed);
        // EXP77 = 1 and EXP78 = 1 (2026-11-12): kill frames re-made GPU-drawn targets the cold
        // sweep had just taken and filled them from memory nobody wrote. Owner Stalchild fight with
        // both on vs off: frames over 22 ms 2.54 -> 2.06 a second, over 28 ms 0.39 -> 0.30, up to
        // 500 uploads (1.5 GB) skipped per 600 frames, device memory steady ~5.5 GB, no glitches
        // seen; owner asked for them as defaults.
        g_experiments[77].store(1, std::memory_order_relaxed);
        g_experiments[78].store(1, std::memory_order_relaxed);
        // EXP82 = 1800 and EXP66 = 1024 (2026-11-12, overnight): proven image pool entries stay
        // 1,800 frames and the pool may hold 1 GiB. Daytime market toggle (orbit, uncapped, zones
        // on): clear 59.6 -> 43.0 ms/s, back busy 7.88 -> 7.45 us per draw, 0.1% low 30.8 ->
        // 35.4; pool 1,000-1,500 taken per 600 frames, held up to ~716 MiB. The pool still
        // empties on device memory pressure (TickFrame).
        g_experiments[82].store(1800, std::memory_order_relaxed);
        g_experiments[66].store(1024, std::memory_order_relaxed);
        // EXP86 = 1 (2026-11-12, overnight): guest reads of GPU-written pages answered from the
        // lock-free page mirrors. Verify mode (2) at the daytime market: ~17,000 reads per 5 s
        // checked, 0 mismatches. Toggle (orbit, uncapped, 180 s warm, 6 cycles): guest lock waits
        // 65.0 -> 3.7 ms/s median, 65.0 -> 66.5 fps, 1% low 44.0 -> 44.9, 0 broken mirrors.
        g_experiments[86].store(1, std::memory_order_relaxed);
        // EXP88 = 3 (2026-11-12, overnight): zortos293/Lichii#1441 parts a-c (Fable's plan). Verify
        // runs: part c skipped 100% of ~800,000 flush ranges a second with 0 that would change,
        // parts a+b 0 differ over ~8 million direct writes. Daytime market toggle (orbit,
        // uncapped): back 7.89 -> 7.43 us per draw, flush work 33.6 -> 16.9 ms/s, 59.8 -> 61.8
        // fps in the same session, 0.1% low 29.2 -> 32.8; picture normal.
        g_experiments[88].store(3, std::memory_order_relaxed);
        // EXP85 = 3 (2026-11-12, overnight): inline uploads and WriteMemory skip the buffer lock on
        // never-GPU-written pages. Alone (EXP86 on) it was inside the noise; with EXP88 = 3 at the
        // daytime market (orbit, uncapped): back 8.09 -> 7.69 us per draw, method 175.4 -> 163.2
        // ms/s, 58.7 -> 60.8 fps same session, 1% low 38.8 -> 40.4; picture normal.
        g_experiments[85].store(3, std::memory_order_relaxed);
        // EXP79 = 1 (2026-11-12, overnight; zortos293/Lichii#1373): guest cores wait for work in
        // hardware. Judged earlier in runs that mixed the market's two phases (no gain); inside the
        // heavy phase only (orbit, uncapped, 8 x 25 s): 56.5 -> 58.6 fps, 1% low 39.2 -> 40.4,
        // back 8.04 -> 7.81 us per draw. Timing only, no rendering change.
        g_experiments[79].store(1, std::memory_order_relaxed);
        // EXP90 = 1 (2026-11-12, overnight; Fable after88 2): an image's LRU node is touched once a
        // frame. Exact (the node ignored same-tick touches anyway). Heavy phase toggle: 92% of
        // touches skipped, back 8.30 -> 8.02 us per draw, 56.0 -> 57.1 fps, 1% low 38.1 -> 39.6.
        g_experiments[90].store(1, std::memory_order_relaxed);
        // EXP99 = 1 (2026-11-12; fable52_answer_chicken 1): evicted image pool entries (OoT's
        // per-NPC depth targets, the chicken hitch) are destroyed in the back's idle wait instead
        // of at the frame's end. Heavy phase toggle: other 143.4 -> 114.2 ms/s, back 7.92 -> 7.72
        // us per draw, 0.1% low 15.6 -> 23.0; every eviction went while idle ("here" 0).
        g_experiments[99].store(1, std::memory_order_relaxed);
        // EXP49 = 1 (2026-11-10, owner's call for v5.7): the async read-back. Stalchild fight,
        // flipped every 30 s while the owner fought: 51.9 -> 56.9 fps, the back's read flushes
        // 35 -> 6 ms/s, guest lock waits halved. No beams in that session or two later ones
        // played with it on from boot.
        g_experiments[49].store(1, std::memory_order_relaxed);
        // EXP39 = 3: publish to a back that waits after 8 words, no snapshot refresh of entries
        // that ran (zortos293/Lichii#805 62db242b34; see the EXP39 notes above). Timing only.
        g_experiments[39].store(3, std::memory_order_relaxed);
        for (int n = 0; n < ExperimentCount; ++n) {
            const std::string name = "EDEN_NX2_EXP" + std::to_string(n);
            if (const char* value = std::getenv(name.c_str())) {
                g_experiments[n].store(std::atoi(value), std::memory_order_relaxed);
            }
        }
        g_experiments_loaded.store(true, std::memory_order_release);
        BumpSwitchGeneration();
        return true;
    }();
    static_cast<void>(loaded);
}

/// The value of experiment switch `n` (EDEN_NX2_EXP<n>), 0 unless set.
[[nodiscard]] inline int Experiment(int n) {
    if (!g_experiments_loaded.load(std::memory_order_acquire)) [[unlikely]] {
        LoadExperiments();
    }
    return g_experiments[n].load(std::memory_order_relaxed);
}

/// The live switch called `name`, or null.
[[nodiscard]] inline std::atomic<int>* FindSwitch(std::string_view name) {
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
    return nullptr;
}

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
