# Fritchii switches

What the live switches and the numbered experiments do. The switches themselves are in
src/video_core/nx2_debug_control.h, the experiments' defaults (with the measurements behind
them) in src/common/fritchii_switch_defaults.cpp. This file is not compiled: editing it
rebuilds nothing.

## Named switches

### EDEN_NX2_DMA_WAIT_WRITERS

EXPERIMENT (not for nx2): EDEN_NX2_DMA_WAIT_WRITERS, read at startup and settable live by the
debug server's `switch` method, so one session can A/B it.

### EDEN_NX2_SEM_FAST

EXPERIMENT, off by default: Ocarina of Time flickers white on the bottom half of Link's house
ladder with it (confirmed toggling it live, 2026-11-06), for +1.4 fps. EDEN_NX2_SEM_FAST=1,
semaphore acquires without draining the back
when the value is already met and no queued write overlaps it (puller.cpp).

### EDEN_NX2_LIST_VERIFY

TEST: EDEN_NX2_LIST_VERIFY=1 (live), each command list the split's front reads on GPU-written
pages is read again as before EXP40 and compared (video_core/nx2_list_verify.h; "list
verify" every 10 s). Changes timing: the front waits for every writer again for each list.

### EDEN_NX2_LOOP_SKIP

EXPERIMENT (not for nx2): EDEN_NX2_LOOP_SKIP=1, the split's front does not send the instance
loop writes the back would ignore (draw.end in an instance loop, unchanged vertex and index
buffer firsts and counts) (maxwell_3d.cpp).

### EDEN_NX2_DMA_SNAPSHOT

EDEN_NX2_DMA_SNAPSHOT (from a ported dma-snapshot fix): macro DMA reads of pages a drained
read flushed whole skip the drain while nothing sent since may write them (Maxwell3D::
FrontDmaRead). 9 = on (default: Lake Hylia 46.5 -> 52.3 fps; verify 0 differences in 235k
reads), 10 = on and verify every skipped read against the drained path, 0 = off.

### EDEN_NX2_ACCURATE_PARTS

EDEN_NX2_ACCURATE_PARTS (bit mask, 0 = none, read once at boot): parts of GPU accuracy High in
Fast mode, the split GPU thread kept, to find which one stops a flicker that Accurate mode
stops. Not live: switching the counter query mode mid-game crashed (queries opened in one
mode were closed in the other).
1 = counter reports written once their host query completes (synced), 2 = precise occlusion
queries, 4 = CPU reads of GPU-written images download them, 8 = CPU writes flush GPU data
before invalidating it, 16 = fences wait for the GPU work before them.

### EDEN_NX2_HOST_PACE

EDEN_NX2_HOST_PACE: the host frame pacer (vk_host_pacer.h). 1 = on (default), 0 = only
measure how long each frame was shown.

### EDEN_NX2_ARB_SPIN

EDEN_NX2_ARB_SPIN (ported; off by default here, the original uses 8): microseconds a
guest thread about to sleep in WaitForAddress first watches the word (KAddressArbiter), 0 =
off. Live: read on every wait, so profile_toggle.py can A/B it in one session.

### EDEN_NX2_IDLE_PAUSE

EDEN_NX2_IDLE_PAUSE=1 (ported; off by default here, on in the original): the guest
core's idle poll (PhysicalCore::Idle, EDEN_NX2_IDLE_SPIN_US) runs PAUSE and reads the clock
every 16 pauses, instead of a SwitchToThread and a clock read per iteration. Live: read on
every Idle.

### EDEN_NX2_GPU_WAKE

EDEN_NX2_GPU_WAKE=1 (live, off by default; ported): the GPU
thread (the split's front) waits for suspended channels in a wait that ends on a wake, instead
of polling them (ResumePending, then yield or sleep per EXP6, ~120-150 ms a second of front
time in Ocarina of Time). A new command (PushCommand) or a GPU write that latches a waiter
(Nx2SemaphoreWaiters::OnWrite) wakes it at once (Tegra::Nx2PendingWake); a guest CPU store to
the word a channel waits on ends the wait within ~2 us (Nx2SemaphoreWaiters::WordStored). For
500 us after the last progress it waits in hardware (MWAITX/UMWAIT, Common::WaitForFlag) with
a full poll every 20 us, then blocks in the OS 500 us at a time (Common::Event::SleepFor).
0 = the polling loop as before (EXP6 applies), and the wake sources return at once. Read on
every wait, so profile_toggle.py can A/B it in one session.

### EDEN_NX2_BACK_FENCE_STATS

EDEN_NX2_BACK_FENCE_STATS=1 (live, diagnostic, off by default; fable52_answer_fences2 4): the
back's SignalFence timed in five parts (CommitCacheFlushes with DownloadStagingBuffer, guard,
the fence's operation, FlushCommands, InvalidateGPUCache), summed per 5 s ("Back fence split
(5 s"), and every fence over 1 ms with its parts ("Back fence over 1 ms", 30 lines per 5 s at
most). Also counts the fences that committed no download (FenceReleaseStats::BackFence).

### EDEN_NX2_RUNAHEAD

EDEN_NX2_RUNAHEAD (live, Ocarina of Time only, default 0; video_core/nx2_runahead.h): 1 = the
split's front answers acquires a release it sent meets without draining and writes the word
early, with the guest's CPU writes snapshotted per boundary for the back; 2 = verify (old path,
chunks compared with guest memory at their stamp); 3 = early answers without writing the word.

### EDEN_NX2_RUNAHEAD_MS

EDEN_NX2_RUNAHEAD_MS (live, default 16): how far the back may lag the oldest early answer.

### EDEN_NX2_BUFFER_CHURN_STATS

EDEN_NX2_BUFFER_CHURN_STATS=1 (live, diagnostic, off by default; Fable's ideal pipeline design
8.3): buffer cache creates and deletes per second by cause, and staging buffer creates by
caller (video_core/buffer_churn_stats.h). "Buffer churn (5 s" every 5 s while on.

### EDEN_NX2_CB_DIRECT

EDEN_NX2_CB_DIRECT (live, default 1; ported): a run of constant
buffer words within one device page outside Switch 2 sparse memory is written through one
translation and a memcpy instead of WriteWordsCached's page walk (Maxwell3D::
CallConstBufferWords). Exact: the same bytes and the same accumulated invalidations; 0 = the
page walk.

### EDEN_NX2_RT_DEPTH_NOUPLOAD

EDEN_NX2_RT_DEPTH_NOUPLOAD (live, bits, default 0; ported): new
depth/stencil images skip the upload from guest memory when the exact rules (EXP56's clear,
EXP8 bit 2's copies) did not already (TextureCache::JoinImages). 1 = render targets with
nothing to join, 2 = images whose every join copy is GPU-written, 3 = both.

### EDEN_NX2_FOLD_INSTANCE

EDEN_NX2_FOLD_INSTANCE (live, default 0; ported): an instance loop
of Ocarina of Time's instanced draw macros (0xe52, 0xea0 by code hash) does not draw its
first instance alone; the loop's deferred draw covers it (Maxwell3D::FrontRunMacro,
DrawManager::FoldFirstInstance). 1 = on.

### EDEN_NX2_OOT_MACROS

EDEN_NX2_OOT_MACROS (live, default 1 since 2026-11-12; ported, its macro_hle_oot.h): Ocarina
of Time's hot 3D macros, found by code hash, run through native C++ handlers on the split
front instead of the MME JIT (Maxwell3D::FrontRunMacro). 1 = handlers, 2 = verify (the JIT's
writes count; the handler runs on copies for the first 4096 calls of each macro and then one
call in 64, and differences are logged: "OoT macro verify"), 3 = verify every call.

### EDEN_NX2_DRAW_PACKET

EDEN_NX2_DRAW_PACKET (live, default 0; Fable's back-thread plan, Build C step 1): the
register writes of an OoT draw macro call (0xe04, 0xe52, 0xea0, identified by code hash) that
the front would send as single Direct entries go as one ring entry the back applies in one
call, in the same order through the same CallMethod (Maxwell3D::ApplyDrawPacket). Constant
buffer words and anything else still go one by one, splitting the packet. 1 = on, 2 = on and
the back checks each packet's registers against the values the front wrote ("Draw packet
verify").

### EDEN_NX2_BANK_SLOT_STATS

EDEN_NX2_BANK_SLOT_STATS (live, default 0; recomp research idea 1, step
1): with the tracked native bank check (EXP9 != 0), counts per constant buffer index (c0, c1,
c2, ... as the shader numbers them; NVN2: c0 driver storage table, c1 program constants, c2
texture handles, c3+ the game's uniform buffers) the bank binds, the pages they cover and
the pages compared and found changed for each reason, logged every 5 s next to the "Native
bank check" line ("Native bank slots"). Counting only (BufferCache::
CheckNativeBankPagesTracked). 1 = on.

### EDEN_NX2_BANK_PAGE_TABLE

EDEN_NX2_BANK_PAGE_TABLE (live, default 0; the recomp's page table,
research idea 1 step 3): the native bank check's page snapshots live in a page-indexed table
(2 MiB regions of 512 page slots, the last region kept) instead of a node map hashed per page
(~1.9% self of the back in the 8 Nov sampler, 720k lookups/s at Castle Town). Exact: the
same snapshots, the same compares; a change of the switch empties both stores, so the next
bind of each page takes a new snapshot and marks it changed, as after the 16384-page reset.
1 = on.

### EDEN_NX2_BANK_SLOTS

EDEN_NX2_BANK_SLOTS (live, default 0; recomp research idea 1): a bit
mask of constant buffer indices (bit n = cn) whose 64 KiB native bank binds rely on the write
flags alone, for the banks NVN2's driver fills only through the command stream (c0 storage
table, c2 texture handles) or never rewrites (c1 program constants). For those binds, with
EXP9=3 (the default): (a) the "hot" rule (pages that changed lately are compared on every
bind even with no flag set) is dropped, and (b) with EDEN_NX2_BANK_PAGE_TABLE=1 a bind whose
whole range has no collected CPU write, no buffer cache write (our own constant buffer words
among them, at FlushCaching), no recheck pending and no CPU-modified page, with every page
already snapshotted, skips the page loop at once (BufferCache::CheckNativeBankPagesTracked).
The game's own banks (c3+) keep the full check: never set their bits (the white flashes,
d79572d28a). Pick the mask from EDEN_NX2_BANK_SLOT_STATS: indices whose "hot" changes stay 0.
0 = off. Suggested after that check: 0x7 (c0, c1, c2).

### EDEN_NX2_BANK_SKIP_VERIFY

EDEN_NX2_BANK_SKIP_VERIFY (live, default 0): with EDEN_NX2_BANK_SLOTS, every bank the skip
passes over and every hot page the slot rule leaves alone is still compared; changes are taken
as before (the page is marked) and counted as "missed", which must stay 0, and pages of a
skipped bank that the page loop finds a reason for are counted as "disagreed" (must stay 0).
Logged every 5 s ("Native bank slot skip"). Costs what the skip saves. 1 = on.

### EDEN_NX2_TEX_BANK_GEN

EDEN_NX2_TEX_BANK_GEN (live, default 0; the recomp's per-stage
texture bank generations, research idea 2): a graphics pipeline whose bindings are all sampled
textures with handles read straight from constant buffers (no handle programs) keeps the view
and sampler ids its last draw resolved, with the generations of the pages that draw read
handle words and TIC/TSC entries from (TextureCache::TexBankPage: watched pages trap CPU
writes; our constant buffer words, buffer cache writes and InvalidateRegion mark them too).
A draw with the same handle constant buffer bindings, the same TextureMemoState (pools,
limits, channel, mappings, no view removed), the same sampler binding mode and every page at
its generation takes those ids: no handle reads, no table reads, no GetSamplerId; the views
are still prepared (PrepareImageView). A miss costs the compares. Vulkan GraphicsPipeline::
ConfigureImpl; not with EXP32. 1 = on, 2 = on and every 16th hit resolves the full way and
compares ("Texture bank memo mismatch", must stay 0). Logged every 10 s ("Texture bank memo").

### EDEN_NX2_STORAGE_SYNC_SCAN

EDEN_NX2_STORAGE_SYNC_SCAN (live, default 0): storage buffer synchronizes (graphics, and
compute while EXP59 is off) walk the memory tracker only from the first to the last word of
each 4 MiB manager that holds a CPU-modified or unprotected page, found by reading the words
(MemoryTrackerBase::ForEachScannedUploadRange). The words left out hold neither, so the full
walk would upload nothing there, re-protect nothing and store back what they hold: the same
result without the stores. Not the CleanWords walk (that one trusts clean_words). The sample
capture of 10 Nov had 72% of the graphics storage binds in that walk over clean words.
1 = on, 2 = the full walk as before, after counting the CPU and unprotected pages the scan
would have left out ("Nx2 storage sync scan verify" every 10 s, must stay 0).

### EDEN_NX2_STORAGE_SYNC_MEMO

EDEN_NX2_STORAGE_SYNC_MEMO (live, default 0): a graphics storage binding synchronized while
the dirty generations of the 4 MiB tracker managers it covers stayed put (WordManager::
dirty_gen: bumped whenever a page gains a CPU-modified or unprotected bit) is clean after it,
and stays clean while that sum holds: the next bind of the same range skips SynchronizeBuffer,
whose walk would find nothing (BufferCache::SynchronizeStorageMemo; 512 entries keyed by
range). Not with EXP48's hot pages (they stay dirty) nor with the CleanWords walk unless
EDEN_NX2_STORAGE_SYNC_SCAN=1 picks the walk. A miss costs a sum over the managers twice.
1 = on, 2 = every hit synchronizes anyway after checking that no word of the range holds a
CPU-modified or unprotected page ("Nx2 storage sync memo verify" every 10 s, must stay 0).

### EDEN_NX2_FEED_MEMORY_OVERLAY

EDEN_NX2_FEED_MEMORY_OVERLAY (live, default 1; fable_answer_pair_2 section 2): the EXP13=3/4
feed takes records whose QMD the copy read from an overlay built from guest memory with a
small upload patched over it (QmdPrediction::Source::OverlayFromMemory; OoT's per-dispatch
12-byte release patch at QMD word 52). Only overlays built while the feed and this switch were
on and no unbounded writer was queued over the QMD (GpuSplit::QueuedGuestWriters) count; the
back verifies every release either way (QmdPrediction::Verify). 0 = refused as before.

### EDEN_NX2_QMD_3D_PATCH

EDEN_NX2_QMD_3D_PATCH (live, default 0; fable_answer_pair_2 3.1): a 3D inline upload over a
QMD the EXP13=2-4 copy holds retires its own 3D method-write mark (GpuSplit::
RangedMethodWrite, which runs right after anyway) before the copy patches the holders
(Nx2PatchQmdCopy). Without it the holder's write-stamp check counts the upload's own method
write as a later writer, so no holder ever takes a 3D patch and the QMD read waits. 1 = on.

### EDEN_NX2_LIGHT_TABLE_ARRAYS

EDEN_NX2_LIGHT_TABLE_ARRAYS=1 (boot, default 0; ported):
a texture handle the shader reads from a storage buffer table at an index each invocation
reads for itself (OoT's clustered lights: each light's shadow map handle, read per pixel from
the light table) becomes a descriptor array of the table's first 64 entries, indexed
NonUniform, instead of binding the null image, whose depth-compare sample is 0:
fully shadowed, the dark grid that follows Link over water. With it on also: depth-compare
samples of uncomparable images (garbage table entries) bind the null image, view lookups skip
repeated table elements, a pipeline's texture arrays fit the sampler limits too, and dual
vertex programs join their handle programs. Read once, on first use (PipelineCache's
constructor, after nx2_switches.txt is applied): it changes what shaders translate to, so it
is not live. Off, translation and binding are exactly what they were before the port.

### FRITCHII_MEMSTAGE_SPLIT_ALLOC

FRITCHII_MEMSTAGE_SPLIT_ALLOC (boot, default on, =0 off; Fable's ideal pipeline design 2.4,
build step 3): the buffer cache runtime gets a VMA allocator of its own (Device::
GetBufferAllocator) and a staging pool of its own on it, so the future memory stage can own
them on its thread. On it: the buffer cache's buffers (EXP104's pool included), quad and
uint8 index buffers, vertex repack and multi-range buffers, the null buffer, the uniform
stream and buffer uploads and downloads. The two list draw buffers (list draw arguments, kept
list words) belong to the memory stage's player (EXP123) and are on the main allocator. Images, the texture cache, query cache, descriptors,
mesh emulation, upscalers and diagnostics stay on the main allocator and staging pool. The
back's pool keeps its full stream buffer (128 MiB, texture uploads up to 8 MiB stay in the
ring); the buffer cache's pool takes half that (64 MiB), so a discrete card reserves 64 MiB
more. On integrated GPUs the two split the one stream buffer instead. EXP104's byte caps count
both pools' recyclers together, and the new allocator takes no memory until used. Read once
when the device is made.

### FRITCHII_MEMSTAGE_SEAM_STATS

FRITCHII_MEMSTAGE_SEAM_STATS=1 (boot, diagnostic): the memory seam's stats line (EXP123) every
10 s also with EXP123 = 0, which shows what the seam carries at today's timing.

### FRITCHII_MEMSTAGE_READ_LOG

FRITCHII_MEMSTAGE_READ_LOG=N (boot, diagnostic, default 0 = off; the memory stage's sub-step 2b,
fable_memstage_step2_spec.md 5.3): with EXP123 = 2 or 3, every Nth draw, dispatch and operation
packet logs what its fill read from game memory (uniforms copied to the stream, buffer uploads
with their staged bytes, cached ranges bound without an upload, storage and table addresses and
size words, compute pointers, the inline page write), and at the old read position (a draw's or
dispatch's first play; an operation's end) reads it all again: bytes and values are compared,
bound ranges are asked whether the CPU wrote them since. Differences are counted by kind; the
first 16 are logged as "Memory seam (5.3)" with the kind, address, size and the programs. Must
read 0. Reads past 1 MiB a packet are not logged (counted as "not logged"). The counts are in the
"memory seam read log" line of EXP123's stats every 10 s. N = 8 is the spec's value; N = 1 checks
every packet (slower). Not FRITCHII_MEMSTAGE_VERIFY, which is 2c's by-value checks (bit flags).

### FRITCHII_MEMSTAGE_VERIFY

FRITCHII_MEMSTAGE_VERIFY (boot, bit flags, default 0; spec 5.4): with EXP123 = 3 the back also
makes its own read and compares it with the packet's value (not the early fill's read log: that
is FRITCHII_MEMSTAGE_READ_LOG). 1 = handle values, 2 = the QMD, 4 = null pointer slots (7 = all).
Checks and mismatches are counted by kind in the "memory seam by value" line; the first 16 mismatches are logged as "Memory seam by value (5.4)" with the
address (a handle word's address, or for a handle program bit 63 set, the program in bits 32-62
and the element in bits 0-31), the kind and the shader (the stage's hash; the program address
for the QMD and null slots). Must read 0: same thread, same instant. "count" (the back took a
different number of handle values than the fill made) is counted always and must be 0 too. The
own reads double the handle reads EDEN_NX2_MEMSTAGE_STATS counts. Later sub-steps may add bits.

### FRITCHII_MEMSTAGE_TICKS

FRITCHII_MEMSTAGE_TICKS (boot, default 0; the memory stage's build step 4,
fable_memstage_step2_spec.md 7.2): what the buffer cache side's stamps are.
- 0: scheduler ticks, as before (SeamClock forwards to the scheduler).
- 1: packet ticks. The memory stage numbers its packets (each draw, dispatch and operation
  packet takes the next number) and every stamp the buffer cache side takes is the number
  current then: buffer usage (and so CanReorderUpload's "used since"), staging buffers and
  the stream ring's lazy region stamps, EXP104's recycler, write ticks, extent probes, list
  page copies, host read flushes. The GPU thread publishes at the end of every scheduler
  tick "every number up to N was recorded in tick K"; a stamp is free when its tick is
  complete. Numbers whose entries still wait for their play stay open past a tick's end (a
  held packet's until its PlayRest), which is what makes EXP123 = 2 and 3 safe: they need
  1 or 2 here and run as EXP123 = 1 without it. Deleted buffers wait their 8 frames and also
  their packet. The stream uniform reuse compares a close count instead of the tick. On one
  thread every answer is the scheduler's. With FRITCHII_MEMSTAGE_SPLIT_ALLOC=0 the texture
  cache's staging (the shared pool) gets packet stamps too, right on one thread.
- 2: as 1, and every answer is checked against the scheduler ticks' answer. The line
  "Memory seam ticks (FRITCHII_MEMSTAGE_TICKS=n)" every 10 s gives packets/s, tick closes/s,
  the published ticks waiting, carried closes and deleted buffers held past their frames (with
  1 too); with 2, "memory seam ticks check" gives early (the packet answer frees or waits
  less: unsafe) and late (holds longer) per check (is free, stream known, wait, host read),
  carried skipped, split (a number handed out in two ticks without a carry), unchecked
  (older than the 2^18-number ring) and epoch off. Early, late, split, epoch off and held
  must all read 0. Disagreements log the first 16 lines with the packet and both ticks.

### FRITCHII_JIT_DISPATCH_BITS

FRITCHII_JIT_DISPATCH_BITS (boot, default 20; fable52_answer_guest 2.1): the A64 JIT's fast
dispatch table holds 2^bits entries of 16 bytes, 12..20 (default 20 = 16 MiB per core, the old
fixed size; 16 = 1 MiB, 12 = 64 KiB). A profile of the game's CPU threads put the read of the
table entry (after `cmp rbx, [r12]` in GenTerminalHandlers) as the hottest instruction in
translated code, 3.2% of JIT time: hot targets spread over 4,096 pages and miss the TLB. The
table only caches block lookups (a miss takes the full lookup and refills the slot), so the
size changes nothing the guest sees, only the hit rate and the cache/TLB footprint. Read once
per process when the first A64 JIT is made (a64_emit_x64.cpp), logged as "JIT fast dispatch
table: 2^N entries"; values outside 12..20 log a warning and keep 20. A32 is unchanged (2^16).

### FRITCHII_BACK_READ_AUDIT

FRITCHII_BACK_READ_AUDIT=1 (boot, default off; Fable's ideal pipeline design 2.5, memory stage
step 2 spec 7.3): counts every read of game memory the GPU back thread makes, by call site, so
the memory stage knows every crossing. The rule for step 7 is that memory the stream writes is
read by the stage only: the list must come down to texture contents, the texture and sampler
tables and shader code; any other site is a crossing to move into the packet. Needs the GPU
split on (only the back counts). Read once when the first GPU address space is made, after
nx2_switches.txt is applied.

Hooked, in the read functions themselves (video_core/back_read_audit.h/.cpp): Tegra::
MemoryManager's Read<T>, GetPointer, ReadHandleWord, GetPointerForDescriptor, ReadBlock,
ReadBlockUnsafe, ReadBlockGuarded, ReadWordsUnsafe, GetSpan and CopyBlock, and the device
memory's (MaxwellDeviceMemoryManager) Read<T>, GetPointer<T>, GetSpan, ReadBlock and
ReadBlockUnsafe. GpuGuestMemory and DeviceGuestMemory read through those, as does a
GpuToCpuAddress followed by a device read. A MemoryManager read counts once, at its caller (its
own device reads are hidden); the managers' writes through the same getters are hidden. Pointer
getters called from outside may still be writes: their kinds end in Mut. Bytes are counted
where the size is known (not for the pointer getters).

Off: one predictable branch on a static bool per read call. On: a thread-local check, then a
lock-free table of 512 call sites (open addressing on the return address). Every 10 s the back
logs the top 40 by reads, as offsets from the exe's base (logged once as "Back read audit: on,
image base ..."):

    Back read audit: 10.0 s, 87 sites, 123456 reads/s, 4567890 bytes/s, 0 reads dropped (table full)
    Back read audit: #01 +0x1a2b3c4 MmReadHandleWord 51234.5 reads/s 204938 bytes/s

`python -I E:\OotRemakeEmulation\work\back_read_audit.py [log] [--exe EXE] [--windows N]
[--chain]` reads those lines (default: the live log), averages the last session's windows and
symbolizes the sites with llvm-symbolizer against the exe that wrote the log (its .pdb next to
it), marking each site CROSSING, allowed (texture cache, descriptor tables, shader code; a guess
from the file name) or helper (a GuestMemory helper that was not inlined: the caller is one
level out). A read function the compiler inlined into its caller (ThinLTO can, across files)
reports the caller's caller: read such a row as "a read somewhere under this call".

## Numbered experiments (EDEN_NX2_EXPn)

Numbered experiment switches EDEN_NX2_EXP0 .. EDEN_NX2_EXP79, live through the debug server:
slots for batches of fixes built together and A/B-tested in one session
(E:\OotRemakeEmulation\work\EXPERIMENTS.md lists which fix uses which slot). 0 = off.

### EXP8

EXP8 (texture upload hitches), bits: 1 = keep idle upload staging buffers up to 32 MiB
(StagingBufferPool::ReleaseLevel), 2 = no guest upload for a new image that GPU copies
overwrite whole (TextureCache::JoinImages), 4 = large 2D levels untiled on the transcode
workers too (UnswizzleImage), 8 = guest memory of uploaded textures not hinted cold
(TextureCache::TickFrame).

### EXP10

EXP10 = 1: macro DMA read snapshots (EDEN_NX2_DMA_SNAPSHOT) skip the drain for compute and
copies sent since whose write ranges miss the pages read (Maxwell3D::FrontDmaRead): copies,
inline uploads and releases as the front notes them, native dispatches as the back records
them once it ran them (the read waits for that much); 2 = the same, but a dispatch still
queued drains as before. +4 (5, 6): compute report semaphores noted as before, 16 bytes or
anything when unmapped (GpuSplit::NoteReleaseWriter). +8 (9, 10, 13, 14; live): the read tests
the bytes it reads, not the whole page: the page scans list the writes since the snapshot that
overlap the page (8 a page, then the two closest merge; GpuSplit::ListedWrites) and the read
drains only when its own bytes overlap one. Verify with EDEN_NX2_DMA_SNAPSHOT=10: "by byte
ranges (N differed)" in the "DMA snapshot EXP10" line must stay 0.

### EXP12

EXP12 (EDEN_NX2_SEM_FAST acquires read current but unmet, puller.cpp ReleaseFencesForReadable):
1 = their fence release waits only for the queued writers the read can't account for, full
drain only if still unmet; 2 = suspend then too, without the full drain; 3 = measure only.

### EXP17

EXP17 = 1: a small streamed uniform buffer binds again any stage or slot's copy of the same
guest range made in the current command buffer, when the bytes still match
(BindSmallUniformReusing; the original's EDEN_NX2_UBO_RANGE_REUSE).

### EXP18

EXP18 (buffer binding memos), bits: 1 = a constant buffer bind of the slot's current
GPU range keeps its translation and resolved buffer (UBO_REBIND), 2 = storage bindings reuse
their last FindBuffer result (STORAGE_MEMO), 4 = buffers touch the LRU once per frame
(default 4: exact, see LoadExperiments). All while the mappings and the set of buffers are
unchanged (BufferCache::binding_epoch).

### EXP19

EXP19 (graphics pipeline lookup without Refresh, PipelineCache::CurrentGraphicsPipelineClean):
1 = as EDEN_NX2_PIPELINE_CLEAN=1, 2 = also the original's patches (blend registers left to the
dual-source bit and Dirty::Blending, attribute types redone alone under Dirty::VertexInput).

### EXP50

EXP50 (live; with EXP19 / EDEN_NX2_PIPELINE_CLEAN on; after a ported fix):
1 = the blend registers leave the clean lookup's word compare and
Dirty::Blending is not looked at; when they moved since the key was made, the attachments
RefreshBlending would build from them (FixedPipelineState::BlendingMatches) and the
dual-source bit are compared with the key's: equal keeps the pipeline without a Refresh,
different patches only the blend part of the key. Reads the registers, so the ported change's writes
without the flag are still caught. Logs "saved by the blend value check" and why the other
lookups refreshed in "EDEN_NX2_PIPELINE_CLEAN per second". Off by default, not measured yet.

### EXP22

EXP22 (split ring traffic; gpu_split.h DeltaWrite and DrawBatch), bits:
1 = 3D state writes that went as Direct macro writes or (EDEN_NX2_RING_PAIRS) as pairs go in
StateDelta blocks, applied in one loop (FRONT_STATE, without its per-register merge),
2 = repeated instance-loop groups go as one DrawBatch entry that counts the repeats
(DRAW_BATCH), 4 = same-value index_buffer.first writes stay on the front
(SKIP_SAME_DRAWREG, as EXP2), 8 = the back checks each delta and batch against the single
writes and keeps theirs (logs "EXP22 check"). 7 = all three. The original's FRONT_CB_ENCODE is not
ported (it holds constant buffer writes off the ring).

### EXP23

EXP23 = 1: trim unused pipeline libraries every 600 frames past 4096 (ported, stage B;
PipelineCache::TickFrame, GraphicsPipelineLibraryCache::TrimUnused). Off by default, not
measured yet.

### EXP24

EXP24 = 1: Ocarina of Time's present poll (`while (nvnEventGetValue(e) != 1) SleepThread(1 ms)`,
from a recomp's research) sleeps until the GPU's next write of a 1 instead, at most 1 ms
(core/hle/kernel/svc/svc_nx2_oot_present.cpp; the site per build id is in its PresentSites
table, or EDEN_NX2_OOT_PRESENT_SITE=\<ret>[:\<blr>] in hex main offsets). Logs "OoT present
poll (EXP24)" counts every 10 s. EXP24 = 2: locate only, sleeps unchanged: short sleeps
(<= 2 ms) are counted by call chain and the top 8 logged every 10 s ("OoT present poll
locate"). EXP24 = 3: no GPU wake, the poll's 1 ms sleep is EDEN_NX2_OOT_PRESENT_SLEEP_US
instead (default 250 us, at most the time asked for). EXP24 = 4 (the default, LoadExperiments)
(fable52_answer_pipeline section 2): as 3, but returns at once when the event's value word
(event object field EDEN_NX2_OOT_PRESENT_FIELD, default 0) already reads 1, and the sleep ends
as soon as it does (Nx2OotPresentPollCheck from KThread::BeginWait: the game's MainThread
writes it, not the GPU). Logs "OoT present poll (EXP24=4" every 10 s.

### EXP25

EXP25 (Ocarina of Time resource poll backoff, ported; needs the built-in patch
placed at launch by EDEN_NX2_OOT_POLL_BACKOFF=1, core/hle/kernel/svc/svc_nx2_oot_poll.cpp):
the resource request thread sleeps before rescanning requests that are not ready.
0 = no sleep (as without the patch, plus one SVC per rescan), 1 = 500 us (the original default),
above 1 = that many us, at most 16000. Off by default, not measured yet.

### EXP26

EXP26 (native uniform bank check trims, ported; BufferCache::
BindHostGraphicsUniformBuffer and CheckNativeBankPagesTracked), bits: 1 = a bind that streams
the buffer (at most 4 KiB, not GPU-modified) skips the check, the stream copies guest memory
itself (BUFCACHE_BANK), 2 = pages compared with an inline SSE2 loop instead of memcmp
(PAGE_CMP), 4 = changed pages copied to the host buffer from their snapshots instead of being
marked CPU-modified (two VirtualProtect calls each), for pages whole in the bound buffer with
no GPU-written data (UploadNativeBank's upload, without its once-a-frame bank shadows: the
check stays per bind, d79572d28a). Off by default; measure 1 and 2 first, then 4. Verify
with EXP9=2 ("missed" stays 0) and the flash watch.

### EXP35

EXP35 (guest writes of never-GPU-written cached pages, Core::Memory::HandleRasterizerWrite):
1 = collected for the split's back (applied at its next FlushCaching) instead of OnCPUWrite
under both caches' mutexes. On by default (2026-11-08): Castle Town guest-core lock spinning
11.6% -> 7.6% of busy (writes 6.1% -> 0.7%), standing 43.0 -> 45.3 fps, route 45.0 fps with
175 flash reports (camera-motion noise ~270 either way).

### EXP36

EXP36 (live): 1 = guest reads of cached memory always take GetFlushArea's locked path (turns
off its lock-free answer for image-free, never-GPU-written pages). For the washed-out hunt.

### EXP37

EXP37 (live): 1 = nx2:gsv answers the driver's armed syncpoint waits right after each guest
syncpoint increment (SyncpointManager guest increment hook), not only at its 500 us poll.

### EXP35 = 2 and EXP60

EXP35=2 (live): every guest write of cached memory is collected for the back, not only writes to
never-GPU-written pages. EXP60 (live): 1 = a guest read whose cache mutexes stay taken after a
short spin is answered with a flush request served by the back (GetFlushArea).

### EXP49

EXP49 (live; async read-back): 1 = a guest read
of GPU-written buffer memory (GPU::OnCPURead) has the GPU thread record and submit the
downloads without waiting for the host GPU (RasterizerVulkan::FlushRegionForRead). The reading
thread waits for them and writes them to guest memory (BufferCache::FinishReadDownloads); a
read whose bytes are already on their way only waits. Until they land the bytes stay
GPU-modified; guest CPU writes keep their bytes (ClearReplacedDownload, pending writes at
landing); writes from the GPU's side (MemoryManager, query results, DMA clears and copies) land
them first. 0 = the synchronous flush (Scheduler::Finish on the GPU thread under the buffer
cache's lock). Both log "Read-back (EXP49) per second" every 10 s (reads, GPU thread time in
read flushes, reader waits). Not ported: the original's early downloads and EDEN_NX2_READ_CHECK.

### EXP27

EXP27 (keep the split's back fed, fewer submits), bits: 1 = the front
publishes before it yields or sleeps polling suspended channels (PUBLISH_WAITS,
GpuSplit::PublishBeforeWait), 2 = retired (was PUBLISH_IDLE, each entry published at once
while the back found the ring empty; EXP39 & 1 replaces it), 4 = the back prefetches the ring 128 words
ahead (RING_PREFETCH), 8 = command list copies after a dispatch are submitted when the front
waits for them, when the back goes idle or with the next submission, instead of at once
(DISPATCH_BATCH, RasterizerVulkan::SubmitNx2ListCopies). Off by default; 1, 2 and 4 change
only when work becomes visible, 8 changes submission timing (the boulder-flash family: flash watch).

### EXP28

EXP28 (fewer fences; FenceManager::SignalFence), bits: 1 = a fence whose
only work is syncing payload releases into host buffers is skipped like an empty one (the
guest word was written at once; the copy goes with the next fence made or wait for idle,
values kept per release since the query slot may be reused: GuestStreamer::
SyncDeferredWrites) (FENCE_TRIM), 2 = a made fence with nothing to wait for on the host GPU
joins the newest queued fence, or with none queued and the fencing thread idle runs its work
on the split's back at once (FENCE_FOLD, FoldStubbedFence). Off by default; timing family
(boulder flashes): flash watch after each. "Fence decisions/s" (GPU profile) counts
both.

### EXP29

EXP29 (a ported micro bundle), bits: 1 = Device::GetDeviceMemoryUsage reuses an
answer younger than 50 ms (MEMSTAT_MS; the buffer and texture caches ask twice per frame on
the back; zone "memory query" times the driver query), 2 = GPU::TickWork checks an atomic
before taking the sync request mutex (OPT_TICKWORK), 4 = native compute engines no longer
record inline upload execs for the old launch method they never run (DROP_NATIVE_UPLOADS).
Default 6: 2 and 4 are exact (2 only allows interleavings the mutex already allowed, 4 drops
a list nothing in native mode reads); 1 is off, since the caches' GC then sees usage up to
50 ms old. EDEN_NX2_EXP29=0 is the behaviour before.

### EXP30

EXP30 = 1 (no env var of its own): a macro DMA read of report memory
with pending query results (the ported white flash fix, RasterizerVulkan::FlushQueryResults) waits
only until the counter reports in its range are written (Tegra::Nx2WaitAsyncReportsSettled,
slots from 12 bytes before the read), not for an operation queued behind every fence, which
also waited for all GPU work sent after those reports (8-15% of the back while walking in
Ocarina, ~11 waits/s of ~3.7 ms, and the front's FrontDmaRead drain with it). Vulkan only
(OpenGL keeps its FlushRegion). 0 = the operation wait as before. Off by default, not measured
yet: compare the back's "query DMA wait" profile zone (ms and waits per second) and the white
flash watch.

### EXP31

EXP31 (render target memo, RT_FAST, ported; TextureCache::
UpdateRenderTargets), bits: 1 = a draw whose render targets are clean (no Dirty::RenderTargets)
skips preparing them again while the same views were prepared for a draw in the same frame,
no image was marked modified, created, deleted, rescaled or CPU written since (modification
tick and bindings_epoch unchanged), and every target image is GPU-modified with no CPU
modification, alias or sparse state, 2 = with 1: such draws run the loop anyway and check it
changed nothing but the targets' own modification ticks (flags, bindings_epoch; logs "RT_FAST
(EXP31) mismatch" and drops the memo on a difference). Logs skips, verified and mismatches
every 600 frames while on. Off by default; measure the back's "render targets" zone (3 = verify
first, then 1).

### EXP32

EXP32 (texture binding memo, the core of a ported BINDMEMO for "the same textures
as the pipeline's previous draw"; Vulkan GraphicsPipeline::TextureMemo, graphics pipelines whose
image bindings are all sampled textures), bits: 1 = a draw that reads the same TIC and TSC
handles as its pipeline's previous draw, with the same pools, channel, GPU mappings and texture
cache epoch (no view removed: image deleted, joined or rescaled), the same guest TIC/TSC
entries and images without CPU writes to upload or aliases, takes that draw's view and sampler
ids and descriptor entries instead of GetSamplerId, FillImageViews and PushImageDescriptors
(handles are still read every draw: the EXP21 storage pointer rule and the per-bind uniform bank
check of d79572d28a are untouched), 2 = verify: every hit for 120 frames after the switch
changes, then one hit in 64, runs the full path instead and compares view ids, sampler ids,
descriptor entries and rescaling bits ("Texture memo mismatch (EXP32)" log, first 32), 4 = a
mismatch turns the memo off until the switch changes. "Texture memo (EXP32=n)" logs hits, misses
by cause, records and verify mismatches every 10 s while set. Off by default; 7 = memo, verify
and auto-off. Measure the back's "read handles", "fill image views" and "push descriptors"
zones (the sampler lookups move from the first to the second with the memo on).

### EXP33

EXP33 = 1 (ported; QueryCacheBase::CounterReport and
FenceManager::SignalFence): at fence behaviour Default without GPU accuracy High (or
EDEN_NX2_ACCURATE_PARTS bit 1), a fence-flagged query report is no longer treated as
synchronized: the query stays registered until its fence runs, and that fence is made even
when empty and runs the report's operation once the host GPU passed the work before it
(VideoCore::g_signaling_query_report_fence), as with delayed fences. Other fences are skipped
as before. Read once per report, live. The ported fix for Ocarina's white flashes on leaves and
walls at low accuracy; ours already has the ported FlushQueryResults wait (EXP30 trims it), so
off by default, not measured yet: white flash watch, and the front's semaphore waits (the
reports land later than with the fence skipped).

### EXP34

EXP34 (EDEN_NX2_ACQUIRE_PRECHECK=2's reset hole, fable_answer_6 6i (b); puller.cpp
PrecheckAudit): where level 2 would answer a GEQ acquire met from guest memory alone,
1 = it takes the normal path instead when the front knows a write to the word is still queued
for the back: a plain release (puller, SEM_EXECUTE, 3D or compute 0x6c3) with any payload, a
noted write over the word (puller semaphores, 3D reports, compute 0x6c3 releases and
reductions, copies, uploads) or, with EXP13 on, a QMD prediction record, and when
SemaphoreReadableNow read the word current but below the value (the Castle Town route's only
cases, 5-13 per 5 s; most likely work it doesn't track ran between that read and level 2's,
the log's first 8 say); 2 = only when that
queued plain release writes less than the word read (a reset); 0 = count only. The counts
are logged either way ("Acquire precheck (level 2)", every 5 s). Default 1: it only turns met
answers into the path level 2 takes for a word below the value, never the other way, so it is
a strict tightening; its cost is the drains it puts back (compare 0, 1 and 2 with
profile_toggle and the AcquiresFast count).

### EXP39

EXP39 (live; keep the split's back fed), bits: 1 = the back
stores its position when it runs out of work (GpuSplit::back_idle_at) and, while that is the
position last published, the front publishes at its next Commit once it holds 8 words, not
after 256 (outside open Pairs and StateDelta blocks; replaces EXP27 & 2), 2 = when an acquire
resumes, the snapshot refresh (DmaPusher::RefreshSnapshot) skips the current list's entries
that ran already and the entry the acquire suspended inside, which reads memory anyway
(CanUseSubmittedCommands); the lists behind it are copied as before. Default 3: both only
change when the back sees work or which words nobody parses again are copied. Measure with
profile_toggle (EXP39 0 vs 3): back "split wait", front "resume pending".

### EXP40

EXP40 (live; command lists on GPU-written pages), bits:
1 = each command list page keeps the 4-byte words GPU work wrote that guest memory lacks
(BufferCache::Nx2ListPage::gpu_words: set by every GPU write noted to the page, the whole range
a dispatch or copy may write, cleared for words the host writes whole and for all of them when
the front flushes the page); once the page was flushed, a list on none of those words reads
guest memory and waits for no copy, and a list on some reads those from the copy and the rest
from guest memory (the "list GPU wait" zone falls; a storage write mark that
EDEN_NX2_WRITTEN_SSBO_MARK_KIB capped counts as written to the end of its last page, since
the shader may write past the cap); 2 = a list on GPU-written pages waits
only for the compute and copy work sent before the last host sync ("list copy writers"
site), not for every one sent so far ("list writers"). Off by default: both read words
earlier than before. Verify with EDEN_NX2_LIST_VERIFY=1 ("list verify ... 0 differ"). Always
on (bug fixes): copy engine copies and clears tell the list pages what they write to guest
memory (their copies went stale), DMAClear checks its whole range (it passed words as bytes),
and a front waiting for a list copy publishes first.

### EXP41

EXP41 (live; command entries on pages no GPU work wrote yet, DmaPusher::Step): 1 = the writer
waits before the port: EDEN_NX2_LIST_WAIT_SKIP=1
(built in) skips every entry's wait, =0 waits for every queued dispatch and copy. 0 (default)
= the new shape: an entry waits only for the compute and copy work sent before the last host
sync (WaitForSyncedWriters), and with LIST_WAIT_SKIP=1 an entry that begins between methods
(a command buffer the CPU writes) doesn't wait unless copy engine work is still queued;
entries inside a method's data (the parameters GPU work writes) still wait. Against the
built-in skip-everything this only adds waits. Entries that didn't wait for every writer are
checked on the back ("EDEN_NX2_LIST_WAIT_SKIP: command list at ... was GPU-written": stays 0).

### EXP42

EXP42 (live; VkImage pool, fable_answer_12 2B; vk_texture_cache.cpp): 1 = a deleted
single-level, single-layer render target's VkImage (and its rescaled copy), once the GPU is done
with it, is kept with its memory for the next image made with the same create info and view
formats, instead of vkDestroyImage + vkCreateImage. Only device-local, non-dedicated images of
256 KiB or more; the pool holds at most 512 MiB, 180 frames each, and empties when the switch
goes off or an image does not fit in device memory. A taken image starts uninitialized
(UNDEFINED layout, contents discarded) as a new one does. Its bytes are left out of the
collector's device memory figure, as deleted images waiting for the GPU are. Logs "Image pool
(EXP42) last 600 frames" and counts pooled creates in "Slow clear texture cache". 1 is the default
(LoadExperiments); 0 = off.

### EXP79

EXP79 (live, 0 = off; ported; Kernel::PhysicalCore::Idle):
a guest core going idle polls its interrupt flag in a hardware wait (MONITORX/MWAITX on AMD,
UMONITOR/UMWAIT on Intel, Common::WaitForFlag; yields elsewhere) instead of a spin of yields
(or EDEN_NX2_IDLE_PAUSE's pauses), so Interrupt's store ends the poll at once and the core
issues nothing meanwhile. The poll length adapts per core (Common::NextPollWindow, the
cpuidle-haltpoll rule): a period a longer poll would have caught doubles it, one past
EDEN_NX2_IDLE_SPIN_US (the limit, 100 us for Switch 2) halves it, down to an eighth of the
limit. EDEN_NX2_IDLE_STATS=1 logs each core's idle periods every 10 s (polled, blocked, wake
latency). Blocking still uses EDEN_NX2_CORE_LOCK's address wait when that is on. In OoT the
guest threads' handoffs sit on the frame's critical path. Off by default.

### EXP87

EXP87 (live, read per draw, 0 = off; ported; GraphicsPipeline::
ConfigureImpl): draw templates. A pipeline whose bindings are all sampled textures with handles
read from constant buffers keeps each draw's views, samplers, texture descriptors and rescaling
words in a shared table (8,192 slots, 4-way, keyed on the pipeline, the address space, the
texture and sampler pools and the handle words). A later draw with the same key, also in a
later frame, takes them instead of resolving its views and samplers and pushing descriptors,
while the template's TIC and TSC bytes are still in memory, no mapping changed
(HostPointerSerial), no image changed its scale (ScaleEpoch) and none of its views was deleted
or moved (ImageViewGeneration). Not with EXP32 or EDEN_NX2_TEX_BANK_GEN on, for mesh pipelines,
or with the OoT FSR LOD bias on a sampler. 1 = on; 2 = verify: draws a template holds for are
resolved in full and compared with it, nothing taken from templates ("Draw template verify in
10 s: ... mismatched" must stay 0). With EXP87 != 0, "Draw templates in 10 s" logs hits,
misses, kept templates and rejections every 10 s.

### EXP86

EXP86 (live, 0 = off; fable52_answer_gpuwritten; RasterizerVulkan::GetFlushArea): a guest
read of pages the GPU ever wrote (no image there, EXP36 off) skips the caches' locks
when the buffer cache's page mirrors (gpu_written_mirror.h) say the locked path would answer
"preemptive" and change nothing: no GPU-modified byte on the pages now, and a Preflushable
mark or no armed read trap there. The mirrors change only under the buffer cache's mutex; a
sequence counter is odd from a hold's first change to its release, and a busy counter sends
the read to the locks. 1 = the fast path; 2 = verify: locked reads compare the mirrors' answer
with what the locked path decides ("mirror clean but cache not" must stay 0), no fast answers.
"GPU-written page mirror" logs the answers every 5 s, also while off.

### EXP88

EXP88 (live, bits, 0 = off; ported, parts a to
c, fable52_answer_next70 3; nx2_cb_path.h): 1 = part c: RasterizerVulkan::InnerInvalidation
(FlushCaching's ranges) and, on the back, OnCacheInvalidations (collected guest writes) skip
BufferCache::WriteMemory and its lock where it would change nothing: no page ever left the
CPU-modified state (MemoryTrackerBase::IsRegionEverSynchronized), no GPU write ever marked one
and no enabled vertex binding overlaps (BufferCache::WriteReachesBuffers); its three page marks
stay (NoteWriteUnlocked: the bank check reads g_buffer_write_pages). The collected writes also
skip the texture lock and the pipeline cache where their page filters are clear, and the
ClaimCpuWrites lock where no GPU-modified mark ever reached. 2 = parts a and b: constant buffer
data (CallConstBufferWords in place of CB_DIRECT, ProcessCBMultiData in place of
WriteBlockCached) goes to the bound buffer's device address, translated once per CB_ADDRESS,
CB_SIZE and memory map (Maxwell3D::WriteBoundConstantBuffer), its device ranges logged for
FlushCaching (MemoryManager::WriteConstantBufferData); buffers that aren't one mapped device
range and Switch 2 sparse memory take the old path. 4 = verify: skipped ranges take the locked
WriteMemory anyway after checking it changes nothing; direct writes are read back and their
translation redone at FlushCaching. Both mismatch counts must stay 0. Same family as EXP85
(its inline writes reach FlushCaching with & 1) and CB_DIRECT (& 2 replaces it). "Constant
buffer path (EXP88" logs the counters every 5 s while not 0.

### EXP90

EXP90 (live, 0 = off; Fable, fable52_answer_after88 2; TextureCache::TouchImageLru): an image's
LRU node is touched once a frame, decided from a frame tick kept on the image
(ImageBase::lru_touched_tick) instead of reading the node (a cache miss) on every
PrepareImage, the way EXP18 & 4 does it for buffers. Exact: the node already ignores a second
touch with the same tick, so the LRU order and the ticks eviction and the sweeps read stay the
same. "Image LRU touch (EXP90" logs touches made and skipped every 5 s while not 0.

### EXP89

EXP89 (live, 0 = off; N > 0 = frames; Fable's lingering protection, fable52_answer_after88
section 1; core DeviceMemoryManager::UpdatePagesCachedCountNoLock): when a page's cached
count falls to 0 it stays protected and is noted with the frame; when a cache counts it again
within N frames nothing is called (no unprotect, no protect). RasterizerVulkan::TickFrame's
sweep (DeviceMemoryManager::ReleaseLingering) really uncaches the pages noted more than N
frames ago; Map, Unmap and UnregisterProcess uncache their pages at once. Exact: a lingering
page is a protected page no cache tracks, whose collected writes reach the caches as today.
Switching to 0 live releases every noted page at the next sweep. Start with 4. Measure first
with EDEN_NX2_PROTECT_STATS=1 and EXP89=0: "Page protection frames since the last unprotect"
gives the share of protects each N would cancel; "Lingering protection (EXP89" logs the
cancelled calls, releases, pages held and, as the verify, pages left lingering after an
Unmap (must stay 0). Watch "writes collected" in the Guest access line: if it rises much,
lower N.

### EXP91

EXP91 (live, 0 = off; fable52_answer_uniformkeeps stage 0; counting only, nothing bound
changes): RasterizerVulkan::InnerInvalidation notes every cached write range it gets
(FlushCaching's, EXP88's constant buffer data among them) in a ring of 128 with a sequence
number, before any early return (BufferCache::NoteCachedWrites). Each of today's slot rebinds
in BindSmallUniformReusing (same bytes, same tick) is then judged as a keep would be: K when
the slot's last bind was the same range and no range noted since overlaps it, with nothing
pending in MemoryManager (accumulator, EXP88's ranges); else not kept, by reason (range
written, ring overflow, pending cached writes, other range, no keep yet). "Uniform keeps
count (EXP91" logs small binds, rebinds and those counts per second every 5 s while not 0.
Gain of the later stages is about K x 55 ns; under 100,000 a second, stop there.

### EXP102

EXP102 (live, pauses, 0 = none; fable52_answer_yield 7): with EXP98=1, an early-returned
yield pauses this many times, keeping the old cadence (try 6, 16, 64).

### EXP103

EXP103 (live, pauses, 0 = off): common WaitForFlag's fallback (no MONITORX/WAITPKG, as on
this owner's PC; EXP79's idle wait uses it) polls with N pauses between checks instead of
SwitchToThread (try 64).

### EXP104

EXP104 (live, 0 = off, 1 = on with entries kept 180 frames, N > 1 = kept N frames; Fable's
ideal pipeline design 8.3, step 3b; renderer_vulkan/vk_buffer_recycler.h): a pool of VkBuffers
by size class for the buffer cache's buffers and the staging buffers. Buffer cache buffers
are made at their size class (powers of two from 64 KiB, up to 32 MiB; sparse-compatible ones
never) and, when the delayed destruction ring lets a Buffer go, its VkBuffer is pooled instead
of destroyed; staging buffers the staging pool releases are pooled by memory usage and size.
Creates take a pooled one of their class first. Exact: an entry carries the scheduler tick
current when it was returned and is handed out only once the GPU passed it (IsFree); the
cache clears a new Buffer's range as before. 256 MiB held at most (oldest go first); entries
past their age, over the cap or left when the switch goes off are destroyed by the back in its
split idle hook (EXP99's way), at the frame's end without the split or past 256 MiB waiting.
Costs up to 2x memory per buffer cache buffer (the class rounding). Gate (Fable): real creates
("missed") near 0 a second in steady play, device memory flat after 10 minutes. With
EDEN_NX2_BUFFER_CHURN_STATS=1 the "Buffer churn (5 s" line ends with the pool's counters.

### EXP106

EXP106 (live, 0 = off; Fable's delta draw port plan, fable_answer_backthread_threads.md 10,
step D2): every graphics pipeline keeps its last draw (GraphicsPipeline::KeptLastDraw: raw
handle words, computed handle values, views, samplers, view generations, image descriptor
entries per stage, rescaling words, ScaleEpoch), through a second instantiation of
ConfigureImpl. Records only, no draw changes; for the delta draw steps after D2.
EDEN_NX2_FASTDRAW_COUNT=1 keeps them too. Off, PrepareDraw reads this switch once a draw.

### EXP107

EXP107 (live, 0 = off; ported): inline
constant buffer data (Maxwell3D::CallConstBufferWords, ProcessCBMultiData) is compared with
the words memory holds, on the back, in order (Maxwell3D::CompareConstantData; one GPU page,
not sparse). Equal and safe: not written at all, so no invalidation, no buffer cache mark and
no delta-draw break (OoT rewrites ~258k blocks a second with the same words, often over
texture handles). Safe means the original conditions: no GPU-written data may be on the pages
(VideoCore::MayHoldGpuWrites, always true outside Switch 2 titles) and no guest read download
may land there; plus ours: not the texture header or sampler pools (also through an alias)
and no run-ahead used this session. Equal but not safe: written as before. With
EDEN_NX2_FASTDRAW_COUNT=1 equal writes are neutral for the count, and the 5 s line ends with
how many changed nothing and how many were not written.

### EXP108

EXP108 (live, 0 = off; ported):
the GPU's own cached writes (MemoryManager::FlushCaching: inline constant buffer data, its
accumulated ranges) mark their pages CPU-modified but leave them tracked
(MemoryTracker::MarkRegionAsWrittenTracked) instead of untracking them as a guest CPU write
does; the upload clears the bits and has nothing to track again. Saves a VirtualProtect pair
per page per write cycle (~4% of the GPU thread at OoT's Kokiri water, +1.4 fps on the
author's PC). Only on pages with no GPU-modified range and no GPU-modified bit (a tracked
page's GPU bits count); never in Nx2Perf verify. A guest CPU access to such a page takes the
slow path, as for any tracked page; a guest write there untracks it as before.

### EXP109

EXP109 (live, 0 = off; ported): a guest write
that untracks buffer cache pages (RasterizerVulkan::OnCPUWrite) changes their host protection
after dropping buffer_cache.mutex instead of under it (the back waited for those system calls
before draws). The word state changes under the lock as before; the page count decrements are
collected per thread (VideoCommon::DeferredPageCaching) and applied in order before OnCPUWrite
returns, so the guest's write still finds the page unprotected. Ours defers only decrements
(increments, e.g. an upload inside the write's slow path, apply at once): late decrements only
keep a page protected a little longer; a late increment could let a page the texture cache
tracks lose its protection in between. Never in Nx2Perf verify.

### EXP110

EXP110 (live, 0 = off, 1 = on at 512 MiB, N > 1 = on at N MiB; fable52_answer_stall3 3, the
ghost pools' "unproven first + takes"): the EXP42 pool's eviction rule. Over the cap, oldest
first, the first entry whose shape (exact create info) was never taken from the pool goes;
failing that the one whose shape has the fewest takes, then the least recent take, then the
oldest. Takes and the last take are kept per shape in a map that outlives entries, evictions
and the emptying on memory pressure (bumped on a take only, never aged; at 4,096 shapes the
3,072 most recently taken stay). Ages as before (180 frames, EXP82 for proven shapes, proof
looked up per shape when tested); EXP65 is not applied. The cap replaces EXP66's and is shared
with EXP99's idle backlog: the pool holds the cap less the backlog, the backlog is bounded at
128 MiB instead of 512 and, past 64 MiB, loses two images a frame end too. Memory pressure,
EXP42 off and a lowered cap still trim by the plain oldest-first loop. The "Image pool (EXP42)"
line ends with the rule, the backlog in MiB and the frame-end destroys. Ghost run (orbit,
182-212 s): 66-68% of lookups taken at 512 MiB against 60-64% for today's rule at 1,024 MiB.

### EXP111

EXP111 (live; 0 = Windows' default, 1 = above normal, 2 = high; vk_rasterizer.cpp TickFrame):
the process priority class, applied when the value changes. The owner gets more fps with the
window focused (Windows' foreground boost); this keeps the game's threads ahead of other
normal-priority programs when it isn't. Watch for a stream encoder starving at 2.

### EXP112

EXP112 (live, 0 = off; fable52_answer_fountain 4; BufferCache::DownloadMemoryForRead, with
EXP49 on): a guest read that must flush GPU-written buffer memory also downloads the
GPU-written run around its pages in the same round trip: the pages after it, then before it,
that hold GPU-modified bytes in a row, up to 256 KiB with the read's own pages, inside the
read's buffer, ending at a neighbouring Preflushable page (the read's own pages may be marked:
with preflushing on, the read marks them first). Exact: only GPU-modified bytes not already on
their way, the same read download, fence and landing. The run's read trap marks go at once,
so once it lands EXP86's mirrors answer the neighbouring pages' reads without the lock (a GPU
write there arms them again). At the great fairy fountain one game job read ~16 such pages a
frame, each a round trip (~0.4 ms). Check FRITCHII_READ_RUNS=1's "Read runs" line first: the
run histogram says whether the reads have neighbours to take.

### EXP113

EXP113 (live; 0 = off, 1 = no poll, N > 1 = poll at most N us; PhysicalCore::Idle): OoT's
guest cores block at once when idle instead of polling for up to 100 us (EXP79's hardware wait
or, without MONITORX/WAITPKG as on the owner's PC, a SwitchToThread loop; ~795 ms/s over six
cores). With SMT those polls share physical cores with the GPU threads. Timing only. An
explicit EDEN_NX2_IDLE_SPIN_US still wins.

### EXP114

EXP114 (live; 0 = off, N > 0 = N us; gpu_thread.cpp): while channels wait on semaphores, the
GPU front polls them with a SwitchToThread between polls for 500 us after the last progress.
With N it waits N us with PAUSE instead, polling less often and leaving its SMT sibling free.
Measured elsewhere at the market: 2 us best (1, 5, 10, 20 less). Not under EDEN_NX2_GPU_WAKE,
whose wait is in hardware already. Timing only.

### EXP115

EXP115 (live; 0 = off, 1 = after 8 misses, N > 1 = after N; uniform_stream_reuse.h): a
(stage, constant buffer) slot whose streamed bytes differed from its copy that many binds in a
row copies guest memory straight into the stream buffer for its next 64 binds: no compares,
no extra copy into the reuse store. Exact (a stream copy of guest memory, as with reuse off);
it only gives up rebinds those 64 binds could have found.

### EXP116

EXP116 (live, read each frame; 0 = 4096 bytes as before, N = N bytes from 256 to 16384;
Vulkan only): the size up to which uniform buffers are streamed and reused instead of going
through the cached buffers. OoT's skinned models bind 7-13 KiB bone palettes about 8 times a
frame with the same bytes. Try 16384. The reuse store is sized for 16 KiB copies either way.

### EXP117

EXP117 (live, 0 = off; TextureCache::RescaleRenderTargets; only matters above 1X): a pass
with a square depth target and no colour target (a shadow map) is drawn unscaled. Scaling a
shadow map does not sharpen the picture and costs the square of the scale in memory (OoT's
12288x12288 sun map: 648 MiB at 1.5x); the per-object maps stay square under Shadow Size
Steps and EXP68, so they are left unscaled too. Switching it on scales such images down once.

### EXP118

EXP118 (delta path for draws that sample nothing; live, but only if it is on at boot: that
raises the breaks EDEN_NX2_FASTDRAW_COUNT counts with, which then stay on for the session; off
by default; src/video_core/fritchii_delta_path.h, fable52_answer_potroom3 sections 2 and 4).
A draw whose previous draw used the same pipeline, with only light state changed in between (no
break: the count's rules), whose pipeline has no texture, image, texture buffer or image buffer
descriptor, keeps the pipeline instead of looking it up and configures it without the
descriptor pools' sync, handle reads, sampler lookups and image views (nothing to bind).
Uniform, storage, vertex and index binds, the descriptor push, render targets (EXP31 still
decides whether they are prepared again), dynamic state, the host query, the draw record and
the reports run as for every draw. With the vertex input moved, the pipeline must take it
dynamically and the attribute types its programs read must be its key's; the cache's types are
then left as a lookup would leave them. Never with EXP32, EXP87 or EDEN_NX2_TEX_BANK_GEN on.
Values: 1 = on; 2 = verify only (decide, draw the full way, the full lookup must give the pipeline
the delta draw would have kept); 3 = on, and every delta draw also looks the pipeline up (another
pipeline: drawn the full way) and synchronizes the descriptor pools after its configure (nothing
may change). "Delta path (EXP118=n) per frame over 600 frames" logs draws decided, delta draws
(or would-be ones), those with the vertex input moved, the full draws by reason, and with 2 or 3
the lookups compared and the differences (must be 0; the first 16 are logged with the shaders).
FRITCHII_DRAW_KINDS also counts, per class, the draws whose pipeline samples nothing and, with
EDEN_NX2_FASTDRAW_COUNT=1, the would-be delta draws. Measure with breaks on at boot on both
sides (EXP118=2 at boot, then toggle 0/1); the breaks' own cost needs a fresh-launch A/B.

### EXP119

EXP119 (report runs: the report side of the occlusion tests; live, off by default;
src/video_core/fritchii_report_runs.h, fable52_answer_potroom3 sections 2 and 4, step 2). A
sample count report (type 21) that is not a fence, and the amend reset of a sample counter
reset, no longer queue a std::function each in the fence manager's list: they become entries
of a run, one operation that writes them in one loop when the fence's results are in. An entry
joins a run only while the run's operation is still the last one queued and no fence took the
list since (the fence manager's operation stamp, checked under its list lock), so each entry
runs exactly where its own operation would have: same fence, same neighbours, same order
against the counter's other operations (presync amend, replicated query, slot-limit resets).
The report is still listed in the query cache's map from the moment it is made until its fence
ran (the split's front, guest reads and render enable rely on it), with one hash fewer; it is
unlisted and freed as before. Render enable: the Conditional mode skips its map lookup (the
runtime only uses the address), IfEqual / IfNotEqual do both lookups under one lock. Today's
path for fences, other report types and traced reports (EDEN_NX2_QUERY_TRACE,
EDEN_NX2_SEM_WATCH). Values: 1 = on; 2 = verify: today's operations still write (queued keeping
the run's stamp, so the runs take the shape they have with 1), the runs only compute outcome,
address and value, and each report's own operation compares (the run must have run before it,
with the same result); render enable does both forms and compares the lookups. Mismatches must
be 0 (a guest write to a report between its run and its operation would count one; the first
16 are logged). "Report runs (EXP119=n) per frame over 600 frames" logs reports, those in runs,
runs started and ran, counter resets (in runs), today's path by reason, render enable calls
(without lookup, with one lock) and with 2 the compares and mismatches. Toggle live 0/1 with
profile_toggle.py; zones "query report", "counter reset", "render enable" (GPU profile 2).

### EXP120

EXP120 (live, 0 = off; DmaPusher's NoteEngineWriter, fable_answer_long_drains 8.4): the
KeplerMemory and Fermi2D methods the GPU thread split sends are noted for the macro DMA read
snapshots (EDEN_NX2_DMA_SNAPSHOT) by what the back really writes, instead of every one as an
unranged writer, which ends every snapshot ("unranged" misses in the "DMA snapshot" line). With
1: a method that only stores a register (every KeplerMemory method but data, every Fermi2D
method but the blit launch, pixels_from_memory.src_y0 + 1) and a single data word that is not
its method's last (the upload is written with the last one) are not noted; the data of a linear
upload notes the bytes it may write (from the upload registers as sent: line_length_in,
line_count, address, pitch, and the exec's copy size). Blits (with 1), block linear uploads
and anything else stay unranged. With 2, also a blit launch notes its whole destination surface
(the registers as sent: address, layout, pitch or block linear size from width, height, depth
and block sizes, the format's bytes per pixel; as SoftwareBlitEngine sizes it, whose covered
rows stay inside it; the texture cache's copy writes an image, not guest memory). A range is
noted as the device ranges its mapped pages make (unmapped pages are written by nothing), when
it is up to 1 GiB and makes at most 16 of them; otherwise it stays unranged. "DMA snapshot
unranged notes (EXP120=n; single/run)" logs, every 5 s with the "DMA snapshot" line, each kind
of method by call (single method / non-incrementing run): noted unranged, ranged, not noted
(with 2, why blits stayed unranged: a destination register not sent yet, format, empty, over
1 GiB, unmapped, over 16 device ranges);
and the reads EXP120 let skip the drain (an unranged note it spared since the page's snapshot)
with their own "differed" count, which EDEN_NX2_DMA_SNAPSHOT=10 checks and must stay 0. With 0
the counts show who notes the unranged writers. Toggle live 0/1 (or 0/2) with profile_toggle.py.

### EXP122

EXP122 (live, 0 = off; GpuSplit's page watch, fable_answer_long_drains 8.2 (a'),
fable52_answer_kills 3.4; needs EXP10 on): the device pages the macro DMA read snapshots
(EDEN_NX2_DMA_SNAPSHOT) are served from are watched. Each compute or copy range the fronts note
(NoteWriterRange, NoteReleaseWriter) is tested against those pages when it is noted, and the
bytes it covers in a watched page are kept with its position (8 a page, the rest folded into one
union). A read of a watched page then answers "may a writer since the snapshot write these
bytes" from that list, with no scan of the writer range ring and no scan limit: before, every
"past the scan" drain of the session was the scan giving up (4,096 ranges at ~11,500 a second),
never an overlap. Pages that are not watched (256 at most, least recently used replaced), or
whose watch started after the snapshot asked about (another channel's snapshot), take the ring
scan as before; the ring's other readers are unchanged. Unbounded writers still count for every
page. 1 = on. 2 = on, and every "no write" answer is checked by the old scan without its limit;
an overlap it finds that the watch did not report is a verify mismatch (the first 16 logged as
"Writer page watch (EXP122=2)"), which must stay 0. EXP10's "DMA snapshot EXP10" line logs reads
answered by the page watch (and how many said "may write"), watched pages, byte-list overflows,
verified answers and verify mismatches. Off by default, not measured yet: run 2 first (0
mismatches in a kill fight and at the pot room), then toggle 0/1 with profile_toggle.py and
compare the "past the scan" count and the macro dma read drains.

### EXP121

EXP121 (live, 0 = off; fable52_answer_kills 3, fix 1): the texture cache's collector, once device
memory is at its expected threshold, downloads each GPU-written image it evicts and waits for
the GPU every time (2 to 3 ms, then ten more short waits, in fight frames). With 1 it records
the download into a staging buffer without waiting and deletes the image at a later frame end,
once the GPU is past the copy: guest memory written, untracked, unregistered, deleted, as the
waiting path does. Until then the image stays registered (a guest read takes the usual read
path) and counts as freed for the collector. The queued download is dropped and the image kept
if it is used again (its LRU tick moved), written again (by the GPU, the CPU, or a guest read
download), deleted by another path, or if newer GPU data reaches its guest memory (another
image's or a DMA buffer download, or an overlapping image written since). Still waiting: an
allocation that failed (the memory is needed now), sparse images, and past 64 queued downloads
or 256 MiB of staging (EXP57 still limits those). With 2, each landing is also compared with a
fresh waited download (mismatches must be 0; the first are logged with the image). "Texture
cache last 600 frames: collector downloads" logs, with EXP121 on or EDEN_NX2_TC_LOG, the waits
left, queued, landed (and frames from record to landing), dropped by reason, fallbacks and
mismatches. Toggle live 0/1 (or 0/2) with profile_toggle.py in a scene at the memory threshold
(8 GB cards, long fights).

### EXP123

EXP123 (SEAM, live, 0 = off; the memory stage's step 2a, fable_memstage_step2_spec.md 6.1):
every Vulkan command the buffer cache runtime records (uploads, copies and their barriers,
clears, the quad index tables, the null buffer's fill, index, vertex and transform feedback
binds, the uint8, quad and vertex repack compute passes, and its uniform, storage and texel
buffer descriptors) is pushed as an entry; MemPlayer::Exec (vk_mem_stage.cpp) is the only code
that turns an entry into scheduler and descriptor queue calls. With 0 each entry runs the
moment it is pushed: today's calls at today's moments, in today's order. With 1 the entries of
a fill bracket wait in the arena and are played where the call site plays them: after the
geometry binds (UpdateGraphicsBuffers, BindHostGeometryBuffers), after each stage's
BindHostStageBuffers, after BindHostComputeBuffers, after each call that returns a buffer
(ObtainBuffer for a GPU count, an indirect dispatch, kept list words, a DMA image copy;
GetDrawIndirectBuffer and Count; the query cache's ObtainCPUBuffer) and at the end of each
single operation (downloads, write and invalidation ranges, runahead chunks, unmaps, DMA
copies and clears, inline uploads, list page copies and flushes, the frame tick, the fence
commit). The picture must stay identical. Read once a frame; a change makes the player forget
the remembered index and vertex buffers (a few rebinds). The rebind checks against the
recording epoch and the list draw facility (list draw arguments, kept list words) are the
player's; finishes, waits, submits for host reads, the multi-range bind and the kept words wait
go through SeamClock, which plays what is pending first.

Verify, always counted: "around the packet" (5.1, something recorded or added a descriptor
inside a fill without going through the player or a sync: a runtime path that was not
converted) and "late plays" (5.2, entries still pending when something else recorded: a call
site misses its play). Both must stay 0; the first 16 are logged as "Memory seam: ..." with the
use and its source location. "Scheduler uses not through the clock" counts flushes, finishes
and state resets inside a fill that did not go through SeamClock. With EXP123 = 1 (or
FRITCHII_MEMSTAGE_SEAM_STATS=1) "Memory seam (EXP123=n)" logs every 10 s: draws, dispatches and
fence commits a second with entries each, deferred entries and plays a second, the counters
above, entries a second by kind (which kinds OoT uses at all: convert, multi-range, texel
buffers), fills by site with entries each, syncs by kind, and entries run outside any fill by
kind. Compare 0 and 1 with profile_toggle.py ("back us a draw", heavy-phase market, 4 cycles).

EXP123 = 2 (sub-step 2b, spec 4 and 6.2): as 1, and each draw and dispatch is filled once,
before the texture work. Needs FRITCHII_MEMSTAGE_TICKS=1 or 2 (boot): without packet ticks 2 and
3 run as 1, logged once ("SEAM 2/3 need FRITCHII_MEMSTAGE_TICKS=1 (or 2); running SEAM 1") and
counted in frames on the "Memory seam" line. The draw's fill (MemStage::FillDraw, at the rasterizer's draw site after
the vertex repack layout) runs SetUniformBuffersState, every prepared stage's storage binds, the
texture buffer unbinds, UpdateGraphicsBuffers, BindHostGeometryBuffers and every stage's
BindHostStageBuffers; the dispatch's (MemStage::FillDispatch, before ComputePipeline::Configure)
the uniform state, the storage binds and shader hash, the texture buffer unbinds,
UpdateComputeBuffers and BindHostComputeBuffers. Configure no longer calls the buffer cache: it
plays the packet's sections where the calls were (the geometry after the texture buffer binds,
each stage after Acquire and before its image descriptors, the dispatch's section before its
image descriptors). A sync inside the fill (the multi-range bind, the kept words wait) plays only
what it needs (copies, uploads, barriers, conversions); the index, vertex and xfb binds and the
descriptors wait for the packet's plays. When Configure returns, true or false, whatever is left is played (the cache
believes those uploads are on their way); descriptors left then are dropped with the draw.
Descriptor entries take the pipeline's form (descriptor buffer or not), as the fill runs before
Acquire. Pipelines with texture or image buffers keep their fill in Configure as with 1
("late fill": the binds need the texture cache's views). The indirect dispatch's ObtainBuffer, the
extent probes, list page copies and draw indirect buffers stay where they are (fill and play at
the call, as with 1). There is no tick guard: packet ticks keep a held packet's numbers open.
"Flushed while held" (5.5) counts held packets whose fill and PlayRest are in different command
buffers (a flush or finish in between, syncs included; the first 16 logged with the caller):
expected near 0, as a vertex binding the fill did not see dirty is not bound again in the new
command buffer. The stats line gets a "memory seam 2b" line: held and
late fill draws and dispatches a second, packets not held (arena busy on another thread), flushed
while held, rest plays (and descriptors dropped), descriptor form mismatches (must be 0), and fill
and play time with EDEN_NX2_GPU_PROFILE. The GPU profile gets two zones, "memory fill" (every fill
at 2, the buffer cache's own work included: what leaves the back in step 6) and "memory play".
FRITCHII_MEMSTAGE_READ_LOG adds the read log (5.3). Compare 1 and 2 with profile_toggle.py
("back us a draw", heavy-phase market, 4 cycles).

EXP123 = 3 (sub-step 2c, spec 6.3 and 9.1-9.3; as 2: the recorder, the player and the early
fill): what the back reads from game memory that the stream writes comes in the packet by value. Handle values: the
raw handle of every image element a draw or dispatch binds, in the order the handle loops read
them (per stage: texture buffers, image buffers, textures with handle tables expanded, images;
one value per element, EXP21's buffer address words keep their place unread), made by one
function for both sides (texture_handle_values.h: RawTextureHandle, FillStageHandleValues), so
handle programs' constant buffer words, storage buffer addresses and storage words are read by
the fill. The fill always runs: draws whose handle loop the back skips (TEX_BANK_GEN hits) now
pay for those reads and are counted. The handle values are not part of 2b's early fill yet: the
pipeline's Configure fills them at the top of its handle work ("filled in Configure"). The QMD: the
native QMD scheduler's 256-byte read goes through the dispatch packet (MemStage::Qmd), which
opens there; the error path's 512-byte read stays direct. Null pointer slots: the compute
pipeline lookup's pointer read (StaleNullPointerSlot) goes through the open dispatch packet too
(MemStage::ReadPointerSlot). With 3 the stats line adds "memory seam by value": draws and
dispatches with values a second (values each, how many filled in Configure, how many the back
skipped and their values), QMDs a second (and reads that ran no dispatch), null pointer slots,
and the verify counters below. Values below 3 change nothing here.

### EXP101

EXP101 (live, 0 = off, 1 = runs, 2 = cached page pointer + prefetch; fable52_answer_presenter): nvhost_gpu::SubmitGPFIFOImpl's pushbuffer
snapshot reads runs of entries whose GPU addresses follow each other with one ReadBlockUnsafe
instead of one per entry (OoT's presenting thread submits ~6,500 entries a frame; the loop is
on the frame's critical path). Exact: same words at the same offsets. "Pushbuffer snapshot
(EXP101" logs submits, entries, reads and loop time every 5 s while on.

### EXP105

EXP105 (live, 0 = every entry; N > 1 = 1 in N): DmaPusher::Step's check that a command list
the front read early was not GPU-written by queued work (EDEN_NX2_LIST_WAIT_SKIP log). It only
reads and logs, never fired in OoT on 2026-11-12, and cost ~8 ms/s of the back (try 64).

### EXP99

EXP99 (live, 0 = off; fable52_answer_chicken 1): evicted EXP42 pool images (OoT's per-NPC
depth targets, ~3 a frame while a chicken passes or an enemy dies) are destroyed on the image
free thread (2; needs EDEN_NX2_ASYNC_IMAGE_FREE=1) or, with 1, by the back itself in its idle hook
(up to 2 each time it runs out of work), instead of at the frame's end. The pool line every 600
frames counts where they went; the timeline shows "back: image pool tick".

### EXP100

EXP100 (live, 0 = off; fable52_answer_imagereads, Fable's F1; RasterizerVulkan::GetFlushArea):
a guest read of pages that hold an image skips the caches' locks when no image there was
GPU-modified. TextureCache::gpu_image_filter counts per 64 KiB page every registered,
non-sparse image that was GPU-modified since it was registered (CountGpuModified at
MarkModification, SynchronizeAliases, JoinImages and RegisterImage; removed at
UnregisterImage; clears of the flag are ignored, so it is conservative), under the texture
cache's mutex, with a sequence counter (image_seq, odd while a hold changes the count, closed
by the mutex's release hook). Nothing counted on the pages, no sparse image ever there, the
counter even and unmoved, and the buffer side clean as today (no GPU write ever marked the
pages, or EXP86 = 1's mirrors say clean): the locked path would answer the same preemptive
area and change nothing. Not with EXP36 or EXP81 on or EXP43 = 4. 1 = the fast path; 2 =
verify: locked reads compare the count with the texture cache's locked look, no fast answers
(mismatches must stay 0, the first 16 are logged with the image's range, flags and
nx2_gpu_counted). "Guest reads locked because" logs the answers and why reads stayed locked.

### EXP96

EXP96 (live; fable52_answer_guest 4): the HLE reply hold (k_server_session.cpp HoldHleReply,
written for Mario Tennis Fever, active in every title). 0 = EDEN_NX2_IPC_MIN_US or 50 us;
1 = off; N > 1 = N us. In OoT the service thread that presents waits in it ~14 ms/s.

### EXP97

EXP97 (live; fable52_answer_guest 3a): spins of a contended KSpinLock (the scheduler lock)
before it sleeps. 0 = 128 as before; N > 0 = N. EDEN_NX2_KLOCK_STATS=1 logs its locks every
5 s ("Kernel spin lock (5 s").

### EXP98

EXP98 (live; fable52_answer_yield): guest yields (svcSleepThread 0 and -1) skip the scheduler
lock when it would find nothing to do (k_scheduler.cpp YieldIsKnownNoOp). A change counter
per core, raised under the lock for the cores of every thread whose scheduling changes; a
yield that did nothing stamps its core and that core's counter and raises nothing; the early
return reads the lock word, then the counter. 0 = the code as before (per-process count);
1 = early return; 2 = verify: always take the lock, check the stamp's prediction under it
and count mismatches (must be 0, the first 16 are logged). While not 0, "Guest yields
(EXP98" logs yields by form, early returns, lock-held checks and locked outcomes every 5 s.
-2 (YieldToAnyThread) is only counted.

### EXP94

EXP94 (live, 0 = off; fable52_answer_front 1): RasterizerVulkan::HasCachedReports answers
"none" from the query cache's page filter before taking its mutex (the front asks for every
macro DMA read, ~5 ms/s of the front in the market heavy phase). Exact: the filter marks every
page a query was ever registered on.

### EXP95

EXP95 (live, 0 = EDEN_NX2_OOT_PRESENT_SLEEP_US or 250; N > 0 = N us; fable52_answer_front 2):
the short sleep of OoT's present poll with EXP24 on. The event word is written by the game's
main thread, so a sleep only ends at its timeout; shorter sleeps end the poll sooner after
the write (try 100).

### EXP93

EXP93 (live, 0 = as EDEN_NX2_UBO_INPLACE says, 1 = on; fable52_answer_writer 3): small
uniform buffers are compared with their slot straight in guest memory instead of through a
scratch copy (VideoCommon::UniformInPlaceCompareEnabled). Judge on "bind stage buffers".

### EXP92

EXP92 (live, 0 = off; fable52_answer_uniformkeeps 3 and 4, idea from a ported change, part d;
BufferCache::BindSmallUniformKeeping): a small uniform slot bound again with the same range,
in the same command buffer, with nothing written to that range since, binds the slot's stream
copy again without reading or comparing guest memory (a "keep"). Written-since is known two
ways: tier 1, the back's cached writes by exact range (EXP91's ring, filled while EXP91 or
EXP92 is on; nothing pending in MemoryManager); tier 2, a generation per device page
(Common::UniformKeepPages) raised after the bytes land by every other writer (MemoryManager
writes, device memory writes, the GPU write listener, in-place spans, run-ahead words, and
guest CPU stores, which trap because a kept range's pages are watched: one
UpdatePagesCachedCount +1 each, given back at RasterizerVulkan::UnmapMemory and at
DeviceMemoryManager Map/Unmap; a page stored to more than 64 times a frame is denied). Plus
the device translation generation. Not on native bank binds or behind a run-ahead frontier.
1 = keeps on, every 256th kept bind verified (the tripwire): one stale bind logs an error and
turns keeps off for the session. 2 = verify: a keepable bind runs the normal path anyway, which
must rebind the same copy ("same"); stale ones are counted, the first 32 logged, and bound
from their fresh bytes. "Uniform keeps (EXP92" logs binds, kept, not kept by reason, watched
pages, guest stores on them, denied pages, verified and stale every 5 s while not 0. Stale
must stay 0. 1 -> 0 at any time drops every keep (watched pages stay watched until unmapped).

### EXP85

EXP85 (live, bits, 0 = off; fable52_answer_microcuts 1 and 2b): 1 = a small inline upload to
pages no GPU write ever marked skips the buffer cache's lock (InlineMemory would find nothing
there; the list note and the query cache as before), 2 = BufferCache::WriteMemory answers
"no GPU-written data on these pages" from the GPU-written bit before the range set. 3 = both.

### EXP84

EXP84 (live, 0 = off; fable52_answer_microcuts 3.4; RasterizerVulkan::GetFlushArea): a guest
read skips the caches' locks when no image is registered on its pages now (TextureCache::
image_count_filter, counted at register and unregister, plus the pages sparse images ever
covered) instead of never (page_filter, whose bits never clear while the game's targets move
every frame) and no GPU write ever marked them. Daytime market: guest cores waited 22-54 ms/s
for those locks, ~3,500 locked reads a second, 65 of them needing a flush. "Guest reads
locked because" logs the reasons every 5 s, also while off.

### EXP83

EXP83 (live, 0 = off; fable52_answer_microcuts 3.2; RasterizerVulkan::UpdateVertexInput): a
vertex input state identical to the last one recorded in this command buffer is not sent
again (any vertex buffer address or stride write resent all 32 bindings). The memo drops on
StateTracker invalidation (new command buffer, input or whole state; helper passes call
InvalidateState) and when a pipeline without dynamic vertex input is drawn. "Vertex input
(EXP83" logs sends and repeats every 10 s, also while off.

### EXP82

EXP82 (live, frames, 0 = off; fable52_answer_after7778 1; TextureCacheRuntime::TickFrame):
proven EXP42 pool entries (a shape taken from the pool before) stay this many frames instead
of 180, so per-NPC depth targets whose size comes back later find their image. Unproven
entries and the byte cap (EXP66) are unchanged. Try 1800. Judge on the pool's taken/missed
line and the clear zone.

### EXP81

EXP81 (live, frames, 0 = off; fable52_answer_readback_card 3; BufferCache TickFrame and
GetFlushArea): with EXP69's Preflushable reset, unmark only the pages the CPU has not read for
this many frames. Every CPU read, locked (GetFlushArea) or EXP86's lock-free clean answer,
sets its pages in a page bitmap that TickFrame folds into per-page stamps every frame
(fable52_answer_fences2 2), so it works with EXP86 on. Unmarking every page made each page
the game still reads need one async read (~40 a second at the market) to be marked again; a
mark only decides whether a GPU write is copied at the fence or at the read, so it is exact.
"Preflush reset" logs the KiB kept. Try EXP69=4 with EXP81=4, then 2 and 8.

### EXP78

EXP78 (live, 0 = off; fable52_answer_killuploads 3-4; TextureCache::TcSkipStale): what
EDEN_NX2_TC_SKIP_STALE=1 does at boot, switchable: a deleted GPU-written image marks its whole
pages stale (CPU writes clear them) and a new image whose range is all stale (and not
buffer-written) skips its upload from guest memory nobody wrote. Fight kill bursts: 30-105
such uploads inside draws, 3-10 ms. Marks start when it turns on. "stale uploads skipped" in
the texture cache's 600-frame line.

### EXP77

EXP77 (live, 0 = off; fable52_answer_keeptex_2 2b; TextureCache): a GPU-written chain target
the cold sweep deletes leaves its shape at its address; a texture header that asks for that
address again marks the shape (and address) GuardSampled, so the sweep keeps that shape from
then on instead of deleting it and remaking it from stale guest bytes. Fight: 197 GPU-written
images a window remade after a sweep. Only keeps longer. Counted in the "Cold sweep" line.

### EXP76

EXP76 (live, bits; fable52_answer_keeptex 2a; TextureCache): kill frames remade textures the
game already had (fight: 767 of 833 new sampled images in 10 s had bytes made before; a BC5
2448x3056 one 129 times). 1 = FindImage keeps an image the game's remap flagged when the same
texture is asked for at the same GPU and device address with the same shape (never
GPU-written, no alias or sparse state): no delete, create or new views; it is uploaded as a new
one would be. 2 = whole uploads of 64 KiB or more store a CityHash128 of the guest bytes (read
after the upload, dropped if the image was CPU-written meanwhile); the next whole upload whose
bytes hash the same is skipped. Any other refresh path clears the hash. The
EDEN_NX2_IMAGE_HASH_LOG=1 summary logs "Remaps (EXP76=n)": images flagged, kept, hash checks,
uploads skipped. 4 (with 2; measure only, fable52_answer_keeptex_2 step 1): hashed images are
indexed by (shape, content hash); each new sampled image hashes its guest bytes and the
"Content reuse" line counts which reuse would serve it (live idle twin, live busy twin,
kept deleted image, nothing) and the hash cost in GB/s. GPU writes clear the hash
(MarkModification, SynchronizeAliases). Off by default.

### EXP75

EXP75 (live; us, 0 = off; fable52_answer_pipeline section 3; Core::Timing::CoreTiming's timer
thread): precise guest timers. The thread spins the last this-many us before a precise event
(the kernel's hardware timer: guest sleeps and timeouts, EventType::precise) instead of waking
on the OS timer, whose 0.5 ms tick made the game's 100 us sleeps last ~420 us and its 1 ms
present poll 1.45 ms. Other events stay on the timer. Costs a share of one logical CPU (kept
off the GPU threads' cores). Default 600 (LoadExperiments).

### EXP74

EXP74 (live; us, 0 = off; fable52_answer_perdraw 1; FenceManager::ReleaseThreadFunc): the fencing
thread lets the texture/buffer cache lock go between two pops after holding it this long, so a
batch of 20-60 fences no longer keeps the back waiting 1-13 ms.

### EXP73

EXP73 (live; 0 = off; fable52_answer_bosswave_2 3b; Maxwell3D macro DMA snapshot miss path): the
report flush after a drained read runs only when reports are listed over the bytes read; else it
was an empty round trip to the back (~2,850 a second in the Stalchild fight).

### EXP72

EXP72 (live; 0 = off; fable52_answer_bosswave 3; BufferCache::GetFlushArea, WaitFenceDownload):
a preflushed guest read whose GPU-written bytes are all in a submitted fence download waits (up
to 20 ms) for that download to land instead of a new copy behind everything recorded since.

### EXP70

EXP70 (live; ms, 0 = off; fable52_answer_acquire 4; PresentManager::GetRenderFrame): after this
long without a free render frame, the renderer drops the oldest queued frame that is not at
the present thread's front and reuses it. With EDEN_NX2_PRESENT_EXTRA_FRAMES (boot) giving more
render frames than swapchain images. Display stalls (acquire blocked 76-140 ms) stalled emulation.

### EXP69

EXP69 (live; frames, 0 = off; fable52_answer_stalls 5.1; BufferCache::TickFrame): every N frames
the Preflushable marks GetFlushArea set go; a range the CPU still reads is trapped on its next
read and marked again. Marks never cleared before: ~100 MB of fence downloads a second.

### EXP68

EXP68 (live; bits, 0 = off; fable52_answer_npcdips 6): bit 1 = a depth clear whose scissor is
the real width (zeta width rounded up to 16, same 64-byte column) looks up the scissor-wide image
the draws then use, so OoT's per-NPC depth targets are one image a frame, not two plus an upload
and alias copies. The next depth bind at the address checks the size; a miss turns it off there.
Bit 4 = images at an address whose target keeps changing size skip the EXP42 pool (their dims
never come back; the pool sat at 512 MiB during NPC passes).

### EXP67

EXP67 (live; bucket in px, 0 = off): depth render targets at an address whose target changes
dims get a host image rounded up to the bucket (ImageFlagBits::HostPadded), so the EXP42 pool
reuses the VkImage for the next frames' nearby dims instead of a create + 0.2-1.3 ms depth bind
(OoT per-NPC depth targets). A texture lookup that finds a padded image stops padding there.

### EXP66

EXP66 (live; MiB, 0 = built in): the EXP42 pool's cap (512 MiB; 1,536 with EXP42=2), trimmed
at once when lowered. NPC passes held it at its cap (~0.5 of the owner's +0.8 GB).

### EXP65

EXP65 (live; 0 = off; vk_texture_cache.h): EXP42 pool entries whose exact create info was never
taken back age out after this many frames (proven shapes keep the 180). Per-NPC targets sized by
screen bounds filled the pool to 512 MiB with dims that never repeat.

### EXP64

EXP64 (live; N = min frames untouched, default 16, 0 = off; fable_answer_npcdips 3, NPC dips): a new render target that a clear
overwrites whole retires the images at its exact address that the GPU wrote in an earlier
frame (TextureCache::RetireSlotPredecessors) instead of leaving them as bad overlaps for EXP58's
sweep 60 frames later. Ocarina draws each passing NPC into a D32 target sized by its screen
bounds at a fixed slot, so each frame made a new image and the old ones piled up (80-95 bad
overlaps walked per clear, ~2,000 live images, the pool full at 512 MiB, VMA growing: device
memory +0.8 GB while the chicken passes). Not for shapes the CPU read or that were sampled in a
later frame (the relocation guard's GuardCpuRead / GuardSampled), images bound, queued for
download, aliased, scaled or of the excluded kinds. Logs "Slot retire (EXP64) last 600 frames".

### EXP45

EXP45 (live; 1 = off): the native compute write extent self-check (BufferCache::ExtentTrust,
fable_answer_25): a program is marked in full until a dispatch at its dims showed its proof
holds, and for good once one wrote past it (Ocarina of Time's glowing triangles, 2026-11-09).

### EXP43

EXP43 (live; render target relocation, fable_answer_12 2A; TextureCache::JoinImages): a render
target a clear overwrites whole, with no copies, aliases or remapped images among its overlaps
and exactly one predecessor among its bad overlaps (an image of the same single-level,
single-layer 2D shape it drifted off by less than its size, not written in this frame: Ocarina
of Time's bump-allocated per-frame targets while an NPC moves), takes that image's place;
bad overlaps of other shapes stay as for a new image. 2 = relocate: the image moves to the new
address (stale at the old one, untracked, unregistered, its overlap links dropped, registered
again; its VkImage, memory, views and framebuffers stay). 3 = as 2, and with several
predecessors the most recently modified one moves (lets a chain made before the switch went
on stop growing and age out). 1 = retire early: the predecessor is deleted at once and a new
image made, as with 0. 0 = off: candidates are only counted. 4 = verify (fable_answer_16 1c): as
3 under the render-frame clock (as EDEN_NX2_RENDER_CLOCK=1), and each old range is kept as a
tombstone for 2 render frames; reads that land on one are counted ("Relocation tombstones
(EXP43=4) last 600 frames: N hits", first 8 logged), which must stay 0. Default 3 (LoadExperiments). Never for a shape
whose GPU-written contents were sampled or rendered to without a clear in a later frame, or
read by the CPU (sticky per shape for the session: "Render target relocation guard (EXP43)"),
nor for rescaled, sparse, converted or CPU-read images, ones with a download queued, bound as
another target of the clear or in another channel. Logs each shape's first candidate and
relocation and "Render target relocation (EXP43=n) last 600 frames".

### EXP51

EXP51 (live; RAM, fable_answer_26 C2c; Vulkan PipelineCache::EvictUnusedComputePipelines):
compute pipelines unused for 9000 frames are destroyed, least recently used first, while more
than 500 are held (checked every 600 frames). Their next dispatch translates them again on the
GPU thread, binding the pointers they bound, and builds them from the driver's pipeline cache
(no new disk cache record). 1 = only programs dispatched this session, 2 = also boot builds no
dispatch used yet. Never one still building or with a rebuild pending; destroyed through
retired_compute_pipelines once the GPU and their worker build are done. Off by default, not
measured yet: compare the 10 s memory line's compute count and the working set over a long
session, and the "Evicted N compute pipelines" log's "dispatched again" count (each is a
GPU-thread rebuild, a possible hitch).

### EXP52

EXP52 (live; with EXP19 / EDEN_NX2_PIPELINE_CLEAN on; fable52_answer_1 section 1; Vulkan
PipelineCache::CurrentGraphicsPipelineClean): 1 = when only the bound programs changed since
the last key (every other Refresh input stands), the key's state is kept and only what the
programs feed is redone (with dynamic vertex input, the attribute types masked by what the
programs read), then Next / the cache map looks the new key up, without
FixedPipelineState::Refresh. 2 = as 1, and each such key is also built by the full Refresh and
compared byte for byte; Refresh's is kept, the first 16 differences are logged ("Programs-only
key (EXP52=2) differs"). The "EDEN_NX2_PIPELINE_CLEAN per second" line counts "programs only"
and, at 2, "verified N, M differ". With EXP52 on its "programs" reason stays 0 (a change of
programs together with other state counts under that state). Off by default, not measured yet:
prove 2 shows 0 differ first (under EXP19=2 without EXP50, blend writes without
Dirty::Blending already leave the clean lookup's attachments stale, and verify reports that
too), then compare the lookup zone with profile_toggle (EXP52 0 vs 1).
