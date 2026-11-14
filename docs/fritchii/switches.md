# Fritchii switches

What the live switches and the numbered experiments do. The switches themselves are in
src/video_core/nx2_debug_control.h, the experiments' defaults (with the measurements behind
them) in src/common/fritchii_switch_defaults.cpp. This file is not compiled: editing it
rebuilds nothing.

## Named switches

### EDEN_NX2_MEM_SAMPLE

TEST, off by default (2026-11-14): EDEN_NX2_MEM_SAMPLE=1 (read at startup) lets the MemoryProbe
thread sample which guest pages are resident every 10 s (HostMemory::SampleResident) for the
"Memory owners" line. The walk holds kernel locks for 35-48 ms and showed up inside a 217 ms hitch
in the owner's playtest, so it only runs for a memory hunt; without it the line's resident and
guest-unique figures read 0. EDEN_NX2_MEM_HEAPS=1 (the heap walk, ~250 ms) is separate.

### EDEN_NX2_STALL_SNAPSHOT

On by default since 2026-11-14; EDEN_NX2_STALL_SNAPSHOT=0 turns it off (read as each thread is
named, so nx2_switches.txt applies). A present gap of 100 ms or more logs a "Stall snapshot:"
line after its Hitch line: for the GPU back, front, Vulkan worker, fencing, present and GPUMem
threads the span each is in now and for how long, the longest span each spent in the gap, the
master semaphore's ticks, and whether the present thread is between acquire and present
(common/nx2_hitch.h).
- Cost (2026-11-14, fable_front_drains 8.4): it no longer opens the profiler's zone gate and reads
  no clock. Spans take their times from a tick the OS keeps (Windows: the kernel's interrupt time,
  0.5 ms steps with Fritchii's timer resolution, 15.6 ms at worst; Linux: the coarse monotonic
  clock), so marking a span is a few loads and stores. Before, every coarse profiler zone called
  out of line and read the wall clock twice (83 ms/s on the front at the market).
- Only waits mark the slots: the timeline's spans (fence, worker, present and host GPU waits),
  the profiler's wait zones (idle, pending sleep, split drain, writer wait, list waits, query DMA
  wait, read flush, image create, pool destroy here) and the front's per-list "push". The
  per-draw, per-dispatch, per-macro and per-fence zones no longer do: time in them reads as the
  span around them ("push" on the front) or "(no span)" (busy outside every wait, the back's
  draws and dispatches mostly). Times are good to about one tick.
- With EDEN_NX2_GPU_PROFILE or a timeline recording the zones are timed exactly as before.

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

### FRITCHII_DELAY_MACRO_DRAIN_US, FRITCHII_DELAY_POLL_DRAIN_US, FRITCHII_DELAY_BEFORE

Delay tests (live, default 0 = nothing changes; Coz-style causal profiling, fable_front_drains
section 3): each adds a busy spin of n microseconds on the front (steady_clock with a pause, not a
sleep) at one kind of front drain, to see whether that wait is on the frame's critical path. If
fps drops by about what the added ms a second predict, it is; if not, shortening it won't help.
- FRITCHII_DELAY_MACRO_DRAIN_US=n: a macro DMA read snapshot miss (Maxwell3D::FrontDmaRead, the
  split.Drain(MacroDmaRead) of EDEN_NX2_DMA_SNAPSHOT's miss path, the 0xe2c occlusion reads'
  "noted write" misses among them). Fable's T1: 50 us adds about 120 ms/s at the market.
- FRITCHII_DELAY_POLL_DRAIN_US=n: a suspended acquire's poll that falls through to the full drain
  (Puller::CanResumeAcquire's DrainBack, the "acquire" site). Fable's T2: 600 us adds about 111
  ms/s. The spin is outside the poll drain's own time in the "Fence writer resumes" line.
- Only drains that wait spin: the back had not run everything sent when the drain began.
- Where: right after the drain returns (the test: the back has nothing queued then, so the spin
  is added front time). FRITCHII_DELAY_BEFORE=1 moves both spins to right before the drain (the
  control: the spin overlaps the back's backlog, which keeps shrinking, so it should cost little).
  Nothing is published before a spin.
- "Front waits for the back by site" (every 5 s) ends with "delay tests added (after/before the
  drain, FRITCHII_DELAY_BEFORE=b): FRITCHII_DELAY_MACRO_DRAIN_US=n x/s y ms/s,
  FRITCHII_DELAY_POLL_DRAIN_US=n x/s y ms/s" and the three work tests below, while a delay
  switch is set or spins were counted.
- Measure with profile_toggle.py in one session (0 against n).

### FRITCHII_DELAY_E2C_US, FRITCHII_DELAY_DISPATCH_US, FRITCHII_DELAY_DRAW_US

Three more delay tests (live, default 0 = nothing changes; fable_front_drains section 6, Fable's
W, K and D), the same busy spin of n microseconds. The drain spins above can't tell "the front
paces" from "the back paces" (right after a full drain the back is empty, so any front spin there
is critical by construction); these add work to one thread in the middle of its pass:
- FRITCHII_DELAY_E2C_US=n ("W", front): every call of the 0xe2c occlusion macro (hits and
  misses, about 33,000 a second at the market), at the start of the call, before its read,
  whether the OoT handler or the JIT runs it (Maxwell3D::FrontRunMacroFeatures, where the call's
  macro slot is identified; with every macro switch off the plain body runs and there is no
  spin). n = 3 is about 100 ms/s of front work.
- FRITCHII_DELAY_DISPATCH_US=n ("K", back): every compute dispatch, at the start of
  RasterizerVulkan::DispatchCompute. n = 12 is about 96 ms/s.
- FRITCHII_DELAY_DRAW_US=n ("D", back): every 3D draw call (not per instance), at the start of
  RasterizerVulkan::Draw and DrawIndirect (list draws included; not DrawTexture blits). n = 1 is
  about 110 ms/s.
- K and D spin only on the GPU back thread (GpuSplit::OnBackThread): never on GPUMem with
  FRITCHII_MEMSTAGE_THREAD=2 (a draw GPUMem serves is still played on the back, which spins
  there), and not at all without the split. The spin sits outside the profile's "draw" and
  "compute" zones.
- Counted in the same "Front waits for the back by site" ending: "FRITCHII_DELAY_E2C_US=n x/s
  y ms/s (front), FRITCHII_DELAY_DISPATCH_US=n ... (back), FRITCHII_DELAY_DRAW_US=n ... (back)".
- Reading them: a spin on the thread that paces a pass costs about its ms/s in fps; one on a
  thread with slack there costs little.
- Off (0), a call, draw or dispatch pays one relaxed load of its switch and a branch, inline at
  the call site (GpuSplit::DelayWorkIf), no function call.

### FRITCHII_DELAY_RUN_US, FRITCHII_DELAY_RUN_KIND

The memory stage's delay test (live, default 0 = nothing changes; fable_stage_levelpoints 5.1),
with FRITCHII_MEMSTAGE_THREAD=2: GPUMem busy-spins n microseconds right after a level run's end
(its StageWaitFor of the run returned, before the return's flag take), so the back stays parked
that much longer. It tells whether GPUMem's time parked in runs is on the frame's path: if fps
drops by about what the added ms a second predict, every ms/s of parked time cut is a ms/s of
frame.
- FRITCHII_DELAY_RUN_KIND: 0 (default) spins after runs without a compute launch, 1 after runs
  with one. The launches are counted with the GPU profile on (EDEN_NX2_GPU_PROFILE) or while
  FRITCHII_DELAY_RUN_US is set (memstage_shadow.cpp's count_launch).
- Fable's reading: 5 us at kind 0 adds about 47 ms/s (9,471 runs a second); a frame 0.85 ms
  longer means parked time is fully on the path. 20 us at kind 1 adds about 21 ms/s.
- The spin is outside the run's own time ("runs ... us each on GPUMem"); counted at the end of
  the "Memory stage passes" line ("delay test FRITCHII_DELAY_RUN_US=n kind k: x/s y ms/s").
- Off (0): one relaxed load and a branch after each run.
- Measure with profile_toggle.py in one session (0 against n).

### FRITCHII_E2C_CENSUS (the "Front 0xe2c misses" line)

FRITCHII_E2C_CENSUS (live, default 0 since 2026-11-14 = no work at all on the miss path; 1 = the
line below). It cost the front about 20 ms/s at the market's 2,600 misses a second (the walk of
the noted writes and the query cache look at each miss; fable_front_drains 8.4). A miss more than
10 s after the previous counted one starts a new window, so turning it on live gives honest rates.

"Front 0xe2c misses (noted write) 10 s" (HW_GPU, Info): the macro DMA reads of the 0xe2c
occlusion macro (hash 0x3b26ab7eb3fc9863, its 0x158 read of six words; OoT macro slots, so only
with a macro switch on, as by default) that miss their page snapshot because a write the front
noted since overlaps the read's bytes (EDEN_NX2_DMA_SNAPSHOT's "noted write" misses). Counted only
at those misses (fable_front_drains 1.3 A: the counter that tells its answer from the second
answer's same-frame sample counters). Per second:
- the misses; how far behind the back was at each (ring words between `consumed` and what the
  front sent: average and maximum); the misses with a report the query cache still lists on the
  read's bytes (the hit path's own test, 12 bytes back); the misses where more writes were noted
  since the snapshot than the table remembers (the per-range answers are then "none").
- For bytes 0-8, 8-16 and 16-24 of the read (the macro compares the 64-bit words at 0 and 16;
  8-16 is the first report's timestamp): the newest noted write since the snapshot over them, by
  kind: "payload report (16 bytes)" (report Payload with the timestamp; the front had the payload
  in its registers but notes 16 unknown bytes), "payload release (4 bytes)" (a short payload
  release, its word known), "counter or sample report" (any other report type: written when the
  host query completes), "other" (puller semaphores, uploads, unknown ranges), "none" (no note
  over that range; the miss came from another one). Each as "known a/b unknown c/d": the front
  knows its words or not, the back had run it (a, c) or not (b, d) when the read missed.
- Reading it: misses blocked by payload reports with the back far behind are answer A
  (bookkeeping on the front); counter or sample reports the back had not run, or a listed report
  on the bytes, are answer B (the value does not exist yet at the front).
- With FRITCHII_FRONT_RAN_READS on, it counts only the misses that switch did not spare (with 2:
  would not spare).

### FRITCHII_FRONT_RAN_READS

FRITCHII_FRONT_RAN_READS (live, default 0 = exactly as before; the market's "Front 0xe2c misses"
line: about 90% of the noted-write misses wait for writes the back already ran). A macro DMA read
(any macro, EDEN_NX2_DMA_SNAPSHOT's 0x158 reads) that misses its page snapshot only because a write
was noted over its bytes since the snapshot (reason "noted write") reads guest memory at once,
without the drain, when all of these hold (Maxwell3D::FrontDmaRead, RanReadCheck):
- (a) the back ran every write noted over the read's bytes since the snapshot: none of those notes
  is past the back's position (`consumed`, loaded first). The note table is in ring order, so the
  scan stops at the first note at or before that position; every older one ran, the ones the table
  dropped included. If the table ends before the scan gets there (the "notes past the table"
  case), a dropped note may still be queued: it drains (GpuSplit::NotedWritesRan);
- (e) the read is on one 4 KB device page;
- (f) the page never held GPU-modified buffer data (the never-cleared GPU-written bit), or no
  indirect grid placeholder was ever handed out: a 3D inline upload with placeholders is the one
  noted write whose run leaves its bytes in the buffer cache only (the grid's DMA copy), which the
  drained path's FlushRegion would bring back. Every other noted write writes guest memory itself:
  3D reports and payload releases, puller semaphores (both through the query cache, at their run
  or when the host query or fence completes), inline uploads (guest memory first, then the caches),
  KeplerMemory data noted by EXP120;
- (c) no async report is owed to the bytes: every report issued to their 16-byte slots completed
  (Nx2AsyncReportsSettled, the rule SEM_FAST uses for semaphore words) and the newest async note
  over them ran. This covers every way a report's value lands after the back ran it: counter and
  sample reports, timed and fence releases, any accuracy or fence setting, EXP33's kept fence
  reports, EXP119's report runs (all issue on the back before it passes the report's entry, and
  complete only after the words are in guest memory). A report starting up to 15 bytes before the
  read counts;
- (b) the query cache lists no report on the bytes (the hit path's own test, 12 bytes back);
- the other miss reasons (compute and copies, unranged writers; with EXP10 the dispatch check and
  byte lists, which come after the noted-write test) are asked as for a hit and still drain.

The page snapshot is left alone: the checks hold for the bytes read, not for the rest of the page,
where a noted write may still be queued or owe a report, and the hit path asks neither (a) nor (c).
So later reads of the page take the same check (a short scan) until a drain renews the snapshot.

- With FRITCHII_MEMSTAGE_THREAD=2 the back still runs every report and semaphore entry (GPUMem only
  fills draws; a lone gather or puller release it serves runs on the back before the back passes
  it), and `consumed` is stored after the entries before it ran, so it is the right position.
  Notes and the async slots are by device address, so another GPU address of the same bytes finds
  them.
- 2 = verify: the same decision, then the read drains anyway; the bytes read at once are compared
  with the drained ones, which are used. "Front ran reads (must be 0): macro m read of n words at
  a (device d) differs from the drained read: word i x, drained y" for the first 16. A guest CPU
  write to the bytes between the two reads would also show (none expected on report memory).
- "Front ran reads (FRITCHII_FRONT_RAN_READS=n) 10 s" (HW_GPU, Info): noted-write misses checked
  a second, passed (with 2: would be served), of those drained after all for another writer (the
  EXP10 dispatch check or byte lists), and the drained ones by first failed check: a note still
  queued, notes past the table, two pages, GPU-written page, an async report owed, a listed
  report; then the compute/copy checks by cause (unbounded writer, writer range overlap, writer
  ranges past the scan, writer ranges left the ring, dispatch queued, dispatch write overlap,
  unbounded dispatch, the dispatch scan's limit passed by the stamps / refused by a block stamped
  since / by a wide write since, listed byte overlap); the reads served (with 2: that would be)
  by what was spared (a noted write, the dispatch scan's limit, both); verify compared and
  differed (must be 0, with the session total).
- Served reads count as "skipped the drain" in the "DMA snapshot" line.
- Expected at the market: about 1,500 to 2,000 of the 2,300 drains a second go (the 1,520 counter
  or sample reports and 548 "other" notes the back had run, minus the ones that owe a report or
  are listed), 45 us each: about 70 to 90 ms/s of front waiting.

The dispatch scan's limit (any read, with the switch on; EXP10's "past the scan"). EXP10's
dispatch check compares the dispatch writes the back recorded since the page was last checked, at
most 2,048 of a ring of 8,192; past that, or when the ring dropped some, it answers "may write"
and the read drains (about 1,660 a second at the market once the noted writes pass, because the
pages are read rarely). By then the check has made every dispatch sent since the snapshot run
(it waits for that, or answers "queued" first), and every write the back recorded for them
(GpuSplit::AppendDispatchWrite: storage and texture buffers the buffer cache marks written,
written images' guest ranges, indirect arguments, QMD releases) stamped its 64 KB blocks with the
record's position before the record was counted. So the read passes when no block stamp of the
page is at or after the snapshot (GpuSplit::DispatchStampsMissSince): the answer the scan would
give without its limit, or stricter (the table is hashed by block, 65,536 slots; a collision only
makes a stamp newer). It then counts as the scan's miss (the page's checked count moves on).
- Still drains: a block stamped since, and a write that stamps everything (an unbounded or
  unconfirmed dispatch, an image without a guest range, a range over 4 GB). Unbounded compute and
  copy writers, writer ranges, unranged writers and dispatches still queued are asked before and
  drain as before.
- Not the GPU-written page bit: it is set for the buffer cache's GPU-modified ranges only, not for
  images a dispatch writes (texture cache) nor QMD releases (written into guest memory by the
  fence callback), and it is never cleared, so a page written once would never pass.
- The stamps start over with each split session (positions start at 0).
- 2 verifies these reads as the others.

### FRITCHII_ACQUIRE_MET_BY (the "Acquire resumes met by" line)

FRITCHII_ACQUIRE_MET_BY (live, default 0 since 2026-11-14 = one relaxed load per resume; 1 = the
line below). On, each resume pays a page table lookup, the release tables, a scan of the noted
writes and a clock read. A resume more than 5 s after the previous counted one starts a new window.

"Acquire resumes met by" (HW_GPU, Info, every 5 s; fable_front_drains 2.3 items 2-3): for every
suspended semaphore acquire that resumes (Puller::ResumeAcquire), the release that met it, as the
front's tables tell, by kind, each as "known/unknown" a second: whether the front knew the value
when it sent it. Two groups: resumed by the back's latch (it ran a release that met the value),
and by the other routes (word read, front answer, poll drain, lagged check).
- Which release, in order: with EXP13 on, the newest release the back made for the word since
  the suspension (the QMD copy's table, as its own "met by" asks), when it meets the acquire or is
  a reduction; else the release recorded for the word (GpuSplit::NoteRelease: puller releases,
  3D payload releases and reports, compute 0x6c3 releases) when its payload meets it; else the
  newest write noted over the word in the last 2^18 ring words; else "not seen by the front" (the
  guest CPU, or a writer the front does not note). It looks, it does not prove: a table slot
  shared with another word, or an older release with the same payload, can name the wrong one.
- Kinds: puller release, 3D payload release (4 bytes), 3D payload report (16 bytes), compute
  0x6c3 release, QMD release, QMD reduction, copy semaphore, counter or sample report, other
  noted write, not seen by the front.
- Known at send: payload releases the front records (it had the payload in its registers), QMD
  and copy releases while the QMD copy predicts them (EXP13 2-4; reductions never: bounds only),
  noted writes with known words. Counter and sample reports never.
- Reading it: latch resumes by releases known at send are what resolving acquires at the front
  (resume when the newest queued release over the word has a known value that meets it) would
  take off the front's blocked time.
- With FRITCHII_ACQUIRE_TABLE_FIRST on, its resumes count under "the other routes".

### FRITCHII_ACQUIRE_TABLE_FIRST

FRITCHII_ACQUIRE_TABLE_FIRST (live, default 0 = exactly as before; fable_front_drains 2.3 item 2
and 6.3). A suspended acquire's poll (Puller::CanResumeAcquire) asks the front's release table
right after the latch: before the word read (SEM_FAST's read, which waits for the noted writers
of the word when it can't overlay them) and above the unmet-poll throttle (16 polls or 1 ms, and
EXP12=5/6's 1 ms lagged check), which held MetByQueuedRelease back until then. Every poll point
the front has asks (each command list's end, the idle loop), so the first poll after another
channel sent the release asks too. Never inside a list: that would change the channels' order.
- 1: met when the newest release recorded for the word (GpuSplit::NoteRelease) is a puller
  release (WriteLong, SEM_EXECUTE) or a 3D payload release (4 bytes) or report (16 bytes), all
  known at send, still queued, with a payload that meets the acquire and nothing sent after it
  that may write the word: MetByQueuedRelease's rule, which the acquire's parse and the throttled
  poll already use. The resumed channel's entries go into the ring after the release, so the
  back runs the release first, and the answer is the one a later poll would give from the word
  once the back ran it. Off with delayed fences (the release lands at a later fence), and for the
  16-byte report with accurate queries (its operation waits for a fence too). Compute 0x6c3
  releases stay with the throttled path.
- 2: also the QMD feed (EXP13=3/4, MetByQmdPrediction) at the same place, before the word read
  (today it is asked after it, also above the throttle). Its own rules, unchanged. When a
  prediction can be wrong: the QMD words the front's copy read differ from what the back reads at
  the launch (a writer of the QMD the copy did not see, an overlay from guest memory), the
  copy's scheduler differs from the back's (dependency counts, chained dependents, decrements),
  or a max release or reduction (a bound, not a value). The checks: the back compares every
  release it makes with its record (the feed turns itself off for the session at the first
  mismatch, an ERROR line) and the feed audits each acquire it met ("acquire wrongly met"). Both
  find a wrong answer only after the resumed channel ran, so 2 stays a test mode. When the table
  does not meet, the poll asks the feed again after the word read (that read may have waited), so
  the feed's gate counters count both asks.
- Must be 0 (1's resumes): each resume watches its word with a waiter of its own
  (Nx2SemaphoreWaiters, added before the last check); once the back ran the release (`consumed`
  past it, its async report settled) the waiter must have latched, that is the back wrote a value
  that meets the acquire. If the word already met when the waiter was added (the release landed
  first), the watch is not judged. 16 watches at most; one not settled within 2 s is dropped
  ("unresolved"). "Acquire table first (must be 0): device d op o value v: the release it was
  resumed on (payload p, ring r) ran, and no write that met it was seen; the word is w now" for the
  first 8.
- "Acquire table first (FRITCHII_ACQUIRE_TABLE_FIRST=n) 5 s" (HW_GPU, Info): resumes a second by
  kind (puller release, 3D payload release (4 bytes), 3D payload report (16 bytes), QMD feed
  exact, QMD feed other), the watches (met at the write, landed before the watch, unwatched,
  unresolved after 2 s, open) and the must-be-0 count with the session total.
- Expected: small. At the market the 256 3D payload releases and 228 QMD releases a second that
  met parked acquires were resumed by the latch: the back had run them by the next poll point,
  which the front reaches only at a list's end. What 1 can take are the polls where the release
  was queued but the word read could not show it, or waited for it (a 16-byte note over the word,
  the throttle).

### FRITCHII_GATHER_ONE_LOCK

FRITCHII_GATHER_ONE_LOCK (live, default 0 = exactly as before; fable_front_drains 8.2 G1). The
game's collected writes, applied at a gather on the back (RasterizerVulkan::OnCacheInvalidations,
EXP88's path), took the buffer cache's lock and opened and closed a memory seam bracket
(MemStage::FillPlay) once per range: 63,000 to 73,000 ranges a second at the market, 2.5 ms/s in
the lock and 7.3 ms/s closing brackets (5.4 of it one shared atomic in Fill::~Fill).
- 1: two passes over the gather's ranges. First the texture cache's WriteMemory and the pipeline
  cache's InvalidateRegion for every range, as before. Then the buffer cache's part (the reach
  test, NoteWriteUnlocked or WriteMemory, the verify check) for every range, under one hold of its
  lock and one bracket, taken at the first range that needs them (a gather whose ranges all skip
  takes neither, as before).
- Why it is exact: each cache gets the same calls in the same order. Across caches nothing
  depends on order. The texture and pipeline calls change only their own state. The buffer
  cache's WriteMemory changes its own state, guest memory (a page download, after the claim) and
  records commands; the other two read none of that. The shared part is the device pages' cached
  counts, which all three only lower here: the sum does not depend on the order, and a page is
  unprotected only when its last user lowers it, in the same call that marks that user's data
  CPU-modified. The bracket plays its entries before anything else is recorded (SeamHook::OnUse),
  so one bracket records the same commands in the same order as one per range. Other threads that
  want the buffer lock wait for the whole buffer pass instead of one range.
- What changes besides time: the seam's counters (fills per site; one Invalidation fill per
  gather) and, with packet ticks, one operation packet number per gather instead of per range.
- Count: "N gathers under one buffer lock (FRITCHII_GATHER_ONE_LOCK)" in the "Constant buffer
  path (EXP88=n, 5 s)" line.
- Expected: about 8 to 9 ms/s off the back at the market (stage off), the per-range lock and
  bracket close; the bracket's atomic runs about 1,400 times a second instead of 68,000.

### FRITCHII_GATHER_EMPTY_TEST

FRITCHII_GATHER_EMPTY_TEST (live, default 0 = exactly as before; fable_front_drains 8.2 G3). About
20,000 gathers a second reach GPU::InvalidateGPUCache at the market, 93% of them with nothing
collected; asking the four game cores' managers (two mutexes each, a std::function, a vector)
was 4.4 ms/s of the back.
- 1: before the gather (after run-ahead's own check), each manager answers
  GPUDirtyMemoryManager::NothingCollected, one lock-free 8-byte load of its `current` word. When
  all four say nothing, the gather is not made. Nothing is added on the game threads' side: the
  flag is `current` itself.
- Why it is exact: a manager holds a collected write that no gather took only while `current` is
  not the empty word {~0, 0}. Collect leaves `current` holding the write (its CAS) or, when it
  moves to another 2 KB block, pushes the old word to the back list and stores the new one in the
  same locked section; only Gather stores the empty word, in the locked section that also takes
  the back list. A write collected before the test (any chain of locks or atomics from the
  game thread's Collect to the back, which the game's submit always is) is, by the atomics'
  coherence, seen by the test as that Collect's store or a later one: a later Collect (not empty)
  or a Gather's empty word, which took it. A write collected after the test is taken by the next
  gather, as today. A second gatherer (a FlushRegion off the back) no longer waits for another
  one's collection in progress; it never waited for that one's application either, and the
  pending-write list (EXP49) still closes the gap for landing downloads.
- "Gather empty test (FRITCHII_GATHER_EMPTY_TEST=n) 10 s: gathers asked N/s, skipped as empty
  M/s (P%)" (HW_GPU, Info).
- Expected: most of the 4.4 ms/s.

### FRITCHII_FENCE_COMMIT_FAST

FRITCHII_FENCE_COMMIT_FAST (live, default 0 = exactly as before; fable_front_drains 8.3 row 3).
Every fence's FenceManager::CommitCacheFlushes takes the texture cache's lock and then the buffer
cache's (with a memory seam bracket) to find out whether a download waits to be committed:
10,606 fences a second in the market's compute stretch found none (5.3 ms/s).
- 1: each half is skipped, lock and bracket included, when its cache's note says it has nothing:
  TextureCache::MayHaveAsyncFlushWork (a download or download buffer queued since the last commit)
  and BufferCache::MayHaveUncommittedFlushes (a fence download range queued, QueueFenceDownload).
  Both notes are kept always (also at 0): raised where the work is queued, under the cache's
  lock, and set again from the containers under the lock after each commit.
- Why it is exact: a note is raised in the same locked section that adds to the containers its
  cache's test reads (texture: uncommitted_downloads, uncommitted_async_buffers; buffer:
  uncommitted_gpu_modified_ranges, which AccumulateFlushes moves to committed_gpu_modified_ranges)
  and lowered only under the lock from the containers themselves, so "note down" means the
  containers were empty at that point. With the containers empty the half does nothing but take
  the lock and open an empty bracket (should_flush false, nothing committed). Work queued on the
  fencing thread is seen in program order; work queued on another thread that does not happen
  before the fence is ordered after it, as when that thread takes the lock second. The skipped
  bracket held no entries; pending entries are played before anything else is recorded either way.
- "Fence commit fast (FRITCHII_FENCE_COMMIT_FAST=n) 10 s: commits N/s, texture half skipped T/s,
  buffer half skipped B/s" (HW_GPU, Info).
- Expected: about 3 to 4 ms/s (both lock takes at most fences in the compute stretch).

### FRITCHII_CB_WORDS_BLOCK

FRITCHII_CB_WORDS_BLOCK (live, default 0 = exactly as before; fable_perdraw_b 3.1). About 1.36
million 3D constant buffer data words a second (0x8e4 to 0x8f3: the command list's increasing
uploads, the draw macros' base vertex and instance words) each went through the split's ring as
its own Method entry.
- 1: Maxwell3D::FrontWrite puts such a word in the open CbWords block (GpuSplit::CbWord) as the
  very Method word its entry would have carried (Direct and last-call from a macro, the command
  list's last-call bit otherwise), up to 64 words a block. Everything else in FrontWrite stays:
  the front's register copy and offset per word, the NoteWrite, RangeLastWrite and
  NoteMethodWrite per word, each at the word's own position (every word is committed as its
  entry was). The block ends at anything else reserved (another entry, an engine switch, held
  back work first), at a publish, and when the sent position is read (SentPosition). The back
  runs the block as one entry: its words go through Execute as the entries in a row did, whose
  constant buffer gather takes them in one CallConstBufferWords (or word by word where the
  gather is off). The memory stage follows the block as one entry, each word through the same
  CallMethod as a Method entry.
- 2: as 1, and each block carries the front's address of its first word. The back compares it
  with its registers after the sink (the address the front's notes used), then applies the
  block, records the offset, the 16 data registers, their shadow RAM words and the dirty flags,
  checks memory holds the words' values (when it wrote them itself, not served by the memory
  stage), puts the state back and applies the words one entry at a time, and compares. Not
  applied twice: one word, a deferred draw waiting (it can draw once), the gather off.
- Why it is exact:
  - Order and position: the words sit in the ring in the same order between the same other
    entries. The back stores `consumed` only at entry ends, so a note at a word's position inside
    a block counts as run exactly when the whole block ran: waits end at the block's end, never
    before the word ran. The notes themselves (range, data, position) are the same calls.
  - Dirty flags: each word still goes through ProcessShadowRam and ProcessDirtyRegisters, in
    order; the first word issues a waiting deferred draw before any word is uploaded, as its own
    entry did. Checked by =2.
  - Same-value skip: constant buffer data is executable, so neither the front's skip nor the
    back's (state registers only) ever applied; EXP107's memory compare runs per
    CallConstBufferWords call on the back, in order, as with the gather (a call over different
    words only changes whether equal bytes are written again).
  - Shadow RAM: the block carries the raw argument as the Method word did; the back and the stage
    apply the shadow RAM per word. The mode cannot change inside a block (shadow_ram_control is
    another method, which ends it).
  - Macro reads of the offset and data registers read the front's copy, which FrontWrite still
    updates per word. Macro DMA reads see each word's note at its position, as before.
  - EXP88: WriteBoundConstantBuffer gets the same calls the gather made; its device ranges merge
    where they touch, and FlushCaching invalidates them at the next draw, which cannot come
    inside a block.
  - FRITCHII_FRONT_RAN_READS: the noted writes it scans (reports, inline uploads) are never made
    for constant buffer words, and a page snapshot's position is SentPosition, which ends the
    block, so no snapshot or noted write position falls inside a block.
  - Packet ticks: no draw, closure or seam bracket can fall between a block's words, so the
    words' ranges reach the same draw's FlushCaching (back) or the same fill (stage) as before.
  - Memory stage (FRITCHII_MEMSTAGE_THREAD=2): the block is one entry, served or lent as a whole.
    A word that makes it hard (a watched page, GPU-written data, EXP107) makes all of it hard:
    the back then writes every word, the ones the stage wrote already being the same bytes.
- "CB words block (FRITCHII_CB_WORDS_BLOCK=n) 10 s: front blocks B/s with W words/s (W/B a
  block), ring entries saved S/s, blocks ended full F/s; ranged write overflows O/s; back
  totals: blocks, words; must be 0: broken blocks, bad blocks; verify (=2): addresses checked,
  differ; applied both ways, differ, not applied twice" (HW_GPU, Info). Broken blocks (a commit
  after a block that did not end it), bad blocks (words that are not constant buffer data) and
  both differ counts must stay 0.
- Expected: the back's constant buffer gather already took words that arrived together, so the
  back saves the per-entry decode and the splits at early publishes (the front published every 8
  words to a waiting back); the front saves the engine select and reserve per word. The ranged
  write overflows are not expected to drop: RangeLastWrite already extends the last range for
  each next word, so they count runs to other places (the line shows them for the A/B).

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

EDEN_NX2_GPU_WAKE=1 (live, ON by default since 2026-11-14, built in with EXP12 = 4; ported): the GPU
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

### EDEN_NX2_SWEEP_UNMAPPED

EXPERIMENT, off by default (2026-11-14): EDEN_NX2_SWEEP_UNMAPPED (read at startup;
TextureCache::SweepUnmapped). An unmap of GPU memory flags the images in it Remapped and
CpuModified; lookups skip them, so most sit until the collector's age limit (50 frames, 10 near
the budget). Ocarina of Time at 3440x1440 unmaps and maps its render targets again every frame
or two in the Castle Town guard house and the dead sets overflowed a 12 GB card. 0 = today. 1 =
at each frame end, delete flagged images untouched for 2 frames, without a download, when
nothing can still need their contents: still CpuModified (nothing uploaded into it or wrote it
whole since the unmap, so neither the collector nor a guest read would ever download it), no
aliases, not among the last 8 images shown, not a bound render target, no download queued, no
decode in flight, not sparse or unbacked, not taken by EXP121. 2 = verify: deletes nothing,
marks the images 1 would delete and counts what reached them later (found again by EXP76 & 1,
uploaded into, and of those GPU-written before the unmap, presented, written whole, deleted
unused by another path). The "Texture cache last 600 frames: GPU memory unmaps" line (with
EDEN_NX2_TC_LOG or this switch) counts unmaps that hit images, images newly flagged, found
again, uploaded into, downloaded, deaths by path (collector, join, unmap, other) with how many
were unused since the unmap, frames and KiB of the collector's unused deaths, and the memory
flagged images hold (now, average, peak). With 1 or 2 a second line, "unmapped sweep", gives
deletions and MiB, entries looked at, and 2's counts. Judge with 2 first: "uploaded into ...
GPU-written before the unmap" and "presented" should stay 0 (an upload into a deleted image's
replacement may be skipped as stale where the kept image would load guest memory, both garbage
next to the lost GPU data), then 1 against 0 for device memory and fps on 8 GB cards. With
EDEN_NX2_IMAGE_HASH_LOG, "unmap sweep" in the 10 s delete lines shows sampled images made again
where the sweep deleted one.

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

### FRITCHII_BANK_CENSUS

FRITCHII_BANK_CENSUS (live, default 0; counters only, nothing the check does changes;
fable_perdraw_b cut 5 step 1). The tracked native bank check (EXP9) compares a page on every bind
for HotFrames frames after any change ("hot"), because some writer of the game's banks sets no
flag: 637,555 hot compares a 5 s window found 328 changes (21.7 ms/s of compares on the back).
The census says, for each change the check finds, what touched the page since its last compare.
- 1: changes by the reason that found them, and for the unflagged ones (hot, frame check, verify
  only, EDEN_NX2_BANK_SKIP_VERIFY's missed) the evidence: binds since a collected CPU write or
  buffer cache write flag was last taken on the page (a trapping writer marks before it stores:
  its bytes can land after the compare that took the flag and after the one recheck), an upload
  (UploadMemory) that read the page from guest memory after the last compare (a guest store
  between that compare and the upload: the host buffer may hold it already), a GPU write mark
  ever on the page, and "unseen" when none of these. Per constant buffer index (c3+ are the
  game's).
- 2: 1, and the write ring turned on (all emulator writes to game memory, with their site): the
  newest ring entry over the page since its last compare, by site and write function. The ring
  costs every emulator write a few stores: fps are not comparable. Not together with
  FRITCHII_MEMSTAGE_READ_LOG (the ring is left on until the census goes below 2 only when the
  census turned it on).
- When no change is found it adds one flag take and two stores per compared page (one more
  shared load at 2), and one load per upload. At 0, one load per bank bind.
- "Native bank census (FRITCHII_BANK_CENSUS=n, 10 s): changes: collected C, buffer cache W,
  recheck R, CPU-modified M, hot H, frame check F, verify only V (slot skip missed S); unflagged
  U: flag taken <=4 / <=32 / <=256 / more / never binds before a / b / c / d / e (late flag L, of
  them buffer cache B), uploaded since the last compare P (that compare CPU-modified Q), ever
  GPU-written G, ring writer X (ring wrapped Y), unseen N (first compare this frame N1); slots:
  cK n ... | site / via: n ..." (HW_GPU, Info). Counts start at the census' first bind; pages
  first seen before it count their flags from their next compare.
- What it must show before a slot can drop the hot rule (FRITCHII_BANK_FLAGGED, not written):
  "unseen" 0 and no ring writer that sets no flag. Late flags are closed by a flag set after the
  store lands; uploaded-since changes by marking the page for a compare at its next bind.

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
back verifies every release either way (QmdPrediction::Verify). 0 = refused as before. EXP127 = 2
adds sources under this switch (EXP127).

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

Without it, and without FRITCHII_MEMSTAGE_TICKS, EXP123 = 0 runs the seam lean (no switch): a fill
bracket only counts its depth (no arena claim, no watch on the scheduler, no site counts), and the
stats counts are skipped. Entries that run as pushed (EXP123 = 0, and outside a deferring bracket
at 1 to 3) go straight to the player's typed call without being written to the scratch arena.
The same calls in the same order: nothing defers at 0, so what the lean drops is the deferring
modes' checks and the stats. With SEAM_STATS set, EXP123 = 0 keeps the full bracket, so its line
still counts the records inside fills.

### FRITCHII_MEMSTAGE_READ_LOG

FRITCHII_MEMSTAGE_READ_LOG=N (boot, diagnostic, default 0 = off; the memory stage's sub-step 2b,
fable_memstage_step2_spec.md 5.3): with EXP123 = 2 or 3, every Nth draw, dispatch and operation
packet logs what its fill read from game memory (uniforms copied to the stream, buffer uploads
with their staged bytes, cached ranges bound without an upload, storage and table addresses and
size words, compute pointers, the inline page write), and at the old read position (a draw's or
dispatch's first play; an operation's end) reads it all again: bytes and values are compared,
bound ranges are asked whether the CPU wrote them since. Differences are counted by kind; the
first 16 of each class (below) are logged as "Memory seam (5.3) <class>" with the kind, address,
size, the changed bytes, the device range, the place, the writer, the pending mark answer, the
gathers since the fill and the programs. Reads past 1 MiB a packet are not compared (counted as
"not logged"; the ranges still count for the over-read test). The counts are in the "memory seam
read log" line of EXP123's stats every 10 s, with the reads compared by kind. N = 8 is the spec's
value; N = 1 checks every packet (slower). Not FRITCHII_MEMSTAGE_VERIFY, which is 2c's by-value
checks (bit flags).

Game threads write the pages the cache reads all the time, so the gate is "0 unexplained", not 0
(fable_memstage_readlog1.md; step 6 spec 10). With READ_LOG on, from boot, every write emulator
code makes to game memory (device and GPU memory managers, span stores, the GPU write listener,
run-ahead release words, Core::Memory's unsafe writes) goes to a ring of 4,096 entries with its
site, thread and sequence (vk_mem_stage.cpp, "Who wrote", lists every hook). Each difference gets:
- a class: "emulator write, download landing" or "emulator write" (the ring names a write newer
  than the fill over the changed bytes; counted by site), "game write, marked" (a game-side mark
  is pending, GPU::ForEachPendingCpuWrite; for a bound range, the tracker holds it), "game write,
  mark already taken" (a tracked read, an upload or the inline page write, with no mark left: the
  store landed after its mark was gathered, and nothing repairs it), "untracked read changed" (a
  stream uniform or a value: no tracker, no mark expected, the next fill reads it again), "gather
  on the back inside Configure" (a bound range, and the drawing thread applied a gather since the
  fill; the first gather's stack is logged), or "unclassified" (unmapped, or the ring wrapped);
- a place: "over-read" (an upload's changed bytes outside every range the packet binds) or
  "inside a range the packet binds" (for a dispatch, whole storage buffers).
The "memory seam read log classes" line has the counts by class and site, over-read and changed
inside a bound range, ring wraps, and on the holding thread the gathers applied and FlushRegion
calls by cache. The "read log gate" line says passed or FAILED: it fails on unexplained (emulator
writes that are not download landings + unclassified) and on ring wrapped (the run is not valid;
make the ring larger). "mark already taken", "untracked read changed" and "changed inside a range
the packet binds" are reported beside it and are not failures (fable_memstage_readlog2.md). Off,
every hook costs one relaxed load and a branch.

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
- With FRITCHII_MEMSTAGE_AHEAD (the window open, the ring running) each thread stamps with its
  own packet instead of the newest number anyone opened: GPUMem with the draw it fills or the
  operation it runs (under its mark), the back with the packet it holds or fills and after it
  the last one while not published. The check then also counts "stamps on a number already
  published" (must be 0: such a use would be freed early), "splits that raced the tick advance"
  (a split by one tick of a number above what the closing tick's close saw: a stamp taken on the
  other thread between a close and the tick's advance; reported) and GPUMem's "epoch reads
  racing a close" (reported). Carried ranges are kept until older than the stamp ring. Late is
  a warning with two threads (it is the conservative side).

### FRITCHII_MEMSTAGE_SHADOW

FRITCHII_MEMSTAGE_SHADOW (boot, default 0; the memory stage's piece 5b,
fable_memstage_step6_spec.md 6.2): the future memory stage's inputs, run on the back thread and
compared, never used. Off, each hook is one test of a flag (one per ring entry, per draw, per
FlushCaching, per channel bind).
- 1: before the back executes each ring entry, the stage's code classifies it (pass, work or
  level: spec section 3, the Call entries by their stage tag in header bits 32-39) and applies
  pass and work entries to its own copy of each Maxwell3D's registers, shadow RAM, constant
  buffer bindings, root tables and draw manager. The draw manager is the same code on both
  (DrawManager is a template on the engine, instantiated for Maxwell3D and for the stage's copy),
  so the stage finds its draws with the back's own lines. Constant buffer words are computed
  (range and offset), not written. After a level entry the copy is taken from the back again,
  with the back's waiting method sink applied. At every draw the back issues: (a) the stage drew
  at the same entry; (b) the stage's state at its draw equals the back's (the registers the
  buffer cache reads through CacheInputs, the draw's registers, the constant buffer bindings, the
  root tables, the draw state, execute_on, the channel, the engine; every dirty flag the stage
  raised is raised on the back); (c) at the pipeline lookup, the facts table (keyed by the program
  registers) against the pipeline's facts digest; (d) at FlushCaching, the device ranges of the
  stage's constant buffer words against the back's invalidation ranges.
- 2: as 1, and (b) compares the whole register file and the shadow RAM (slower).
Every 10 s, "Memory stage shadow": entries/s pass, work, level; level runs/s and resyncs/s;
draws/s compared (served, render-enable off, flushed from the method sink) and in level entries;
the draw kinds of spec 3.4 (indirect, list, GPU count, texel buffers, diagnostics, host
conditional); facts misses/s; watched-page hits (constant buffer words onto image pages or the
texture/sampler tables: counted, and the entry is treated as level). "Memory stage shadow
compare" gives a (missing, extra, position), b (dirty flags apart), c (conflicts; program keys
that changed code; pipelines without a compared draw), d (missing, extra; flushes, strict ones)
for the window and the session: **all must read 0**. "Memory stage shadow inside draws": what
runs on the back between a draw's fill and its record, FlushRegion by the cache halves that ran
(texture, buffer, query; the buffer half must stay 0 in play) and collected game writes applied.
"Memory stage shadow level entries by kind and channel": the top 12, named by the 3D method, the
engine, the closure (`call +0x...`: its address in the executable, for the symbol file), or the
constant buffer case; this table is the go/no-go for piece 6b. The first 16 differences of each
compare are logged as "Memory stage shadow (a/b/c/d)", (b) naming the registers. The stage
implements the default constant buffer write; with EXP107 on, constant buffer words are level.
Not covered yet: the watched pages leave out shader code and query addresses.

Piece 5c (fable_memstage_6b_plan.md sections 2 to 4 and 6.1), in every mode:
- Classes: pass, work, level ("hard": needs the buffer cache on the back) and neutral (another
  engine's plain register write). clear_report_value, fragment_barrier, tiled_cache_barrier and
  report_semaphore.query are pass, except a fence report (Release, short query), which is hard.
  render_enable.mode is work when the registers decide it (an override, or mode False / True: the
  stage sets execute_on), else hard. Other engines: only their executable methods are hard. The
  front's EXP105 command list check is tagged pass; the list draw's address and keep id are a
  tagged closure (work); the 3D macro data closure (dma_pusher.cpp, IsMemoryDirty) is not sent
  when the front runs the macro (front macros and the native Turing macro engine: the back never
  reads it then). A new tag, StageTag::Cache (hard, no copy after), is on no closure yet.
- Follow: hard entries are applied to the stage's copy as well (the store; falcon[4]'s scratch
  word), so the copy is taken from the back again ("full copy") only after an entry it cannot
  follow: an untagged closure, an unknown tag, a 3D macro run on the back, methods before any
  engine entry. After an undecided render_enable.mode the copy takes execute_on from the back, and
  after a draw the back issued in a hard entry, an indirect draw or a GPU-counted draw the draw
  manager's scalar state, both at the next entry.
- Runs: a run starts at a hard entry and ends after the last hard entry reached through pass and
  neutral entries; a work entry or a draw ends it.
- Compare b's dirty flags: only flags 15 to 48 (the buffer cache's: VertexBuffers, VertexBuffer0
  to 31, IndexBuffer). The copy keeps them as the cache's own state (its stores raise them, a full
  copy takes the back's, what the back's fill clears is cleared on the copy), and they must be
  equal at each draw both ways, or a subset of the back's when the state tracker raised them since
  the back's previous draw. Counted per flag.
- 3: as 2, and the buffer cache's fills read the stage's copy outside runs (CacheInputs pointing
  at the copy's registers, bindings, root tables, draw state and flags): at every draw the back
  issues outside a hard entry whose shadow draw matched, that is not indirect, and whose copy has
  not changed since the shadow's draw. Otherwise the fill reads the engine as before. The cache's
  flags 15 to 48 are then the copy's: at a run's start the back's are set to the copy's, inside a
  run the copy takes the back's at every entry. Needs FRITCHII_MEMSTAGE_FULL_BINDS=1 (the
  scheduler's flag raises on the back are dropped at the hand-over); while it is off, 3 runs as 2.
  The flag part of compare b is skipped at 3. A check of the stage's inputs with the real
  picture: flash-watch route, compare b and the seam's own counters must stay clean.
Lines every 10 s, with the others: "Memory stage 5c runs" (runs/s, entries per run, neutral
entries/s, served draws between two runs as a histogram 0, 1, 2-3, 4-7, 8-15, 16+, full copies/s
and MB/s, the takes, 3D macros run on the back/s, fence reports/s); "Memory stage 5c runs by first
hard kind" (top 12); "Memory stage 5c full copies by cause" (top 8); "Memory stage 5c
render_enable.mode by override and mode" (must show only False / True or an override: the
undecided count must be 0); "Memory stage 5c flags 15-48" (compares, mismatches by flag, at 3
where the fills read from and why not the copy, and the OoT tracker record check below); and
"Memory stage 5c gate: pass/fail", the stop thresholds of plan 6.1, read in the market's window:
runs above 15,000/s, full copies above 2,000/s, or any undecided render_enable.mode in the session.

The OoT temporal trackers (TrackOcarinaCamera, TrackOcarinaActor; plan 5.1) no longer ask the
buffer cache after the fill when the draw has a packet (EXP123 >= 2): while the trackers are on
(published at each frame end), the draw's fill adds a record of the vertex stage's uniform slots
3, 4 and 6 (address, bound size, and whether the bytes the trackers read are GPU-modified) to the
packet, and the trackers read that. Without a packet (EXP123 < 2, texel buffer draws, the first
frame) they ask the cache as before. With FRITCHII_MEMSTAGE_SHADOW on, every such draw compares
the record with today's query ("OoT tracker record vs today's query": mismatches must be 0; the
first 16 log "Memory stage shadow (OoT trackers)"). The shadow's "diagnostics" draw kind is now a
frame dump or a trace only.

### FRITCHII_MEMSTAGE_THREAD

FRITCHII_MEMSTAGE_THREAD (live, read at a frame end, default 0; the memory stage's step 6, pieces
6a and 6b, fable_memstage_step6_spec.md 6.1, 6.3 and section 5): 1 runs the buffer cache's two big
fills on a thread of their own by hand-off; 2 lets that thread read the ring itself (below).
- The thread, "GPUMem", exists while the GPU thread split runs (started and stopped with the
  back) and sleeps at 0. It is not started with FRITCHII_MEMSTAGE_SPLIT_ALLOC=0 (logged).
- 0: the fills run on the back (EXP123 = 2's early fill, as before).
- 1: the draw's fill (MemStage::FillDraw) and the dispatch's (FillDispatch), and the buffer
  recycler's idle destroys, run on GPUMem: the back posts the fill, GPUMem runs it, the back
  waits (about 2,000 pauses of spinning, then a sleep). Same order and timing as 0, another
  thread. The buffer cache's lock goes with the fill. Every other cache call stays on the back,
  where it was. A sync inside a fill (finish, wait, submit for host read, the multi-range bind)
  is posted back to the back, which serves it inside its wait. Needs EXP123 >= 2 (so
  FRITCHII_MEMSTAGE_TICKS >= 1) and the buffer cache's own allocator; without them, or without
  the split, it runs as 0, logged once.
- Expect fps lower than 0: two thread wake-ups a draw. The point is the thread rules, not speed.
- "Memory stage link (FRITCHII_MEMSTAGE_THREAD=1)" every 10 s: hand-offs a second (draw fills,
  dispatch fills, idle destroys), the back's wait in ms/s and per hand-off, GPUMem's run time,
  sleeps on both sides, served syncs a second, and the counters that must be 0: a buffer
  allocator call, a buffer staging request or a seam push by a thread that is not the cache's
  user; a VMA heap budget read off its allocator's user; a scheduler or descriptor queue use on
  GPUMem; an entry GPUMem would run at once (served on the back instead); an entry run at once
  while entries are pending; the memory player reached on GPUMem; the front's waits reached on
  GPUMem; a fill handed off while the back did not hold the buffer cache's lock. The first 16 of
  each log "Memory stage link (must be 0)" with the place. The seam's
  own 5.1/5.2 counters (EXP123's line) must stay 0 too.
- EDEN_NX2_GPU_PROFILE: GPUMem reports as "GPU memory thread" with its "memory fill" zone (the
  fill ms/s of EXP123 = 2, now off the back) and "memory sync wait"; the back shows "memory
  hand-off" for its waits. FRITCHII_BACK_READ_AUDIT adds a "Stage read audit" table for GPUMem.
  The hitch line, the stall snapshot and the CPU trace list GPUMem as a thread of its own.
- Not pinned to a core (spec 5.8, until step 7 measures it).
- 2: the ring-fed stage, level mode (piece 6b, fable_memstage_step6_spec.md 6.3 and
  fable_memstage_6b_plan.md 6.2; the code: memory_stage_link.cpp, memstage_shadow.cpp, ring B in
  vk_mem_stage.cpp). GPUMem reads the ring itself, before the back, with its own cursor
  ("staged"); the back runs only what GPUMem passed on. GPUMem runs piece 5c's code on every
  entry (classify, apply, follow) on its own copy of the 3D registers, and that copy is what the
  fills read:
  - a draw it serves is filled on GPUMem right where its draw manager issues it (the pipeline's
    facts come from a table the back fills; constant buffer words are written to game memory by
    GPUMem, their buffer half applied before the fill), and the packet goes to ring B (4 MiB);
    the back takes the packet at its draw and plays it, without entering the buffer cache;
  - everything GPUMem does not implement is a level run: the cache, its flags and the words'
    ranges are lent to the back, which runs those entries exactly as today, then GPUMem takes the
    flags back (and a full copy only after a closure it cannot follow). Indirect and list draws
    (list draws are served with the window open, FRITCHII_MEMSTAGE_LIST_DRAWS below),
    GPU-counted draws, texel buffers, a facts miss, a frame dump or trace, host conditional
    rendering and constant buffer words onto a watched page make their entry level (a lone
    gather does not with FRITCHII_MEMSTAGE_GATHER=1 and the window open: GPUMem comes level at it
    and the back runs it there, below);
  - level mode (AHEAD = 0): GPUMem works only while the back is parked (it ran everything passed
    on and is not in its idle hook), so the two threads still take turns at every draw. Expect fps
    lower than 0; the point is correctness and what a run costs. Step 7 opens the window.
  - Needs what 1 needs, plus EXP123 = 3 (values by value: a served draw's texture handles are
    read by GPUMem at its fill and come in the packet) and EDEN_NX2_EXP53 off (it runs as 1
    otherwise, logged once); forces FRITCHII_MEMSTAGE_FULL_BINDS on. An exception on GPUMem ends
    the ring cleanly where the back is parked and THREAD = 2 runs as 1 for the rest of the
    session (logged, counted).
  - Ending the ring (2 to 1 or 0, at a frame end) hands the cache's flags and GPUMem's last
    constant buffer ranges back to the back, and plays whatever is left in ring B. Step 7's
    window is FRITCHII_MEMSTAGE_AHEAD (below).
    The constant buffer write modes other than the default (EXP107) make every constant buffer
    entry level. EXP118 (delta draws) is untested with it.
  - What the back would do in the buffer cache inside a served entry (a read flush from a
    texture upload in Configure, its gather, a MustFlushRegion question) runs on GPUMem as a
    served call while the back waits, counted by cause.
  - "Memory stage ring" every 10 s (with the link line): entries a second by class (pass, work,
    level, neutral), runs a second with entries per run and microseconds per run (GPUMem, from the
    lend to the return; the back, from the lend seen to its next packet), packets a second and
    their size, the back's waits for the stage (times and ms/s), GPUMem busy and its waits by
    reason (the front, the back, runs), full copies and flag hand-overs a second, served calls by
    cause, level draws by kind, and the verify. Piece 5c's lines (runs by first hard kind, full
    copies by cause, render_enable.mode, ...) keep coming every 10 s, from GPUMem's state.
  - Must be 0, beside 1's: the back in the buffer cache while not lent (its entry points, by
    function), InnerInvalidation with ranges on the back while not lent, a packet asked for at a
    draw the stage sent none for, a packet left unplayed at a run's start, ring B full, the draw
    counts of both threads differing at a run's end, a packet played whole for a draw the back
    did not prepare, a packet whose entry is past the back's draw (dropped unplayed, the back
    fills the draw), a served draw whose texture handles the back read from game memory, and an
    exception on GPUMem.
  - FRITCHII_MEMSTAGE_RING_VERIFY=1 (boot): compare b with two threads: GPUMem puts a hash of
    compare b's words (the narrow set) in each packet and the back hashes its own at the take;
    differences are counted on the ring line, the first 16 logged. Off for fps runs. The flags
    15-48 are not in it (in served mode the back's are not the cache's: GPUMem's copy holds them,
    and they are handed over at each lend and return).
  - Profile: GPUMem's "stage wait" and "level run" zones; on the back the "split wait" span's
    phase 3 is its wait for the stage.
  - The read log (FRITCHII_MEMSTAGE_READ_LOG) and the texture-write check do not cover ring
    packets in level mode; with FRITCHII_MEMSTAGE_AHEAD the read log does (below).

### FRITCHII_MEMSTAGE_AHEAD

FRITCHII_MEMSTAGE_AHEAD (live, read at a frame end, default 0; the memory stage's step 7, "the
window"; work\fable_memstage_step6_spec.md 1.4 and 6.4): with FRITCHII_MEMSTAGE_THREAD=2, how many
draw packets GPUMem may fill ahead of the back's take. 0 is 6b's level mode, unchanged. Any value
of 65,536 or more is unlimited (the test harness sends 2147483647); the frame end is a level run,
so the lead never passes one frame. Ignored (0) while THREAD is not 2; "Memory stage window
(FRITCHII_MEMSTAGE_AHEAD=n)" is logged when it changes. It runs 0 while EDEN_NX2_RUNAHEAD is on
(the back applies due CPU-write chunks before each entry; logged once), while the delta path
(EXP118, or EDEN_NX2_FASTDRAW_COUNT) was on at boot (a delta draw holds the buffer cache's lock
through Configure, where a read flush is a served call; logged once), and for the rest of the
session after a catch-up could not finish (below; logged).

- GPUMem waits only while that many packets are in ring B that the back has not taken, while
  ring B has no room for another (both after publishing up to the entry), and at level runs (it
  lends and waits for the back to get through the run, as in level mode).
- It comes level inside the window (waits for the back to reach the entry) where running ahead
  would change an answer the back gets from the cache in a served call: an entry that takes a full
  copy of the back's engine (a new or stale copy), a draw whose fill marks buffers GPU-written
  (written storage buffers, transform feedback), and collected constant buffer ranges over
  GPU-written buffer data at the next fill. Counted by reason. Constant buffer words onto
  GPU-written buffer data are tested at the write, before the bytes land: the entry turns hard
  (the back writes them in a level run, in stream order), counted as "entries made hard by
  constant buffer words onto GPU-written data".
- GPUMem never waits on a staging or stream region stamped with a packet not played yet: the
  staging pool never waits for a region (a busy stream takes a new buffer, counted as "new
  staging buffers for GPUMem, the stream busy"). The one wait a draw's fill can reach is the
  multi-range storage bind's (a sync, with its catch-up).
- So what the back reads in a served draw is its position's: texture handles (EXP123=3), uniforms
  and storage addresses come in the packet; with the OoT trackers on, the bytes they read (the
  camera block, the actor's origin and bones) are copied at the fill and come in the packet too.
- GPUMem's loop with the window open (the step 7 fps fixes): the window and the room are tested
  before a draw's fill, not before every entry; GPUMem's busy time is timed per publish batch; a
  run publishes its end once, after GPUMem followed it, and stays lent when it reaches the end of
  what the front published (it waits for more input and ends at the first entry that needs
  GPUMem; "runs kept lent at the end of the input" on the window line); a served draw sends the
  vertex input state only when it may have changed (the back's VertexBuffers flag, not the
  cache's in served mode, is cleared at a packet's draw as the cache's vertex buffer update
  clears it with one thread, and with each send). With EDEN_NX2_GPU_PROFILE=2 GPUMem's
  profile splits its entry loop into "stage entries" and, inside it, "stage cb words", "stage
  pairs" (Pairs and StateDelta blocks' writes), "stage methods" (Method and Multi entries'
  handlers, their words and draws excluded), "stage packets" (Writes-tagged draw packets and
  DrawBatch entries) and "stage draw setup" (a draw the copy issued, served or made level, its
  fill excluded); what stays in "stage entries" is the per-entry bookkeeping (classify, the
  copy's lookup, runs, counters) and the closures, and GPUMem's "other" is the loop around the
  entries (the input's and the back's positions, served calls, publishes); the back's
  "draw" splits off "ring take" (the packet's take from ring B) with "ring take copy" inside it
  (its bytes copied out), "draw locks" and "ring facts" (inside "draw setup": the caches' locks,
  and the facts published under the stage's program key), "draw after configure" (the
  trackers, traces and dumps between Configure and dynamic state) and "rest tail" (PlayRest after
  its play: the held flags, the ticks mark, the watch count).
- The second fps fixes (window open only): GPUMem's wait for the front spins about 200 us before
  it sleeps (level mode keeps about 2,000 pauses; the link line's "slept n/s" adds how many of
  those sleeps waited for the front); the back prefetches the first 1 KiB of the next packet in
  ring B when it takes one; the back's loop calls the ring's per-entry work (BackEntry, the run's
  edges, BackAfter) only at a batch's first entry and where a run starts or ends. A take wakes
  GPUMem only when it sleeps on the window or on room in ring B (not on the back's position, a
  run or the front, which a take does not end); ring B's control fields and the link's counters
  sit on cache lines per thread.
- Other changes it makes: packet ticks keep one unplayed mark per thread (the close takes the
  lowest, and ring B's oldest packet; the ring's start and end clear GPUMem's); the lend's flag
  hand-over happens at the run's first entry; a sync in GPUMem's fill, and a join in it over
  GPU-written data, first let the back play up to the published position (a catch-up); a back
  waiting for the buffer cache's lock GPUMem holds serves GPUMem's requests from that wait (each
  wait counted, the first 16 logged with the caller's frames: read each site once and declare it
  safe for a sync there, or make it a served call); a GPUMem waiting for the lock the back holds
  (EXP118's delta draws) serves the back's call from its clean lock sites; GPUMem answers the
  back's served calls between entries too; the facts table has a cache on each side; a waiter's
  idle hook request runs even while GPUMem is busy.
- "Memory stage window" every 10 s, only while the window is open: packets in ring B at each push
  (average, maximum, and per second by 1, 2-3, 4-7, 8-15, 16-31, 32-63, 64-255, 256+ ahead), the
  back's waits for the stage (times and ms/s: should drop toward 0; spec 8.4's target is under
  20 ms/s, and 6b's run rate makes about 58 ms/s the floor until runs drop), GPUMem's waits for the
  window, for room in ring B (and how often the back's takes woke it from them) and to come
  level (ms/s, comings level a second by reason: full copy,
  writing draw, ranges over GPU-written data), entries made hard at a constant buffer write, new
  staging buffers for GPUMem with the stream busy, served calls
  made while packets were ahead, sync and join catch-ups, lends handed over at the run's start, the
  back's waits for the lock GPUMem held and GPUMem's served calls while waiting for the back's,
  served downloads past the join ring's reach (4,096 joins; not a failure), idle hook runs with
  GPUMem busy, and GPUMem's idle work (EXP104's doomed buffers; it runs only when the back is
  parked as GPUMem runs out of input).
- Must be 0 (on the "Memory stage link" line, the first 16 logged): "a sync of GPUMem's fill sent
  to the back before it played the packets ahead of it" (a catch-up that could not finish: the back
  waited in a served call or for the lock; the window is then refused for the session); "a served
  download over GPU-written data in a buffer a packet ahead of the back joined"; "an image
  registered over a constant buffer write GPUMem made for a later entry" (spec 9.1); "a served
  download over a constant buffer write GPUMem made for a later entry" (on GPU-written bytes). The
  last two need GPUMem's own-write ring, which runs only with the verify on (READ_LOG or
  RING_VERIFY), so the fps gate measures play without it. Also "facts key collisions" (any ring
  mode): the facts table's key is a hash of the program registers (the six pipeline slots and the
  program region), and every entry keeps those registers and compares them on a key match, so
  two programs under one key are told apart (a miss, the draw is level), never mixed up.
- EXP104 (the buffer pool, spec 7.5): with the window open GPUMem creates and destroys buffers while
  the back records, so the gate's area load should run with EXP104 on and off (or watch its hitch
  lines). GPUMem has no role in thread_pinning.h: the gate runs it unpinned, and the pinning choice
  of spec 5.8 stays open.
- Verify (spec 6.4) with FRITCHII_MEMSTAGE_READ_LOG=N: every Nth ring packet's fill logs its reads
  of game memory on GPUMem, and the back reads them again at the take and compares, leaving out
  the bytes GPUMem itself wrote after the fill (its constant buffer words for later entries, from
  its own ring; while the window is open they are kept out of the emulator's write ring, which they
  would wrap). The rest is classed as every read log difference; the read log gate line is the
  gate (unexplained 0, ring wrapped 0: GPUMem's own ring and the mark ring count too). Bound ranges
  have no tracker on the back: they differ when an emulator write noted since the fill, a mark the
  cache took since (a served gather or invalidation, a game core's trapped store: a small ring
  while the compare runs) or a pending game-side mark covers them; a difference over a mark the
  cache took is "game write, marked". A value is left out only when every changed byte is
  GPUMem's. The read log line adds "ring packets compared at the take", "left out as GPUMem's own
  later writes", ring packets "with a CPU-written mark since the fill", "async GPU completions
  over-read" (a query result GPUFencingThread wrote when the GPU finished, on uploaded bytes the
  packet does not bind: reported, not a failure; inside a range the packet binds it fails as an
  emulator write), "GPUMem's writes kept out
  of the write ring" and "CPU-written marks noted" (marks taken anywhere: served and inline gathers
  and invalidations, InnerInvalidation, game cores' trapped stores). Keep
  FRITCHII_MEMSTAGE_RING_VERIFY (compare b's hash) on with it.
- FRITCHII_MEMSTAGE_VERIFY & 1 (5.4, handle values) with the window open: the back's own read is
  at play, after GPUMem wrote constant buffer words for later entries, so a handle value whose
  words GPUMem wrote since the packet's fill (from its own-write ring; for a handle program, any
  enabled constant buffer of the stage) is left out and counted apart on the by-value line
  ("handle values left out as GPUMem's own later writes"); any other difference is a mismatch.
  The QMD and null pointer slot checks read twice at once inside level runs and are unchanged.
- Order of the verify runs: 1, 8, 64, unlimited (work\run_step7_verify.sh), then the fps gate
  against THREAD 0 (work\run_step7_gate.sh).

### FRITCHII_MEMSTAGE_LIST_DRAWS

FRITCHII_MEMSTAGE_LIST_DRAWS (live, read at a frame end with FRITCHII_MEMSTAGE_AHEAD, default 1;
fix 6 of work\fable_memstage_step7_fps.md, work\fable_memstage_6b_plan.md 7.2): with
FRITCHII_MEMSTAGE_THREAD=2 and the window open (AHEAD > 0), GPUMem serves list draws (OoT's
culled object draws, nx2_list_draw.h) instead of making each one a level run. 0, AHEAD 0 and
THREAD 0/1 keep them as before (level draws on THREAD 2, the back's own fill otherwise).

- Which: a list draw GPUMem's copy issues outside a hard entry, when the back would draw it
  indirectly from the two words. The ones the back reads back on the CPU (EDEN_NX2_LIST_DRAW other
  than 1, quads, a base instance, transform feedback, EDEN_NX2_MESH_FIRST; one test,
  ListDrawReadsBack, on the copy and on the back) stay level, and so do a refused fill (facts
  miss, texel buffers, no room in ring B) and a frame dump or trace.
- How: GPUMem fills it as any served draw, non-indexed, with the copy's indirect parameters in the
  cache's inputs, so the fill's geometry section binds and synchronizes the two words exactly as
  the back's own fill does after SetDrawIndirect; the packet then carries their buffer and offset
  (what the back's Configure asked the cache for, GetDrawIndirectBuffer). The back takes the
  packet in PrepareDraw and records the copy into the draw command from that buffer: it does not
  enter the cache. The copy is recorded at the draw's place in the stream, so the GPU reads the
  words the dispatches before it wrote; the buffer outlives the packet's play (packet ticks), as
  every vertex buffer a packet binds does.
- "Memory stage list draws (FRITCHII_MEMSTAGE_LIST_DRAWS=n, window open/closed)" every 10 s on
  THREAD 2: list draws a second served, and level by why (the window closed or the switch off,
  read back on the CPU, the fill refused it, a frame dump or trace); runs started by a list draw
  against all runs (each level one starts a run); the back's draws from the packet's buffer
  (equals served). List draws in an entry that is hard already are the ring line's "late" level
  draws. The ring line's level draw kind 0 is "indirect" only now, and 5c's "runs by first hard
  kind" names the level draw kinds ("level draw (list read back)", ...).
- Must be 0 (the link line): "a served list draw the back would read back" (the two sides decided
  differently: their state differs), "a served list draw whose packet has no list words buffer"
  (the back asked the cache) and "a list draw packet at a back draw that asked for no list
  arguments" (the other direction). A served list draw whose packet is missing (PacketMissing)
  is filled by the back as before, with the cache's indirect parameters set for it.
- At the market: about 1,600 runs a second start at a list draw (step 7 run, s7g); with the
  switch on they should go, and with them their lends, returns and the back's wait for GPUMem in
  each (step 7 fps list: about -30 ms/s for GPUMem parked, -23 ms/s of the back's wait).
  Compare 0 and 1 with profile_toggle.py in one session.
- Not the second starter at the same rate, "cache call +0xc744a0" (s7g; +0xc8c700 in the
  2026-11-14 build's stage-again run, symbolized against that build's PDB): it is the puller's
  semaphore release (WriteLong, Puller::ProcessSemaphoreTriggerMethod: the query cache's
  payload report, a fence when RELEASE_SIZE is set on Switch 2), not a list-words closure. The
  kept list words closure (KeplerCompute's list dispatch) runs about 850 a second and sits in
  front of its dispatch, inside the dispatch's run. Left as they are here.

### FRITCHII_MEMSTAGE_FAST_LOOP

FRITCHII_MEMSTAGE_FAST_LOOP (live, read at a frame end with FRITCHII_MEMSTAGE_AHEAD, default 1;
lever A of work\fable_stage_drains.md): with FRITCHII_MEMSTAGE_THREAD=2 and the window open
(AHEAD > 0), GPUMem's entry loop takes two shortcuts that leave its copy in exactly the state the
loop leaves it in without them. 0, AHEAD 0 and THREAD 0/1 run the loop as before (the shadow on
the back never takes them). "Memory stage fast entry loop (FRITCHII_MEMSTAGE_FAST_LOOP=n)" is
logged when it changes.

- Pairs and StateDelta blocks (3.7 million writes a second in 1.45 million blocks at the market)
  are applied in one loop (memstage_shadow.cpp, PairsFast) instead of StateWrite per write. While
  a deferred draw waits the writes still go one by one through StateWrite (the first stored write
  that is not an instance loop method draws it, reading the copy as it is then); after it no
  write in the block can issue a draw, so the loop reads the shadow RAM's mode once (again after a
  tracked write to shadow_ram_control's own word, the one write that changes it), keeps the
  same-value skip, gathers each stored register's two dirty flags and raises them on the copy
  (and in the flags raised since its last draw) at the block's end, and advances the copy's
  generation by the number of stores. Registers, shadow RAM, every dirty flag, the generation and
  the draws end the same; nothing reads the copy inside a block.
- The input's end (the front's `published`, written at every publish) is read again only where
  GPUMem reaches the last value it read, not at every entry. Entries before it stay published and
  that read's acquire made them visible, so GPUMem reads the same entries in the same order and
  stops for input at the same positions.
- The verify: the memstage gate (work\memstage_gate.py) with FRITCHII_MEMSTAGE_RING_VERIFY (compare
  b's hash at every served draw) and READ_LOG, once with 0 and once with 1; every must-be-0 line
  stays 0. Then 0 against 1 with profile_toggle.py at the market (GPUMem's "stage pairs" and "other"
  at EDEN_NX2_GPU_PROFILE=2, its busy time, fps).

### FRITCHII_MEMSTAGE_GATHER

FRITCHII_MEMSTAGE_GATHER (live, read at a frame end with FRITCHII_MEMSTAGE_AHEAD, default 0;
lever B of work\fable_stage_drains.md): with FRITCHII_MEMSTAGE_THREAD=2 and the window open
(AHEAD > 0), a gather (3D invalidate_texture_data_cache, 0x3dd: the game's collected CPU writes
into the caches, then a wait for idle) no longer starts a level run. GPUMem comes level at its
entry (StageLevel, reason "gather") and the back, parked there, runs the method's whole handler as
a request GPUMem waits for; the back's own run of the entry then skips it. 0, AHEAD 0 and THREAD
0/1 keep every gather as before (a run on THREAD 2, the back's own code otherwise).

- Why not GPUMem's own work (the plan's first form: GPUMem takes the game's writes, applies the
  buffer half, sends the ranges to the back for the other caches): the method's wait for idle
  syncs query results into buffers (QueryCache::NotifyWFI, SyncValues: buffer lookups and uploads
  in the buffer cache) and groups the async flushes (SignalOrdering), both in the buffer cache at
  the gather's place in the stream, with the back's query state there. At the market that sync is
  due at nearly every gather (about 68,000 reports a second, each leaves a pending sync until the
  next wait for idle). GPUMem ahead of the back can't give the back that cache state, and a served
  call would answer from GPUMem's later state (joins and uploads ahead). The gather line counts how
  often the sync was due ("the wait for idle synced query results") and how often a game write
  range lay by GPU-written buffer data, so the cost of that first form can be judged.
- Which: a gather that is the whole entry (a lone Direct method entry) of the copy, outside a run
  and outside a hard entry, with no draw before it in the entry. Level instead (a run, as before):
  a gather inside a multi-word or packet entry (the back would run what precedes it in the entry
  after the request), one the back's pending deferred draw would come before (its sink or the
  gather's own store flushes it at the entry: tested by the request, on the back), one right
  before a hard closure the front has published (a Cache or untagged Call, a semaphore release or
  a fence: the gather's run takes it in, so coming level first would only add a wait), no room
  for its record (256), GPUMem stopping.
- Exact: once level, the back has run every entry before the gather's and played every packet;
  GPUMem waits outside any fill, so the back serving the request is the buffer cache's only user,
  at that place in the stream, and it runs InvalidateTextureDataCache's default, the same code
  THREAD 0 runs there. The cache's flags 15-48 go to the back's engines before the gather and
  back to the copies after it, inside the request on the back (GPUMem waits; the back's idle hook
  flush, which may run while it is parked, is on the same thread), as a run of that one entry
  hands them over (the request may mark bindings dirty). The record
  (the entry's position and the back's draw count) makes the back skip the method when it runs
  the entry: nothing of the entry comes before it there (a lone method, no deferred draw), and its
  register store and dirty flags are the same either way.
- What it saves against the run: the lend and the return (flag hand-overs, the run's start and end
  on the back, FlushCaching), GPUMem parked through the back's run of the entry and of the entries
  the run was extended over (GPUMem follows those itself now). What it keeps: GPUMem waits until the back reaches the gather (its lead is gone there, as
  at a run), then for the request.
- "Memory stage gathers (FRITCHII_MEMSTAGE_GATHER=n, window open/closed)" every 10 s on THREAD 2:
  gathers a second run at their entry and level by why (switch off or window closed, not alone in
  their entry, after a deferred draw, before a hard closure, no ring backend, no record room,
  GPUMem stopping), inside a level run, in an entry hard already; runs started by a gather against all runs; GPUMem's ms a second in the
  served ones (the wait to be level and the request, and us each); of those, how often the wait
  for idle synced query results and a game write range lay by GPU-written buffer data, and the
  game write ranges a second; the gathers the back found run at their entry (equals served; the
  ones inside a level run should be 0, a lone gather's entry has nothing after it to turn hard). The window line's "level
  inside the window" has a "gather" reason, and the link line's "served syncs" include the
  back's runs of the served gathers. 5c's "runs by first hard kind" shows the gathers that
  are left.
- Must be 0 (the link line): "a served gather with no record of GPUMem's" (the back ran it in a
  served entry: the two sides decided differently; it then runs it as before, which is not in
  order), "a gather GPUMem had the back run at an entry the back passed without it" (a record
  dropped at a later gather, a lend, the ring's end or start) and "a back draw between a gather
  GPUMem had run and its entry" (the back's draw count moved between the request and its entry).
  The first market run (build 44a2974be6) counted about one "served gather with no record" every
  10 s with the switch off: not the gathers but the window itself. The back read lent_from and
  lent_until with two loads, and a lend between them gave it the last run's start with the next
  run's end: it took the entries between the two runs as lent, stayed "in a run" and at the next
  run's start skipped that start (no flag hand-over, no lend ranges, its first entry run with the
  back served; a gather is the usual first entry). BackLoad now reads one run's pair (lent_until
  stored with release after lent_from, lent_from read again), and a run start that finds the back
  still in another run is a must-be-0 of its own ("a run start while the back was still in the
  last run"), whose end is then taken there. "Level after a deferred draw" is not one of them: the shadow applies sunk writes at once and
  may issue that draw at an earlier entry, while the back issues it when it consumes its sink at
  the gather's entry.
- The record is pushed before the request and GPUMem publishes past the entry right after it (as
  an entry with a packet): the back, parked there, goes on at once. If the request throws, the
  ring ends at the entry and the record stays: the back's run of the entry without the ring
  still skips the gather (a record at or past the ring's end is honoured), so it never runs
  twice; the cache's flags come back to the copies in a scope guard either way. "The window open"
  is GPUMem's own read of it at its loop top (what StageLevel goes by), not the live value.
- The constant buffer ranges GPUMem collected and has not applied yet stay pending through the
  request, as on THREAD 0, where the memory manager's cached ranges wait for FlushCaching at the
  next draw (nothing in the gather, the wait for idle, the query sync or a fence calls it); a lent
  run applies them at its start, earlier.
- At the market (ms46, 2026-11-14): 4,579 runs a second start at a gather (43% of runs, 15 us
  parked each). With the switch on they should go from the run starters; the gain is the run
  overhead per gather, not the whole 15 us (GPUMem still waits to be level). Compare 0 and 1 with
  profile_toggle.py in one session; the gather line has GPUMem's ms a second in the served ones.
- "cache call +0xc8d340" (ms46, the second starter at 1,705 a second; symbolized against the
  2026-11-14 03:09 release PDB): the closure at puller.cpp:1751, the puller's semaphore release
  (WriteLong, Puller::ProcessSemaphoreTriggerMethod: QueryCache::CounterReport with the payload, a
  fence when RELEASE_SIZE is set on Switch 2, so SignalFence and the fence manager), the same
  closure as the earlier "+0xc744a0" and "+0xc8c700". Tagged Cache already, which is right: it
  reaches the fence manager (its releases commit the buffer cache's async flushes) and the query
  cache, and leaves a pending query sync. Not Pass (it writes the semaphore word into game memory
  at the back's place, which uploads GPUMem made ahead could miss, and a fence commits cache
  state). It could be run at its entry the way gathers are here (GPUMem level, the back runs the
  closure as a request), with a skip in the back's entry loop for Call entries:
  FRITCHII_MEMSTAGE_RELEASE below.

### FRITCHII_MEMSTAGE_RELEASE

FRITCHII_MEMSTAGE_RELEASE (live, read at a frame end with FRITCHII_MEMSTAGE_AHEAD, default 0): with
FRITCHII_MEMSTAGE_THREAD=2 and the window open (AHEAD > 0), the puller's semaphore release closure
(Puller::ProcessSemaphoreTriggerMethod's WriteLong: QueryCache::CounterReport with the payload, a
fence when RELEASE_SIZE is set on Switch 2; "cache call +0xc8d340" in ms46, the second run starter
at 1,705 a second) no longer starts a level run, the same way FRITCHII_MEMSTAGE_GATHER handles
gathers: GPUMem comes level at its entry (StageLevel, reason "semaphore release"), the back,
parked there, runs the closure as a request GPUMem waits for, and GpuSplit::Execute skips the
entry when it gets there. 0, AHEAD 0 and THREAD 0/1: as before.

- Which: the closure carries StageCacheCall::SemaphoreRelease in its header (bits 40-47 of a Cache
  tagged Call entry; Execute and every other reader ignore them). Outside a level run, with the
  switch on and the window open. Kept as a run: a Cache or untagged closure right after it that
  the front has published (the run takes it in, so coming level first would only add a wait and
  a second run), no room for its record (256), GPUMem stopping.
- Exact: once level, the back has run every entry before the release's and played every packet,
  and GPUMem waits outside any fill: the back serving the request is the buffer cache's, the query
  cache's and the fence manager's only user, at the release's place in the stream, and it makes
  the call Execute makes (the entry's thunk on its payload, which stays in ring A until the back
  passes the entry): the report, the fence and what it commits, the semaphore word written to
  game memory, all where THREAD 0 does them. On the back a Call entry is only its call (no method
  sink is consumed, no deferred draw is issued there), so nothing of the entry comes before it,
  and there is no lone-method or deferred-draw exception as for the gathers. The cache's flags
  15-48 go to the back's engines around the call and back to the copies (the fence's commits and
  the query sync may delete buffers and mark bindings dirty), inside the request, as a run of
  that entry hands them over. The front's bookkeeping of the release (NoteRelease, its waiters)
  reads positions and the word: the word is written while the back is at the entry, as before;
  `consumed` passes the entry later than the write, as before.
- "Memory stage semaphore releases (FRITCHII_MEMSTAGE_RELEASE=n, window open/closed)" every 10 s
  on THREAD 2: per second run at their entry and level by why (switch off or window closed,
  before a hard closure, no record room, GPUMem stopping), inside a level run; runs started by a
  release against all runs; GPUMem's ms a second in the served ones (and us each); the releases
  the back found run at their entry (equals served). The window line's "level inside the window"
  has a "semaphore release" reason; the link line's "served syncs" include the back's runs of
  them.
- Must be 0 (the link line): "a served semaphore release with no record of GPUMem's", "a
  semaphore release GPUMem had the back run at an entry the back passed without it" and "a back
  draw between a semaphore release GPUMem had run and its entry", as the gathers'. The record
  first, the publish right after and the record honoured past the ring's end as for the
  gathers: a release is never made twice (a second report, fence and semaphore word write).
- With FRITCHII_MEMSTAGE_GATHER on too, a gather right before a release keeps its run, which
  takes the release in (one run instead of two level waits).
- Compare 0 and 1 with profile_toggle.py in one session (5c's run starters lose "cache call"
  at the release's address).
- Since fable_stage_p3 section 3 all three puller releases are one closure with a fixed layout
  (puller.cpp's ReleaseCall: the rasterizer, the GPU address, the payload and the
  QueryPropertiesFlags, the channel's memory manager; memstage_shadow.h's StageReleaseWord) and
  carry StageCacheCall::SemaphoreRelease: WriteLong as before, and ProcessSemaphoreRelease and
  SemaphoreExecute's release (both fence-flagged), which were untagged closures before: each one
  made GPUMem's copies stale (a full copy at the next 3D entry; very likely the "call +0xc99160"
  behind most of the full copies). They are cache calls now (hard, no copy after) and candidates
  for this switch's request path too.

### FRITCHII_MEMSTAGE_PASS

FRITCHII_MEMSTAGE_PASS (live, read at a frame end with FRITCHII_MEMSTAGE_AHEAD, default 0;
work\fable_stage_levelpoints.md sections 4 to 6, with Fable's amendment and fable_stage_p3.md
section 1): with FRITCHII_MEMSTAGE_THREAD=2 and the window open (AHEAD > 0), bit 1 (=1) makes a
lone gather (3D invalidate_texture_data_cache, 0x3dd) or wait for idle (0x44) a passed entry
instead of a level run: GPUMem does what the entry does to the buffer cache itself, at its
position, and goes on; the back runs the entry later, served, with its own code, and finds
nothing to do in the buffer cache there. 0, AHEAD 0 and THREAD 0/1: as before (the passes line
is then dry). Bit 2 (fence entries, fable_stage_p3) is not built: it is masked off.
- Which (tried before FRITCHII_MEMSTAGE_GATHER; one that does not pass goes to what that switch
  says, a run otherwise): a lone method entry outside a run, no draw before it in the entry,
  - rule Q: no query result sync can be pending at the entry (the mirror in the passes line);
  - rule C: no read download tracking (EXP49 off, none pending) and no uncommitted buffer flush
    (the note, else the sets under the lock): only a fill that marks GPU-written data adds one,
    and those come level first or run lent, so a set empty at GPUMem's position is empty when
    the back gets there;
  - room for its record (256).
- A passed gather (rule G): GPUMem takes the game's collected writes (nothing when every
  collector says empty) and, under the buffer cache's lock, tests their pages (page-aligned, as
  WriteMemory's recording branch reads whole pages: the page bits, then the cache). No
  GPU-written data there: the served gather's buffer half at once (claims, WriteMemory per range:
  on such pages only marks, no command; the vertex-stale marks reach GPUMem's copy through the
  mark hook). GPU-written data on a page: nothing applied, the record goes whole and the entry is
  a level run (its gather applies the ranges with the full path before it takes more).
- Every passed entry pushes a record (its position; a gather's ranges). The back asks for it
  first at the entry (RasterizerVulkan::InvalidateTextureDataCache, GpuSplit::Execute for a
  release, Maxwell3D's wait_for_idle), so FRITCHII_MEMSTAGE_GATHER's / _RELEASE's "served ...
  with no record" stay real must-be-0s. At the gather, RasterizerVulkan::InvalidateGPUCache
  applies the record's other half (the texture and pipeline caches), then the back's own gather
  runs. A record lives until the back asks for a later entry's (one it never asked for is late,
  applied there), or is dropped at a lend (before the run's first entry), the ring's start and
  end (one at or past the end stays for the back's run of that entry without the ring).
- The back in a served entry: its own gathers (GPU::Impl::InvalidateGPUCache, read tracking off)
  return at once when every collector is empty; otherwise the take and the buffer half run on
  GPUMem in one served call (RasterizerVulkan::GatherOnStage), the other caches' half on the back: only
  GPUMem takes while the back is served. Its wait for idle and flush grouping (SignalOrdering)
  find nothing to do: a pending sync there, or a set uncommitted-flush note, is a must-be-0 (then
  they run as today).
- Must be 0 (the link line), each turning the switch to 0 for the session from the next frame
  end (logged once): "a query result sync pending at a passed entry's wait for idle", "uncommitted
  buffer flushes at a passed entry's flush grouping", "a pass record of an entry the back passed
  without asking for it", "a passed gather's buffer half recorded a command" (the seam arena grew
  in its bracket), "a served wait for idle with no pass record".
- Exactness, in short: the back's query state at the entry has no pending sync (every source of
  one is a 3D report GPUMem counts or a hard entry); the uncommitted set is empty there (above);
  the game writes taken at the gather were collected before GPUMem read the list, and both halves
  land at the entry, the buffer half before GPUMem's next fill (THREAD 0's order). A write made
  while the list runs has no position the game can count on and is taken at the next gather.
- The passes line counts passed ones and the first failed test per method, records and ranges a
  second, the records the back applied, and the back's own served gathers taken on GPUMem.
- Compare 0 and 1 with profile_toggle.py in one session; expected (Fable, an estimate): runs
  from about 10,500 to about 7,000 a second, GPUMem's run wait down 30 to 45 ms/s, +1 to +2 fps
  at the market.

### Memory stage passes (the "Memory stage passes" line)

Every 10 s on THREAD 2 (fable_stage_levelpoints 5.2), "Memory stage passes (dry, window
open/closed)": the counts P1 (passed gathers and waits for idle) and P3 (fence entries) are
judged by. Dry means decided and counted, the entries behave as before.
- Per method (gathers 0x3dd, waits for idle 0x44), each lone one GPUMem reaches: passed (dry: it
  would pass P1's tests up to the flush test), or the first test it failed: window closed, in a
  run, late (its entry hard already), not first (not a lone method entry, or a draw before it),
  sync may be pending (the mirror below), read tracking (EXP49 or a pending read download),
  flushes uncommitted (the buffer cache's uncommitted set, its note first, then under the lock),
  no record room, GPU-written page (both P1 only). "Of the sync-pending ones, clear in the
  alternative mirror and ready": how many more would pass with that mirror.
- The mirror (6.1) of the back's query_cache.HasPendingSync() at GPUMem's place: unknown at the
  ring's start, after every run's return and after every hard or unfollowed entry; a 3D report
  that is not a fence (Release or ReportOnly) makes a sync pending; a lone 0x3dd or 0x44 that
  passed (dry: would pass) or that the back ran at its entry (FRITCHII_MEMSTAGE_GATHER) clears
  it. The alternative mirror is the same, but a lone 0x3dd or 0x44 that runs as a level run
  clears it too (the back runs its wait for idle at that entry), and a run's return leaves it as
  it is (every entry of the run went through GPUMem, so the hard ones set it themselves). Dry, a
  sync point that would pass and a run it starts are taken as P1 would (cleared, no run).
- "The mirror at the gathers the back ran at their entry": with FRITCHII_MEMSTAGE_GATHER on, each
  served gather checks the mirror against the back's own answer (it had query results to sync):
  "clear but a sync found" must be 0 for the passes, for each mirror.
- Fence entries outside a run (the puller's release closures and the 3D fence reports): whether
  rule C holds there (no read tracking, no uncommitted flush).
- Puller releases on the front (WriteLong): timestamped and fence-flagged (RELEASE_SIZE), each
  with its unmapped ones.
- GPUMem from a run's lend to its return: waiting for the back (the run's end), for the front
  while the run is kept lent, following (the rest); then by compute launches a run (0, 1, 2-7,
  8-63, 64-511, 512+; launches are counted with the GPU profile or the delay test on), runs a
  second with back/front/following ms a second.
- The back's waits for the stage between a run's end and its next packet (its true idle after a
  level point), count and ms a second.
- The delay test FRITCHII_DELAY_RUN_US's spins.
- Then pass records and their game write ranges a second, the records the back applied, and the
  back's own gathers in served entries taken on GPUMem (FRITCHII_MEMSTAGE_PASS on).

"Memory stage passes P3 (dry)" (fable_stage_p3 section 6, after the passes line): per kind
(gathers, waits for idle, the puller's releases, 3D fence reports), whether P3 would pass it and
otherwise the first reason, in this order: not lone (a run, late, not a lone method entry; a
fence report inside a packet or batch); for a release, unmapped (passes: the back's Query returns
at once) or not fence-flagged (passes: no fence, no cache; its page joins the mirror); EXP28 & 1
(fence kinds: the fence may not sync); read tracking; flushes uncommitted; for a sync point with
no report pending, pass (P1's); reports unknown (after a run's return or a hard or unfollowed
entry); the fence behaviour Accurate; then the targets: the report pages GPUMem saw since the
last sync point (and the entry's own) as maximal runs of adjacent 4 KiB device pages, more than 8
runs, a run no existing buffer holds on its first page ("no buffer": the sync would create one;
should fall to near 0 after the first seconds, else pages are being collected or re-made), a run
past that buffer, a run with CPU-modified bytes (the sync would upload: the count that decides
P3's worth). Then the report pages per sync point that reached the target test (average and
maximum). The target test is BufferCache::ProbeSyncTarget under the cache's lock: it changes
nothing (no usage mark in dry). P3's mirror is kept as P3 would keep it: an entry it would pass
does not count as hard, and the run such an entry starts does not count as a run.

### FRITCHII_MEMSTAGE_MARKS_AHEAD

FRITCHII_MEMSTAGE_MARKS_AHEAD (live, read at a frame end with FRITCHII_MEMSTAGE_AHEAD, default 0;
fable_stage_compute.md part 1 with its section 10): with FRITCHII_MEMSTAGE_THREAD=2 and the window
open (AHEAD > 0), a draw whose fill writes buffers (written storage buffers, transform feedback)
no longer makes GPUMem come level first (the window line's "writing draw" reason goes to 0, about
310 waits a second at the market). 0, AHEAD 0 and THREAD 0/1: as before.

- How: every set the buffer cache had keeps meaning "as of the back" (gpu_modified_ranges, the
  uncommitted and committed fence downloads, the taken downloads, the read downloads, the read
  traps). A write mark GPUMem's own fill makes runs at once only what its later fills need (the
  dispatch write record, the buffer's write tick, the tracker's GPU-modified bit, the vertex
  buffers' stale flags, the page bits that never clear) and puts the rest on a list in the
  buffer cache, keyed by the push index of the ring B packet the fill makes (ring B's packet
  order: the back's issue order, deferred draws included). The packet's header carries a flag;
  right after the back played such a packet it takes the cache's lock and the rest runs, with
  today's code, in packet order ("promotion"). Every reader of the real sets promotes first.
  GPUMem's own work asks both the real sets and the list (upload skips, the index fast path, the
  bank page copies, join tests, the trackers' record, the level tests).
- The back never blocks inside the lock for a promotion: while GPUMem holds it (its fill) the
  back retries, serving GPUMem's requests, and GPUMem promotes for it when it waits for the back
  meanwhile (a sync's or a join's catch-up), instead of giving up.
- Still level, the test on both sets: claims (collected constant buffer ranges over GPU-written
  data; the list on the ranges' pages), joins over GPU-written data, constant buffer words onto
  GPU-written data. After a claim's wait, a mark of a draw the back still holds deferred on those
  pages refuses the fill (the entry is a run). The list's bound, 4,096 marks, waits as a writing
  draw did.
- The fence commit (FenceManager::CommitCacheFlushes, SignalOrdering) promotes first and runs on
  the back under the lock, also while served (the link's user checks accept it there).
- Exact: a mark enters the real sets after the back recorded its draw and before anything else
  can read or change them (all promote first, under the same lock): against every other event
  (a landing, a game thread's read, a later fence) it falls where THREAD 0 puts it, give or take
  that one packet's play, and nothing after the draw is signalled to the game in that time. The
  tracker's bit is set again at the promotion (a served write at the back's position may have
  dropped it since the fill, and THREAD 0 has it set after such a write); a landing keeps the bit
  on pages with a mark on the list. A fill whose packet never went into ring B, or a packet the
  back drops unplayed, takes its marks back: the back fills that draw itself.
- "Memory stage marks ahead" every 10 s while on: marks put on the list, promoted by the back at
  the play and by others, taken back; the list's length (average, maximum); the back's waits for
  the lock at a promotion (count, ms); level waits for claims (and refused after the wait) and at
  the bound; the must-be-0 totals: a level function on GPUMem's own work over a page with a mark
  on the list, a promotion out of order, the list not empty at a ring's start or end, a mark due
  at a run's first entry, a due mark a reader could not promote, a mark in a call GPUMem serves
  for the back, a mark on GPUMem's own work outside a draw fill, a pushed packet that is not the
  fill's. Each is logged as "Memory stage marks ahead (must be 0)" (the first 16).
- Gate: memstage_gate.py with the read log at market and fight; the fight with kills (the
  read-back counters and the hitch log must not move); the flash watch route; an area load.
  Compare 0 and 1 with profile_toggle.py in one session.

### FRITCHII_MEMSTAGE_FULL_BINDS

FRITCHII_MEMSTAGE_FULL_BINDS (live, read at a frame end, default 0; spec 4.2): every draw binds
every enabled vertex binding, not only the dirty ones. The ones that were not dirty go in runs of
their own, without a usage mark, and the memory player drops a run still bound in the current
command buffer, so the recorded commands stay the same; repacked bindings are left out (binding
one again runs its conversion). It is what the stage will need instead of the scheduler's
command buffer flags (step 6b). With EXP123 >= 1 the seam's stats line adds "memory seam full
vertex bindings": vertex bind runs recorded and dropped as still bound a second. Independent of
FRITCHII_MEMSTAGE_THREAD 0 and 1; THREAD = 2 turns it on by itself.

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

### FRITCHII_VERTEX_INPUT_COMPACT

FRITCHII_VERTEX_INPUT_COMPACT (live, default 0 = exactly as before; fable_perdraw_b 3.2).
RasterizerVulkan::UpdateVertexInput sent all 32 vertex bindings, used or not, about 59,000 times
a second (only about 1,500 of them equal to the one before, so EXP83's memo cannot help), in a
command that captured two 32-entry arrays: about 2 KB of chunk space copied for every send, and
a full 32 KB chunk (one worker dispatch) every 16 sends.
- 1: only the bindings that an attribute in the list names are described, and the command keeps
  12 bytes a binding or attribute in one of three size classes (up to 4 bindings and 8
  attributes, up to 8 and 16, else 32 and 32); the worker builds the Vulkan structs. The
  attribute list is the same as at 0: every non-constant location the registers define.
- Why it is exact: vkCmdSetVertexInputEXT replaces the whole vertex input state. A draw reads
  vertex data only through its shader's input locations; each location reads the binding its
  attribute description names. Every location the registers define is described as before, and
  so is the binding it names, with the same stride, input rate and divisor. A binding that no
  attribute names has no effect on any draw: a buffer bound to it (vkCmdBindVertexBuffers2) is
  ignored, and leaving it out also drops its stride and divisor checks (an unused stream's
  divisor of 0, a big unused stride). The repack path (stride 72) only changes bindings that
  attributes name; mesh emulation does not use this state.
- A later pipeline and a left-out binding: the described set depends on the registers only,
  not on the pipeline, so it cannot happen that a later pipeline reads a binding an earlier
  compact command left out. A pipeline reads locations, every defined location is described
  with its binding, and a register write that moves an attribute to another binding raises
  VertexInput and sends the state again. That is also why the attribute list is not cut to the
  locations the bound pipeline's vertex program reads: the state outlives that pipeline,
  UpdateVertexInput returns at once when no input register changed whatever pipeline is bound,
  and delta draws skip it, so such a cut would need a send whenever a pipeline reads a location
  the last send left out.
- EXP83's memo compares the lists that were built, so it works the same in both modes (a switch
  flip mid-game just makes the next compare differ and sends).
- "Vertex input compact (FRITCHII_VERTEX_INPUT_COMPACT=n) last 10 s: N recorded, bindings
  described B a send (referenced R), attributes A a send, command C bytes a send (full F)"
  (Render_Vulkan, Info, next to the EXP83 line). At 0 it says what 1 would save: "referenced"
  is what 1 describes.
- Expected: about half of the 24 ms/s UpdateVertexInput takes on the back, fewer chunk
  dispatches (and worker wakes), and a cheaper SetVertexInputEXT on the worker. Picture check
  across areas before it becomes a default (vertex layouts differ by scene).

### FRITCHII_WORKER_SPIN

FRITCHII_WORKER_SPIN (live, value = microseconds, default 0 = exactly as before; fable_perdraw_b
3.6). Scheduler::DispatchWork wakes the Vulkan worker with condition_variable::notify_all when it
sleeps, a system call: about 12 ms/s on the back, since the worker waits about half its time.
- n > 0: with an empty queue the worker first spins outside the queue lock for up to n us,
  watching dispatch_serial (raised under the lock with every push), before it waits as before.
  While it spins worker_waiting stays false, so a dispatch skips the notify. Try 50 to 100.
- Why no wakeup is lost: worker_waiting is set and read only under queue_mutex. After the spin
  (work seen or time up) the worker takes the lock again and the condition wait tests the queue
  under it before sleeping, so a chunk pushed during the spin is popped there. A chunk pushed
  after that test finds worker_waiting true and notifies, which the waiter, registered before it
  released the lock, receives. The spin only changes when the worker sleeps, never what it runs
  or in which order. WaitWorker's waits are untouched (the worker still notifies on each pop),
  and the spin holds no lock, so they are not held up by it.
- Cost: worker CPU while it spins (the worker runs at critical priority on a performance core);
  the process already uses about 7.5 CPUs, so judge it on a toggle with the 1% lows.
- "Worker wake (FRITCHII_WORKER_SPIN=n) 10 s: dispatches D/s, notified N/s, skipped B/s busy +
  S/s spinning; worker spins W/s, H/s found work, spinning M ms/s" (Render_Vulkan, Info; counted
  in both modes, at 0 the spin numbers are 0). "busy" is the old skip (the worker was running a
  chunk), "spinning" is the new one.
- Expected: 6 to 11 ms/s on the back (most of the notifies), at the price of the worker's spin
  time shown on the line.

### FRITCHII_BIND_ADDR_MEMO

FRITCHII_BIND_ADDR_MEMO (live, default 0 = exactly as before; fable_perdraw_b 3.7).
GpuToCpuAddress costs the back about 30 ms/s, 16 of it in the vertex, index and uniform binds
(UpdateVertexBuffer 8.4, BindGraphicsUniformBuffer 4.1, UpdateIndexBuffer 3.5), which translate
the same few pages again and again.
- 1: each vertex stream, the index buffer and each graphics uniform slot (stage, slot) keeps its
  last translation: the manager (unique_identifier), the small GPU page (page_bits), its device
  page and the manager's map_serial when it was looked up (Tegra::MemoryManager::
  GpuToCpuAddressMemo). A hit needs all three equal and costs one load of map_serial. A disabled
  vertex stream is not translated at all (it becomes a null binding whatever the translation
  gives).
- Why it is exact: a translation is a function of the address and this manager's page tables
  only, and those change only in PageTableOp and BigPageTableOp (Map, MapSparse, Unmap and the
  remaps through Map), which bump map_serial before their first table write and after their
  last (MapChange). So an unmoved serial means the tables are the ones the remembered lookup
  read. A lookup is kept only when no change was running when it started and none started
  before it ended (BeginMapRead, MapReadStable, the sequence lock the EXP63 page cache uses), so
  a half-written table is never remembered. Within a small page the translation is linear (both
  tables map whole small pages), so base plus offset is what the walk returns. Another address
  space (a channel switch) has another unique_identifier and misses. Unmapped results are not
  kept. A change running at the moment of a hit is ordered after it, as for a walk that races it.
- "Bind address memo (FRITCHII_BIND_ADDR_MEMO=n) 10 s: hits H/s, translations T/s (U/s not kept:
  a map change overlapped), disabled vertex streams not translated D/s" (HW_GPU, Info; only
  while the switch is on).
- Expected: 8 to 12 ms/s on the back. The memos sit beside EXP18's uniform bind memo, which
  skips the whole bind; this one only shortens the translation when EXP18's memo misses or is
  off.

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

EXP12 (DEFAULT 4 since 2026-11-14, with EDEN_NX2_GPU_WAKE=1 built in; EDEN_NX2_SEM_FAST
acquires read current but unmet, puller.cpp ReleaseFencesForReadable):
1 = their fence release waits only for the queued writers the read can't account for, full
drain only if still unmet; 2 = suspend then too, without the full drain; 3 = measure only;
4 = as 2 without the wait for those writers (the "fence writers" site): the fence release is
queued, the word read once more, and an unmet acquire suspends at once. Exact for the reason 2
is: the waiter is registered before the fence release is queued, the back latches it at any
release that meets the value, and a suspended channel resumes only on that latch or a met read
of the word (CanResumeAcquire). Live. The risk is speed only: an acquire the latch does not
resume within 16 polls or 1 ms gets a poll drain (the "acquire" site), which costs more than
the wait did.

Lines, every 5 s:
- "Fence writer waits (EXP12=N)": as before. At 4, "writer waits" is 0 and "met there" counts
  acquires met at the read without the wait.
- "Fence writer resumes (EXP12=N)", every mode: writer waits skipped (4) and how many of them
  had the back short of the target (overstates: the back stores its position only every few
  thousand words of a batch and at its end); the
  fence path acquires resumed from a suspension, per route (the first answer of the two
  CanResumeAcquire calls a resume makes; latch: the back latched the
  waiter; word read: a read of the word met it; front answer: a queued release, the QMD
  prediction or the GEQUAL bound; poll drain: met after a poll drained the back), as a
  histogram per second of the time from acquire to resume (under 100 us / 500 us / 1 ms /
  5 ms / longer) with the mean; poll drains of these acquires and their time, split by
  pending under 2 ms or longer, and those of the other acquires; at 4, the reduction bounds of
  resumed acquires checked and not; then per channel: fence path acquires, met there,
  suspended, resumed by route, and the mean time to resume.
- At 4 the QMD prediction's "met by" for an acquire suspended there is counted at its resume
  instead of at once (there would be no "unmet"). Its reduction bound is checked there only
  when a word read met it within 1 ms of the acquire, as 2's read after its wait; by a latch or
  later the word may already be a later write's. Compare "safety: N checked" in the QMD
  prediction's reduction bound line between 2 and 4.
- With EDEN_NX2_GPU_WAKE=0 (the default before 2026-11-14) a latch does not wake the GPU
  thread: once nothing moved for 500 us it sleeps 1 ms at a time (EXP6=0), so a latched resume
  can wait out one sleep. 4 leans on latches for the acquires 2 met after its wait.

Must stay 0 at 4: the 2 s "waits on semaphore" warning. Judge 2 against 4 with a live toggle:
"fence writers" in the front waits by site to 0, the "acquire" site under 20 ms/s, "from
acquire to resume" not higher.

5 = 4, and the poll of a suspended acquire whose word reads current (CanResumeAcquire's
readable branch) no longer drains the back in full (fable_fence_writers_2 section 3). Once 1 ms
passed since the last check (the 16-poll count no longer applies there), a lagged check
publishes what was sent, waits only for what was sent by the previous check (by the suspension
for the first: the "lagged check" drain site), then asks the latch and the word again and the
queued release, QMD prediction and GEQUAL tests, without ReadSemaphoreWord. Exact: the drain
added only the latch, asked again, and ReadSemaphoreWord, which differs from the plain read
only when the buffer cache holds GPU-written bytes of the word; those mark its page GPU-written
(never cleared), which makes the word unreadable, so that poll takes the drain as before. No
hang: everything sent is published, and what was sent by one check has run by the next, where
it latched, shows in the word, or marked the page. Only with EDEN_NX2_GPU_WAKE on (read live)
and default fences; otherwise the drain as for 4 (without the wake an idle front sleeps 1 ms
at a time, which the drain hid). Unreadable words keep the drain path.

Lines at 5: the "Fence writer resumes" line adds lagged checks/s, how many waited and their
time (expect near 0), and how many found the acquire met; the "lagged check" route in the
histogram and per channel; "pending over 100 ms" (every mode, counted once per acquire, fence
path and others; not a hard zero: acquires the guest CPU releases can wait that long during
loads and pauses, so compare with EXP12=2 in the same scene). Judge 4 against 5 (GPU_WAKE=1 on
both): poll drains of these
acquires pending under 2 ms to near 0, the split drain zone down, latch and lagged resumes
under 1 ms up, channel 2's time to resume down.

6 = 5, but the acquire waits when something queued can write its word itself
(fable_fence_writers_2 section 4): the target is the newest of the QMD prediction's record for
the word that has not landed, the block and page stamps over the word alone
(GpuSplit::BytesStamp: WriteStamp without its floors, i.e. writers of anything, 3D method
writes, unknown writes, functions, the overflow) and a queued puller release to it, taken
before the fence release is queued. Past the back's position: wait for it (drain site "fence
word writers"), then the latch and the word, as 2. Otherwise suspend at once, as 4. While the
write stamps are off (before the front's QMD copy started them) it waits for 2's target. Exact
as 4: it only chooses when to look, and a wait skipped is never a wrong answer. The polls check
lagged as 5. With EDEN_NX2_GPU_WAKE=0, 6 keeps its wait at the acquire and only the polls fall
back to 4's drain. The "Fence writer resumes" line adds word writer waits/s, which term set
their target (unlanded prediction record, block/page stamp, queued release), how many had the
stamps off, how many acquires had none queued, and (counting only) the term the full
WriteStamp would have waited for, by StampTerm, which tells real writers (block, page stamp)
from floors. Judge 5 against 6 (GPU_WAKE=1).

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

### FRITCHII_PIPELINE_KEY_HASH

FRITCHII_PIPELINE_KEY_HASH (live, default 0): the graphics pipeline lookup after a key change
(GraphicsPipeline::Next: the current pipeline's key, then each transition's, a full key compare
each) compares a 64-bit fingerprint of the keys first (GraphicsPipelineCacheKey::Fingerprint,
over exactly the bytes the key compare reads) and runs the full compare only where the
fingerprints match.

- 0: Next as before.
- 1: fingerprints first. The pipelines' fingerprints are made once (their keys never change),
  the current key's once per lookup, from the key as the lookup has it.
- 2: both, Next's answer kept; the answers that differ are counted (must be 0) and the first 16
  logged.

Exact: equal keys have equal fingerprints, so a candidate skipped by its fingerprint is one the
full compare rejects, and the candidates are tried in Next's order. Works with or without
EDEN_NX2_PIPELINE_CLEAN, EXP52 and EXP55 (they decide when the lookup runs, this how it compares).

Counts, one line every 10 s while on: "Pipeline key hash": lookups/s, keys looked at (and per
lookup), full compares, compares skipped by the fingerprint, fingerprint matches with another key,
and the verify's checked / differ.

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

### FRITCHII_READ_RUNS

FRITCHII_READ_RUNS=1 (boot, diagnostic, default off; read_run_stats.h): the guest reads of
GPU-written memory that go to the back or wait for a download (GPU::OnCPURead past its
preemptive answer). Every 600 frames these lines:

- "Read runs": each read's kind (flushed, waited for a read download on its way, waited for a
  fence download on its way), whether it is on the page after the previous one, the
  GPU-written run around its page, the wait, distinct pages a frame, and EXP112's run flushes.
- "Read overlap" (fable52_answer_kills 2b): whether the read's own bytes [address, address +
  size) overlap the GPU-written bytes inside its pages that it waits for, or only share the
  page(s) with them. Checked in BufferCache::GetFlushArea under the buffer cache's mutex it
  already holds, against: for a fence download wait (EXP72), pending_download_ranges up to the
  newest download the read needs; for a read download wait (EXP49 in_flight),
  read_download_ranges; for a flush, gpu_modified_ranges at the read (what the flush
  downloads). Per kind as overlap/page only/unknown (unknown: the buffer cache did not answer,
  e.g. image memory, or EXP60 handed the read off). For page only, the gap to the nearest
  GPU-written byte (<64 B, <256 B, <1 KiB, <4 KiB; "more" must stay 0). The read size
  (<=16 B, <=64, <=256, <=4096, more). The wait split by overlap/page only/unknown, in ms a
  frame and ms a read. "gpu_modified_ranges answers otherwise" counts waits where the download
  ranges and gpu_modified_ranges disagree (should be near 0).
- "Read writers", up to 8 "Read writer k/n" lines and "Read runs for B1"
  (fable52_answer_readback 3): which write made the bytes each read waits for. Every GPU write
  mark (BufferCache::MarkWrittenBuffer, on the thread that binds: the back, or the memory stage)
  goes into a ring of 16,384 marks with its range, host time, the dispatch count at the time
  and its writer: a dispatch's storage or texture buffer (the compute program's hash, the same
  hash as the write extent checks use, and the binding index), a draw's (stage and binding), or
  other (copies, queries, transform feedback; no bind named it). The binds set the writer in a
  thread-local around their marks. At a read whose pages hold GPU-written bytes, GetFlushArea
  (reading thread, cache mutex held, before anything lands) takes the newest mark over the
  read's bytes and copies up to 64 of the read's bytes from guest memory; OnCPURead, once the
  read is answered, compares them with guest memory again.
  - "Read writers": reads with a mark over their bytes, with one only on their page, with none
    in the ring, and not GPU-written in the buffer cache; the write's age at the read (from the
    mark to the read's ask) as <1 ms / 1-1.5 / >=1.5 per kind; reads whose bytes differed at
    landing out of those compared; marks a frame (the ring covers about 16,384 / that frames).
  - "Read writer k/n", the 8 writers with the most wait: reads by kind; the age (<0.5, <1, <2,
    <4 ms, more); the wait in ms a frame and a read; the read's offset in the marked range (the
    bound range, or a multi-range segment) as lowest to highest, distinct offsets, and the most
    common step from the writer's previous read; how many reads' bytes differed at landing; the
    dispatches bound between the write and the read.
  - "Read runs for B1": the GPU-written runs around the reads (GetFlushArea's run, up to 65
    pages) kept 16 frames after their last read and merged where they touch: how many and how
    big a frame, and the bytes marked GPU-written inside them a frame (all writers, and by
    dispatches). That is about what fence downloads would add if B1 marked those runs
    Preflushable (an upper bound: a range marked twice before a fence downloads once).

  Reading it for fable52_answer_readback 2: flushed reads whose writes are 1.5 ms or more old at
  the read are the ones B1 turns into landed reads (the bytes were written long before the read;
  only the missing mark sent them to the back); flushed reads under 1 ms only become fence
  waits with B1, and the fence download kind only gains from B2 (reaching the dispatch run
  sooner). A writer whose bytes never differ at landing is the one to read the code of. A steady
  step in the offsets means the read walks an array: mark ahead by a page instead of the run.
  Costs with it on: a lock and a ring slot per write mark ("marks a frame"), a scan of up
  to 16,384 marks and a 64-byte copy per read; off: one test of a constant at each site.

The GPU-written ranges are what the cache marks: for a dispatch or draw that is the bound
writable buffer range (cut only by the native store bound, or by EDEN_NX2_WRITTEN_SSBO_MARK_KIB,
which must be unset for this), not only the bytes the shader stored. So "page only" is as exact
as the cache's own tracking (no written binding covers the read's bytes), while "overlap" means
a written binding covers them, not that the shader changed them. Reads after the first one in an area are answered from it (memory.cpp's per-core area)
and are not seen. Off: one test of a constant at each site.

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

EXP120 (live, 0 = off, DEFAULT 2 since 2026-11-13: market 1.89 M reads skipped, 0 differed, 1%
low 37.7 -> 39.2; DmaPusher's NoteEngineWriter, fable_answer_long_drains 8.4): the
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

EXP122 (live, 0 = off, DEFAULT 1 since 2026-11-13: 0 mismatches in 4.39 M verified answers;
GpuSplit's page watch, fable_answer_long_drains 8.2 (a'),
fable52_answer_kills 3.4; needs EXP10 on): the device pages the macro DMA read snapshots
(EDEN_NX2_DMA_SNAPSHOT) are served from are watched. Each compute or copy range the fronts note
(NoteWriterRange, NoteReleaseWriter) is tested against those pages when it is noted, and the
bytes it covers in a watched page are kept with its position (8 a page, the rest folded into one
union). A read of a watched page then answers "may a writer since the snapshot write these
bytes" from that list, with no scan of the writer range ring and no scan limit: before, every
"past the scan" drain of the session was the scan giving up (4,096 ranges at ~11,500 a second),
never an overlap. Pages that are not watched (2,048 at most, 256 sets of 8, least recently used
replaced; the first table held 256 and kept evicting the rarely read pages, exactly the ones
that need it, fable52_answer_exp122), or
whose watch started after the snapshot asked about (another channel's snapshot), take the ring
scan as before; the ring's other readers are unchanged. Unbounded writers still count for every
page. 1 = on. 2 = on, and every "no write" answer is checked by the old scan without its limit;
an overlap it finds that the watch did not report is a verify mismatch (the first 16 logged as
"Writer page watch (EXP122=2)"), which must stay 0. EXP10's "DMA snapshot EXP10" line logs reads
answered by the page watch (and how many said "may write"), watched pages, byte-list overflows,
verified answers and verify mismatches, then the checks left to the ring (page not watched,
watched only after the check's snapshot, over two pages) and the pages replaced; with the
watch working, "page not watched", "pages replaced" and "past the scan" stay near flat. Off by default, not measured yet: run 2 first (0
mismatches in a kill fight and at the pot room), then toggle 0/1 with profile_toggle.py and
compare the "past the scan" count and the macro dma read drains.

### EXP121

EXP121 (live, 0 = off, DEFAULT 1 since 2026-11-13: verified 0 mismatches, fight +1.7 fps twice;
fable52_answer_kills 3, fix 1): the texture cache's collector, once device
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
descriptors wait for the packet's plays. When Configure ends, true, false or by an exception (a
scope guard at the call site, MemStage::RestAtEnd), whatever is left is played (the cache
believes those uploads are on their way); descriptors left then are dropped with the draw.
Descriptor entries take the pipeline's form (descriptor buffer or not), as the fill runs before
Acquire. Pipelines with texture or image buffers keep their fill in Configure as with 1
("late fill": the binds need the texture cache's views). The indirect dispatch's ObtainBuffer, the
extent probes, list page copies and draw indirect buffers stay where they are (fill and play at
the call, as with 1). There is no tick guard: packet ticks keep a held packet's numbers open.
"Flushed while held" (5.5) counts held packets whose fill and PlayRest are in different command
buffers (a flush or finish in between, syncs included; the first 16 logged with the caller):
expected near 0, as a vertex binding the fill did not see dirty is not bound again in the new
command buffer. Texture writes while held: the texture cache writes game memory inside Configure
(GC downloads, EXP121's landed downloads, a flushed download), after the early fill read. Each
held fill keeps the device ranges it read (uniform copies, uploads, bound ranges, storage and
table addresses); every such write is counted, and a write over one of those ranges is counted
apart and the first 16 logged as "Memory seam: texture write over fill read". That count must
read 0 for 2 to be exact. The stats line gets a "memory seam 2b" line: held and
late fill draws and dispatches a second, packets not held (arena busy on another thread), flushed
while held, texture writes while held (over fill reads, must be 0), rest plays (and descriptors dropped), descriptor form mismatches (must be 0), and fill
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

### Compute counters (fable_memstage_compute_brief 3)

Counting only, no switch; for one market run with FRITCHII_MEMSTAGE_THREAD=2, AHEAD unlimited
and EDEN_NX2_GPU_PROFILE=2. Nothing is counted without the GPU profile.
- "Memory stage runs by channel n s" (THREAD 2 with EDEN_NX2_GPU_PROFILE, every 10 s after the
  "Memory stage ring" line):
  GPUMem's level runs by the channel the stage had bound at the run's first entry (0 to 6, then
  "other"): runs a second, entries and compute launches each (0xaf with bit 1, 0xb0 above 1,
  0x107, as the compute front notes dispatch writers), and GPUMem's run time in ms/s (from the
  lend to the return). Then runs a second and ms/s by entries a run (1, 2-7, 8-63, 64-511, 512+)
  and by compute launches a run (0, 1, 2-7, ...): the ms/s in compute lists against small runs.
- Fine zones (GPU profile 2): "inline upload" (AccelerateInlineToMemoryImpl, inside "method"),
  "dispatch fill" (MemStage::FillDispatchHere, on whichever thread fills) and "qmd signal" (the
  QMD scheduler's signal with its QMD reads and releases; the dispatches it runs stay
  "compute"). Zones are exclusive, so "method" and "compute" lose what they cover.
- Fine counters (GPU profile 2, each thread's "fine counters per second" line). The front
  thread: compute upload data words by path, "upload words (macro emit)", "(method)"
  (FrontCallMethod) and "(multi-method)" (FrontCallMultiMethod). The back: inline uploads by
  size ("inline uploads 4 B", "5-64 B", "65-256 B", "over 256 B") and by branch ("to never
  GPU-written pages", EXP85's branch; "uploaded in place"; "not uploaded"; "unmapped");
  dispatches by what their fill marks ("marking a buffer GPU-written" with "dispatch marked
  bytes", "writing images only", "marking nothing": the buffer cache's GPU-written marks made
  while DispatchCompute holds its lock), "indirect dispatches", "dispatches with texel buffers"
  (late fill), "dispatches skipped" (DispatchCompute ended before the record: unconfigured, grid
  over the device's limit, an exception), "dispatches replaced" (the upscaler), "dispatches
  copying list pages", "dispatches with extent probes"; and served calls made inside a dispatch
  by cause ("served in a dispatch: FlushRegion", ... in the "Memory stage ring" line's order).

### EXP124

EXP124 (upload words as one entry; live, 0 = off, DEFAULT 1 since 2026-11-13: market 56.2 -> 58.4
fps, 1% low 38.8 -> 40.0; fable_memstage_compute_brief 4).
Today the compute macro's emit sends every inline upload data word (KeplerCompute 0x6d) as its
own Direct MacroWrite: about 457,000 ring entries a second at the Castle Town market, each
followed by GPUMem as a hard entry, and on the back each a last call that Upload::State flushes
alone (AccelerateInlineToMemory with 4 bytes). With 1, the words a front macro call emits
collect and go as one Multi entry marked Direct (GpuSplit::MacroWords; pieces of 8,192 words;
never for the 3D engine, whose Multi entries the memory stage applies without the Direct bit),
sent before any other write the macro emits and at its end (or when the macro call ends in an
exception, so none waits for the next call); after the entry, in this order, the
upload's writer note (FrontNoteWriter: its whole range, as each word noted it) and with EXP13=2
the copy's batch at the entry's position (FrontPatchOverlays). FrontApply still runs at each
word (a later register read of the macro sees it). FrontCallMethod's and FrontCallMultiMethod's
words are unchanged. The memory stage follows the entry as one hard entry (an executable method
of another engine), as it followed each word. On the back the entry calls
KeplerCompute::CallMacroWords: every word goes through CallMethod as before (registers end the
same), but only each entry's last word is a last call (a batch over 8,192 words is several
entries, each ending with one), and where the flush writes in place (native
partial transfers, linear, one line) it goes to AccelerateInlineWords, which writes the words as
their own calls would have: word by word while a grid placeholder is live (the scan sees x, y
and z in one write, x alone in a word's) or when the destination is off a 4-byte boundary (the
list note clears only words a write covers whole); otherwise split at 4 KiB pages of the GPU
address, a piece in one call where InlineSkipsCaches (EXP85 & 1, deferral on, no GPU-written
mark on its page) holds and it is mapped and not on sparse memory, word by word otherwise. Words
an earlier non-last call left unflushed are flushed with the first word's call, as before;
block linear, multi-line and non-native uploads keep a last call per word.
Why one call per piece is the same as its words' calls: the write lands the same bytes (guest
memory between two words is touched by nothing that reads the destination on a never
GPU-written page); the list note bumps a page's write count once instead of once a word, and
every user of that count compares it for equality only; the query filter is per page and the
invalidation drops queries by overlap; the invalidation accumulator is fed word by word (it
keeps a range per call where one does not extend the last, so a single range could differ).
Values: 1 = on; 2 = on, and before each one-call piece every word is asked as its own call
would ask (its translation and the page test); a piece whose words disagree goes word by word.
GPU profile 2 fine counters: front "upload word batches (EXP124)" (one per batch sent; a ring
entry per 8,192 words of it) and "upload batch words"; back
"upload pieces in one call", "upload pieces word by word (page may be GPU-written)", "(unmapped
or sparse)", "upload flushes word by word (grid placeholder live)", "(not word-aligned)",
"upload pieces checked (EXP124=2)" and "upload pieces whose words disagree (must be 0)" (the
first 16 also logged as "Upload words (EXP124=2)"). Toggle live 0/1 with profile_toggle.py (a
macro call reads it once). Fields that should move at the market: the ring's level entries
(about 500,000 to 50,000 a second), entries each run, the shadow's KeplerCompute 0x6d rows
(about 9,000 a second, as multi entries), GPUMem's stage entries, the back's "method" and new
"inline upload" counts, the front's "macro" zone. Must stay 0 with THREAD 0 and THREAD 2 (AHEAD
unlimited), at the market and the fight: the memstage gate, "DMA snapshot ... 0 differed", the
QMD release prediction's payload differences and unpredicted counts, the compute front verify,
FRITCHII_MEMSTAGE_VERIFY=2's QMD mismatches, the flash-watch route.

### EXP126

EXP126 (live, 0 = off, DEFAULT 1 since 2026-11-14: trial 0 differed, market 57.7 -> 58.7 fps;
fable_qmd_wait_brief piece 2): the ranged ring's overflow
(below) out of GpuSplit::WriteStamp. With 0 a dropped ranged write counts as a write to every
address for the write stamps, so with the back more than about 5,000 ring words behind (the
Castle Town market) every QMD read the front's copy cannot serve waits for the whole backlog
("qmd read", about 95 waits and 125 to 150 ms a second), and small patches cannot become
overlays. With 1 it does not: RangeLastWrite stamps each ranged write's bytes before it touches
the ring, so a dropped write is still found over its own bytes, and stamps only err newer (a
collision makes one newer, never older). Unranged 3D writes still count for every address;
QueuedGuestWriters and the macro DMA reads (MemoryWriterTarget) scan the ring and keep the
overflow as before. An overlay from memory made while the overflow is queued is kept out of the
EXP13 feed (its unbounded-writer check sees the overflow; EXP127 not 0 changes that, see
EXP127). Aliases: with EXP126 not 0 every
ranged write is stamped by device address as well as by GPU address (the page translated once
and cached, as NoteWrite does), so a QMD or semaphore read of the same memory through another
GPU address finds it in WriteStamp (the QMD sites, PatchHolders and the semaphore feed check all
ask it with the device address). With 0 they are stamped by GPU address only, as before (the
overflow floor covered the dropped ones; the newest 256 had that hole already). While a queued
function may remap the range, or for sparse pages, the write counts for every address instead
(the stamp floor) until the back ran it. When EXP126 turns on, the ranged writes sent before were
not stamped by device address: with 1 they count for every address until the back ran them.
The ring scans (QueuedGuestWriters, ExactWriterCheck) still compare GPU addresses only; they
keep the overflow, so their answers are as before.
With 2 WriteStamp keeps the overflow as with 0, but the device stamps are on as with 1, so its
answers are the same as 0's or newer (never served earlier than 0; a block collision can add a
wait). The table below is cleared whenever EXP126 changes, so a switch into 2 starts from 1's copy
and this one being the same. The QMD copy follows what 1's copy would hold where it
differs, in a trial table (never served): the overlay 1 makes from memory at a patch nothing but
the overflow refused, and an overlay or upload shadow block that takes a patch at 1 only (a
holder still_holds refused for the overflow alone), each patched further as 1's PatchHolders
would. At every read it works out what 1 would serve (an overlay, a shadow block, guest memory
at once, or a wait, with 1's target and those holders) and, where that is something else than
this copy serves or the same without the wait, compares the 256 bytes with what this copy
reads (after its wait, if it waits). "differed" on the line below must be 0 (the first 16 are
logged as errors with the word and the source). A change 1 would make that the table does not
follow (a holder 1 drops as newer than a patch, an overlay this copy makes where 1 keeps its
own) marks the block: its reads are counted as not compared until the back passes it (every read
of a marked block, also those 1 would have waited for, so the count overstates); entries
dropped at the table's limit (4,096 blocks, only after those the back passed) count as lost.
Not followed at all: the overlay table's capacity at 1 (its evictions and evicted floor). To
judge 1: profile_toggle.py on EXP126,
0 against 1, market heavy phase uncapped with EDEN_NX2_GPU_PROFILE, and watch "qmd read" in the
front waits by site, the EXP13 lines' "after a wait" and "ranged overflow" waits, overlays from
memory, the fence writers wait and fps; with 1 the EXP13 checks (payload differs, unpredicted,
resyncs, auto-off, bound violated) and "DMA snapshot ... differed" must stay 0.

### QMD wait floor line

"QMD wait floor (EXP126=n)" (HW_GPU, every 5 s with EDEN_NX2_GPU_PROFILE or EXP126 on;
fable_qmd_wait_brief piece 1,
counting only): why the front's QMD scheduler copy (EXP13 = 2 to 4) waits at "qmd read". The
front keeps at most 256 ranged 3D writes (constant buffer words, reports; EDEN_NX2_RANGED_MAX);
at the next one the oldest is dropped and its position, `ranged_overflow_position`, counts for
every address in the ring scans. It used to raise the "3D method write" floor; it is a term of
its own now (the EXP13=2 wait cause line says "ranged overflow" for it), with the same effect.
The line gives:
- ranged ring overflows a second; QMD reads a second, how many with the overflow still queued,
  and the ring words from it to the write position then (average, most);
- small QMD patches a second by outcome: a holder (overlay or shadow) took it, an overlay from
  memory made (and of them, made with the overflow queued: 0 unless EXP126 = 1), refused as
  unmapped or GPU-written, an overlay queued, an evicted inline QMD, or by the stamps, by the
  term that refused it (block stamp, page stamp, stamps on since/over 1 MiB, unknown write,
  unbounded writer, unranged writer, queued function, 3D method write, ranged overflow), and how
  many of those nothing but the overflow refused;
- QMD read waits a second by the term that set the target (the same terms, and evicted inline
  QMD) with the ring words each waited, and how many nothing but the overflow made wait;
- the stamp terms, the "overflow alone" counts and the exact scan below are worked out only with
  the line on (EDEN_NX2_GPU_PROFILE or EXP126 not 0; WriteStampWhy and the scan cost the front
  otherwise); without it refusals count under "by the stamp" with no terms;
- for refusals and waits set by a block or page stamp: whether the exact scan over the same bytes
  (GpuSplit::QueuedGuestWriters) finds the writer at the stamp, only an older or unbounded one, or
  nothing (a hash collision, or a page stamp of a write elsewhere in the page), as counts for
  the window: patches a/b/c, reads a/b/c;
- macro DMA read waits a second (WaitForMemoryWriters with a range) and those whose target is the
  ranged overflow, with the ring words each;
- EXP126 not 0: ranged writes stamped by device address a second, and those counted for every
  address instead (a queued function, sparse pages, no memory manager given);
- EXP126 = 2: reads compared, by what 1 serves (guest memory, an overlay, an upload shadow
  block), how many differed, trial overlays and blocks built, reads not compared and trial
  entries lost (counts for the window).

### EXP127

EXP127 (live, 0 = off, default 0; fable_fence_writers_brief piece 2): QMD sources the EXP13=3/4
feed refuses today although the copy read them as safely as one it takes. 1 = audit, 2 = on.
With 0 nothing changes. With 1 and 2 the copy tags the records with the sources below and the
"QMD feed sources" line (next section) is written; with 1 the feed takes nothing new and counts
what 2 would have met; with 2 it takes them (EDEN_NX2_FEED_MEMORY_OVERLAY on, as for overlays
from memory). The value is read at each feed decision, so a record tagged while 1 or 2 was on
is refused again at 0.
- Overlays from memory built while the ranged overflow was queued (2a; needs EXP126 = 1). The
  belt that keeps an overlay from memory out of the feed when an unbounded writer is queued over
  the QMD (GpuSplit::QueuedGuestWriters, EDEN_NX2_FEED_MEMORY_OVERLAY) counted the ranged ring's
  overflow as one, so with the back far behind (the market) every such overlay was "not checked"
  (220 to 233 a second). With EXP127 not 0 and EXP126 = 1 the belt counts the overflow only for
  the dropped writes no device stamp covers (those sent before the device stamps started; its
  other unbounded writers count as before), and an overlay that only the overflow kept out gets
  the source "overlay from memory (the stamped overflow queued)"
  (QmdPrediction::Source::OverlayFromMemoryOverflow). Why it is exact: the belt is for writers
  the write stamps cannot see, and the overflow is not one. RangeLastWrite stamps a ranged write
  by GPU address and, with EXP126 not 0, by device address before it enters the ring, and the
  overlay is built only when WriteStamp over the QMD (both addresses, with EXP126 = 1 the writes
  sent before the device stamps as a floor) is not newer than the back's position. So no dropped
  write still queued writes the QMD's bytes; EXP126's trial compared about 560 reads a window, 0
  differed. With EXP126 at 0 or 2 the belt is as before.
- Plain memory reads (2b): a QMD read from guest memory with nothing queued over it
  (Source::Memory, 174 to 202 refusals a second at the market with EXP126 = 1) passes the guards
  an overlay from memory passes: mapped, not GPU-written, no inline QMD of it queued or evicted
  (a queued overlay of it would serve the read or set its target past the back), no upload
  shadow block of it queued (it would serve the read), no write stamp over it newer than the
  back. With EXP127 not 0 (EXP13 3 or 4, EDEN_NX2_FEED_MEMORY_OVERLAY on) and EXP126 not 0 (at 0
  ranged writes are stamped and scanned by GPU address only, so a write through another GPU
  address of the same memory would be missed) the read asks the GPU-written check again (the
  first one ran before the back's position was loaded: a GPU writer the back ran in between
  would be at or below it, past the stamps) and the overlays' belt (QueuedGuestWriters: no
  unbounded writer queued; the overflow as above with EXP126 = 1); with both clean its records
  get the source "memory, checked"
  (QmdPrediction::Source::MemoryChecked). Exact on the same ground as the overlays from memory:
  the copy reads the QMD at the same signal the back reads it at, so the back reads the same
  bytes unless something outside the stream (the guest CPU, a read-back landing) writes the QMD
  in between, and the overlays have that exposure too (their bytes come from memory at the
  patch, earlier still). The back checks every release against its record, and the feed turns
  itself off at the first one that differs.
To judge: one market and one fight window at 1 (the "QMD feed sources" line: the new sources'
back checks must show 0 payload differs and 0 unpredicted, and their would-be decisions 0
wrongly met), then profile_toggle.py on EXP127 1 against 2 at the market, heavy phase,
uncapped, GPU_PROFILE on. At 2 these must stay 0 / on: the EXP13 feed line's "feed on", "0
violated", "0 wrongly met", "0 not fed (check table full)"; the EXP13 line's "back QMD: ... 0
payload differs, 0 unpredicted" and "0 resyncs/s". Should move at 2: the feed line's "not fed
for their QMD source" (from memory, overlays from memory; with 1 these count the new sources
too, as at 0) and "built/s with an unbounded writer queued" to
near 0, "met by a queued plain release" up by 80 to 100 a second, the fence writer line's
readable acquires unmet down by 105 to 120 a second, the EXP13 line's "would be met by a
predicted queued release" to near 0.

### QMD feed sources line

"QMD feed sources (EXP127=n, EXP126=m)" (HW_GPU, every 5 s with EXP127 not 0 and EXP13 2 to 4):
- the back's check of the copy's QMD records by the source their QMD was read from, per second
  as matched/payload differs/unpredicted (an unpredicted release has no record: it counts under
  the source of the copy's last QMD read), for every source: inline QMD, patched inline QMD,
  overlay from memory, upload shadow, memory, memory after a wait, overlay from memory (not
  checked for the feed), overlay from memory (the stamped overflow queued), memory, checked;
  releases and reductions together;
- overlays from memory built a second with the stamped overflow as the only unbounded writer
  queued (with EXP127 not 0 these no longer count in the feed line's "built/s with an unbounded
  writer queued");
- QMD reads a second from memory with nothing queued over them (the EXP13 line's "from memory")
  whose belt was clean ("memory, checked"), those with an unbounded writer queued, and those
  whose page was GPU-written by the time the back's position was loaded (EXP13 3 or 4 with
  EDEN_NX2_FEED_MEMORY_OVERLAY on and EXP126 not 0; 0 otherwise);
- EXP127 = 1: acquires a second (resume polls included) the feed would have met through the new
  sources, by source, as queued plain/run but unlanded/reduction bound (the plain and narrow
  gates as at 2; the bound's check table is not asked), and those decisions checked once their
  record landed as the feed's own audit checks its decisions: right, wrongly met (must be 0; the
  first 8 are logged as errors "QMD feed sources (EXP127=1) audit"), unknown (left the table, or
  not landed in 2 s), dropped unchecked (the 4,096-decision ring, checked every 256 decisions
  and at each report, overwrote one not judged yet; should be 0), and session totals;
- EXP127 = 2: acquires a second fed through the new sources, by source.

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
