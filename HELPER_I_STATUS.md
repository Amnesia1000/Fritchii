# Helper I (front ops, S2-S4): status

Branch front-ops (worktree E:\lichii-frontops), merged up to oot-gather-fix 8217d3d516.
All three steps written and committed. Compile-checked only (check_compile.py); not built, not run.

## Commits
- aa4a67c9d7 step 1: FrontWrite split into FrontWriteDecide / FrontWriteSend / FrontWriteNotes, no behaviour change.
- e3623b6ef2 step 2: FRITCHII_ONEF_FRONT_OPS (S2 + S3 + checks), macro_ops_oot.h, FRITCHII_ONEF_DEAD_STORES counter.
- 2c6ad264f7 step 3a: StageTag::Ops, the memory stage follows op entries.
- e255514587 step 3b: FRITCHII_ONEF_FILL_COUNT (S4 count).
- a09ce2fbe3 review fable_b2 1.1-1.3 folded in: 0xe3a added, S3 on under the GPU profile, new checks
  (front skipped E differs, cb word addresses, S3 routes + indexed flag), coverage by reason; switches.md.
- 134f476c16 / last: this file.

## Switches (all live, default 0 = old path)
- FRITCHII_ONEF_FRONT_OPS: 1 ops, E expands (S2); 2 = 1 + check; 3 E applies directly (S3); 4 = 3 + check.
  Ops: 0xe02 0xe1c 0xe36 0xea0 0xe08 0xe52 0xe3a. Per call, only with shadow Track, EXP22/EXP2/CB_WORDS_BLOCK/
  draw packet off, MACRO_SINK only with RING_PAIRS (owner config has both = fine), no list guard, no
  trace/dry/seq/verify tools, no EXP53 predicate, no E2C mode, no FOLD_INSTANCE first-instance fold.
  0xea0/0xe52: up to 1 instance while EDEN_NX2_FOLD_INSTANCES is on (its default), 16 otherwise.
- FRITCHII_ONEF_DEAD_STORES: 1 = count only (27.7). The drop is not built.
- FRITCHII_ONEF_FILL_COUNT: 1 = count only (28.5), needs FRONT_OPS >= 1 (samples every 16th 0xe02/0xe08 bind, first 4 KiB).

## Log lines
- Front, once a second: "Front ops (FRITCHII_ONEF_FRONT_OPS=N): ... op entries, ... writes sent, ... dropped on the
  front; per op sent/sent nothing/old path: e02 a/b/c ...; old path by reason: lanes, tools, stopped, declined or too
  long; ... fallbacks; ... count differences (must be 0)[, stopped]"
- E, check modes, once a second: "Front ops check: ... ops checked, ... write list differences, ... register differences,
  ... skipped writes checked, ... front skipped E differs, ... cb word addresses checked, ... differ; S3 ... direct,
  ... compared with S2, ... differences, ... routes checked, ... effect differences (all differences must be 0)"
  (modes 1/3 sample the skipped-write check every 64th op; a mismatch there logs an error and stops ops).
- Errors (16 max each kind): "Front ops check: op X expands to ...", "... skipped M = V as unchanged, E holds ...",
  "... writes a constant buffer word at A on E, the front noted B", "... applied directly (S3): <route / not indexed>", "... left register ... on E, the front holds ...",
  "... applied directly (S3) leaves other registers, shadow words or dirty flags ...",
  "Front ops (FRITCHII_ONEF_FRONT_OPS): stopped for this session, <why>".
- "Dead stores (FRITCHII_ONEF_DEAD_STORES): ..." and "Fill count (FRITCHII_ONEF_FILL_COUNT): ..." once a second.

## Exactness argument (S2)
- The front's state work is FrontWriteDecide, the same code FrontWrite runs; the conditions above exclude every
  branch of it that sends by itself (EXP22 delta/batch, fold count) and every send lane except Direct MacroWrite and
  PairWrite, so decide-all-then-send equals decide/send interleaved (send touches no state decide reads besides
  sink_pending, which the recorder sets the same way). The fold-count draw.begin is caught before decide and the call
  falls back (recorded writes sent the old way, in order).
- E applies exactly the sent writes in order: ConsumeSink then CallMethod per Direct write, ApplyPairs per run of
  Pairs words (independent stores, as across any block boundary), Direct cb words grouped as Execute's CbBatch (a run
  split differently is the same, as CallConstBufferWords is built to be).
- Notes (NoteMethodWrite, NoteWrite/RangeLastWrite for cb words) are made after the op entry, which is where its
  writes run. The entry's effect is the Combine of the Direct writes' effects (Pairs blocks always noted None).
- The data RAM words a handler reads are read on the front before it runs (none of the six writes a word it reads).
- S3 calls the leaf CallMethod reaches for each write (WriteStateRegister / CallMethodFast / ApplyPairs' store /
  CallConstBufferWords), only when CallMethod's fast-route conditions hold (fast_maxwell3d, executing_macro 0,
  no perf verify, profile off, draw kinds off), sink consumed once (nothing in between fills it).

## Open risks
- Register check vs front: if the back legitimately differs from the front's copy for one of these registers
  (e.g. const_buffer.offset when a word does not fit the buffer), check mode stops ops on the first op; that would be a
  front/back divergence the old path has too, worth a look, not an op bug.
- S3 gain is per-call overhead only (virtual calls, sink checks, profile tests); the bigger cuts of 28.4 (final values
  + dirty flags once, bind/draw called directly, dead stores dropped) are not built.
- S3 check (4) compares S2 vs S3 state only for state-only ops (0xe1c, 0xe36, 0xe3a mostly; profile off); ops with
  executables get the route check, the indexed-flag check and the register check. A full effect-call list (bind
  stage/slot/address/size, run address/count/words, DrawBegin/End) from a dry S3 is not built: it needs hooks in
  ProcessCBBind / CallConstBufferWords / DrawManager. S3 reaches the same leaves as S2 by construction, so the route
  check is the part that can differ.
- S2 alone is expected ~0 on both threads (fable_b2 1.4); S3 is the E cut; the front cut needs a lean decide (not built).
- Fill count reads guest memory raw on the front and on E (no flush); the memory stage may write ahead of E, which can
  skew "same" counts. Count only.
- The op entry carries the engine pointer and does not select the engine (as the draw packet).

## Test order (step-in first, one value at a time)
1. FRITCHII_ONEF_FRONT_OPS=2 (pot room, short): "Front ops check" all differences 0, "count differences" 0, not stopped;
   look at the coverage (ops vs old path) per id; picture fine.
2. FRITCHII_ONEF_FRONT_OPS=1: front us/draw and fps (toggle 0/1).
3. FRITCHII_ONEF_FRONT_OPS=4: check line incl. "compared with S2" differences 0.
4. FRITCHII_ONEF_FRONT_OPS=3: E method time / fps (toggle 1/3).
5. FRITCHII_ONEF_DEAD_STORES=1 (with ops 0 or 1): read the Dead stores line.
6. FRITCHII_ONEF_FILL_COUNT=1 (with ops 1): read the Fill count line ("differ with none" expected 0).
