# Helper I (front ops, S2-S4): status at the account switch

Branch front-ops, from 7c2ab4b124. No code is changed yet: this session went on reading the code
and working out the design. Nothing was compiled, built or run.

## Done (reading, design)
- Read fable_b_one_follower.md 27 and 28, plus the code: macro_hle_oot.h (the handlers),
  maxwell_3d.cpp FrontWrite / FrontRunMacroFeatures / OotFrontContext / draw packet
  (FrontCapturePacket, ApplyDrawPacket), GpuSplit::Execute (gpu_split.cpp ~2323), the ring entry
  API (gpu_split.h 327-580), Maxwell3D CallMethod / CallMethodFast / WriteStateRegister /
  ApplyPairs / CallConstBufferWords, and the memory stage's Call tags (memstage_shadow.cpp ~3240).
- Where the code differs from Fable:
  - Owner config has EDEN_NX2_MACRO_SINK=1 + EDEN_NX2_RING_PAIRS=1. So a macro's state writes go as
    PairWrite (Pairs blocks), not MacroWrite, and only executable writes go as Direct MacroWrite.
  - 0xe02's 4th write (0x904+8s) is bind_groups[s].raw_config: executable (ProcessCBBind).
  - E has no data RAM of its own. native_data_ram belongs to the front while macros run there, so
    an op has to carry values the front already resolved from the data RAM (0xe02's stage when
    p0 >= 8, 0xe1c's and 0xe36's addresses, 0xe08's cache test). E cannot run the handler again.
  - E already applies Pairs, Direct writes and the sink the same way per state write
    (WriteStateRegister semantics), so an op applied as "ConsumeSink, then each sent write" matches
    the old path. Exception: MACRO_SINK without RING_PAIRS (sunk writes get applied later), so the
    op path needs !MacroSink() || RingPairs().

## Design (not written yet)
- FRITCHII_ONEF_FRONT_OPS (live, default 0). 1 = ops (S2), 2 = ops + check, 3 = E consumes
  directly (S3), 4 = S3 + check. A separate value or switch for the 27.7 dead-store counter and the
  drop.
- Front: run the existing handler with an OpContext. Its Emit does FrontWrite's
  "decide + apply front state" half (shadow, same-value test, regs, executable bookkeeping,
  effect) and records (method, value, sent bit). Then ONE CallWithPayload entry goes out:
  [engine, system, id | sent mask | flags, resolved args], plus the notes (cb NoteWrite /
  RangeLastWrite) after it. A call that sends nothing gets no entry. FrontWrite should be split
  into decide/apply and send, so both paths share the same code and the sent set is the same by
  construction.
- Per-call conditions, else the old path: Track/TrackWithFilter shadow, exp22 == 0,
  same_experiment == 0, CB_WORDS_BLOCK == 0, !MacroSink() || RingPairs(), packet off,
  list_draw_stale == 0, no e2c_mode/shadow_unknown, no guard_list_reads, no trace / dry / seq /
  verify, and for 0xea0/0xe52 instances <= 1 (or fold off).
- Whitelist, by call rate: 0xe02 (497k/s), 0xe1c (410k), 0xe36 (355k), 0xea0 (127k), 0xe08 (102k),
  0xe52 (24.5k). 0xe2c stays on the old path (its 0x158 read runs inside the call; E2C_ON_E and
  EXP53 interplay). Count covered vs not per id.
- E (S2): expand(id, args) is a shared header-only function (also for the memory stage), applied
  as Execute would: ConsumeSink, state writes via CallMethod (or WriteStateRegister), runs of cb
  data words grouped as Execute's CbBatch does (count 1 -> CallMethod unless CbLean&1 on E). The
  old path's grouping already depends on where the ring was published, so any split of a cb run is
  treated as the same.
- Check (2): the payload also carries the front's full write list and final values. E compares its
  expansion with it and the registers after applying, both must be 0, logged once a second.
- S3: a direct apply per id, the same as CallMethodFast with the method known (shadow +
  ProcessDirtyRegisters + ProcessCBBind / ProcessCBData / draw_manager.DrawBegin/DrawEnd), only
  when CallMethod would take the fast path (fast_maxwell3d, executing_macro == 0, !perf_verify,
  profile off, draw kinds off); otherwise S2. Check (4): apply state on a snapshot, restore, run S2,
  compare regs/shadow/dirty.
- Memory stage: add a StageTag for ops (memstage_shadow.h:33), expanded with the same function.
  Today an unknown tag means Unfollow (correct, slower).
- 27.7 dead stores: count on the front (0xea0's trailing 8e3=0x640, 8e4=0, 8e5=0 followed by the
  next ring entry being an 0xea0 upload of the same 12 bytes, same 8e0-8e2). A drop by holding them
  as a pending flush is NOT provably exact: front-side guest memory reads between two ring entries
  (QMD reads, list reads) could see them. Keep it behind its own value, default off.
- S4 count: not started (28.5 counter: bound ranges whose bytes differ between the front's parse
  and E's fill, outside queued uploads).

## Next steps
1. Split FrontWrite into decide/apply + send (no behaviour change), check_compile.
2. OpContext + op encode for the 6 ids, the CallWithPayload entry, Apply on E (S2) + check mode +
   once-a-second line.
3. memstage tag; S3 direct apply + check; 27.7 counter; S4 counter.
4. Compile maxwell_3d.cpp, gpu_split.cpp, memstage_shadow.cpp, plus one OpenGL file.

## Open questions
- Whether EXP14 (early_same) is on in the test set: doesn't matter for exactness (the same
  condition gets copied), only for the profile counters.
- Helper B overlap: none so far (no edits to vk_rasterizer.cpp or DmaPusher::Step planned).
