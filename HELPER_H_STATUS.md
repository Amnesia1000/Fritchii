# Helper H status (paused for account switch)

## Done
- cab330cd22 FRITCHII_JIT_FLAG_BRANCH (boot, default 0 = byte-identical old code). Compile-checked
  (a64_emit_x64.cpp, emit_x64.cpp, emit_x64_data_processing.cpp), not built, not run.
  - 1: If terminal branches on host flags when the block's last A64SetNZCV stored the value whose
    lahf+seto snapshot is still in the flags (Add/Sub with NZCV, GetNZCVFromOp's own test, which
    now clears AL with mov instead of xor). Any IR instruction not on the flag-neutral list
    (register gets/sets, pseudo ops, identity, SetNZCV) drops the record. cmc before ja/jna for
    HI/LS, same as the old reload. Off with cycle counting (single core) and verbose debug.
  - 2: also A64SetCheckBit tests the stored byte, CheckBit terminal skips its cmp (cbz/tbz).
  - Exactness: the stored NZCV is a snapshot of these exact flags; the old path rebuilt the same
    SF/ZF/CF (sahf) and OF (cmp al,0x81). reg_alloc only emits mov/movd/movq/movaps/xchg,
    except xor for immediate 0, which the list excludes.

## Half done
- Pinning: design only, E:/OotRemakeEmulation/work/helper_h_pinning.md. No code.

## Not started
- FRITCHII_JIT_VERIFY.

## Next steps
1. Test FLAG_BRANCH=1 then 2 (boots, OoT runs, no wrong branches; pot room fps).
2. Pinning step 1: rbp+rsi (Windows, needs NO_FRAME_POINTER=1), store/reload hook in
   CallFunction/CallLambda, load in GenRCP, store at dispatcher exit.
3. Verify mode idea: a block-exit self-check under the switch (compare pinned regs vs state at
   the dispatcher exit), cheap but only catches pin write-back bugs.

## Overlap with helper G
- Touched only the If and CheckBit terminal bodies in a64_emit_x64.cpp (one line each).
