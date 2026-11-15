# Helper G status (JIT dense code, branch jit-layout), paused for account switch

## Done
- Worktree E:/lichii-jitlayout on branch jit-layout from 7c2ab4b124.
- Read Fable A sections 1, 2, 4.1 and the JIT code (a64_emit_x64.cpp, emit_x64.cpp,
  block_of_code.*, a64_interface.cpp, the A64 translator's terminals, the fastmem patch code).
- block_of_code.h: far-region API declared (EnableFarCode, Switch{To}Far/NearCode,
  GetFarCodeCurr, SmallestRegionSize, members). No definitions yet, nothing calls them.
  Default behaviour unchanged.

## Not started / half done
1. block_of_code.cpp: define the far API. ClearCache resets both regions. SpaceRemaining =
   min(near, far) when far is on. EnsureMemoryCommitted must commit the two regions separately
   (factor the retry loop into CommitPrefix(base, committed, needed, limit); with far off it must
   stay the same logic). EnableWriting/DisableWriting (NO_EXECUTE builds) also protect the far range.
2. FRITCHII_JIT_FAR_CODE=1: in A64EmitX64's constructor, before PreludeComplete, call
   code.EnableFarCode(total * 3 / 8) when the cache is 64 MiB or more. In Emit: no int3 after the
   terminal, SwitchToFarCode, run the deferred emits, int3, SwitchToNearCode, then RegisterBlock.
   PrecompileBlock's 256 MiB check should become SmallestRegionSize()/2 when far is on.
   Checked: every hot-to-deferred jump already uses T_NEAR. Jumps back from deferred code go to
   labels that are already defined. Fastmem patch info takes rips in any order, and the exception
   handler covers the whole code space.
3. FRITCHII_JIT_DUMP (1 = flush every 4096 blocks, 2 = flush only at cache clear and shutdown),
   FRITCHII_JIT_DUMP_CORE (default 0), FRITCHII_JIT_DUMP_FILE (default jit-dump-core{N}.bin in
   cwd; the tests should set it under work\measure), FRITCHII_JIT_DUMP_LIST (hex pc per line).
   Planned format: 32-byte header "FJITDMP1", version, core, code base, code size. Then 56-byte
   records: kind (1 block, 2 prelude), flags, descriptor, guest pc, near addr, far addr, guest
   count, near size, far size, reserved. After each record: guest words, near bytes, far bytes.
   Host bytes are read at flush time, so links show patched. Flush before ClearCache and in the
   destructor.
4. FRITCHII_JIT_CHAIN bits: 1 = chain the fall-through successor (needs FAR_CODE, implies 4),
   2 = halt check only on backward links, 4 = compact link slots. Depth: FRITCHII_JIT_CHAIN_DEPTH,
   default 4.
   - Compact slot = 5-byte jmp. When unlinked it jumps to a far stub per target that sets pc and
     goes to the dispatcher. A checked link becomes `cmp halt,0; jne halt_stub(target); slot`.
     When the target sits right behind the slot, the patch writes nop5 (fall through).
     Create both stubs before AddPatchLocation.
   - Chain: in far mode Emit records the last slot. a64_interface GetBlock then translates and
     emits the successor at that exact pointer (no align, only if getCurr matches and the code is
     readable), up to the depth limit. If/CheckBit swap their leaves so the pc+4 leaf comes last
     (inverted cond = cond ^ 1). Run() must skip its +15 & ~15 rounding when chaining is on.
   - Bit 2: direct links keep or gain the check only when target.PC <= block start PC. That
     includes adding it to backward LinkBlockFast (B, BL), which today has no check. Indirect
     terminals (FastDispatchHint, PopRSBHint) gain `cmp halt; jne ForceReturn` (pc is already
     stored). Proof: between checks, the path only takes forward edges, so block start pcs
     strictly increase and the path is bounded by the number of compiled blocks. Off with cycle
     counting.

## Open questions
- Chaining compiles code before it runs. Guest code written without IC maintenance would go
  stale (UB on hardware). Translating data after a function's end could hit translator asserts.
  Worth a check.
- Bit 2: halts raised inside a block (unmapped-access PrefetchAbort without the debugger) stop a
  few forward blocks later. Memory aborts with the debugger stay immediate (EmitCheckMemoryAbort).
- Far split 5/8 near, 3/8 far is a guess, check fill with a counter.
