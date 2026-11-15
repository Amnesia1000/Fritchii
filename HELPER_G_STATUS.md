# Helper G status (JIT dense code, branch jit-layout): done, compile clean, not built or run

## Commits
- 82f4ad46ab WIP: far-region API declared (previous account)
- 138c11fc44 FRITCHII_JIT_FAR_CODE: far region in BlockOfCode + deferred emits there
- 6c43ea650c FRITCHII_JIT_DUMP (pcs as image-base offsets) + hot list takes offsets
- 1e14a27e16 FRITCHII_JIT_CHAIN (bits 1/2/4, CHAIN_DEPTH)
- b97ae9ff54 FAR_CODE logs near/far fill at each full cache clear
- bc9508386d merge oot-gather-fix (helper H's pins + FLAG_BRANCH)
- (last) docs/fritchii/switches.md entries + this file

## Switches (all boot env, default 0 = old code)
- FRITCHII_JIT_FAR_CODE=1: last 3/8 of the 512 MiB cache = far region for deferred emits.
  Xbyak's maxSize_ is the near region's end while near code is emitted (overflow throws like the
  end of the space); SetCodePtr picks the limit by the new position's region, so patches work
  anywhere. Windows commits each region separately (CommitPrefix). No int3 after the terminal.
- FRITCHII_JIT_CHAIN bits: 4 compact links (needs FAR_CODE), 2 halt check only on backward links
  (+ indirect branches/returns gain one), 1 chain successors (needs FAR_CODE, implies 4).
  FRITCHII_JIT_CHAIN_DEPTH (default 4). Off with cycle counting.
- FRITCHII_JIT_DUMP=1/2, FRITCHII_JIT_DUMP_CORE (0, -1 all), FRITCHII_JIT_DUMP_FILE,
  FRITCHII_JIT_DUMP_LIST. Header carries the image base, records the abs pc and pc - base.
- FRITCHII_JIT_HOT_LIST: pcs below the image base are offsets (+ this launch's base); "{core}" in
  the path; FRITCHII_JIT_HOT_LIST_ORDER=1 keeps file order.
- Image base: UserConfig::image_base = KProcess::GetEntryPoint() (rtld's load address; main is
  +0x20000 in OoT), set in ArmDynarmic64::MakeJit (interfaces are made after Initialize).

## Merge with helper H (pins, FLAG_BRANCH)
- Emit sets terminal_nzcv_in_host_flags / terminal_check_bit_in_host_zf around EmitTerminal, then
  runs the far deferred emits. Chain swaps in If/CheckBit use EmitCondFromHostFlags / skip the cmp
  the same way. All CHAIN halt checks sit inside leaves, after the condition jump.
- Pin call hook stays on during far deferred emits (set once in the ctor). Stubs jump to the
  dispatcher / force return, which handle the pins.

## Compile-checked (check_compile.py, all OK after the merge)
block_of_code.cpp, a64_emit_x64.cpp, a64_interface.cpp, a32_interface.cpp, emit_x64.cpp,
arm_dynarmic_64.cpp (physical_core.cpp before the merge).

## Test order
1. FRITCHII_JIT_DUMP=2 with nothing else (cheap, default code): check the file parses, offsets
   match work\fable_a_gc\hotlist-300-rel.txt. Then DUMP=2 + FAR_CODE=1 to see the far bytes.
2. FRITCHII_JIT_FAR_CODE=1 alone: boot, market, fps vs off (fresh-launch A/B, boot switch);
   watch for crashes in slow paths (NaN/fallback code) and the fill line at clears.
3. FAR_CODE=1 + CHAIN=4 (compact links): exactness first (boot, play a minute), then fps.
4. + CHAIN=6 (backward halt checks): check pausing, interrupts, loading screens, shutdown still
   respond (halts must still land).
5. + CHAIN=7 (chaining), depth 4; then depth 0/8 if it helps.
6. Hot pack with the per-core rel lists: FRITCHII_JIT_HOT_LIST=...\hotlist-300-core{core}-rel.txt,
   FRITCHII_JIT_HOT_LIST_ORDER=1 (optional; Fable expects 0 to 1%).

## Open risks
- Far code: a forward Xbyak label from far code to a label defined later in near code would
  throw (rewrite past maxSize_). Checked none exist (deferred code only jumps back to defined
  labels); a new deferred emit doing that would abort at emit with a clear message.
- Short forward jumps over leaves (If/CheckBit) are the same kinds as before; a leaf over 127
  bytes would throw at emit (true before too).
- CHAIN=2: a halt raised inside a block (unmapped access without the debugger) stops a few
  forward blocks later. Memory aborts with the debugger stay immediate.
- CHAIN=1 compiles code before it runs: guest code changed without cache maintenance would be
  stale (UB on hardware too). Only readable successors are chained.
- Far split 5/8 : 3/8 is a guess; read the fill line.
- Dump mode 1 writes host bytes early (links shown as of then); mode 2 holds records in memory
  until a clear or shutdown.
