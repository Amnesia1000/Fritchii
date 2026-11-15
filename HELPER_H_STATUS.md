# Helper H status (done, nothing built or run)

## Commits (branch jit-code)
- cab330cd22 FRITCHII_JIT_FLAG_BRANCH (boot 0..2): conditional terminals branch on the host flags.
- 3ded56dd35 FRITCHII_JIT_PIN (boot), FRITCHII_JIT_VERIFY (boot, with PIN), FRITCHII_JIT_PIN_STATS (boot, dry).

All default 0 = byte-identical old code. Compile-checked (check_compile.py): a64_emit_x64.cpp,
block_of_code.cpp, a64_interface.cpp, emit_x64.cpp, a64_emit_x64_memory.cpp,
a32_emit_x64_memory.cpp, a32_interface.cpp, a64_jitstate.cpp, backend/exception_handler_windows.cpp.

## Switches
- FRITCHII_JIT_FLAG_BRANCH=1|2 (see the cab330cd22 message; 1 If, 2 + CheckBit).
- FRITCHII_JIT_PIN=1 (Windows only): guest registers from FRITCHII_JIT_PIN_REGS (comma list of
  sp, x0..x30; default "sp,x0") in rsi, and in rbp only with FRITCHII_JIT_NO_FRAME_POINTER=1
  (without it only the first one is pinned, logged). Log line "FRITCHII_JIT_PIN=1: sp in rsi, x0 in rbp".
- FRITCHII_JIT_VERIFY=1 (needs PIN=1): write-through + compare at dispatcher entries, exit and
  every call out. Line "JIT pin check ... mismatches N new, M total (must be 0)" once a second
  per core. On a mismatch the state block's value wins (that is the old code's value).
- FRITCHII_JIT_PIN_STATS=1: counts guest register reads/writes in generated code, top 8 per core
  every 10 s ("JIT register use ..."). Turns FLAG_BRANCH off. Dry, costs fps, use to pick PIN_REGS.

## How PIN works
- BlockOfCode has a new last ctor parameter PinConfig (A32 passes none). Loads after rcp in both
  prologues, store at return_to_caller_mxcsr_already_exited (every exit path goes there).
- CallFunction/CallLambda (all ArgCallback/SimpleCallback/Devirtualize calls) and the raw call in
  EmitCallHostFunction: store before, reload after. Raw calls to the fastmem/page-table fallback
  thunks and the 128-bit accessors are covered because those thunks call through CallFunction.
- Hook off for the exception handler code (emitted in the EmitX64 base ctor, before
  A64EmitX64's body turns it on) and for the run_code prologue (GetTicksRemaining before rcp),
  and off again after the exit store (AddTicks).
- Emit drops the pinned host regs from the allocator set. Get = mov from host reg,
  Set = mov into it (imm32 sign extends like the old qword store, SetW zero extends).
- Jump sizes: dispatcher exit jumps and the exclusive-store je(end) are T_NEAR when pins exist
  (Xbyak throws on a short forward jump that ends up too far). Audited the other forward short
  jumps over call sites: none.

## Exactness argument
- Host reg == the value the state block would hold in old code at every point in generated code:
  loaded from the state at entry; only the Set emitters write it (no emitter or shared handler
  names rsi/rbp, the allocator cannot pick them, C callees preserve them, Windows ABI);
  everything that reads or writes guest registers outside the Get/Set emitters is C code
  reached through a call out, which sees a store before and a reload after; the exit stores.
- Only Get/Set emitters touch A64JitState::reg/sp in generated code (grep).
- PIN code adds no flag writes in blocks (mov only), so FLAG_BRANCH's flag-neutral list holds.
  VERIFY's compares are only at calls/dispatcher/exit, where flags are dead.

## Open risks
- Anything reading guest registers from another thread while the core runs sees stale values
  for pinned regs (diagnostics only; already racy before).
- Stack walkers following rbp (only with NO_FRAME_POINTER=1, which already left rbp undefined).
- A future emitter naming rsi/rbp, or a future raw `code.call` that bypasses CallFunction.
  VERIFY catches both live.
- New short forward jumps over call sites would throw at emit with PIN on (crash at boot or first
  block); the audit found none today.
- Helper G overlap: block_of_code.h (structs, ctor param, CallFunction/CallLambda bodies, new
  members/methods), block_of_code.cpp (ctor, two Load lines after rcp, dispatcher entries,
  exit, new methods at the end before SwitchMxcsrOnEntry), a64_emit_x64.cpp (ctor hook block,
  the reg_alloc.Reset lambda, Get/Set emitters, new functions after FlagBranchMode). Not
  touched: Emit's terminals, far code, chaining, EnsurePatchLocationSize.

## Test order
1. FLAG_BRANCH=1, then 2 (boot, OoT runs, pot room fps).
2. PIN_STATS=1 one short session (pot room): read which registers lead. Off for fps.
3. PIN=1 + NO_FRAME_POINTER=1 + VERIFY=1 with the PIN_STATS winners: boots, mismatches stay 0
   through a load and the market.
4. PIN=1 + NO_FRAME_POINTER=1 without VERIFY: fps A/B (boot switch, fresh launches).
5. If it wins: more pins need rbx/rdi/r12 freed in the shared handlers (helper G area), step 2.
