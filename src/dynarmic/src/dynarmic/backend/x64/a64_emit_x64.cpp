// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/x64/a64_emit_x64.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <cstdlib>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include "common/assert.h"
#include "common/common_types.h"
#include "common/logging.h"
#include "dynarmic/ir/terminal.h"
#include "dynarmic/mcl/integer_of_size.hpp"
#include <boost/container/static_vector.hpp>

#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/backend/x64/abi.h"
#include "dynarmic/backend/x64/block_of_code.h"
#include "dynarmic/backend/x64/devirtualize.h"
#include "dynarmic/backend/x64/emit_x64.h"
#include "dynarmic/backend/x64/nzcv_util.h"
#include "dynarmic/backend/x64/perf_map.h"
#include "dynarmic/backend/x64/stack_layout.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/frontend/A64/a64_types.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/cond.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"

// TODO: Have ARM flags in host flags and not have them use up GPR registers unless necessary.
// TODO: Actually implement that proper instruction selector you've always wanted to sweetheart.

namespace Dynarmic::Backend::X64 {

using namespace Xbyak::util;

A64EmitContext::A64EmitContext(const A64::UserConfig& conf, RegAlloc& reg_alloc, IR::Block& block, boost::container::stable_vector<Xbyak::Label>& shared_labels)
    : EmitContext(reg_alloc, block, shared_labels)
    , conf(conf)
{}

A64::LocationDescriptor A64EmitContext::Location() const {
    return A64::LocationDescriptor{block.Location()};
}

bool A64EmitContext::IsSingleStep() const {
    return Location().SingleStepping();
}

FP::FPCR A64EmitContext::FPCR(bool fpcr_controlled) const {
    return fpcr_controlled ? Location().FPCR() : Location().FPCR().ASIMDStandardValue();
}

namespace {

// Fritchii: FRITCHII_JIT_DISPATCH_BITS, the A64 fast dispatch table holds 2^bits entries
// (12..20, default 20, which is the old fixed size). The table only caches block lookups (a
// miss takes the full lookup and refills the slot), so its size changes nothing the guest
// sees. Read once per process; logged once.
unsigned FastDispatchTableBits(unsigned default_bits, unsigned min_bits) {
    static const unsigned bits = [default_bits, min_bits] {
        unsigned value = default_bits;
        const char* text = std::getenv("FRITCHII_JIT_DISPATCH_BITS");
        if (text != nullptr && *text != '\0') {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(text, &end, 10);
            if (end != text && parsed >= min_bits && parsed <= default_bits) {
                value = unsigned(parsed);
            } else {
                LOG_WARNING(Core_ARM, "FRITCHII_JIT_DISPATCH_BITS={} is outside {}..{}, using {}",
                            text, min_bits, default_bits, default_bits);
            }
        }
        LOG_INFO(Core_ARM, "JIT fast dispatch table: 2^{} entries ({} KiB per core)", value,
                 (size_t{1} << value) * 16 / 1024);
        return value;
    }();
    return bits;
}

// Fritchii: a whole-number boot switch, read once per process.
int EnvSwitch(const char* name) {
    const char* text = std::getenv(name);
    return text == nullptr ? 0 : std::atoi(text);
}

}  // namespace

int A64EmitX64::IndirectCacheMode() {
    static const int mode = [] {
        const int value = EnvSwitch("FRITCHII_JIT_INDIRECT_CACHE");
        if (value < 0 || value > 3) {
            LOG_WARNING(Core_ARM, "FRITCHII_JIT_INDIRECT_CACHE={} is outside 0..3, using 0", value);
            return 0;
        }
        if (value != 0) {
            LOG_INFO(Core_ARM, "FRITCHII_JIT_INDIRECT_CACHE={}: {}", value,
                     value == 1   ? "dry, per-site last-target counters"
                     : value == 2 ? "per-site caches at indirect branches"
                                  : "per-site caches at indirect branches, returns pop inline");
        }
        return value;
    }();
    return mode;
}

bool A64EmitX64::OmitFramePointer() {
    static const bool omit = [] {
        const bool value = EnvSwitch("FRITCHII_JIT_NO_FRAME_POINTER") != 0;
        if (value) {
            LOG_INFO(Core_ARM, "FRITCHII_JIT_NO_FRAME_POINTER=1: blocks do not set up rbp");
        }
        return value;
    }();
    return omit;
}

bool A64EmitX64::FarCodeEnabled() {
    static const bool enabled = [] {
        const bool value = EnvSwitch("FRITCHII_JIT_FAR_CODE") != 0;
        if (value) {
            LOG_INFO(Core_ARM, "FRITCHII_JIT_FAR_CODE=1: slow paths in a far region (3/8 of the code cache)");
        }
        return value;
    }();
    return enabled;
}

int A64EmitX64::ChainBits() {
    static const int bits = [] {
        const int value = EnvSwitch("FRITCHII_JIT_CHAIN");
        if (value < 0 || value > 7) {
            LOG_WARNING(Core_ARM, "FRITCHII_JIT_CHAIN={} is outside 0..7, using 0", value);
            return 0;
        }
        if (value != 0) {
            LOG_INFO(Core_ARM, "FRITCHII_JIT_CHAIN={}: chain successors {}, halt checks only backward {}, compact links {}",
                     value, (value & 1) != 0 ? "on" : "off", (value & 2) != 0 ? "on" : "off",
                     (value & 5) != 0 ? "on" : "off");
        }
        return value;
    }();
    return bits;
}

int A64EmitX64::ChainDepth() {
    static const int depth = [] {
        const char* text = std::getenv("FRITCHII_JIT_CHAIN_DEPTH");
        const int value = text == nullptr ? 4 : std::atoi(text);
        return std::clamp(value, 0, 64);
    }();
    return depth;
}

A64EmitX64::A64EmitX64(BlockOfCode& code, A64::UserConfig conf, A64::Jit* jit_interface)
        : EmitX64(code), conf(conf)
        , fast_dispatch_table_size{size_t{1} << FastDispatchTableBits(fast_dispatch_table_default_bits, fast_dispatch_table_min_bits)}
        , fast_dispatch_table_mask{u32((fast_dispatch_table_size - 1) * sizeof(FastDispatchEntry))}
        , fast_dispatch_table{new FastDispatchEntry[fast_dispatch_table_size]}
        , jit_interface{jit_interface} {
    // Fritchii: the per-site caches fall back on the fast dispatch table.
    indirect_cache_mode = conf.HasOptimization(OptimizationFlag::FastDispatch) ? IndirectCacheMode() : 0;
    omit_frame_pointer = OmitFramePointer();
    if (indirect_cache_mode != 0) {
        indirect_slots.reset(new FastDispatchEntry[indirect_slot_capacity]);
    }
    GenMemory128Accessors();
    GenFastmemFallbacks();
    GenTerminalHandlers();
    // Fritchii (FRITCHII_JIT_FAR_CODE): 5/8 of the space for blocks, 3/8 for their slow paths.
    if (FarCodeEnabled() && code.GetTotalCodeSize() >= (size_t{64} << 20)) {
        code.EnableFarCode(code.GetTotalCodeSize() / 8 * 3);
        far_code = true;
    }
    // Fritchii (FRITCHII_JIT_CHAIN): the halt checks it moves stand in for cycle counting's
    // checks too, so not with cycle counting; compact links need the far region for their stubs.
    if (const int chain = ChainBits(); chain != 0 && conf.HasOptimization(OptimizationFlag::BlockLinking)) {
        if (conf.enable_cycle_counting) {
            LOG_WARNING(Core_ARM, "FRITCHII_JIT_CHAIN={} is off with cycle counting", chain);
        } else {
            backward_halt_checks = (chain & 2) != 0;
            if ((chain & 5) != 0) {
                if (far_code) {
                    compact_links = true;
                    chain_successors = (chain & 1) != 0;
                } else if (conf.processor_id == 0) {
                    LOG_WARNING(Core_ARM, "FRITCHII_JIT_CHAIN={}: compact links and chaining need FRITCHII_JIT_FAR_CODE=1, off", chain);
                }
            }
        }
    }
    code.PreludeComplete();
    ClearFastDispatchTable();

    if (conf.fastmem_pointer.has_value()) {
        exception_handler.SetFastmemCallback([this](u64 rip_) {
            return FastmemCallback(rip_);
        });
    }
}

A64EmitX64::~A64EmitX64() = default;

// Fritchii: a throw here (an Xbyak error) used to end the process without a word: it is called
// from generated code (the dispatcher's block lookup), whose frames the unwinder cannot cross.
A64EmitX64::BlockDescriptor A64EmitX64::Emit(IR::Block& block, bool chained) noexcept try {
    if (conf.very_verbose_debugging_output) [[unlikely]] {
        std::puts(IR::DumpBlock(block).c_str());
    }

    code.EnableWriting();
    this->reg_alloc.Reset([this] {
        std::bitset<32> gprs = any_gpr;
        if (conf.fastmem_pointer)
            gprs.reset(size_t(HostLoc::R13));
        if (conf.page_table)
            gprs.reset(size_t(HostLoc::R14));
        return gprs;
    }(), any_xmm);

    A64EmitContext ctx{conf, reg_alloc, block, shared_labels};

    // Start emitting.
    if (!chained) {
        code.align();
    }
    const auto* const entrypoint = code.getCurr();
    // Fritchii: FRITCHII_JIT_NO_FRAME_POINTER leaves rbp as the dispatcher set it (no block
    // allocates rbp either way, reg_alloc.cpp): only stack walkers that follow rbp see a change.
    if (!omit_frame_pointer) {
        code.mov(code.qword[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, abi_base_pointer)], rbp);
        code.lea(rbp, code.ptr[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, abi_base_pointer) - 8]);
    }

    DEBUG_ASSERT(block.GetCondition() == IR::Cond::AL);
    typedef void (EmitX64::*EmitHandlerFn)(EmitContext& context, IR::Inst* inst);
    constexpr EmitHandlerFn opcode_handlers[] = {
#define OPCODE(name, type, ...) &EmitX64::Emit##name,
#define A32OPC(name, type, ...)
#define A64OPC(name, type, ...)
#include "dynarmic/ir/opcodes.inc"
#undef OPCODE
#undef A32OPC
#undef A64OPC
    };
    typedef void (A64EmitX64::*A64EmitHandlerFn)(A64EmitContext& context, IR::Inst* inst);
    constexpr A64EmitHandlerFn a64_handlers[] = {
#define OPCODE(...)
#define A32OPC(...)
#define A64OPC(name, type, ...) &A64EmitX64::EmitA64##name,
#include "dynarmic/ir/opcodes.inc"
#undef OPCODE
#undef A32OPC
#undef A64OPC
    };

    for (auto& inst : block.instructions) {
        auto const opcode = inst.GetOpcode();
        // Call the relevant Emit* member function.
        switch (opcode) {
#define OPCODE(name, type, ...) case IR::Opcode::name: goto opcode_branch;
#define A32OPC(name, type, ...)
#define A64OPC(name, type, ...) case IR::Opcode::A64##name: goto a64_branch;
#include "dynarmic/ir/opcodes.inc"
#undef OPCODE
#undef A32OPC
#undef A64OPC
        default:
            UNREACHABLE();
        }
opcode_branch:
        (this->*opcode_handlers[size_t(opcode)])(ctx, &inst);
        goto finish_this_inst;
a64_branch:
        (this->*a64_handlers[size_t(opcode) - std::size(opcode_handlers)])(ctx, &inst);
finish_this_inst:
        ctx.reg_alloc.EndOfAllocScope();
#ifndef NDEBUG
        if (conf.very_verbose_debugging_output)
            EmitVerboseDebuggingOutput(reg_alloc);
#endif
    }

    reg_alloc.AssertNoMoreUses();
    if (conf.enable_cycle_counting)
        EmitAddCycles(block.CycleCount());
    if (!omit_frame_pointer) {
        code.mov(rbp, code.qword[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, abi_base_pointer)]);
    }
    current_end_location = block.EndLocation();
    chain_target.reset();
    EmitTerminal(block.terminal, ctx.Location().SetSingleStepping(false), ctx.IsSingleStep());
    if (chain_target && code.getCurr() != chain_slot_end) {
        chain_target.reset(); // something follows the last slot
    }
    last_far_entry = nullptr;
    last_far_size = 0;
    if (far_code) {
        // Fritchii (FRITCHII_JIT_FAR_CODE): the terminal ends in a jump; the slow paths go to the
        // far region (their jumps from the block are near jumps, their jumps back go to labels
        // already defined), so the next block follows this one's terminal.
        if (!ctx.deferred_emits.empty()) {
            code.SwitchToFarCode();
            last_far_entry = code.getCurr();
            for (auto& deferred_emit : ctx.deferred_emits)
                deferred_emit();
            code.int3();
            last_far_size = size_t(code.getCurr() - static_cast<const u8*>(last_far_entry));
            code.SwitchToNearCode();
        }
    } else {
        code.int3();
        for (auto& deferred_emit : ctx.deferred_emits)
            deferred_emit();
        code.int3();
    }

    const size_t size = size_t(code.getCurr() - entrypoint);

    const A64::LocationDescriptor descriptor{block.Location()};
    const A64::LocationDescriptor end_location{block.EndLocation()};

    const auto range = boost::icl::discrete_interval<u64>::closed(descriptor.PC(), end_location.PC() - 1);
    block_ranges.AddRange(range, descriptor);

    auto bdesc = RegisterBlock(descriptor, entrypoint, size);
    code.DisableWriting();
    shared_labels.clear();
    return bdesc;
} catch (const std::exception& error) {
    ::Common::Log::FmtCrashMessage(__FILE__, __LINE__, __func__,
                                   "JIT emit of block {:#x} (core {}) threw: {}; indirect sites {}",
                                   A64::LocationDescriptor{block.Location()}.PC(), conf.processor_id,
                                   error.what(), indirect_slots_used);
    std::abort();
}

void A64EmitX64::ClearCache() {
    EmitX64::ClearCache();
    block_ranges.ClearCache();
    ClearFastDispatchTable();
    ClearIndirectSlots(true);
    fastmem_patch_info.clear();
    link_stubs.clear();
    halt_stubs.clear();
    chain_target.reset();
}

std::optional<IR::LocationDescriptor> A64EmitX64::ChainCandidate() const {
    if (!chain_successors || !chain_target || code.getCurr() != chain_slot_end) {
        return std::nullopt;
    }
    return chain_target;
}

CodePtr A64EmitX64::LinkStub(const IR::LocationDescriptor& target) {
    if (const auto it = link_stubs.find(target.Value()); it != link_stubs.end()) {
        return it->second;
    }
    code.SwitchToFarCode();
    const CodePtr stub = code.getCurr();
    code.mov(rax, A64::LocationDescriptor{target}.PC());
    code.mov(qword[code.ABI_JIT_PTR + offsetof(A64JitState, pc)], rax);
    code.jmp(code.GetReturnFromRunCodeAddress(), code.T_NEAR);
    code.SwitchToNearCode();
    link_stubs.emplace(target.Value(), stub);
    return stub;
}

CodePtr A64EmitX64::HaltStub(const IR::LocationDescriptor& target) {
    if (const auto it = halt_stubs.find(target.Value()); it != halt_stubs.end()) {
        return it->second;
    }
    code.SwitchToFarCode();
    const CodePtr stub = code.getCurr();
    code.mov(rax, A64::LocationDescriptor{target}.PC());
    code.mov(qword[code.ABI_JIT_PTR + offsetof(A64JitState, pc)], rax);
    code.ForceReturnFromRunCode();
    code.SwitchToNearCode();
    halt_stubs.emplace(target.Value(), stub);
    return stub;
}

void A64EmitX64::EmitFritchiiLink(const IR::LocationDescriptor& target, IR::LocationDescriptor initial_location, bool checked_by_default) {
    // FRITCHII_JIT_CHAIN=2: only a link that can close a loop checks for a halt.
    const bool checked = backward_halt_checks
                           ? A64::LocationDescriptor{target}.PC() <= A64::LocationDescriptor{initial_location}.PC()
                           : checked_by_default;
    const auto next_bb = GetBasicBlock(target);
    if (compact_links) {
        // Both stubs exist before the slot is recorded: Unpatch rewrites it to the link stub.
        static_cast<void>(LinkStub(target));
        if (checked) {
            const CodePtr halt_stub = HaltStub(target);
            code.cmp(dword[code.ABI_JIT_PTR + offsetof(A64JitState, halt_reason)], 0);
            code.jne(halt_stub);
        }
        AddPatchLocation(target, EmitX64::PatchKind::Jmp);
        EmitPatchJmp(target, next_bb ? next_bb->entrypoint : nullptr);
        chain_target = target;
        chain_slot_end = code.getCurr();
        return;
    }
    if (checked) {
        // LinkBlock's shape.
        code.cmp(dword[code.ABI_JIT_PTR + offsetof(A64JitState, halt_reason)], 0);
        AddPatchLocation(target, EmitX64::PatchKind::Jz);
        if (next_bb) {
            EmitPatchJz(target, next_bb->entrypoint);
        } else {
            EmitPatchJz(target);
        }
        code.mov(rax, A64::LocationDescriptor{target}.PC());
        code.mov(qword[code.ABI_JIT_PTR + offsetof(A64JitState, pc)], rax);
        code.ForceReturnFromRunCode();
    } else {
        // LinkBlockFast's shape.
        AddPatchLocation(target, EmitX64::PatchKind::Jmp);
        if (next_bb) {
            EmitPatchJmp(target, next_bb->entrypoint);
        } else {
            EmitPatchJmp(target);
        }
    }
}

void A64EmitX64::InvalidateCacheRanges(const boost::icl::interval_set<u64>& ranges) {
    // Fritchii: a slot may hold a block just removed; the table entries go one by one (Unpatch),
    // the slots all go once after (code removal is rare, they refill from the table).
    indirect_bulk_invalidation = true;
    InvalidateBasicBlocks(block_ranges.InvalidateRanges(ranges));
    indirect_bulk_invalidation = false;
    ClearIndirectSlots(false);
}

A64EmitX64::FastDispatchEntry* A64EmitX64::AllocateIndirectSlot() {
    if (!indirect_slots || indirect_slots_used >= indirect_slot_capacity) {
        return nullptr;
    }
    return &indirect_slots[indirect_slots_used++];
}

void A64EmitX64::ClearIndirectSlots(bool release) {
    if (!indirect_slots) {
        return;
    }
    std::fill_n(indirect_slots.get(), indirect_slots_used, FastDispatchEntry{});
    if (release) {
        indirect_slots_used = 0;
    }
}

void A64EmitX64::EmitCalculateLocationDescriptor() {
    // This calculation has to match up with A64::LocationDescriptor::UniqueHash
    code.mov(rdi, qword[code.ABI_JIT_PTR + offsetof(A64JitState, pc)]);
    code.mov(rcx, A64::LocationDescriptor::pc_mask);
    code.and_(rcx, rdi);
    code.mov(ebx, dword[code.ABI_JIT_PTR + offsetof(A64JitState, fpcr)]);
    code.and_(ebx, A64::LocationDescriptor::fpcr_mask);
    code.shl(rbx, A64::LocationDescriptor::fpcr_shift);
    code.or_(rbx, rcx);
}

void A64EmitX64::LogIndirectCounters() {
    const s64 now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    if (indirect_log_ns == 0) {
        indirect_log_ns = now;
        indirect_counters_logged = indirect_counters;
        return;
    }
    if (now - indirect_log_ns < 10'000'000'000LL) {
        return;
    }
    const double seconds = double(now - indirect_log_ns) / 1e9;
    std::array<u64, IndirectCounterCount> d{};
    for (size_t i = 0; i < IndirectCounterCount; ++i) {
        d[i] = indirect_counters[i] - indirect_counters_logged[i];
    }
    const auto share = [](u64 part, u64 whole) {
        return whole == 0 ? 0.0 : 100.0 * double(part) / double(whole);
    };
    if (indirect_cache_mode == 1) {
        LOG_INFO(Core_ARM,
                 "JIT indirect sites (FRITCHII_JIT_INDIRECT_CACHE=1) core {} {:.1f} s: indirect "
                 "branches {:.0f}/s, same target as the site's last {:.1f}%; returns {:.0f}/s, same "
                 "target as the site's last {:.1f}%, return buffer hits {:.1f}%; sites {}",
                 conf.processor_id, seconds, double(d[IndirectRuns]) / seconds,
                 share(d[IndirectSameTarget], d[IndirectRuns]), double(d[ReturnRuns]) / seconds,
                 share(d[ReturnSameTarget], d[ReturnRuns]), share(d[ReturnRsbHits], d[ReturnRuns]),
                 indirect_slots_used);
    } else {
        LOG_INFO(Core_ARM,
                 "JIT indirect sites (FRITCHII_JIT_INDIRECT_CACHE={}) core {} {:.1f} s: slot misses "
                 "{:.0f}/s; sites {}",
                 indirect_cache_mode, conf.processor_id, seconds, double(d[SlotMisses]) / seconds,
                 indirect_slots_used);
    }
    indirect_log_ns = now;
    indirect_counters_logged = indirect_counters;
}

void A64EmitX64::ClearFastDispatchTable() {
    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        std::fill_n(fast_dispatch_table.get(), fast_dispatch_table_size, FastDispatchEntry{});
    }
}

void A64EmitX64::GenTerminalHandlers() {
    // PC ends up in rcx, location_descriptor ends up in rbx
    //static_assert(ABI_ALL_CALLEE_SAVE.test(size_t(HostLoc::R12)));
    const auto calculate_location_descriptor = [this] {
        // TODO: Optimization is available here based on known state of fpcr.
        EmitCalculateLocationDescriptor();
    };

    Xbyak::Label fast_dispatch_cache_miss;
    Xbyak::Label rsb_cache_miss;

    code.align();
    terminal_handler_pop_rsb_hint = code.getCurr<const void*>();
    calculate_location_descriptor();
    code.mov(eax, dword[code.ABI_JIT_PTR + offsetof(A64JitState, rsb_ptr)]);
    code.sub(eax, 1);
    code.and_(eax, u32(A64JitState::RSB_PTR_MASK));
    code.mov(dword[code.ABI_JIT_PTR + offsetof(A64JitState, rsb_ptr)], eax);
    code.cmp(rbx, qword[code.ABI_JIT_PTR + offsetof(A64JitState, rsb_location_descriptors) + rax * sizeof(u64)]);
    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        code.jne(rsb_cache_miss, code.T_NEAR);
    } else {
        code.jne(code.GetReturnFromRunCodeAddress());
    }
    code.mov(rax, qword[code.ABI_JIT_PTR + offsetof(A64JitState, rsb_codeptrs) + rax * sizeof(u64)]);
    code.jmp(rax);
    PerfMapRegister(terminal_handler_pop_rsb_hint, code.getCurr(), "a64_terminal_handler_pop_rsb_hint");

    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        code.align();
        terminal_handler_fast_dispatch_hint = code.getCurr<const void*>();
        calculate_location_descriptor();
        code.L(rsb_cache_miss);
        terminal_handler_fast_dispatch_after_calc = code.getCurr<const void*>();
        code.mov(r8, u64(fast_dispatch_table.get()));
        //code.mov(r12, qword[code.ABI_JIT_PTR + offsetof(A64JitState, pc)]);
        code.mov(r12, rbx);
        if (code.HasHostFeature(HostFeature::SSE42)) {
            code.crc32(r12, r8);
        }
        code.and_(r12d, fast_dispatch_table_mask);
        code.lea(r12, ptr[r8 + r12]);
        code.cmp(rbx, qword[r12 + offsetof(FastDispatchEntry, location_descriptor)]);
        code.jne(fast_dispatch_cache_miss, code.T_NEAR);
        code.jmp(ptr[r12 + offsetof(FastDispatchEntry, code_ptr)]);
        code.L(fast_dispatch_cache_miss);
        code.mov(qword[r12 + offsetof(FastDispatchEntry, location_descriptor)], rbx);
        code.LookupBlock();
        code.mov(ptr[r12 + offsetof(FastDispatchEntry, code_ptr)], rax);
        code.jmp(rax);
        PerfMapRegister(terminal_handler_fast_dispatch_hint, code.getCurr(), "a64_terminal_handler_fast_dispatch_hint");

        code.align();
        fast_dispatch_table_lookup = code.getCurr<FastDispatchEntry& (*)(u64)>();
        code.mov(code.ABI_PARAM2, u64(fast_dispatch_table.get()));
        if (code.HasHostFeature(HostFeature::SSE42)) {
            code.crc32(code.ABI_PARAM1, code.ABI_PARAM2);
        }
        code.and_(code.ABI_PARAM1.cvt32(), fast_dispatch_table_mask);
        code.lea(code.ABI_RETURN, code.ptr[code.ABI_PARAM2 + code.ABI_PARAM1]);
        code.ret();
        PerfMapRegister(fast_dispatch_table_lookup, code.getCurr(), "a64_fast_dispatch_table_lookup");
    }

    // Fritchii: FRITCHII_JIT_INDIRECT_CACHE's shared parts (the sites are in EmitTerminalImpl).
    // Each ends in the fast dispatch handler right after its descriptor calculation, so what a
    // site's guest code reaches is decided by the same table and lookup as before.
    if (indirect_cache_mode == 1) {
        // Dry, rax = the site's slot: count a target equal to the site's last, then remember it.
        const auto count_same_target = [this](IndirectCounter runs, IndirectCounter same) {
            Xbyak::Label differs;
            code.mov(r8, u64(indirect_counters.data()));
            code.inc(qword[r8 + runs * sizeof(u64)]);
            code.cmp(rbx, qword[rax + offsetof(FastDispatchEntry, location_descriptor)]);
            code.jne(differs);
            code.inc(qword[r8 + same * sizeof(u64)]);
            code.L(differs);
            code.mov(qword[rax + offsetof(FastDispatchEntry, location_descriptor)], rbx);
        };

        code.align();
        terminal_handler_indirect_dry = code.getCurr<const void*>();
        calculate_location_descriptor();
        count_same_target(IndirectRuns, IndirectSameTarget);
        code.jmp(terminal_handler_fast_dispatch_after_calc);
        PerfMapRegister(terminal_handler_indirect_dry, code.getCurr(), "a64_terminal_handler_indirect_dry");

        if (conf.HasOptimization(OptimizationFlag::ReturnStackBuffer)) {
            // As terminal_handler_pop_rsb_hint, counting its hits (r8 still holds the counters).
            code.align();
            terminal_handler_pop_rsb_dry = code.getCurr<const void*>();
            calculate_location_descriptor();
            count_same_target(ReturnRuns, ReturnSameTarget);
            code.mov(eax, dword[code.ABI_JIT_PTR + offsetof(A64JitState, rsb_ptr)]);
            code.sub(eax, 1);
            code.and_(eax, u32(A64JitState::RSB_PTR_MASK));
            code.mov(dword[code.ABI_JIT_PTR + offsetof(A64JitState, rsb_ptr)], eax);
            code.cmp(rbx, qword[code.ABI_JIT_PTR + offsetof(A64JitState, rsb_location_descriptors) + rax * sizeof(u64)]);
            code.jne(terminal_handler_fast_dispatch_after_calc);
            code.inc(qword[r8 + ReturnRsbHits * sizeof(u64)]);
            code.mov(rax, qword[code.ABI_JIT_PTR + offsetof(A64JitState, rsb_codeptrs) + rax * sizeof(u64)]);
            code.jmp(rax);
            PerfMapRegister(terminal_handler_pop_rsb_dry, code.getCurr(), "a64_terminal_handler_pop_rsb_dry");
        }
    } else if (indirect_cache_mode >= 2) {
        // A site's slot did not match (rbx = descriptor, rax = slot): the table entry as the fast
        // dispatch handler finds it; on a table hit the slot takes it, else the handler's own
        // miss path (lookup, table refill) runs and the slot is filled at the next miss.
        code.align();
        terminal_handler_indirect_slot_miss = code.getCurr<const void*>();
        code.mov(r8, u64(indirect_counters.data()));
        code.inc(qword[r8 + SlotMisses * sizeof(u64)]);
        code.mov(r8, u64(fast_dispatch_table.get()));
        code.mov(r12, rbx);
        if (code.HasHostFeature(HostFeature::SSE42)) {
            code.crc32(r12, r8);
        }
        code.and_(r12d, fast_dispatch_table_mask);
        code.lea(r12, ptr[r8 + r12]);
        code.cmp(rbx, qword[r12 + offsetof(FastDispatchEntry, location_descriptor)]);
        code.jne(terminal_handler_fast_dispatch_after_calc);
        code.mov(rcx, qword[r12 + offsetof(FastDispatchEntry, code_ptr)]);
        code.mov(qword[rax + offsetof(FastDispatchEntry, location_descriptor)], rbx);
        code.mov(qword[rax + offsetof(FastDispatchEntry, code_ptr)], rcx);
        code.jmp(rcx);
        PerfMapRegister(terminal_handler_indirect_slot_miss, code.getCurr(), "a64_terminal_handler_indirect_slot_miss");
    }
}

void A64EmitX64::EmitPushRSB(EmitContext& ctx, IR::Inst* inst) {
    if (!conf.HasOptimization(OptimizationFlag::ReturnStackBuffer)) {
        return;
    }

    EmitX64::EmitPushRSB(ctx, inst);
}

void A64EmitX64::EmitA64SetCheckBit(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const Xbyak::Reg8 to_store = ctx.reg_alloc.UseGpr(code, args[0]).cvt8();
    code.mov(code.byte[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, check_bit)], to_store);
}

void A64EmitX64::EmitA64GetCFlag(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr(code).cvt32();
    code.mov(result, dword[code.ABI_JIT_PTR + offsetof(A64JitState, cpsr_nzcv)]);
    code.shr(result, NZCV::x64_c_flag_bit);
    code.and_(result, 1);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetNZCVRaw(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 nzcv_raw = ctx.reg_alloc.ScratchGpr(code).cvt32();

    code.mov(nzcv_raw, dword[code.ABI_JIT_PTR + offsetof(A64JitState, cpsr_nzcv)]);

    if (code.HasHostFeature(HostFeature::FastBMI2)) {
        const Xbyak::Reg32 tmp = ctx.reg_alloc.ScratchGpr(code).cvt32();
        code.mov(tmp, NZCV::x64_mask);
        code.pext(nzcv_raw, nzcv_raw, tmp);
        code.shl(nzcv_raw, 28);
    } else {
        code.and_(nzcv_raw, NZCV::x64_mask);
        code.imul(nzcv_raw, nzcv_raw, NZCV::from_x64_multiplier);
        code.and_(nzcv_raw, NZCV::arm_mask);
    }

    ctx.reg_alloc.DefineValue(code, inst, nzcv_raw);
}

void A64EmitX64::EmitA64SetNZCVRaw(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const Xbyak::Reg32 nzcv_raw = ctx.reg_alloc.UseScratchGpr(code, args[0]).cvt32();

    code.shr(nzcv_raw, 28);
    if (code.HasHostFeature(HostFeature::FastBMI2)) {
        const Xbyak::Reg32 tmp = ctx.reg_alloc.ScratchGpr(code).cvt32();
        code.mov(tmp, NZCV::x64_mask);
        code.pdep(nzcv_raw, nzcv_raw, tmp);
    } else {
        code.imul(nzcv_raw, nzcv_raw, NZCV::to_x64_multiplier);
        code.and_(nzcv_raw, NZCV::x64_mask);
    }
    code.mov(dword[code.ABI_JIT_PTR + offsetof(A64JitState, cpsr_nzcv)], nzcv_raw);
}

void A64EmitX64::EmitA64SetNZCV(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const Xbyak::Reg32 to_store = ctx.reg_alloc.UseScratchGpr(code, args[0]).cvt32();
    code.mov(dword[code.ABI_JIT_PTR + offsetof(A64JitState, cpsr_nzcv)], to_store);
}

void A64EmitX64::EmitA64GetW(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr(code).cvt32();

    code.mov(result, dword[code.ABI_JIT_PTR + offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg)]);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetX(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();
    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr(code);

    code.mov(result, qword[code.ABI_JIT_PTR + offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg)]);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetS(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = qword[code.ABI_JIT_PTR + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm(code);
    code.movd(result, addr);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetD(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = qword[code.ABI_JIT_PTR + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm(code);
    code.movq(result, addr);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetQ(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = xword[code.ABI_JIT_PTR + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm(code);
    code.movaps(result, addr);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetSP(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr(code);
    code.mov(result, qword[code.ABI_JIT_PTR + offsetof(A64JitState, sp)]);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetFPCR(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr(code).cvt32();
    code.mov(result, dword[code.ABI_JIT_PTR + offsetof(A64JitState, fpcr)]);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

static u32 GetFPSRImpl(A64JitState* jit_state) {
    return jit_state->GetFpsr();
}

void A64EmitX64::EmitA64GetFPSR(A64EmitContext& ctx, IR::Inst* inst) {
    ctx.reg_alloc.HostCall(code, inst);
    code.mov(code.ABI_PARAM1, code.ABI_JIT_PTR);
    code.stmxcsr(code.dword[code.ABI_JIT_PTR + offsetof(A64JitState, guest_MXCSR)]);
    code.CallFunction(GetFPSRImpl);
}

void A64EmitX64::EmitA64SetW(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();
    const auto addr = qword[code.ABI_JIT_PTR + offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg)];
    if (args[1].FitsInImmediateS32()) {
        code.mov(addr, args[1].GetImmediateS32());
    } else {
        // TODO: zext tracking, xmm variant
        const Xbyak::Reg64 to_store = ctx.reg_alloc.UseScratchGpr(code, args[1]);
        code.mov(to_store.cvt32(), to_store.cvt32());
        code.mov(addr, to_store);
    }
}

void A64EmitX64::EmitA64SetX(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();
    const auto addr = qword[code.ABI_JIT_PTR + offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg)];
    if (args[1].FitsInImmediateS32()) {
        code.mov(addr, args[1].GetImmediateS32());
    } else if (args[1].IsInXmm(ctx.reg_alloc)) {
        const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(code, args[1]);
        code.movq(addr, to_store);
    } else {
        const Xbyak::Reg64 to_store = ctx.reg_alloc.UseGpr(code, args[1]);
        code.mov(addr, to_store);
    }
}

void A64EmitX64::EmitA64SetS(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = xword[code.ABI_JIT_PTR + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(code, args[1]);
    const Xbyak::Xmm tmp = ctx.reg_alloc.ScratchXmm(code);
    // TODO: Optimize
    code.pxor(tmp, tmp);
    code.movss(tmp, to_store);
    code.movaps(addr, tmp);
}

void A64EmitX64::EmitA64SetD(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = xword[code.ABI_JIT_PTR + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm to_store = ctx.reg_alloc.UseScratchXmm(code, args[1]);
    code.movq(to_store, to_store);  // TODO: Remove when able
    code.movaps(addr, to_store);
}

void A64EmitX64::EmitA64SetQ(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = xword[code.ABI_JIT_PTR + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(code, args[1]);
    code.movaps(addr, to_store);
}

void A64EmitX64::EmitA64SetSP(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const auto addr = qword[code.ABI_JIT_PTR + offsetof(A64JitState, sp)];
    if (args[0].FitsInImmediateS32()) {
        code.mov(addr, args[0].GetImmediateS32());
    } else if (args[0].IsInXmm(ctx.reg_alloc)) {
        const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(code, args[0]);
        code.movq(addr, to_store);
    } else {
        const Xbyak::Reg64 to_store = ctx.reg_alloc.UseGpr(code, args[0]);
        code.mov(addr, to_store);
    }
}

static void SetFPCRImpl(A64JitState* jit_state, u32 value) {
    jit_state->SetFpcr(value);
}

void A64EmitX64::EmitA64SetFPCR(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.HostCall(code, nullptr, {}, args[0]);
    code.mov(code.ABI_PARAM1, code.ABI_JIT_PTR);
    code.CallFunction(SetFPCRImpl);
    code.ldmxcsr(code.dword[code.ABI_JIT_PTR + offsetof(A64JitState, guest_MXCSR)]);
}

static void SetFPSRImpl(A64JitState* jit_state, u32 value) {
    jit_state->SetFpsr(value);
}

void A64EmitX64::EmitA64SetFPSR(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.HostCall(code, nullptr, {}, args[0]);
    code.mov(code.ABI_PARAM1, code.ABI_JIT_PTR);
    code.CallFunction(SetFPSRImpl);
    code.ldmxcsr(code.dword[code.ABI_JIT_PTR + offsetof(A64JitState, guest_MXCSR)]);
}

void A64EmitX64::EmitA64SetPC(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const auto addr = qword[code.ABI_JIT_PTR + offsetof(A64JitState, pc)];
    if (args[0].FitsInImmediateS32()) {
        code.mov(addr, args[0].GetImmediateS32());
    } else if (args[0].IsInXmm(ctx.reg_alloc)) {
        const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(code, args[0]);
        code.movq(addr, to_store);
    } else {
        const Xbyak::Reg64 to_store = ctx.reg_alloc.UseGpr(code, args[0]);
        code.mov(addr, to_store);
    }
}

void A64EmitX64::EmitA64CallSupervisor(A64EmitContext& ctx, IR::Inst* inst) {
    ctx.reg_alloc.HostCall(code, nullptr);
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ASSERT(args[0].IsImmediate());
    const u32 imm = args[0].GetImmediateU32();
    Devirtualize<&A64::UserCallbacks::CallSVC>(conf.callbacks).EmitCall(code, [&](RegList param) {
        code.mov(param[0], imm);
    });
    // The kernel would have to execute ERET to get here, which would clear exclusive state.
    code.mov(code.byte[code.ABI_JIT_PTR + offsetof(A64JitState, exclusive_state)], u8(0));
}

void A64EmitX64::EmitA64ExceptionRaised(A64EmitContext& ctx, IR::Inst* inst) {
    ctx.reg_alloc.HostCall(code, nullptr);
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ASSERT(args[0].IsImmediate() && args[1].IsImmediate());
    const u64 pc = args[0].GetImmediateU64();
    const u64 exception = args[1].GetImmediateU64();
    Devirtualize<&A64::UserCallbacks::ExceptionRaised>(conf.callbacks).EmitCall(code, [&](RegList param) {
        code.mov(param[0], pc);
        code.mov(param[1], exception);
    });
}

void A64EmitX64::EmitA64DataCacheOperationRaised(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.HostCall(code, nullptr, {}, args[1], args[2]);
    Devirtualize<&A64::UserCallbacks::DataCacheOperationRaised>(conf.callbacks).EmitCall(code);
}

void A64EmitX64::EmitA64InstructionCacheOperationRaised(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.HostCall(code, nullptr, {}, args[0], args[1]);
    Devirtualize<&A64::UserCallbacks::InstructionCacheOperationRaised>(conf.callbacks).EmitCall(code);
}

void A64EmitX64::EmitA64DataSynchronizationBarrier(A64EmitContext&, IR::Inst*) {
    code.mfence();
    code.lfence();
}

void A64EmitX64::EmitA64DataMemoryBarrier(A64EmitContext&, IR::Inst*) {
    code.mfence();
}

void A64EmitX64::EmitA64InstructionSynchronizationBarrier(A64EmitContext& ctx, IR::Inst*) {
    if (!conf.hook_isb) {
        return;
    }

    ctx.reg_alloc.HostCall(code, nullptr);
    Devirtualize<&A64::UserCallbacks::InstructionSynchronizationBarrierRaised>(conf.callbacks).EmitCall(code);
}

void A64EmitX64::EmitA64GetCNTFRQ(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr(code).cvt32();
    code.mov(result, conf.cntfrq_el0);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetCNTPCT(A64EmitContext& ctx, IR::Inst* inst) {
    ctx.reg_alloc.HostCall(code, inst);
    if (!conf.wall_clock_cntpct) {
        code.UpdateTicks();
    }
    Devirtualize<&A64::UserCallbacks::GetCNTPCT>(conf.callbacks).EmitCall(code);
}

void A64EmitX64::EmitA64GetCTR(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr(code).cvt32();
    code.mov(result, conf.ctr_el0);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetDCZID(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr(code).cvt32();
    code.mov(result, conf.dczid_el0);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetTPIDR(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr(code);
    if (conf.tpidr_el0) {
        code.mov(result, u64(conf.tpidr_el0));
        code.mov(result, qword[result]);
    } else {
        code.xor_(result.cvt32(), result.cvt32());
    }
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetTPIDRRO(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr(code);
    if (conf.tpidrro_el0) {
        code.mov(result, u64(conf.tpidrro_el0));
        code.mov(result, qword[result]);
    } else {
        code.xor_(result.cvt32(), result.cvt32());
    }
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64SetTPIDR(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const Xbyak::Reg64 value = ctx.reg_alloc.UseGpr(code, args[0]);
    const Xbyak::Reg64 addr = ctx.reg_alloc.ScratchGpr(code);
    if (conf.tpidr_el0) {
        code.mov(addr, u64(conf.tpidr_el0));
        code.mov(qword[addr], value);
    }
}

std::string A64EmitX64::LocationDescriptorToFriendlyName(const IR::LocationDescriptor& ir_descriptor) const {
    const A64::LocationDescriptor descriptor{ir_descriptor};
    return fmt::format("a64_{:016X}_fpcr{:08X}",
                       descriptor.PC(),
                       descriptor.FPCR().Value());
}

namespace {
/// Fritchii (FRITCHII_JIT_CHAIN): the leaf is a direct link to `location`.
bool LeafLinksTo(const IR::Term::LeafTerminal& leaf, const IR::LocationDescriptor& location) {
    if (const auto x = std::get_if<IR::Term::LinkBlock>(&leaf)) {
        return x->next == location;
    }
    if (const auto x = std::get_if<IR::Term::LinkBlockFast>(&leaf)) {
        return x->next == location;
    }
    return false;
}

bool EmitTerminalImpl(A64EmitX64& e, IR::Term::ReturnToDispatch, IR::LocationDescriptor, bool) {
    e.code.ReturnFromRunCode();
    return true;
}

bool EmitTerminalImpl(A64EmitX64& e, IR::Term::LinkBlock terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    // Used for patches and linking
    if (e.conf.HasOptimization(OptimizationFlag::BlockLinking) && !is_single_step) {
        if (e.compact_links || e.backward_halt_checks) {
            // Fritchii (FRITCHII_JIT_CHAIN; never with cycle counting).
            e.EmitFritchiiLink(terminal.next, initial_location, true);
            return true;
        }
        if (e.conf.enable_cycle_counting) {
            e.code.cmp(qword[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, cycles_remaining)], 0);
            e.AddPatchLocation(terminal.next, EmitX64::PatchKind::Jg);
            if (const auto next_bb = e.GetBasicBlock(terminal.next)) {
                e.EmitPatchJg(terminal.next, next_bb->entrypoint);
            } else {
                e.EmitPatchJg(terminal.next);
            }
        } else {
            e.code.cmp(dword[e.code.ABI_JIT_PTR + offsetof(A64JitState, halt_reason)], 0);
            e.AddPatchLocation(terminal.next, EmitX64::PatchKind::Jz);
            if (const auto next_bb = e.GetBasicBlock(terminal.next)) {
                e.EmitPatchJz(terminal.next, next_bb->entrypoint);
            } else {
                e.EmitPatchJz(terminal.next);
            }
        }
        e.code.mov(rax, A64::LocationDescriptor{terminal.next}.PC());
        e.code.mov(qword[e.code.ABI_JIT_PTR + offsetof(A64JitState, pc)], rax);
        e.code.ForceReturnFromRunCode();
    } else {
        e.code.mov(rax, A64::LocationDescriptor{terminal.next}.PC());
        e.code.mov(qword[e.code.ABI_JIT_PTR + offsetof(A64JitState, pc)], rax);
        e.code.ReturnFromRunCode();
    }
    return true;
}

bool EmitTerminalImpl(A64EmitX64& e, IR::Term::LinkBlockFast terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    if (e.conf.HasOptimization(OptimizationFlag::BlockLinking) && !is_single_step && (e.compact_links || e.backward_halt_checks)) {
        // Fritchii (FRITCHII_JIT_CHAIN).
        e.EmitFritchiiLink(terminal.next, initial_location, false);
    } else if (e.conf.HasOptimization(OptimizationFlag::BlockLinking) && !is_single_step) {
        e.AddPatchLocation(terminal.next, EmitX64::PatchKind::Jmp);
        if (auto next_bb = e.GetBasicBlock(terminal.next)) {
            e.EmitPatchJmp(terminal.next, next_bb->entrypoint);
        } else {
            e.EmitPatchJmp(terminal.next);
        }
    } else {
        e.code.mov(rax, A64::LocationDescriptor{terminal.next}.PC());
        e.code.mov(qword[e.code.ABI_JIT_PTR + offsetof(A64JitState, pc)], rax);
        e.code.ReturnFromRunCode();
    }
    return true;
}

bool EmitTerminalImpl(A64EmitX64& e, IR::Term::PopRSBHint, IR::LocationDescriptor, bool is_single_step) {
    if (e.conf.HasOptimization(OptimizationFlag::ReturnStackBuffer) && !is_single_step) {
        if (e.backward_halt_checks) {
            // FRITCHII_JIT_CHAIN=2: a return may go back (the pc is already stored).
            e.code.cmp(dword[e.code.ABI_JIT_PTR + offsetof(A64JitState, halt_reason)], 0);
            e.code.jne(e.code.GetForceReturnFromRunCodeAddress());
        }
        if (e.indirect_cache_mode == 1 && e.terminal_handler_pop_rsb_dry != nullptr) {
            // FRITCHII_JIT_INDIRECT_CACHE=1: the dry handler counts this site's targets.
            if (auto* const slot = e.AllocateIndirectSlot()) {
                e.code.mov(rax, u64(slot));
                e.code.jmp(e.terminal_handler_pop_rsb_dry);
                return true;
            }
        } else if (e.indirect_cache_mode == 3) {
            // FRITCHII_JIT_INDIRECT_CACHE=3: terminal_handler_pop_rsb_hint's instructions here, so
            // this site's jump has its own prediction; a miss continues in the fast dispatch
            // handler as the shared one's does.
            e.EmitCalculateLocationDescriptor();
            e.code.mov(eax, dword[e.code.ABI_JIT_PTR + offsetof(A64JitState, rsb_ptr)]);
            e.code.sub(eax, 1);
            e.code.and_(eax, u32(A64JitState::RSB_PTR_MASK));
            e.code.mov(dword[e.code.ABI_JIT_PTR + offsetof(A64JitState, rsb_ptr)], eax);
            e.code.cmp(rbx, qword[e.code.ABI_JIT_PTR + offsetof(A64JitState, rsb_location_descriptors) + rax * sizeof(u64)]);
            e.code.jne(e.terminal_handler_fast_dispatch_after_calc);
            e.code.jmp(ptr[e.code.ABI_JIT_PTR + offsetof(A64JitState, rsb_codeptrs) + rax * sizeof(u64)]);
            return true;
        }
        e.code.jmp(e.terminal_handler_pop_rsb_hint);
    } else {
        e.code.ReturnFromRunCode();
    }
    return true;
}

bool EmitTerminalImpl(A64EmitX64& e, IR::Term::FastDispatchHint, IR::LocationDescriptor, bool is_single_step) {
    if (!e.conf.HasOptimization(OptimizationFlag::FastDispatch) || is_single_step) {
        e.code.ReturnFromRunCode();
        return true;
    }
    if (e.backward_halt_checks) {
        // FRITCHII_JIT_CHAIN=2: an indirect branch may go back (the pc is already stored).
        e.code.cmp(dword[e.code.ABI_JIT_PTR + offsetof(A64JitState, halt_reason)], 0);
        e.code.jne(e.code.GetForceReturnFromRunCodeAddress());
    }
    if (e.indirect_cache_mode != 0) {
        if (auto* const slot = e.AllocateIndirectSlot()) {
            if (e.indirect_cache_mode == 1) {
                // FRITCHII_JIT_INDIRECT_CACHE=1: the dry handler counts this site's targets.
                e.code.mov(rax, u64(slot));
                e.code.jmp(e.terminal_handler_indirect_dry);
                return true;
            }
            // FRITCHII_JIT_INDIRECT_CACHE>=2: the slot holds a descriptor and the entrypoint the
            // fast dispatch table had for it (copied on a table hit; every slot is emptied when
            // any block is removed), so a match jumps where the table would have.
            Xbyak::Label miss;
            e.EmitCalculateLocationDescriptor();
            e.code.mov(rax, u64(slot));
            e.code.cmp(rbx, qword[rax + offsetof(A64EmitX64::FastDispatchEntry, location_descriptor)]);
            e.code.jne(miss);
            e.code.jmp(ptr[rax + offsetof(A64EmitX64::FastDispatchEntry, code_ptr)]); // unsized: jmp m64 takes no REX.W (qword fails Xbyak's size check)
            e.code.L(miss);
            e.code.jmp(e.terminal_handler_indirect_slot_miss);
            return true;
        }
    }
    e.code.jmp(e.terminal_handler_fast_dispatch_hint);
    return true;
}

bool EmitTerminalImpl(A64EmitX64& e, IR::Term::If terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    switch (terminal.if_) {
    case IR::Cond::AL:
    case IR::Cond::NV:
        e.EmitLeafTerminal(terminal.then_, initial_location, is_single_step);
        break;
    default:
        if (e.chain_successors && !is_single_step && LeafLinksTo(terminal.else_, e.current_end_location) &&
            !LeafLinksTo(terminal.then_, e.current_end_location)) {
            // FRITCHII_JIT_CHAIN=1: the fall-through leaf last, so it can be chained.
            Xbyak::Label fail = e.EmitCond(IR::Cond(u32(terminal.if_) ^ 1));
            e.EmitLeafTerminal(terminal.then_, initial_location, is_single_step);
            e.code.L(fail);
            e.EmitLeafTerminal(terminal.else_, initial_location, is_single_step);
            break;
        }
        Xbyak::Label pass = e.EmitCond(terminal.if_);
        e.EmitLeafTerminal(terminal.else_, initial_location, is_single_step);
        e.code.L(pass);
        e.EmitLeafTerminal(terminal.then_, initial_location, is_single_step);
        break;
    }
    return true;
}

bool EmitTerminalImpl(A64EmitX64& e, IR::Term::CheckBit terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    Xbyak::Label fail;
    e.code.cmp(e.code.byte[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, check_bit)], u8(0));
    if (e.chain_successors && !is_single_step && LeafLinksTo(terminal.then_, e.current_end_location) &&
        !LeafLinksTo(terminal.else_, e.current_end_location)) {
        // FRITCHII_JIT_CHAIN=1: the fall-through leaf last, so it can be chained.
        Xbyak::Label pass;
        e.code.jnz(pass);
        e.EmitLeafTerminal(terminal.else_, initial_location, is_single_step);
        e.code.L(pass);
        e.EmitLeafTerminal(terminal.then_, initial_location, is_single_step);
        return true;
    }
    e.code.jz(fail);
    e.EmitLeafTerminal(terminal.then_, initial_location, is_single_step);
    e.code.L(fail);
    e.EmitLeafTerminal(terminal.else_, initial_location, is_single_step);
    return true;
}

bool EmitTerminalImpl(A64EmitX64& e, IR::Term::CheckHalt terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    e.code.cmp(dword[e.code.ABI_JIT_PTR + offsetof(A64JitState, halt_reason)], 0);
    e.code.jne(e.code.GetForceReturnFromRunCodeAddress());
    e.EmitLeafTerminal(terminal.else_, initial_location, is_single_step);
    return true;
}

}

bool A64EmitX64::EmitLeafTerminal(IR::Term::LeafTerminal const& terminal, IR::LocationDescriptor initial_location, bool is_single_step) noexcept {
    if (auto const x = std::get_if<IR::Term::ReturnToDispatch>(&terminal))
        return EmitTerminalImpl(*this, *x, initial_location, is_single_step);
    if (auto const x = std::get_if<IR::Term::LinkBlock>(&terminal))
        return EmitTerminalImpl(*this, *x, initial_location, is_single_step);
    if (auto const x = std::get_if<IR::Term::LinkBlockFast>(&terminal))
        return EmitTerminalImpl(*this, *x, initial_location, is_single_step);
    if (auto const x = std::get_if<IR::Term::PopRSBHint>(&terminal))
        return EmitTerminalImpl(*this, *x, initial_location, is_single_step);
    if (auto const x = std::get_if<IR::Term::FastDispatchHint>(&terminal))
        return EmitTerminalImpl(*this, *x, initial_location, is_single_step);
    UNREACHABLE();
}

bool A64EmitX64::EmitTerminal(IR::Term::Terminal const& terminal, IR::LocationDescriptor initial_location, bool is_single_step) noexcept {
    if (auto const x = std::get_if<IR::Term::LeafTerminal>(&terminal))
        return EmitLeafTerminal(*x, initial_location, is_single_step);
    if (auto const x = std::get_if<IR::Term::If>(&terminal))
        return EmitTerminalImpl(*this, *x, initial_location, is_single_step);
    if (auto const x = std::get_if<IR::Term::CheckBit>(&terminal))
        return EmitTerminalImpl(*this, *x, initial_location, is_single_step);
    if (auto const x = std::get_if<IR::Term::CheckHalt>(&terminal))
        return EmitTerminalImpl(*this, *x, initial_location, is_single_step);
    UNREACHABLE();
}

void A64EmitX64::EmitPatchJg(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr) {
    const CodePtr patch_location = code.getCurr();
    if (target_code_ptr) {
        code.jg(target_code_ptr);
    } else {
        code.mov(rax, A64::LocationDescriptor{target_desc}.PC());
        code.mov(qword[code.ABI_JIT_PTR + offsetof(A64JitState, pc)], rax);
        code.jg(code.GetReturnFromRunCodeAddress());
    }
    code.EnsurePatchLocationSize(patch_location, 23);
}

void A64EmitX64::EmitPatchJz(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr) {
    const CodePtr patch_location = code.getCurr();
    if (target_code_ptr) {
        code.jz(target_code_ptr);
    } else {
        code.mov(rax, A64::LocationDescriptor{target_desc}.PC());
        code.mov(qword[code.ABI_JIT_PTR + offsetof(A64JitState, pc)], rax);
        code.jz(code.GetReturnFromRunCodeAddress());
    }
    code.EnsurePatchLocationSize(patch_location, 23);
}

void A64EmitX64::EmitPatchJmp(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr) {
    const CodePtr patch_location = code.getCurr();
    if (compact_links) {
        // Fritchii (FRITCHII_JIT_CHAIN): one 5-byte slot; falls through when the block follows.
        if (target_code_ptr == static_cast<const u8*>(patch_location) + 5) {
            code.nop(5);
        } else if (target_code_ptr) {
            code.jmp(target_code_ptr, code.T_NEAR);
        } else {
            const auto stub = link_stubs.find(target_desc.Value());
            ASSERT(stub != link_stubs.end());
            code.jmp(stub->second, code.T_NEAR);
        }
        code.EnsurePatchLocationSize(patch_location, 5);
        return;
    }
    if (target_code_ptr) {
        code.jmp(target_code_ptr);
    } else {
        code.mov(rax, A64::LocationDescriptor{target_desc}.PC());
        code.mov(qword[code.ABI_JIT_PTR + offsetof(A64JitState, pc)], rax);
        code.jmp(code.GetReturnFromRunCodeAddress());
    }
    code.EnsurePatchLocationSize(patch_location, 22);
}

void A64EmitX64::EmitPatchMovRcx(CodePtr target_code_ptr) {
    if (!target_code_ptr) {
        target_code_ptr = code.GetReturnFromRunCodeAddress();
    }
    const CodePtr patch_location = code.getCurr();
    code.mov(code.rcx, u64(target_code_ptr));
    code.EnsurePatchLocationSize(patch_location, 10);
}

void A64EmitX64::Unpatch(const IR::LocationDescriptor& location) {
    EmitX64::Unpatch(location);
    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        code.DisableWriting();
        (*fast_dispatch_table_lookup)(location.Value()) = {};
        code.EnableWriting();
    }
    // Fritchii (FRITCHII_JIT_INDIRECT_CACHE): a block removed outside a range invalidation (the
    // fastmem fault handler's recompile, emit_x64_memory.cpp.inc) may sit in a slot; the slots all
    // go, as the table entry did.
    if (!indirect_bulk_invalidation) {
        ClearIndirectSlots(false);
    }
}

}  // namespace Dynarmic::Backend::X64
