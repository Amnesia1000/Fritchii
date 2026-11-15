// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include "common/container/unordered_map.h"
#include "common/container/unordered_set.h"
#include <boost/container/static_vector.hpp>

#include "dynarmic/backend/block_range_information.h"
#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/backend/x64/emit_x64.h"
#include "dynarmic/backend/x64/reg_alloc.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/interface/A64/config.h"
#include "dynarmic/ir/terminal.h"

namespace Dynarmic::Backend::X64 {

struct A64EmitContext final : public EmitContext {
    A64EmitContext(const A64::UserConfig& conf, RegAlloc& reg_alloc, IR::Block& block, boost::container::stable_vector<Xbyak::Label>& shared_labels);

    A64::LocationDescriptor Location() const;
    bool IsSingleStep() const;
    FP::FPCR FPCR(bool fpcr_controlled = true) const override;

    bool HasOptimization(OptimizationFlag flag) const override {
        return conf.HasOptimization(flag);
    }

    const A64::UserConfig& conf;
};

class A64EmitX64 final : public EmitX64 {
public:
    A64EmitX64(BlockOfCode& code, A64::UserConfig conf, A64::Jit* jit_interface);
    ~A64EmitX64() override;

    /// Emit host machine code for a basic block with intermediate representation `block`.
    /// @note block is modified.
    BlockDescriptor Emit(IR::Block& block) noexcept;

    void ClearCache() override;

    void InvalidateCacheRanges(const boost::icl::interval_set<u64>& ranges);

//protected:
    struct FastDispatchEntry {
        u64 location_descriptor = 0xFFFF'FFFF'FFFF'FFFFull;
        const void* code_ptr = nullptr;
    };
    static_assert(sizeof(FastDispatchEntry) == 0x10);
    // Fritchii: the table's size is chosen once per JIT (FRITCHII_JIT_DISPATCH_BITS, entries =
    // 2^bits, 12..20, default 20 = 0x100000 entries). The mask is a byte offset into the table,
    // so its low 4 bits are 0 (16-byte entries): default 0xFFFFF0.
    static constexpr unsigned fast_dispatch_table_default_bits = 20;
    static constexpr unsigned fast_dispatch_table_min_bits = 12;

    void ClearFastDispatchTable();
    void GenMemory128Accessors();
    void GenFastmemFallbacks();
    void GenTerminalHandlers();

    // Microinstruction emitters
    void EmitPushRSB(EmitContext& ctx, IR::Inst* inst);
#define OPCODE(...)
#define A32OPC(...)
#define A64OPC(name, type, ...) void EmitA64##name(A64EmitContext& ctx, IR::Inst* inst);
#include "dynarmic/ir/opcodes.inc"
#undef OPCODE
#undef A32OPC
#undef A64OPC

    // Helpers
    std::string LocationDescriptorToFriendlyName(const IR::LocationDescriptor&) const override;

    // Fastmem information
    using DoNotFastmemMarker = std::tuple<IR::LocationDescriptor, unsigned>;
    /// Where a fastmem access that faults goes: code addresses are offsets in the code space.
    struct FastmemPatchInfo {
        u32 rip;
        u32 resume_rip;
        u32 callback;
        u32 inst_name : 31;
        u32 recompile : 1;
        IR::LocationDescriptor location;
    };
    std::optional<DoNotFastmemMarker> ShouldFastmem(A64EmitContext& ctx, IR::Inst* inst) const;
    void AddFastmemPatch(CodePtr location, CodePtr resume_location, void (*callback)(), const DoNotFastmemMarker& marker, bool recompile);
    FakeCall FastmemCallback(u64 rip);

    // Memory access helpers
    void EmitCheckMemoryAbort(A64EmitContext& ctx, IR::Inst* inst, Xbyak::Label* end = nullptr);
    template<std::size_t bitsize, auto callback>
    void EmitMemoryRead(A64EmitContext& ctx, IR::Inst* inst);
    template<std::size_t bitsize, auto callback>
    void EmitMemoryWrite(A64EmitContext& ctx, IR::Inst* inst);
    template<std::size_t bitsize, auto callback>
    void EmitExclusiveReadMemory(A64EmitContext& ctx, IR::Inst* inst);
    template<std::size_t bitsize, auto callback>
    void EmitExclusiveWriteMemory(A64EmitContext& ctx, IR::Inst* inst);
    template<std::size_t bitsize, auto callback>
    void EmitExclusiveReadMemoryInline(A64EmitContext& ctx, IR::Inst* inst);
    template<std::size_t bitsize, auto callback>
    void EmitExclusiveWriteMemoryInline(A64EmitContext& ctx, IR::Inst* inst);

    // Terminal instruction emitters
    bool EmitLeafTerminal(IR::Term::LeafTerminal const& terminal, IR::LocationDescriptor initial_location, bool is_single_step) noexcept override;
    bool EmitTerminal(IR::Term::Terminal const& terminal, IR::LocationDescriptor initial_location, bool is_single_step) noexcept override;

    // Patching
    void Unpatch(const IR::LocationDescriptor& target_desc) override;
    void EmitPatchJg(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) override;
    void EmitPatchJz(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) override;
    void EmitPatchJmp(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) override;
    void EmitPatchMovRcx(CodePtr target_code_ptr = nullptr) override;

//data
    const A64::UserConfig conf;
    RegAlloc reg_alloc; //reusable reg alloc
    BlockRangeInformation<u64> block_ranges;
    const size_t fast_dispatch_table_size;
    const u32 fast_dispatch_table_mask;
    std::unique_ptr<FastDispatchEntry[]> fast_dispatch_table;
    std::vector<FastmemPatchInfo> fastmem_patch_info;  // sorted by rip
    ::Common::unordered_map<std::tuple<bool, size_t, int, int>, void (*)()> read_fallbacks;
    ::Common::unordered_map<std::tuple<bool, size_t, int, int>, void (*)()> write_fallbacks;
    ::Common::unordered_map<std::tuple<bool, size_t, int, int>, void (*)()> exclusive_write_fallbacks;
    ::Common::unordered_set<DoNotFastmemMarker> do_not_fastmem;
    boost::container::stable_vector<Xbyak::Label> shared_labels;
    const void* terminal_handler_pop_rsb_hint = nullptr;
    const void* terminal_handler_fast_dispatch_hint = nullptr;
    FastDispatchEntry& (*fast_dispatch_table_lookup)(u64) = nullptr;

    // Fritchii: FRITCHII_JIT_INDIRECT_CACHE (boot; fable_a_gameceiling_answer 12.2). 1 = dry: each
    // indirect branch and return site keeps its last target in a slot and the shared handlers
    // count how often the next target equals it; 2 = each indirect branch site compares the
    // target with its slot and jumps straight to the cached block (a miss takes the shared
    // table, which refills the slot); 3 = 2, and each return site pops the return stack buffer
    // itself instead of jumping to the shared handler. 0 elsewhere (or without FastDispatch).
    /// Called from the constructor (the boot switch, read once per process).
    static int IndirectCacheMode();
    /// A free slot, or nullptr when all are taken (the site then uses the shared handler).
    FastDispatchEntry* AllocateIndirectSlot();
    /// Every slot back to the empty entry; `release`: the sites are gone too (ClearCache).
    void ClearIndirectSlots(bool release);
    /// rbx = the location descriptor of the guest pc and fpcr (rdi, rcx clobbered), as the
    /// shared handlers compute it.
    void EmitCalculateLocationDescriptor();
    /// The dry counters' line, at most every 10 s (from Jit::Run, on the owner thread).
    void LogIndirectCounters();
    enum IndirectCounter : size_t {
        IndirectRuns,       ///< =1: indirect branches run through a dry site
        IndirectSameTarget, ///< of them, same target as the site's last
        ReturnRuns,         ///< =1: returns run through a dry site
        ReturnSameTarget,   ///< of them, same target as the site's last
        ReturnRsbHits,      ///< of them, served by the return stack buffer
        SlotMisses,         ///< >=2: slot compares that failed (the shared table then)
        IndirectCounterCount,
    };
    int indirect_cache_mode = 0;
    static constexpr size_t indirect_slot_capacity = size_t{1} << 18; // 4 MiB per JIT
    std::unique_ptr<FastDispatchEntry[]> indirect_slots;
    size_t indirect_slots_used = 0;
    std::array<u64, IndirectCounterCount> indirect_counters{};  // written by generated code
    std::array<u64, IndirectCounterCount> indirect_counters_logged{};
    s64 indirect_log_ns = 0;
    /// Inside the fast dispatch handler, right after the descriptor is computed (rbx).
    const void* terminal_handler_fast_dispatch_after_calc = nullptr;
    const void* terminal_handler_indirect_dry = nullptr;  ///< rax = the site's slot
    const void* terminal_handler_pop_rsb_dry = nullptr;   ///< rax = the site's slot
    const void* terminal_handler_indirect_slot_miss = nullptr; ///< rbx = descriptor, rax = slot

    A64::Jit* jit_interface = nullptr;
    void (*memory_read_128)() = nullptr;
    void (*memory_write_128)() = nullptr;
    void (*memory_exclusive_write_128)() = nullptr;
};

}  // namespace Dynarmic::Backend::X64
