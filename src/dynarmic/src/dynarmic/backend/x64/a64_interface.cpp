// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <exception>
#include <memory>
#include <mutex>

#include <boost/icl/interval_set.hpp>
#include "common/assert.h"
#include "common/logging.h"
#include "common/nx2_hitch.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/common/llvm_disassemble.h"
#include <bit>

#include "dynarmic/backend/x64/a64_emit_x64.h"
#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/backend/x64/block_of_code.h"
#include "dynarmic/backend/x64/devirtualize.h"
#include "dynarmic/backend/x64/jitstate_info.h"
#include "dynarmic/common/atomic.h"
#include "dynarmic/frontend/A64/translate/a64_translate.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opt_passes.h"

namespace Dynarmic::A64 {

using namespace Backend::X64;

static RunCodeCallbacks GenRunCodeCallbacks(A64::UserCallbacks* cb, CodePtr (*LookupBlock)(void* lookup_block_arg), void* arg, const A64::UserConfig& conf) {
    return RunCodeCallbacks{
        ArgCallback(LookupBlock, reinterpret_cast<u64>(arg)),
        ArgCallback(Devirtualize<&A64::UserCallbacks::AddTicks>(cb)),
        ArgCallback(Devirtualize<&A64::UserCallbacks::GetTicksRemaining>(cb)),
        conf.enable_cycle_counting,
    };
}

static std::function<void(BlockOfCode&)> GenRCP(const A64::UserConfig& conf) {
    return [conf](BlockOfCode& code) {
        if (conf.page_table) {
            code.mov(code.r14, std::bit_cast<u64>(conf.page_table));
        }
        if (conf.fastmem_pointer) {
            code.mov(code.r13, *conf.fastmem_pointer);
        }
    };
}

namespace {

/// Fritchii: FRITCHII_JIT_DUMP (boot, default 0; Fable A 4.1 and 8.5). Writes the code each
/// block was compiled to, with its guest words, so the host code can be read offline.
/// 1 = written every 4096 blocks, 2 = only at a full cache clear and at shutdown. Host bytes
/// are read when written, so links show as patched by then.
/// FRITCHII_JIT_DUMP_CORE: the core that dumps (default 0, -1 = every core).
/// FRITCHII_JIT_DUMP_FILE: the file (default jit-dump-core{core}.bin in the working directory;
/// "{core}" is replaced by the core, and with every core dumping ".core{N}" is appended when
/// the name has no "{core}").
/// FRITCHII_JIT_DUMP_LIST: only these blocks, one hex guest pc per line (first word of a line,
/// so hot lists work too). Values below the image base are offsets from it (the lists made by
/// earlier launches, as the base is random per launch), others absolute.
/// File: a 64-byte header ("FJITDMP2", version 2, core, code space base and size, image base,
/// far region start or 0, far code flag, entry count of the list), then 64-byte records, each
/// followed by its guest words, near bytes and far bytes (see CodeDumpRecord).
struct CodeDumpSettings {
    int mode = 0;
    int core = 0;
    std::string file;
    std::vector<u64> list; // as written in the list file
};

const CodeDumpSettings& GetCodeDumpSettings() {
    static const CodeDumpSettings settings = [] {
        CodeDumpSettings s;
        const auto text = [](const char* name) {
            const char* value = std::getenv(name);
            return value != nullptr ? std::string{value} : std::string{};
        };
        s.mode = std::atoi(text("FRITCHII_JIT_DUMP").c_str());
        if (s.mode < 0 || s.mode > 2) {
            s.mode = 0;
        }
        if (s.mode == 0) {
            return s;
        }
        s.core = std::atoi(text("FRITCHII_JIT_DUMP_CORE").c_str());
        s.file = text("FRITCHII_JIT_DUMP_FILE");
        if (const std::string list_path = text("FRITCHII_JIT_DUMP_LIST"); !list_path.empty()) {
            if (std::FILE* const list = std::fopen(list_path.c_str(), "r")) {
                char line[256];
                while (std::fgets(line, sizeof(line), list) != nullptr) {
                    char* end = nullptr;
                    const u64 value = std::strtoull(line, &end, 16);
                    if (end != line) {
                        s.list.push_back(value);
                    }
                }
                std::fclose(list);
            }
            LOG_INFO(Core_ARM, "FRITCHII_JIT_DUMP_LIST: {} pcs from {}", s.list.size(), list_path);
        }
        LOG_INFO(Core_ARM, "FRITCHII_JIT_DUMP={}: core {}, {}", s.mode, s.core,
                 s.mode == 1 ? "written every 4096 blocks" : "written at cache clears and shutdown");
        return s;
    }();
    return settings;
}

#pragma pack(push, 1)
struct CodeDumpHeader {
    char magic[8];
    u32 version;
    u32 core;
    u64 code_base;
    u64 code_size;
    u64 image_base;
    u64 far_begin;     ///< address of the far region, 0 without one
    u32 flags;         ///< bit 0: far code
    u32 list_entries;  ///< FRITCHII_JIT_DUMP_LIST entries (0 = every block)
    u64 reserved;
};
struct CodeDumpRecord {
    u16 kind;          ///< 1 block, 2 prelude (dispatcher, handlers, constant pool)
    u16 flags;         ///< bit 0: far part present
    u32 guest_count;   ///< guest instruction words that follow
    u64 descriptor;    ///< location descriptor (pc, fpcr, single step)
    u64 guest_pc;      ///< absolute guest pc
    s64 guest_offset;  ///< guest pc minus the image base (the same in every launch)
    u64 near_addr;
    u64 far_addr;      ///< 0 without a far part
    u32 near_size;
    u32 far_size;
    u64 sequence;      ///< emission order since the JIT was made
};
#pragma pack(pop)
static_assert(sizeof(CodeDumpHeader) == 64 && sizeof(CodeDumpRecord) == 64);

class CodeDump {
public:
    CodeDump(const UserConfig& conf, const BlockOfCode& code) : conf{conf}, code{code} {
        const CodeDumpSettings& s = GetCodeDumpSettings();
        if (s.mode == 0 || (s.core != -1 && s.core != int(conf.processor_id))) {
            return;
        }
        mode = s.mode;
        std::string name = s.file.empty() ? std::string{"jit-dump-core{core}.bin"} : s.file;
        if (const size_t at = name.find("{core}"); at != std::string::npos) {
            name.replace(at, 6, std::to_string(conf.processor_id));
        } else if (s.core == -1) {
            name += ".core" + std::to_string(conf.processor_id);
        }
        path = std::move(name);
        list_entries = u32(s.list.size());
        for (const u64 value : s.list) {
            list.insert(value < conf.image_base ? value + conf.image_base : value);
        }
    }

    ~CodeDump() {
        Flush();
        if (file != nullptr) {
            std::fclose(file);
        }
    }

    bool Active() const {
        return mode != 0;
    }

    /// After a block was emitted. `get_code` reads its guest words.
    template<typename GetCode>
    void Record(const IR::Block& block, CodePtr near_addr, size_t near_size, CodePtr far_addr, size_t far_size, GetCode&& get_code) {
        ++sequence;
        const A64::LocationDescriptor location{block.Location()};
        const u64 pc = location.PC();
        if (!list.empty() && list.count(pc) == 0) {
            return;
        }
        Pending& p = pending.emplace_back();
        p.record.kind = 1;
        p.record.flags = far_size != 0 ? 1 : 0;
        p.record.descriptor = location.UniqueHash();
        p.record.guest_pc = pc;
        p.record.guest_offset = s64(pc - conf.image_base);
        p.record.near_addr = u64(reinterpret_cast<uintptr_t>(near_addr));
        p.record.far_addr = far_size != 0 ? u64(reinterpret_cast<uintptr_t>(far_addr)) : 0;
        p.record.near_size = u32(near_size);
        p.record.far_size = u32(far_size);
        p.record.sequence = sequence;
        const u64 end_pc = A64::LocationDescriptor{block.EndLocation()}.PC();
        const u64 count = end_pc > pc ? (end_pc - pc) / 4 : 0;
        p.words.reserve(size_t(count));
        for (u64 i = 0; i < count; ++i) {
            p.words.push_back(get_code(pc + i * 4).value_or(0));
        }
        p.record.guest_count = u32(p.words.size());
        if (mode == 1 && pending.size() >= 4096) {
            Flush();
        }
    }

    /// Writes what is pending (the host bytes as they are now). Call before the code goes.
    void Flush() {
        if (pending.empty() || !Open()) {
            pending.clear();
            return;
        }
        for (const Pending& p : pending) {
            WriteRecord(p.record, p.words.data());
        }
        std::fflush(file);
        pending.clear();
    }

private:
    struct Pending {
        CodeDumpRecord record{};
        std::vector<u32> words;
    };

    bool Open() {
        if (file != nullptr) {
            return true;
        }
        if (failed) {
            return false;
        }
        file = std::fopen(path.c_str(), "wb");
        if (file == nullptr) {
            failed = true;
            LOG_WARNING(Core_ARM, "FRITCHII_JIT_DUMP: can't open {}", path);
            return false;
        }
        CodeDumpHeader header{};
        std::memcpy(header.magic, "FJITDMP2", 8);
        header.version = 2;
        header.core = conf.processor_id;
        header.code_base = u64(reinterpret_cast<uintptr_t>(code.getCode()));
        header.code_size = code.GetTotalCodeSize();
        header.image_base = conf.image_base;
        header.far_begin = code.HasFarCode() ? u64(reinterpret_cast<uintptr_t>(code.getCode())) + code.FarCodeBegin() : 0;
        header.flags = code.HasFarCode() ? 1 : 0;
        header.list_entries = list_entries;
        std::fwrite(&header, sizeof(header), 1, file);
        // The prelude: everything before the first block, as it never moves.
        CodeDumpRecord prelude{};
        prelude.kind = 2;
        prelude.near_addr = header.code_base;
        prelude.near_size = u32(static_cast<const u8*>(code.GetCodeBegin()) - code.getCode());
        WriteRecord(prelude, nullptr);
        LOG_INFO(Core_ARM, "FRITCHII_JIT_DUMP: core {} writes {} (image base {:#x})",
                 conf.processor_id, path, conf.image_base);
        return true;
    }

    void WriteRecord(const CodeDumpRecord& record, const u32* words) {
        std::fwrite(&record, sizeof(record), 1, file);
        if (record.guest_count != 0) {
            std::fwrite(words, sizeof(u32), record.guest_count, file);
        }
        if (record.near_size != 0) {
            std::fwrite(reinterpret_cast<const void*>(uintptr_t(record.near_addr)), 1, record.near_size, file);
        }
        if (record.far_size != 0) {
            std::fwrite(reinterpret_cast<const void*>(uintptr_t(record.far_addr)), 1, record.far_size, file);
        }
    }

    const UserConfig& conf;
    const BlockOfCode& code;
    int mode = 0;
    std::string path;
    u32 list_entries = 0;
    ::Common::unordered_set<u64> list;
    std::vector<Pending> pending;
    u64 sequence = 0;
    std::FILE* file = nullptr;
    bool failed = false;
};

} // namespace

static Optimization::PolyfillOptions GenPolyfillOptions(const BlockOfCode& code) {
    return Optimization::PolyfillOptions{
        .sha256 = !code.HasHostFeature(HostFeature::SHA),
        .vector_multiply_widen = true,
    };
}

struct Jit::Impl final {
public:
    Impl(Jit* jit, UserConfig conf)
        : conf(conf)
        , block_of_code(GenRunCodeCallbacks(conf.callbacks, &GetCurrentBlockThunk, this, conf), JitStateInfo{jit_state}, conf.code_cache_size, GenRCP(conf))
        , emitter(block_of_code, conf, jit)
        , polyfill_options(GenPolyfillOptions(block_of_code))
    {
        ASSERT(conf.page_table_address_space_bits >= 12 && conf.page_table_address_space_bits <= 64);
    }

    ~Impl() = default;

    HaltReason Run() {
        ASSERT(!is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&jit_state.halt_reason)));
        is_executing = true;
        // TODO: Check code alignment
        const CodePtr current_code_ptr = [this] {
            // RSB optimization
            const u32 new_rsb_ptr = (jit_state.rsb_ptr - 1) & A64JitState::RSB_PTR_MASK;
            if (jit_state.GetUniqueHash() == jit_state.rsb_location_descriptors[new_rsb_ptr]) {
                jit_state.rsb_ptr = new_rsb_ptr;
                return CodePtr(jit_state.rsb_codeptrs[new_rsb_ptr]);
            }
            return CodePtr((uintptr_t(GetCurrentBlock()) + 15) & ~uintptr_t(15));
        }();
        const HaltReason hr = block_of_code.RunCode(&jit_state, current_code_ptr);
        PerformRequestedCacheInvalidation(hr);
        is_executing = false;
        // Fritchii: FRITCHII_JIT_INDIRECT_CACHE's counters, written by this thread's code.
        if (emitter.indirect_cache_mode != 0 && (++runs_since_counter_look & 0x3ff) == 0) [[unlikely]] {
            emitter.LogIndirectCounters();
        }
        return hr;
    }

    HaltReason Step() {
        ASSERT(!is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&jit_state.halt_reason)));
        is_executing = true;
        const HaltReason hr = block_of_code.StepCode(&jit_state, GetCurrentSingleStep());
        PerformRequestedCacheInvalidation(hr);
        is_executing = false;
        return hr;
    }

    void ClearCache() {
        std::unique_lock lock{invalidation_mutex};
        invalidate_entire_cache = true;
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void InvalidateCacheRange(u64 start_address, size_t length) {
        std::unique_lock lock{invalidation_mutex};
        const auto end_address = static_cast<u64>(start_address + length - 1);
        const auto range = boost::icl::discrete_interval<u64>::closed(start_address, end_address);
        invalid_cache_ranges.add(range);
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void Reset() {
        ASSERT(!is_executing);
        jit_state = {};
    }

    void HaltExecution(HaltReason hr) {
        Atomic::Or(&jit_state.halt_reason, static_cast<u32>(hr));
    }

    void ClearHalt(HaltReason hr) {
        Atomic::And(&jit_state.halt_reason, ~static_cast<u32>(hr));
    }

    u64 GetSP() const {
        return jit_state.sp;
    }

    void SetSP(u64 value) {
        jit_state.sp = value;
    }

    u64 GetPC() const {
        return jit_state.pc;
    }

    void SetPC(u64 value) {
        jit_state.pc = value;
    }

    u64 GetRegister(size_t index) const {
        if (index == 31)
            return GetSP();
        return jit_state.reg.at(index);
    }

    void SetRegister(size_t index, u64 value) {
        if (index == 31)
            return SetSP(value);
        jit_state.reg.at(index) = value;
    }

    std::array<u64, 31> GetRegisters() const {
        return jit_state.reg;
    }

    void SetRegisters(const std::array<u64, 31>& value) {
        jit_state.reg = value;
    }

    Vector GetVector(size_t index) const {
        return {jit_state.vec.at(index * 2), jit_state.vec.at(index * 2 + 1)};
    }

    void SetVector(size_t index, Vector value) {
        jit_state.vec.at(index * 2) = value[0];
        jit_state.vec.at(index * 2 + 1) = value[1];
    }

    std::array<Vector, 32> GetVectors() const {
        std::array<Vector, 32> ret;
        static_assert(sizeof(ret) == sizeof(jit_state.vec));
        std::memcpy(ret.data(), jit_state.vec.data(), sizeof(jit_state.vec));
        return ret;
    }

    void SetVectors(const std::array<Vector, 32>& value) {
        static_assert(sizeof(value) == sizeof(jit_state.vec));
        std::memcpy(jit_state.vec.data(), value.data(), sizeof(jit_state.vec));
    }

    u32 GetFpcr() const {
        return jit_state.GetFpcr();
    }

    void SetFpcr(u32 value) {
        jit_state.SetFpcr(value);
    }

    u32 GetFpsr() const {
        return jit_state.GetFpsr();
    }

    void SetFpsr(u32 value) {
        jit_state.SetFpsr(value);
    }

    u32 GetPstate() const {
        return jit_state.GetPstate();
    }

    void SetPstate(u32 value) {
        jit_state.SetPstate(value);
    }

    void ClearExclusiveState() {
        jit_state.exclusive_state = 0;
    }

    bool IsExecuting() const {
        return is_executing;
    }

    std::string Disassemble() const {
        const size_t size = reinterpret_cast<const char*>(block_of_code.getCurr()) - reinterpret_cast<const char*>(block_of_code.GetCodeBegin());
        auto const* p = reinterpret_cast<const char*>(block_of_code.GetCodeBegin());
        return Common::DisassembleX64(p, p + size);
    }

private:
    static CodePtr GetCurrentBlockThunk(void* thisptr) {
        Jit::Impl* this_ = static_cast<Jit::Impl*>(thisptr);
        return this_->GetCurrentBlock();
    }

    IR::LocationDescriptor GetCurrentLocation() const {
        return IR::LocationDescriptor{jit_state.GetUniqueHash()};
    }

    CodePtr GetCurrentBlock() {
        if (conf.counters) {
            ++conf.counters->block_lookups;
        }
        return GetBlock(GetCurrentLocation());
    }

public:
    bool PrecompileBlock(u64 location) {
        if (is_executing) {
            return false;
        }
        // Fritchii (FRITCHII_JIT_HOT_PACK): a whole-cache clear asked for since the last run is
        // done now, as the next Run would do it first, instead of compiling into the code it
        // throws away.
        bool clear_pending = false;
        {
            std::unique_lock lock{invalidation_mutex};
            clear_pending = invalidate_entire_cache;
        }
        if (clear_pending) {
            PerformRequestedCacheInvalidation(HaltReason::CacheInvalidation);
        }
        const IR::LocationDescriptor descriptor{location};
        if (emitter.GetBasicBlock(descriptor)) {
            return false;
        }
        // Fritchii (FRITCHII_JIT_FAR_CODE): half of the smaller region, as 256 MiB is of 512.
        const size_t keep_free = block_of_code.HasFarCode() ? block_of_code.SmallestRegionSize() / 2
                                                            : (size_t{256} << 20);
        if (block_of_code.SpaceRemaining() < keep_free) {
            return false;
        }
        static_cast<void>(GetBlock(descriptor));
        return true;
    }

private:

    CodePtr GetCurrentSingleStep() {
        return GetBlock(A64::LocationDescriptor{GetCurrentLocation()}.SetSingleStepping(true));
    }

    CodePtr GetBlock(IR::LocationDescriptor descriptor) {
        if (auto block = emitter.GetBasicBlock(descriptor))
            return block->entrypoint;
        // Fritchii hitch attribution (2026-11-10): new game code compiled while the game waits.
        // Each emulated core has its own JIT (fable52_answer_12): count code another core has.
        const ::Common::Nx2Hitch::Scope hitch_timed{
            ::Common::Nx2Hitch::CompileCause(descriptor.Value(), conf.processor_id)};

        constexpr size_t MINIMUM_REMAINING_CODESIZE = 1 * 1024 * 1024;
        if (block_of_code.SpaceRemaining() < MINIMUM_REMAINING_CODESIZE) {
            // Immediately evacuate cache
            invalidate_entire_cache = true;
            PerformRequestedCacheInvalidation(HaltReason::CacheInvalidation);
        }
        block_of_code.EnsureMemoryCommitted(MINIMUM_REMAINING_CODESIZE);

        // JIT Compile
        const auto get_code = [this](u64 vaddr) { return conf.callbacks->MemoryReadCode(vaddr); };
        // LocationDescriptor ctor() does important ops (like tflags) do not skip
        auto const arch_descriptor = A64::LocationDescriptor{descriptor};
        ir_block.Reset(arch_descriptor);
        // Fritchii: called from generated code (block lookup), so a throw cannot unwind past it
        // and the process would end without a word: log it, then abort (the crash handler's
        // report and dump follow).
        try {
            A64::Translate(ir_block, arch_descriptor, get_code, {conf.define_unpredictable_behaviour, conf.wall_clock_cntpct});
            Optimization::Optimize(ir_block, conf, polyfill_options);
        } catch (const std::exception& error) {
            ::Common::Log::FmtCrashMessage(__FILE__, __LINE__, __func__,
                                           "JIT translate of {:#x} (core {}) threw: {}",
                                           arch_descriptor.PC(), conf.processor_id, error.what());
            std::abort();
        }
        if (conf.counters) {
            ++conf.counters->blocks_compiled;
        }
        const auto bdesc = emitter.Emit(ir_block);
        if (code_dump.Active()) [[unlikely]] {
            code_dump.Record(ir_block, bdesc.entrypoint, bdesc.size, emitter.last_far_entry,
                             emitter.last_far_size, get_code);
        }
        return bdesc.entrypoint;
    }

    void PerformRequestedCacheInvalidation(HaltReason hr) {
        if (Has(hr, HaltReason::CacheInvalidation)) {
            std::unique_lock lock{invalidation_mutex};

            ClearHalt(HaltReason::CacheInvalidation);

            if (!invalidate_entire_cache && invalid_cache_ranges.empty()) {
                return;
            }

            jit_state.ResetRSB();
            if (invalidate_entire_cache) {
                if (conf.counters) {
                    ++conf.counters->cache_clears;
                }
                code_dump.Flush(); // FRITCHII_JIT_DUMP: before the code goes
                block_of_code.ClearCache();
                emitter.ClearCache();
            } else {
                emitter.InvalidateCacheRanges(invalid_cache_ranges);
            }
            invalid_cache_ranges.clear();
            invalidate_entire_cache = false;
        }
    }

    IR::Block ir_block = {LocationDescriptor(0, FP::FPCR(0), false)};
    const UserConfig conf;
    A64JitState jit_state;
    BlockOfCode block_of_code;
    A64EmitX64 emitter;
    Optimization::PolyfillOptions polyfill_options;
    CodeDump code_dump{conf, block_of_code}; // after the code: written out before it goes
    bool is_executing = false;
    u32 runs_since_counter_look = 0;
    bool invalidate_entire_cache = false;
    boost::icl::interval_set<u64> invalid_cache_ranges;
    std::mutex invalidation_mutex;
};

Jit::Jit(UserConfig conf)
        : impl(std::make_unique<Jit::Impl>(this, conf)) {}

Jit::~Jit() = default;

HaltReason Jit::Run() {
    return impl->Run();
}

HaltReason Jit::Step() {
    return impl->Step();
}

void Jit::ClearCache() {
    impl->ClearCache();
}

bool Jit::PrecompileBlock(u64 location) {
    return impl->PrecompileBlock(location);
}

void Jit::InvalidateCacheRange(u64 start_address, size_t length) {
    impl->InvalidateCacheRange(start_address, length);
}

void Jit::Reset() {
    impl->Reset();
}

void Jit::HaltExecution(HaltReason hr) {
    impl->HaltExecution(hr);
}

void Jit::ClearHalt(HaltReason hr) {
    impl->ClearHalt(hr);
}

u64 Jit::GetSP() const {
    return impl->GetSP();
}

void Jit::SetSP(u64 value) {
    impl->SetSP(value);
}

u64 Jit::GetPC() const {
    return impl->GetPC();
}

void Jit::SetPC(u64 value) {
    impl->SetPC(value);
}

u64 Jit::GetRegister(size_t index) const {
    return impl->GetRegister(index);
}

void Jit::SetRegister(size_t index, u64 value) {
    impl->SetRegister(index, value);
}

std::array<u64, 31> Jit::GetRegisters() const {
    return impl->GetRegisters();
}

void Jit::SetRegisters(const std::array<u64, 31>& value) {
    impl->SetRegisters(value);
}

Vector Jit::GetVector(size_t index) const {
    return impl->GetVector(index);
}

void Jit::SetVector(size_t index, Vector value) {
    impl->SetVector(index, value);
}

std::array<Vector, 32> Jit::GetVectors() const {
    return impl->GetVectors();
}

void Jit::SetVectors(const std::array<Vector, 32>& value) {
    impl->SetVectors(value);
}

u32 Jit::GetFpcr() const {
    return impl->GetFpcr();
}

void Jit::SetFpcr(u32 value) {
    impl->SetFpcr(value);
}

u32 Jit::GetFpsr() const {
    return impl->GetFpsr();
}

void Jit::SetFpsr(u32 value) {
    impl->SetFpsr(value);
}

u32 Jit::GetPstate() const {
    return impl->GetPstate();
}

void Jit::SetPstate(u32 value) {
    impl->SetPstate(value);
}

void Jit::ClearExclusiveState() {
    impl->ClearExclusiveState();
}

bool Jit::IsExecuting() const {
    return impl->IsExecuting();
}

std::string Jit::Disassemble() const {
    return impl->Disassemble();
}

}  // namespace Dynarmic::A64
