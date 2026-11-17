// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/storage_write_extent.h"

namespace {
namespace IR = Shader::IR;
using Shader::StorageWriteExtent;

class TestEnvironment final : public Shader::Environment {
public:
    bool native{true};
    TestEnvironment() { stage = Shader::Stage::Compute; }
    bool IsNativeProgram() const override { return native; }
    bool IsCbufStorageBufferPointer(u32 index, u32) override { return index == 0; }
    u64 ReadInstruction(u32) override { return 0; }
    u32 ReadCbufValue(u32, u32) override { return 0; }
    Shader::TextureType ReadTextureType(u32) override { return Shader::TextureType::Color2D; }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return 0; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {64, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}
};

struct KernelProgram {
    Shader::ObjectPool<IR::Inst> pool{256};
    IR::Block entry{pool};
    IR::Block loop{pool};
    IR::Program program;
    TestEnvironment env;
    bool ran{};

    KernelProgram() {
        program.stage = Shader::Stage::Compute;
        program.workgroup_size = {64, 1, 1};
        program.blocks = {&entry};
        program.post_order_blocks = {&entry};
    }

    /// Runs the pass once and returns the descriptor of the buffer whose pointer is at c0[offset]
    const Shader::StorageBufferDescriptor& Run(u32 offset) {
        if (!ran) {
            Shader::HostTranslateInfo host{};
            host.min_ssbo_alignment = 16;
            Shader::Optimization::GlobalMemoryToStorageBufferPass(env, program, host);
            ran = true;
        }
        for (const auto& descriptor : program.info.storage_buffers_descriptors) {
            if (descriptor.cbuf_index == 0 && descriptor.cbuf_offset == offset) {
                return descriptor;
            }
        }
        FAIL("No descriptor for the pointer");
        return program.info.storage_buffers_descriptors.front();
    }

    /// Runs the pass once and returns the read extent of the buffer whose pointer is at c0[offset]
    const StorageWriteExtent& Read(u32 offset) {
        const auto& descriptor{Run(offset)};
        const auto& descriptors{program.info.storage_buffers_descriptors};
        REQUIRE(program.info.storage_read_extents.size() == descriptors.size());
        return program.info
            .storage_read_extents[static_cast<size_t>(&descriptor - descriptors.data())];
    }

    /// FRITCHII_EXTENT_CENSUS / FRITCHII_EXTENT_CBUF: the extra of the buffer at c0[offset]
    const Shader::StorageWriteExtra& Extra(u32 offset) {
        const auto& descriptor{Run(offset)};
        const auto& descriptors{program.info.storage_buffers_descriptors};
        REQUIRE(program.info.storage_write_extras.size() == descriptors.size());
        return program.info
            .storage_write_extras[static_cast<size_t>(&descriptor - descriptors.data())];
    }
};

IR::U32 Word(IR::IREmitter& ir, u32 offset) {
    return ir.GetCbuf(ir.Imm32(0), ir.Imm32(offset));
}

/// The 64-bit pointer at c0[offset] plus a 32-bit byte offset
IR::U64 Address(IR::IREmitter& ir, u32 offset, const IR::Value& bytes) {
    return ir.PackUint2x32(ir.CompositeConstruct(
        IR::U32{ir.IAdd(Word(ir, offset), IR::U32{bytes})}, Word(ir, offset + 4)));
}

std::optional<u64> At(const StorageWriteExtent& extent, std::array<u32, 3> block,
                      std::optional<std::array<u32, 3>> grid) {
    return Shader::ResolveStorageWriteExtent(extent, block, grid);
}
} // Anonymous namespace

TEST_CASE("Stores at constant offsets end at the last byte written", "[shader][storage_extent]") {
    KernelProgram fixture;
    fixture.env.native = GENERATE(false, true);
    IR::IREmitter ir{fixture.entry};
    ir.WriteGlobal32(Address(ir, 0x170, ir.Imm32(0x40)), ir.Imm32(1));
    ir.WriteGlobal128(Address(ir, 0x170, ir.Imm32(0x10)), ir.CompositeConstruct(
        ir.Imm32(0), ir.Imm32(0), ir.Imm32(0), ir.Imm32(0)));
    const auto& descriptor{fixture.Run(0x170)};
    REQUIRE(descriptor.is_written);
    const StorageWriteExtent& extent{descriptor.write_extent};
    CHECK(extent.bounded);
    CHECK_FALSE(extent.guarded);
    CHECK(extent.const_end == 0x44);
    CHECK(extent.local_stride == std::array<u32, 3>{});
    CHECK(extent.group_stride == std::array<u32, 3>{});
    CHECK(At(extent, {64, 1, 1}, std::array<u32, 3>{100, 1, 1}) == 0x44);
    CHECK(At(extent, {64, 1, 1}, std::nullopt) == 0x44);
}

TEST_CASE("Stores indexed by the global invocation id scale with the dispatch",
          "[shader][storage_extent]") {
    // ((ctaid.x << 6) + tid.x) << 8 + 236: one 256-byte record per invocation, as Ocarina of
    // Time's effect and skinning programs write them
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 global{ir.IAdd(ir.ShiftLeftLogical(ir.WorkgroupIdX(), ir.Imm32(6)),
                                 ir.LocalInvocationIdX())};
    const IR::U32 record{ir.ShiftLeftLogical(global, ir.Imm32(8))};
    ir.WriteGlobal64(Address(ir, 0x150, ir.IAdd(record, ir.Imm32(236))),
                     ir.CompositeConstruct(ir.Imm32(0), ir.Imm32(0)));
    const StorageWriteExtent& extent{fixture.Run(0x150).write_extent};
    REQUIRE(extent.bounded);
    CHECK(extent.const_end == 244);
    CHECK(extent.local_stride == std::array<u32, 3>{256, 0, 0});
    CHECK(extent.group_stride == std::array<u32, 3>{16384, 0, 0});
    // 64 invocations: one group
    CHECK(At(extent, {64, 1, 1}, std::array<u32, 3>{1, 1, 1}) == 244 + 256 * 63);
    CHECK(At(extent, {64, 1, 1}, std::array<u32, 3>{3, 1, 1}) == 244 + 256 * 63 + 16384 * 2);
    // An indirect dispatch's group count is unknown
    CHECK_FALSE(At(extent, {64, 1, 1}, std::nullopt).has_value());
}

TEST_CASE("Constant multiplications and masks keep the bound", "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 lane{ir.BitwiseAnd(ir.LocalInvocationIdX(), ir.Imm32(31))};
    const IR::U32 row{ir.IMul(ir.WorkgroupIdY(), ir.Imm32(48))};
    ir.WriteGlobal32(Address(ir, 0x160, ir.IAdd(ir.ShiftLeftLogical(lane, ir.Imm32(2)), row)),
                     ir.Imm32(0));
    const StorageWriteExtent& extent{fixture.Run(0x160).write_extent};
    REQUIRE(extent.bounded);
    CHECK(extent.const_end == 31 * 4 + 4);
    CHECK(extent.local_stride == std::array<u32, 3>{});
    CHECK(extent.group_stride == std::array<u32, 3>{0, 48, 0});
}

TEST_CASE("A multiplication split in 16-bit halves is bounded while its operand fits",
          "[shader][storage_extent]") {
    // Native IMAD lowered through XMAD: lo(id) * 48 + ((hi(id) * 48) << 16)
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 id{ir.IAdd(ir.ShiftLeftLogical(ir.WorkgroupIdX(), ir.Imm32(6)),
                             ir.LocalInvocationIdX())};
    const IR::U32 low_half{ir.BitFieldExtract(id, ir.Imm32(0), ir.Imm32(16), false)};
    const IR::U32 high_half{ir.BitFieldExtract(id, ir.Imm32(16), ir.Imm32(16), false)};
    const IR::U32 product{ir.IAdd(ir.IMul(low_half, ir.Imm32(48)),
                                  ir.ShiftLeftLogical(ir.IMul(high_half, ir.Imm32(48)),
                                                      ir.Imm32(16)))};
    ir.WriteGlobal32(Address(ir, 0x170, product), ir.Imm32(0));
    const StorageWriteExtent& extent{fixture.Run(0x170).write_extent};
    REQUIRE(extent.bounded);
    CHECK(extent.guarded);
    CHECK(extent.local_stride == std::array<u32, 3>{48, 0, 0});
    CHECK(extent.group_stride == std::array<u32, 3>{48 * 64, 0, 0});
    // 1024 groups of 64: the largest id is 0xffff, its high half still zero
    CHECK(At(extent, {64, 1, 1}, std::array<u32, 3>{1024, 1, 1}) == 4 + 48 * (1024 * 64 - 1));
    // 1025 groups: the id reaches 2^16 and its high half is no longer zero
    CHECK_FALSE(At(extent, {64, 1, 1}, std::array<u32, 3>{1025, 1, 1}).has_value());
}

TEST_CASE("Offsets read from memory or other parameters are unbounded",
          "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const bool loaded = GENERATE(false, true);
    // An index loaded from another buffer, or a count or offset parameter from the constant
    // buffer
    const IR::U32 index{
        loaded ? ir.LoadGlobal32(Address(
                     ir, 0x180, IR::U32{ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(2))}))
               : IR::U32{ir.IAdd(ir.LocalInvocationIdX(), Word(ir, 0x1a0))}};
    ir.WriteGlobal32(Address(ir, 0x170, ir.ShiftLeftLogical(index, ir.Imm32(2))), ir.Imm32(0));
    const auto& descriptor{fixture.Run(0x170)};
    CHECK(descriptor.is_written);
    CHECK_FALSE(descriptor.write_extent.bounded);
    CHECK_FALSE(At(descriptor.write_extent, {64, 1, 1}, std::array<u32, 3>{1, 1, 1}));
    if (loaded) {
        // The buffer only read keeps no extent and is not written
        const auto& source{fixture.Run(0x180)};
        CHECK_FALSE(source.is_written);
        CHECK_FALSE(source.write_extent.bounded);
    }
}

TEST_CASE("Loop-carried store offsets are unbounded", "[shader][storage_extent]") {
    KernelProgram fixture;
    fixture.entry.AddBranch(&fixture.loop);
    fixture.loop.AddBranch(&fixture.loop);
    fixture.program.blocks = {&fixture.entry, &fixture.loop};
    fixture.program.post_order_blocks = {&fixture.loop, &fixture.entry};
    IR::IREmitter entry{fixture.entry};
    const IR::U32 first{entry.ShiftLeftLogical(entry.LocalInvocationIdX(), entry.Imm32(2))};
    IR::IREmitter body{fixture.loop};
    fixture.loop.AppendNewInst(IR::Opcode::Phi, {});
    IR::Inst* const cursor{&fixture.loop.back()};
    cursor->SetFlags(IR::Type::U32);
    body.WriteGlobal32(Address(body, 0x170, IR::U32{IR::Value{cursor}}), body.Imm32(0));
    const IR::U32 next{body.IAdd(IR::U32{IR::Value{cursor}}, body.Imm32(256))};
    cursor->AddPhiOperand(&fixture.entry, first);
    cursor->AddPhiOperand(&fixture.loop, next);
    CHECK_FALSE(fixture.Run(0x170).write_extent.bounded);
}

TEST_CASE("Selected offsets take the larger alternative", "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U1 condition{ir.IEqual(ir.LocalInvocationIdX(), ir.Imm32(0))};
    const IR::U32 offset{ir.Select(condition, ir.Imm32(0x20),
                                   ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(4)))};
    ir.WriteGlobal32(Address(ir, 0x170, offset), ir.Imm32(0));
    const StorageWriteExtent& extent{fixture.Run(0x170).write_extent};
    REQUIRE(extent.bounded);
    CHECK(At(extent, {64, 1, 1}, std::array<u32, 3>{1, 1, 1}) == 0x24 + 16 * 63);
}

TEST_CASE("A global store left untracked makes every written buffer unbounded",
          "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    ir.WriteGlobal32(Address(ir, 0x170, ir.Imm32(0)), ir.Imm32(0));
    // An address with no constant-buffer pointer at all stays a global store
    ir.WriteGlobal32(ir.PackUint2x32(ir.CompositeConstruct(ir.LocalInvocationIdX(), ir.Imm32(0))),
                     ir.Imm32(0));
    CHECK_FALSE(fixture.Run(0x170).write_extent.bounded);
}

TEST_CASE("Loads and atomics have read extents, buffers only stored to read nothing",
          "[shader][storage_extent]") {
    KernelProgram fixture;
    fixture.env.native = GENERATE(false, true);
    IR::IREmitter ir{fixture.entry};
    const IR::U32 word{ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(2))};
    const IR::U32 value{ir.LoadGlobal32(Address(ir, 0x180, word))};
    const IR::Value vector{ir.LoadGlobal128(Address(
        ir, 0x190, ir.IAdd(ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(4)),
                           ir.Imm32(0x10))))};
    const IR::U32 old{ir.GlobalAtomicIAdd(Address(ir, 0x1a0, word), ir.Imm32(1))};
    ir.WriteGlobal32(Address(ir, 0x170, ir.Imm32(0)),
                     ir.IAdd(ir.IAdd(value, old), IR::U32{ir.CompositeExtract(vector, 0)}));
    const StorageWriteExtent& loaded{fixture.Read(0x180)};
    REQUIRE(loaded.bounded);
    CHECK(loaded.const_end == 4);
    CHECK(loaded.local_stride == std::array<u32, 3>{4, 0, 0});
    CHECK(At(loaded, {64, 1, 1}, std::array<u32, 3>{1, 1, 1}) == 4 + 4 * 63);
    const StorageWriteExtent& wide{fixture.Read(0x190)};
    REQUIRE(wide.bounded);
    CHECK(wide.const_end == 0x20);
    CHECK(wide.local_stride == std::array<u32, 3>{16, 0, 0});
    const StorageWriteExtent& atomic{fixture.Read(0x1a0)};
    REQUIRE(atomic.bounded);
    CHECK(atomic.const_end == 4);
    CHECK(atomic.local_stride == std::array<u32, 3>{4, 0, 0});
    // Only stored to: empty and proven
    const StorageWriteExtent& stored{fixture.Read(0x170)};
    CHECK(stored.bounded);
    CHECK(At(stored, {64, 1, 1}, std::array<u32, 3>{1, 1, 1}) == 0);
}

TEST_CASE("A load indexed by a loaded value has no read extent", "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 index{ir.LoadGlobal32(
        Address(ir, 0x180, ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(2))))};
    const IR::U32 value{ir.LoadGlobal32(Address(ir, 0x190, ir.ShiftLeftLogical(index, ir.Imm32(2))))};
    ir.WriteGlobal32(Address(ir, 0x170, ir.Imm32(0)), value);
    CHECK_FALSE(fixture.Read(0x190).bounded);
}

TEST_CASE("An access left global leaves every read extent unknown", "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 value{ir.LoadGlobal32(Address(ir, 0x180, ir.Imm32(0)))};
    // An address with no constant-buffer pointer at all stays a global load
    const IR::U32 other{ir.LoadGlobal32(
        ir.PackUint2x32(ir.CompositeConstruct(ir.LocalInvocationIdX(), ir.Imm32(0))))};
    ir.WriteGlobal32(Address(ir, 0x170, ir.Imm32(0)), ir.IAdd(value, other));
    CHECK_FALSE(fixture.Read(0x180).bounded);
    CHECK_FALSE(fixture.Read(0x170).bounded);
}

TEST_CASE("Store extents merge to cover both", "[shader][storage_extent]") {
    StorageWriteExtent lhs{.bounded = true, .const_end = 16, .local_stride = {4, 0, 0}};
    const StorageWriteExtent rhs{.bounded = true, .const_end = 8, .group_stride = {0, 32, 0}};
    Shader::MergeStorageWriteExtent(lhs, rhs);
    CHECK(lhs.bounded);
    CHECK(lhs.const_end == 16);
    CHECK(lhs.local_stride == std::array<u32, 3>{4, 0, 0});
    CHECK(lhs.group_stride == std::array<u32, 3>{0, 32, 0});
    Shader::MergeStorageWriteExtent(lhs, StorageWriteExtent{});
    CHECK_FALSE(lhs.bounded);
}

TEST_CASE("An index from a constant buffer word is a cbuf extent term",
          "[shader][storage_extent]") {
    // c0[0x1a0] * 0x100 + tid.x * 4: an object's record, picked by a uniform
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 record{ir.IMul(Word(ir, 0x1a0), ir.Imm32(0x100))};
    const IR::U32 lane{ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(2))};
    ir.WriteGlobal32(Address(ir, 0x170, ir.IAdd(record, lane)), ir.Imm32(0));
    const auto& descriptor{fixture.Run(0x170)};
    CHECK_FALSE(descriptor.write_extent.bounded);
    const Shader::StorageWriteExtra& extra{fixture.Extra(0x170)};
    CHECK(extra.write_unknown.kind == Shader::ExtentUnknown::OtherCbufWord);
    CHECK(extra.write_unknown.cbuf_index == 0);
    CHECK(extra.write_unknown.cbuf_offset == 0x1a0);
    const Shader::StorageCbufExtent& cbuf{extra.cbuf_extent};
    REQUIRE(cbuf.ids.bounded);
    REQUIRE(cbuf.num_terms == 1);
    CHECK(cbuf.num_guard_terms == 0);
    CHECK(cbuf.terms[0].cbuf_index == 0);
    CHECK(cbuf.terms[0].cbuf_offset == 0x1a0);
    CHECK(cbuf.terms[0].coef == 0x100);
    CHECK(cbuf.ids.const_end == 4);
    CHECK(cbuf.ids.local_stride == std::array<u32, 3>{4, 0, 0});
    const std::array<u32, Shader::MaxExtentTerms> values{3, 0, 0, 0};
    const std::array<u32, Shader::MaxExtentTerms> no_guard{};
    CHECK(Shader::ResolveStorageCbufExtent(cbuf, {64, 1, 1}, std::array<u32, 3>{1, 1, 1}, values,
                                           no_guard) == 4 + 4 * 63 + 3 * 0x100);
}

TEST_CASE("A constant buffer read at a computed offset leaves the cbuf extent unknown",
          "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 lane{ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(2))};
    const IR::U32 index{ir.GetCbuf(ir.Imm32(0), lane)};
    ir.WriteGlobal32(Address(ir, 0x170, ir.ShiftLeftLogical(index, ir.Imm32(2))), ir.Imm32(0));
    const auto& descriptor{fixture.Run(0x170)};
    CHECK_FALSE(descriptor.write_extent.bounded);
    const Shader::StorageWriteExtra& extra{fixture.Extra(0x170)};
    CHECK(extra.write_unknown.kind == Shader::ExtentUnknown::DynamicCbuf);
    CHECK_FALSE(extra.cbuf_extent.ids.bounded);
}

TEST_CASE("Record stores keep a footprint modulo the record stride", "[shader][storage_extent]") {
    // A record picked by a constant buffer word, stride 0x100: stores at +0x20..+0x2f and
    // +0x40..+0x43, never +0x10
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 record{ir.IMul(Word(ir, 0x1a0), ir.Imm32(0x100))};
    ir.WriteGlobal128(Address(ir, 0x170, ir.IAdd(record, ir.Imm32(0x20))),
                      ir.CompositeConstruct(ir.Imm32(0), ir.Imm32(0), ir.Imm32(0), ir.Imm32(0)));
    ir.WriteGlobal32(Address(ir, 0x170, ir.IAdd(record, ir.Imm32(0x40))), ir.Imm32(0));
    const Shader::StorageStoreFootprint& footprint{fixture.Extra(0x170).footprint};
    REQUIRE(footprint.Valid());
    CHECK(footprint.modulus == 0x100);
    CHECK_FALSE(footprint.ids_only);
    CHECK_FALSE(footprint.covers_all);
    REQUIRE(footprint.num_intervals == 2);
    CHECK(footprint.intervals[0].begin == 0x20);
    CHECK(footprint.intervals[0].end == 0x30);
    CHECK(footprint.intervals[1].begin == 0x40);
    CHECK(footprint.intervals[1].end == 0x44);
    CHECK_FALSE(Shader::StoreFootprintHits(footprint, 0x310, 8));
    CHECK(Shader::StoreFootprintHits(footprint, 0x32c, 8));
    CHECK(Shader::StoreFootprintHits(footprint, 0x543, 1));
    CHECK_FALSE(Shader::StoreFootprintHits(footprint, -0xf0, 4)); // +0x10 of the record before
}

TEST_CASE("Stores a lane apart cover every residue", "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 lane{ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(2))};
    ir.WriteGlobal32(Address(ir, 0x170, lane), ir.Imm32(0));
    const Shader::StorageStoreFootprint& footprint{fixture.Extra(0x170).footprint};
    REQUIRE(footprint.Valid());
    CHECK(footprint.modulus == 4);
    CHECK(footprint.ids_only);
    CHECK(footprint.covers_all);
    CHECK(Shader::StoreFootprintHits(footprint, 0x10, 1));
}

TEST_CASE("Constant stores have an exact footprint", "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    ir.WriteGlobal32(Address(ir, 0x170, ir.Imm32(0x40)), ir.Imm32(1));
    ir.WriteGlobal32(Address(ir, 0x170, ir.Imm32(0xfffffff0)), ir.Imm32(1)); // 16 bytes below
    const Shader::StorageStoreFootprint& footprint{fixture.Extra(0x170).footprint};
    REQUIRE(footprint.Valid());
    CHECK(footprint.modulus == (u64{1} << 32));
    CHECK(Shader::StoreFootprintHits(footprint, 0x40, 4));
    CHECK_FALSE(Shader::StoreFootprintHits(footprint, 0x44, 4));
    CHECK(Shader::StoreFootprintHits(footprint, -0x10, 4));
    CHECK_FALSE(Shader::StoreFootprintHits(footprint, -0x20, 4));
}

TEST_CASE("A record stride that is not a power of two keeps its residues",
          "[shader][storage_extent]") {
    // 0x330-byte records picked by a loaded index: stores at +0x40 and (two records on) +0x48
    const std::array<Shader::StoreCongruence, 2> stores{{
        {.c = 0x40, .m = 0x330, .size = 4},
        {.c = -0x330 + 0x48, .m = 0x660, .size = 4},
    }};
    const Shader::StorageStoreFootprint footprint{
        Shader::MakeStoreFootprint(stores.data(), stores.size(), false)};
    REQUIRE(footprint.Valid());
    CHECK(footprint.modulus == 0x330);
    CHECK_FALSE(footprint.PowerOfTwo());
    REQUIRE(footprint.num_intervals == 2);
    CHECK(footprint.intervals[0].begin == 0x40);
    CHECK(footprint.intervals[0].end == 0x44);
    CHECK(footprint.intervals[1].begin == 0x48);
    CHECK(footprint.intervals[1].end == 0x4c);
    CHECK_FALSE(Shader::StoreFootprintHits(footprint, 0x330 * 5 + 0x10, 8));
    CHECK(Shader::StoreFootprintHits(footprint, 0x330 * 5 + 0x48, 1));
    CHECK(Shader::StoreFootprintHits(footprint, -0x330 + 0x40, 4));
}

TEST_CASE("A footprint reduces to its modulus' power-of-two part", "[shader][storage_extent]") {
    // 0x110-byte records, stores at +0x80..+0x8b and +0x108..+0x10b: modulo 0x10 they cover
    // +0x0..+0xb, so a read at +0x10 (residue 0) is inside and +0xc is still outside
    const std::array<Shader::StoreCongruence, 2> stores{{
        {.c = 0x80, .m = 0x110, .size = 12},
        {.c = 0x108, .m = 0x110, .size = 4},
    }};
    const Shader::StorageStoreFootprint footprint{
        Shader::MakeStoreFootprint(stores.data(), stores.size(), true)};
    REQUIRE(footprint.modulus == 0x110);
    const Shader::StorageStoreFootprint reduced{Shader::ReduceStoreFootprint(footprint)};
    REQUIRE(reduced.Valid());
    CHECK(reduced.modulus == 0x10);
    CHECK(reduced.PowerOfTwo());
    REQUIRE(reduced.num_intervals == 1);
    CHECK(reduced.intervals[0].begin == 0);
    CHECK(reduced.intervals[0].end == 0xc);
    CHECK(Shader::StoreFootprintHits(reduced, 0x110 * 3 + 0x10, 4));
    CHECK_FALSE(Shader::StoreFootprintHits(reduced, 0x10c, 4));
}
