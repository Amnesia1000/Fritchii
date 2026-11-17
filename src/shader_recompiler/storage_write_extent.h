// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <compare>
#include <optional>

#include "common/common_types.h"

namespace Shader {

/// How far a program's stores and atomics to one storage buffer can reach, in bytes past the
/// pointer it reads from its constant buffer. Every store ends at or below
///
///   const_end + sum(local_stride[i] * local_id[i]) + sum(group_stride[i] * group_id[i])
///
/// for every invocation, where local_id is gl_LocalInvocationID and group_id gl_WorkGroupID. The
/// bound only holds while each value the stores' offsets split into 16-bit halves (multiplications
/// lowered from native IMAD) stays below 2^16: guard_* is an upper bound of the largest of them in
/// the same form, and the extent is unknown when it can reach 2^16 for a dispatch.
///
/// A native (Switch 2) program's storage pointer carries no size, so the buffer cache binds up to
/// 8 MiB behind it; this is what it can claim the dispatch wrote instead of all of it.
struct StorageWriteExtent {
    /// False: the reach is not known (or the buffer is never written); take the whole binding.
    bool bounded{};
    /// Whether the bound needs the guard below.
    bool guarded{};
    u32 const_end{};
    std::array<u32, 3> local_stride{};
    std::array<u32, 3> group_stride{};
    u32 guard_const{};
    std::array<u32, 3> guard_local_stride{};
    std::array<u32, 3> guard_group_stride{};

    auto operator<=>(const StorageWriteExtent&) const = default;
};

/// Bytes a dispatch's stores can reach past the pointer, or nullopt when unknown. grid is nullopt
/// for an indirect dispatch, whose group count is read by the GPU.
[[nodiscard]] constexpr std::optional<u64> ResolveStorageWriteExtent(
    const StorageWriteExtent& extent, const std::array<u32, 3>& block,
    const std::optional<std::array<u32, 3>>& grid) {
    if (!extent.bounded) {
        return std::nullopt;
    }
    const auto evaluate = [&](u64 constant, const std::array<u32, 3>& local,
                              const std::array<u32, 3>& group) -> std::optional<u64> {
        // Anything past 2^62 bytes is as good as unknown, and stops the sum from overflowing.
        constexpr u64 limit = u64{1} << 62;
        u64 total = constant;
        for (size_t i = 0; i < 3; ++i) {
            if (block[i] == 0) {
                return std::nullopt;
            }
            const u64 local_max = u64{block[i]} - 1;
            if (local[i] != 0 && local_max != 0) {
                if (local_max > limit / local[i]) {
                    return std::nullopt;
                }
                total += u64{local[i]} * local_max;
            }
            if (group[i] == 0) {
                continue;
            }
            if (!grid || (*grid)[i] == 0) {
                return std::nullopt;
            }
            const u64 group_max = u64{(*grid)[i]} - 1;
            if (group_max != 0) {
                if (group_max > limit / group[i]) {
                    return std::nullopt;
                }
                total += u64{group[i]} * group_max;
            }
            if (total > limit) {
                return std::nullopt;
            }
        }
        return total;
    };
    if (extent.guarded) {
        const auto guard =
            evaluate(extent.guard_const, extent.guard_local_stride, extent.guard_group_stride);
        if (!guard || *guard >= 0x10000) {
            return std::nullopt;
        }
    }
    return evaluate(extent.const_end, extent.local_stride, extent.group_stride);
}

/// FRITCHII_EXTENT_CENSUS: why a written storage buffer's store extent is unknown, the first exit
/// of the store analysis (StoreExtentAnalysis). Not stored with the info (info_serialize.cpp):
/// an info from the precompiled cache has Unrecorded.
enum class ExtentUnknown : u8 {
    Unrecorded,        ///< nothing recorded (an info from the precompiled cache)
    Known,             ///< the extent is proven
    OtherCbufWord,     ///< an offset reads another constant buffer word (cbuf_index, cbuf_offset)
    DynamicCbuf,       ///< a constant buffer read at a computed offset (opcode)
    LoadedFromMemory,  ///< an offset loaded from memory (opcode)
    AtomicResult,      ///< an offset is an atomic's result (opcode)
    LoopCarried,       ///< an offset carried around a loop (a cycle through a phi)
    SignedOr64Bit,     ///< a signed or 64-bit computation (opcode)
    VariableProduct,   ///< two non-constant factors multiplied, or a variable subtracted (opcode)
    OtherOp,           ///< an instruction the analysis does not bound (opcode)
    TooDeep,           ///< the expression is deeper than the analysis follows
    GuardFailed,       ///< a lowered multiplication's 16-bit split it cannot bound
    AddressNotTracked, ///< the store's 64-bit address is not split into pointer + low word
    NoPointerTerm,     ///< the low word does not add the pointer word exactly once
    TooLarge,          ///< a constant or stride past the analysis' limits
    UntrackedWrites,   ///< a store of the program stays global (routed by address at run time)
    RuntimeRouted,     ///< the store picks one of several pointers at run time
    GlobalStore,       ///< a driver buffer the program's global stores may land anywhere in
};

struct ExtentUnknownWhy {
    ExtentUnknown kind{};
    u16 opcode{}; ///< Shader::IR::Opcode, for the kinds that name one
    u32 cbuf_index{};
    u32 cbuf_offset{};

    auto operator<=>(const ExtentUnknownWhy&) const = default;
};

[[nodiscard]] constexpr const char* ExtentUnknownName(ExtentUnknown kind) {
    switch (kind) {
    case ExtentUnknown::Unrecorded:
        return "not recorded (no extra: a precompiled-cache info, or a bind path without it)";
    case ExtentUnknown::Known:
        return "proven";
    case ExtentUnknown::OtherCbufWord:
        return "index from a constant buffer word";
    case ExtentUnknown::DynamicCbuf:
        return "constant buffer read at a computed offset";
    case ExtentUnknown::LoadedFromMemory:
        return "index loaded from memory";
    case ExtentUnknown::AtomicResult:
        return "index from an atomic";
    case ExtentUnknown::LoopCarried:
        return "loop-carried index";
    case ExtentUnknown::SignedOr64Bit:
        return "signed or 64-bit math";
    case ExtentUnknown::VariableProduct:
        return "variable times variable (or minus a variable)";
    case ExtentUnknown::OtherOp:
        return "instruction not bounded";
    case ExtentUnknown::TooDeep:
        return "expression too deep";
    case ExtentUnknown::GuardFailed:
        return "16-bit split not bounded";
    case ExtentUnknown::AddressNotTracked:
        return "address not split into pointer + low word";
    case ExtentUnknown::NoPointerTerm:
        return "low word does not add the pointer once";
    case ExtentUnknown::TooLarge:
        return "constant or stride too large";
    case ExtentUnknown::UntrackedWrites:
        return "a store stays global";
    case ExtentUnknown::RuntimeRouted:
        return "pointer picked at run time";
    case ExtentUnknown::GlobalStore:
        return "driver buffer with global stores";
    }
    return "?";
}

/// FRITCHII_EXTENT_CBUF: a constant buffer word in a store's offset, coef times its value.
struct StorageExtentTerm {
    u32 cbuf_index{};
    u32 cbuf_offset{};
    u32 coef{};

    auto operator<=>(const StorageExtentTerm&) const = default;
};
inline constexpr size_t MaxExtentTerms = 4;

/// FRITCHII_EXTENT_CBUF: a store extent whose offsets may also add constant buffer words (an
/// object index read from a uniform times a stride). Every store ends at or below
///
///   ids' bound + sum(terms[i].coef * value of terms[i]'s word)
///
/// and the guard (when ids.guarded) is ids' guard bound plus the guard terms the same way. The
/// words are read at bind time, like the pointer word. Not stored with the info: an info from the
/// precompiled cache has an unbounded one.
struct StorageCbufExtent {
    StorageWriteExtent ids{};
    u8 num_terms{};
    u8 num_guard_terms{};
    std::array<StorageExtentTerm, MaxExtentTerms> terms{};
    std::array<StorageExtentTerm, MaxExtentTerms> guard_terms{};

    auto operator<=>(const StorageCbufExtent&) const = default;
};

/// Bytes a dispatch's stores can reach past the pointer with the terms' word values (`values`
/// for terms, `guard_values` for guard_terms, in order), or nullopt when unknown.
[[nodiscard]] constexpr std::optional<u64> ResolveStorageCbufExtent(
    const StorageCbufExtent& extent, const std::array<u32, 3>& block,
    const std::optional<std::array<u32, 3>>& grid, const std::array<u32, MaxExtentTerms>& values,
    const std::array<u32, MaxExtentTerms>& guard_values) {
    if (!extent.ids.bounded) {
        return std::nullopt;
    }
    constexpr u64 limit = u64{1} << 62;
    if (extent.ids.guarded) {
        // The guard's id part, resolved as an unguarded extent of its own
        const StorageWriteExtent guard_ids{
            .bounded = true,
            .const_end = extent.ids.guard_const,
            .local_stride = extent.ids.guard_local_stride,
            .group_stride = extent.ids.guard_group_stride,
        };
        const auto guard = ResolveStorageWriteExtent(guard_ids, block, grid);
        if (!guard) {
            return std::nullopt;
        }
        u64 total = *guard;
        if (total >= 0x10000) {
            return std::nullopt;
        }
        for (size_t i = 0; i < extent.num_guard_terms && i < MaxExtentTerms; ++i) {
            // A product of 2^16 or more fails the guard by itself (and keeps the sum small)
            const u64 product = u64{extent.guard_terms[i].coef} * guard_values[i];
            if (product >= 0x10000) {
                return std::nullopt;
            }
            total += product;
        }
        if (total >= 0x10000) {
            return std::nullopt;
        }
    }
    StorageWriteExtent ids = extent.ids;
    ids.guarded = false;
    const auto base = ResolveStorageWriteExtent(ids, block, grid);
    if (!base) {
        return std::nullopt;
    }
    u64 total = *base;
    for (size_t i = 0; i < extent.num_terms && i < MaxExtentTerms; ++i) {
        // coef and value are below 2^32: the product fits in 64 bits
        const u64 product = u64{extent.terms[i].coef} * values[i];
        if (product > limit) {
            return std::nullopt;
        }
        total += product;
        if (total > limit) {
            return std::nullopt;
        }
    }
    return total;
}

/// Widens an extent to also cover another (both bounds hold for each component's maximum).
constexpr void MergeStorageWriteExtent(StorageWriteExtent& base, const StorageWriteExtent& other) {
    if (!base.bounded || !other.bounded) {
        base = StorageWriteExtent{};
        return;
    }
    const auto max3 = [](std::array<u32, 3>& lhs, const std::array<u32, 3>& rhs) {
        for (size_t i = 0; i < 3; ++i) {
            lhs[i] = (std::max)(lhs[i], rhs[i]);
        }
    };
    base.const_end = (std::max)(base.const_end, other.const_end);
    max3(base.local_stride, other.local_stride);
    max3(base.group_stride, other.group_stride);
    if (other.guarded) {
        if (base.guarded) {
            base.guard_const = (std::max)(base.guard_const, other.guard_const);
            max3(base.guard_local_stride, other.guard_local_stride);
            max3(base.guard_group_stride, other.guard_group_stride);
        } else {
            base.guarded = true;
            base.guard_const = other.guard_const;
            base.guard_local_stride = other.guard_local_stride;
            base.guard_group_stride = other.guard_group_stride;
        }
    }
}

/// Adds `term` to a term list (the larger coefficient for a word already on it). False when the
/// list is full.
constexpr bool MergeExtentTerm(std::array<StorageExtentTerm, MaxExtentTerms>& terms, u8& count,
                               const StorageExtentTerm& term) {
    for (size_t i = 0; i < count; ++i) {
        if (terms[i].cbuf_index == term.cbuf_index && terms[i].cbuf_offset == term.cbuf_offset) {
            terms[i].coef = (std::max)(terms[i].coef, term.coef);
            return true;
        }
    }
    if (count >= MaxExtentTerms) {
        return false;
    }
    terms[count++] = term;
    return true;
}

/// FRITCHII_EXTENT_CBUF: widens a cbuf extent to also cover another (unbounded when the terms do
/// not fit).
constexpr void MergeStorageCbufExtent(StorageCbufExtent& base, const StorageCbufExtent& other) {
    MergeStorageWriteExtent(base.ids, other.ids);
    if (!base.ids.bounded) {
        base = StorageCbufExtent{};
        return;
    }
    for (size_t i = 0; i < other.num_terms; ++i) {
        if (!MergeExtentTerm(base.terms, base.num_terms, other.terms[i])) {
            base = StorageCbufExtent{};
            return;
        }
    }
    for (size_t i = 0; i < other.num_guard_terms; ++i) {
        if (!MergeExtentTerm(base.guard_terms, base.num_guard_terms, other.guard_terms[i])) {
            base = StorageCbufExtent{};
            return;
        }
    }
}

} // namespace Shader
