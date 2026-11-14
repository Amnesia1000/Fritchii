// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>

#include "common/common_types.h"

/// Page flags of the device address space that writers of game memory set and the native uniform
/// bank check takes (BufferCache::CheckNativeBankPagesTracked). In common so that the writers in
/// common (the write ring's Note, the guest store's after-store point) set them too; core names
/// them Core::g_... (core/gpu_dirty_memory_manager.h).
namespace Common {

/// One bit per device page of the 36-bit device address space (2 MiB), set by writers and taken
/// by the native uniform bank check (BufferCache, EDEN_NX2_EXP9).
class DevicePageFlags {
public:
    static constexpr u64 AddressBits = 36;
    static constexpr u64 PageBits = 12;

    void Mark(u64 address, u64 size) noexcept {
        constexpr u64 limit = u64{1} << AddressBits;
        if (size == 0 || address >= limit) {
            return;
        }
        u64 page = address >> PageBits;
        const u64 last = ((std::min)(address + size, limit) - 1) >> PageBits;
        while (page <= last) {
            const u64 first_bit = page % 64;
            const u64 bits = (std::min<u64>)(64 - first_bit, last - page + 1);
            const u64 mask = (bits == 64 ? ~u64{0} : (u64{1} << bits) - 1) << first_bit;
            std::atomic<u64>& word = words[page / 64];
            if ((word.load(std::memory_order_relaxed) & mask) != mask) {
                word.fetch_or(mask, std::memory_order_seq_cst);
            }
            page += bits;
        }
    }

    /// Whether any page of [address, address + size) is marked, without clearing anything. A
    /// relaxed look, as Take's first: a mark set after it is seen by the next look.
    [[nodiscard]] bool Any(u64 address, u64 size) const noexcept {
        constexpr u64 limit = u64{1} << AddressBits;
        if (size == 0 || address >= limit) {
            return false;
        }
        u64 page = address >> PageBits;
        const u64 last = ((std::min)(address + size, limit) - 1) >> PageBits;
        while (page <= last) {
            const u64 first_bit = page % 64;
            const u64 bits = (std::min<u64>)(64 - first_bit, last - page + 1);
            const u64 mask = (bits == 64 ? ~u64{0} : (u64{1} << bits) - 1) << first_bit;
            if ((words[page / 64].load(std::memory_order_relaxed) & mask) != 0) {
                return true;
            }
            page += bits;
        }
        return false;
    }

    /// Clears the bit of the page holding `address`, returning whether it was set.
    bool Take(u64 address) noexcept {
        if (address >= (u64{1} << AddressBits)) {
            return false;
        }
        const u64 page = address >> PageBits;
        const u64 bit = u64{1} << (page % 64);
        std::atomic<u64>& word = words[page / 64];
        if ((word.load(std::memory_order_relaxed) & bit) == 0) {
            return false;
        }
        return (word.fetch_and(~bit, std::memory_order_seq_cst) & bit) != 0;
    }

private:
    std::array<std::atomic<u64>, (u64{1} << (AddressBits - PageBits)) / 64> words{};
};

/// Pages guest CPU writes went to through the dirty memory managers: the buffer cache sees them
/// only at the next InvalidateGPUCache. Set before the store (Core::Memory's
/// HandleRasterizerWrite) and again after it (UniformKeepPages::AfterGuestStore).
inline DevicePageFlags g_collected_write_pages;
/// Pages whose guest memory the buffer cache saw change (WriteMemory, uploads in place,
/// downloads), and every emulator write the write ring hears of, after its bytes landed
/// (WriteRing::Note, MemoryManager::Write).
inline DevicePageFlags g_buffer_write_pages;
/// EDEN_NX2_TEX_BANK_GEN: pages written by anyone the GPU side hears of: guest CPU writes that
/// trapped (Memory::HandleRasterizerWrite, every one, collected or not), the buffer cache's
/// writes (the same places as g_buffer_write_pages) and TextureCache::WriteMemory. Only the
/// texture cache's watched-page harvest takes them (TextureCache::TexBankPageGen).
inline DevicePageFlags g_watch_write_pages;
/// The bank marks' gate: true while FRITCHII_BANK_FLAGGED or FRITCHII_BANK_CENSUS is non-zero
/// (set by the bank check, BufferCache::CheckNativeBankPagesTracked). Off, each mark below that
/// the hot rule made unneeded (the write ring's Note, MemoryManager::Write, unsafe and span
/// stores reaching Note, the after-store re-mark) is one relaxed load and a branch. The bank check
/// trusts no snapshot from before the gate opened: it drops them all two frames after.
inline std::atomic<bool> g_bank_marks{false};

[[nodiscard]] inline bool BankMarks() noexcept {
    return g_bank_marks.load(std::memory_order_relaxed);
}

/// Guest stores whose ranges the after-store point could not hold (more than a thread records):
/// no page flag was set after them. The bank check with FRITCHII_BANK_FLAGGED drops every
/// snapshot when this moves.
inline std::atomic<u64> g_collected_overflows{0};
/// FRITCHII_BANK_CENSUS=2: pages the write ring recorded a write to (WriteRing::Record), taken by
/// the census at each compare. Only while the ring is on.
inline DevicePageFlags g_ring_write_pages;

} // namespace Common
