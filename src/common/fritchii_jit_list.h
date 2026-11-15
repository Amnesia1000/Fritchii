// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// FRITCHII_JIT_BLOCK_LIST (boot, default 0): a per-title list of the game code blocks the CPU
// recompiler compiled, kept between launches, so the next launch compiles them on idle cores
// before the game reaches them (EDEN_NX2_JIT_WARM's queues, its budget). Port of the idea of
// zortos #1225 (ne1i). In OoT, the first time a code path runs (a save load, an enemy type's first
// kill) compiles hundreds to thousands of blocks in one frame: a 721 ms frame after a save load,
// then ~20 frames of 40-450 ms.
//
// The file: <cache dir>/jit/<program id>-<build id>.blocks. A 64-byte header naming the image,
// then append-only chunks (count, checksum, entries). An entry is a block's location key with its
// pc taken relative to the code image (the image moves every launch). A chunk cut short or with a
// wrong checksum ends the list; the file is cut back to the good part before the next write.

#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <vector>

#include "common/common_types.h"

namespace Common::FritchiiJitList {

/// FRITCHII_JIT_BLOCK_LIST (boot): 0 = off, 1 = record and warm up from the list, 2 = record only
/// (builds the list, compiles nothing ahead: the A side of an A/B).
[[nodiscard]] int Mode();

/// The application's code image is known (before its cores start): loads the list and starts
/// the writer. Does nothing when Mode() is 0.
void Begin(u64 program_id, std::span<const u8> build_id, u64 image_base, u64 image_size,
           const std::filesystem::path& cache_dir);

/// The application stops: writes what is left and stops the writer.
void End();

/// From the recompiler, any core: execution compiled block `key` (an A64 location key). Cheap
/// when the list is off or the key is already listed.
void Note(u64 key);

/// For `core`'s warm-up while it is idle: up to `max` listed keys it has not been handed yet, in
/// list order (appended to `out`). Returns how many.
std::size_t TakePreload(u32 core, std::vector<u64>& out, std::size_t max);

/// Whether a list is loaded for warm-up (Mode() 1 and Begin found entries).
[[nodiscard]] bool Preloading();

} // namespace Common::FritchiiJitList
