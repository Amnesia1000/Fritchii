// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "video_core/nx2_debug_control.h"

// EDEN_NX2_FRONT_HANDLES (live, default 0; nx2_debug_control.h): the split GPU thread's front reads
// the texture handle words a draw's shaders take from their constant buffers and sends them with
// the draw; the back (Vulkan GraphicsPipeline::ConfigureImpl) takes them instead of reading guest
// memory (fable_answer_pair 4.1 item 4, fable_answer_backthread_2 3.2 item 3).
//
// What the front reads. A Layout lists the words, as (stage, constant buffer, offset), in the order
// the back reads them. The back makes it from the pipeline it drew with and publishes it under the
// program registers that picked that pipeline's shaders (Key, ShaderCache::RefreshStages' inputs),
// when a record asks (NeedLayout) or the record's words missed its reads; the front looks the key up
// for every draw. The bound constant buffers come from the front's copy of the binds (Maxwell3D::
// FrontState::cbufs). Nothing about the layout or the binds has to be right for the result to be:
// the back takes a word only for an address equal to the one it computed itself.
//
// Why the words are the back's. A record is made at the ring position of the one entry that draws
// (a draw.end, or a draw.begin whose DrawDeferred ends an instance loop) and the back takes it only
// while it runs that entry (GpuSplit::BackEntryPosition): the entry writes no guest memory before
// its draw (a sunk state write or a deferred draw at most). The front reads guest memory after
// loading the back's position `done` and applies every write still queued after `done` over each
// word, newest last: 3D constant buffer uploads (by GPU address, GpuSplit::RangeLastWrite's words)
// and noted writes (by device address, NoteMemoryWrite / NoteMemoryWriteData). A word is refused
// when such a write's words are unknown; a draw is refused while anything queued after `done` may
// write guest memory in a way the front can't place: an entry of unknown effect, an unranged writer
// (KeplerMemory, Fermi2D, syncpoints), a 3D memory write without a range (or ranged writes past the
// remembered ones), a host sync (draws' writes reach memory only after one), a queued function (a
// map: the front's translation would be stale), more noted writes than remembered, run-ahead's CPU
// write chunks; a word is refused for compute or copy work that may write it, GPU-written pages
// (downloads land there), async report slots, Switch 2 sparse spans and the syncpoint shim.
// The guest CPU's own writes are as racy for the back's read as for the front's.

namespace VideoCommon::FrontHandles {

/// The switch: 0 off, 1 on, 2 verify (the back reads every shipped word too and keeps its own).
[[nodiscard]] inline int Mode() {
    return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_front_handles,
                                            "EDEN_NX2_FRONT_HANDLES");
}

/// Words a record carries at most (a mask bit each); a layout with more ships nothing.
inline constexpr u32 MaxWords = 32;

/// The program registers that pick a draw's shaders (ShaderCache::RefreshStages): per program slot
/// its config word, offset and native address words; the program region, rasterize enable, mesh
/// control; and the address space.
inline constexpr size_t KeyWords = 6 * 4 + 5;
struct Key {
    std::array<u32, KeyWords> words{};

    [[nodiscard]] u64 Hash() const {
        u64 hash = 0xcbf29ce484222325ULL;
        for (const u32 word : words) {
            hash = (hash ^ word) * 0x100000001b3ULL;
        }
        return hash ^ (hash >> 29);
    }
    [[nodiscard]] bool operator==(const Key&) const = default;
};

/// One handle word a pipeline reads: constant buffer `cbuf` of `stage`, at `offset` bytes.
struct LayoutWord {
    u8 stage{};
    u8 cbuf{};
    u16 reserved{};
    u32 offset{};
    [[nodiscard]] bool operator==(const LayoutWord&) const = default;
};

/// What a pipeline reads, in its read order (immutable once published).
struct Layout {
    Key key;
    u64 hash{}; ///< key.Hash()
    u32 count{};
    bool too_many{};
    std::array<LayoutWord, MaxWords> words{};
};

/// Record flags.
inline constexpr u32 NeedLayout = 1; ///< No layout for the key: the back publishes its own

/// The words the front read for one draw entry. Written by the front before the entry is
/// published (`tag` last, release); read by the back only while it runs that entry, when the
/// front can't be reusing the slot (it reuses a slot only once the back consumed past its entry).
struct Record {
    std::atomic<u64> tag{};   ///< Sequence number + 1 of the record in this slot, 0 = none yet
    std::atomic<u64> entry{}; ///< Ring position of the entry whose draw takes it
    u64 done{};               ///< The back's position the front read guest memory after
    const void* engine{};     ///< The Maxwell3D that draws
    u32 count{};              ///< Words in address/value
    u32 flags{};
    u32 valid{};              ///< Bit i: value[i] is what the back would read at address[i]
    u32 reserved{};
    Key key;
    std::array<GPUVAddr, MaxWords> address{};
    std::array<u32, MaxWords> value{};
};

/// Front counters (front thread only), per draw or per word.
enum class FrontCounter : u32 {
    Draws,     ///< Draw entries the front looked at
    Shipped,   ///< Records sent with words
    Words,     ///< Words sent valid
    Overlaid,  ///< Of those, taken from a write still queued for the back
    Learn,     ///< Records that asked the back for a layout
    // Draws sent without a record:
    SideFull,      ///< The record slot's entry not consumed yet
    TooMany,       ///< Layout over MaxWords
    Runahead,      ///< EDEN_NX2_RUNAHEAD on, or its chunks pending
    Function,      ///< A queued function (map)
    Unranged,      ///< An unranged writer queued
    MethodWrite,   ///< A 3D memory write without a range queued (or past the remembered ranges)
    Sync,          ///< A host sync queued
    UnknownWriter, ///< An entry of unknown effect queued
    NotedOverflow, ///< More noted writes queued than remembered
    NoWords,       ///< Every word refused
    // Words sent invalid (the back reads them):
    Disabled,      ///< The front's copy of the bind is disabled
    Unaligned,     ///< Not 4-byte aligned, or across a page
    Sparse,        ///< Switch 2 sparse span or syncpoint shim
    Unmapped,      ///< No device address
    GpuWritten,    ///< GPU-written page
    Async,         ///< A report or timed release may land there
    Writer,        ///< Compute or copy work queued over it
    RangedUnknown, ///< The newest queued 3D write over it has no known words
    NotedUnknown,  ///< The newest queued noted write over it has no known words
    Count,
};
inline constexpr std::array<const char*, static_cast<size_t>(FrontCounter::Count)> front_names{
    "draws",         "shipped",       "words",        "overlaid",       "learn",
    "side_full",     "too_many",      "runahead",     "function",       "unranged",
    "method_write",  "sync",          "unknown_writer", "noted_overflow", "no_words",
    "disabled",      "unaligned",     "sparse",       "unmapped",       "gpu_written",
    "async",         "writer",        "ranged_unknown", "noted_unknown",
};

/// Back counters (written by the back only, read by the front's report).
enum class BackCounter : u32 {
    Matched,    ///< Draws that found their record
    Missing,    ///< Draws with the switch on and no record
    Expired,    ///< Records no draw took
    Used,       ///< Words taken from a record
    Refused,    ///< Words read by the back that the record had invalid
    Misses,     ///< Words read by the back that the record didn't list
    Verified,   ///< Words read by the back to check them (mode 2)
    Mismatches, ///< Of those, different from the record (must stay 0)
    Learned,    ///< Layouts published
    LearnSame,  ///< Republish asked with the same layout already there
    LearnFull,  ///< Layouts not published: registry full
    Count,
};
inline constexpr std::array<const char*, static_cast<size_t>(BackCounter::Count)> back_names{
    "matched", "missing", "expired",  "used",    "refused",    "misses",
    "verified", "mismatches", "learned", "learn_same", "learn_full",
};

class State {
public:
    // Front.

    void Count(FrontCounter counter, u64 amount = 1) {
        front_counts[static_cast<size_t>(counter)] += amount;
    }

    /// The layout published for `key`, or null.
    [[nodiscard]] const Layout* Find(const Key& key) const {
        const u64 hash = key.Hash();
        const size_t base = static_cast<size_t>(hash) & (LayoutSlots - 1) & ~(LayoutWays - 1);
        for (size_t way = 0; way < LayoutWays; ++way) {
            const Layout* const layout = layouts[base + way].load(std::memory_order_acquire);
            if (layout != nullptr && layout->hash == hash && layout->key == key) {
                return layout;
            }
        }
        return nullptr;
    }

    /// The slot for the next record, or null while the back may still read the one in it (its
    /// entry not consumed: `consumed` is GpuSplit::ConsumedNow()).
    [[nodiscard]] Record* BeginRecord(u64 consumed) {
        Record& record = records[next_sequence % RecordSlots];
        if (record.tag.load(std::memory_order_relaxed) != 0 &&
            record.entry.load(std::memory_order_relaxed) >= consumed) {
            return nullptr;
        }
        return &record;
    }

    /// Publishes the record filled in BeginRecord's slot for the entry at `entry`.
    void CommitRecord(Record& record, u64 entry) {
        record.entry.store(entry, std::memory_order_relaxed);
        record.tag.store(++next_sequence, std::memory_order_release);
    }

    // Back.

    void Add(BackCounter counter, u64 amount = 1) {
        if (amount != 0) {
            auto& value = back_counts[static_cast<size_t>(counter)];
            value.store(value.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
        }
    }

    /// The record for the entry at ring position `entry` drawn by `engine`, or null. Records of
    /// earlier entries are passed (counted as expired unless a draw took them).
    [[nodiscard]] const Record* TakeRecord(u64 entry, const void* engine) {
        for (;;) {
            const Record& record = records[cursor % RecordSlots];
            const u64 tag = record.tag.load(std::memory_order_acquire);
            if (tag > cursor + 1) {
                // The front reused slots of records never looked at (the back passed their entries
                // without a draw that asked): continue from the one in this slot.
                Add(BackCounter::Expired, tag - 1 - cursor);
                cursor = tag - 1;
                cursor_taken = false;
            } else if (tag != cursor + 1) {
                return nullptr; // Not written yet
            }
            const u64 record_entry = record.entry.load(std::memory_order_relaxed);
            if (record_entry > entry) {
                return nullptr; // A later entry's
            }
            if (record_entry == entry) {
                if (record.engine != engine) {
                    return nullptr;
                }
                cursor_taken = true;
                return &record;
            }
            if (!cursor_taken) {
                Add(BackCounter::Expired);
            }
            ++cursor;
            cursor_taken = false;
        }
    }

    /// Publishes `words` as the layout for `key` (back). The table is 4-way per key hash; a key
    /// pushed out and asked for again gets the layout made for it before.
    void Publish(const Key& key, std::span<const LayoutWord> words, bool too_many) {
        const u64 hash = key.Hash();
        const u32 count = too_many ? 0U : static_cast<u32>(words.size());
        const auto same = [&](const Layout& layout) {
            return layout.hash == hash && layout.key == key && layout.too_many == too_many &&
                   layout.count == count &&
                   std::equal(words.begin(), words.begin() + count, layout.words.begin());
        };
        const size_t base = static_cast<size_t>(hash) & (LayoutSlots - 1) & ~(LayoutWays - 1);
        for (size_t way = 0; way < LayoutWays; ++way) {
            const Layout* const layout = layouts[base + way].load(std::memory_order_relaxed);
            if (layout != nullptr && same(*layout)) {
                Add(BackCounter::LearnSame);
                return;
            }
        }
        const Layout* made = nullptr;
        for (auto [it, end] = made_by_hash.equal_range(hash); it != end; ++it) {
            if (same(*it->second)) {
                made = it->second;
                break;
            }
        }
        if (made == nullptr) {
            if (owned.size() >= MaxLayouts) {
                Add(BackCounter::LearnFull);
                return;
            }
            auto layout = std::make_unique<Layout>();
            layout->key = key;
            layout->hash = hash;
            layout->count = count;
            layout->too_many = too_many;
            std::copy_n(words.begin(), count, layout->words.begin());
            made = layout.get();
            made_by_hash.emplace(hash, made);
            owned.push_back(std::move(layout));
            Add(BackCounter::Learned);
        }
        // The way holding this key (an older layout of it), else an empty one, else in turn.
        size_t target = LayoutWays;
        for (size_t way = 0; way < LayoutWays && target == LayoutWays; ++way) {
            const Layout* const layout = layouts[base + way].load(std::memory_order_relaxed);
            if (layout != nullptr && layout->hash == hash && layout->key == key) {
                target = way;
            }
        }
        for (size_t way = 0; way < LayoutWays && target == LayoutWays; ++way) {
            if (layouts[base + way].load(std::memory_order_relaxed) == nullptr) {
                target = way;
            }
        }
        if (target == LayoutWays) {
            target = next_victim++ % LayoutWays;
        }
        layouts[base + target].store(made, std::memory_order_release);
    }

    // Either thread once the split stopped (GpuSplit::Stop, the back joined): ring positions
    // start over at 0.
    void Reset() {
        for (Record& record : records) {
            record.tag.store(0, std::memory_order_relaxed);
            record.entry.store(0, std::memory_order_relaxed);
        }
        next_sequence = 0;
        cursor = 0;
        cursor_taken = false;
        for (auto& slot : layouts) {
            slot.store(nullptr, std::memory_order_relaxed);
        }
        made_by_hash.clear();
        owned.clear();
        next_victim = 0;
    }

    /// The front's report (GpuSplit::ReportBatched, every ~10 s): empty when nothing was counted.
    [[nodiscard]] std::string Report(double seconds) {
        std::array<u64, static_cast<size_t>(BackCounter::Count)> back{};
        bool any = false;
        for (size_t i = 0; i < back.size(); ++i) {
            const u64 now = back_counts[i].load(std::memory_order_relaxed);
            back[i] = now - back_reported[i];
            back_reported[i] = now;
            any |= back[i] != 0;
        }
        for (const u64 count : front_counts) {
            any |= count != 0;
        }
        if (!any) {
            return {};
        }
        std::string text = fmt::format(", front_handles.mode {}", Mode());
        for (size_t i = 0; i < front_counts.size(); ++i) {
            text += fmt::format(", front_handles.{} {:.0f}", front_names[i],
                                static_cast<double>(front_counts[i]) / seconds);
        }
        for (size_t i = 0; i < back.size(); ++i) {
            text += fmt::format(", front_handles.back_{} {:.0f}", back_names[i],
                                static_cast<double>(back[i]) / seconds);
        }
        front_counts.fill(0);
        return text;
    }

private:
    /// Over a frame of Castle Town draws (~2,500): how far the front may run ahead with records
    /// (past it, "side_full" counts the draws sent without one).
    static constexpr size_t RecordSlots = 4096;
    static constexpr size_t LayoutSlots = 4096;
    static constexpr size_t LayoutWays = 4;
    static constexpr size_t MaxLayouts = 16384;

    std::array<Record, RecordSlots> records{};
    std::array<std::atomic<const Layout*>, LayoutSlots> layouts{};
    // Back: every layout made (never freed while the split runs: the front holds them), by hash.
    std::vector<std::unique_ptr<Layout>> owned;
    std::unordered_multimap<u64, const Layout*> made_by_hash;
    size_t next_victim{};
    // Front.
    u64 next_sequence{};
    std::array<u64, static_cast<size_t>(FrontCounter::Count)> front_counts{};
    std::array<u64, static_cast<size_t>(BackCounter::Count)> back_reported{};
    // Back.
    u64 cursor{};
    bool cursor_taken{};
    alignas(64) std::array<std::atomic<u64>, static_cast<size_t>(BackCounter::Count)> back_counts{};
};

inline State g_state;

[[nodiscard]] inline State& Get() {
    return g_state;
}

} // namespace VideoCommon::FrontHandles
