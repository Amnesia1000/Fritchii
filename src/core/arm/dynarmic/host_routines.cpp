// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>
#include <zstd_errors.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "common/logging.h"
#include "core/arm/dynarmic/host_routines.h"
#include "core/core.h"
#include "core/hle/kernel/k_process.h"
#include "core/memory.h"

namespace Core::HostRoutines {
namespace {

// ------------------------------------------------------------------ sites

constexpr u64 OotProgramId = 0x0400486024DEC000ULL;
// main E0F3CCFA... (2.0.0, v131072), the build the offsets below were read from.
constexpr std::array<u8, 20> OotBuildId{0xE0, 0xF3, 0xCC, 0xFA, 0xF5, 0x00, 0x65, 0xAE, 0x4E, 0xA0,
                                        0x1E, 0xB5, 0xAE, 0xB1, 0x3B, 0x1A, 0x45, 0xDA, 0x33, 0x48};
// main is loaded right after rtld's 0x20000 bytes (the process's entry point is rtld's).
constexpr u64 MainOffset = 0x20000;

enum class Routine : u32 {
    ZstdFrames = 0, ///< whole-buffer decode (the resource loaders, ARes_Loading threads)
    ZstdStream = 1, ///< the stream decode wrapper (ParallelDecompThread)
    Count = 2,
};

struct Site {
    Routine routine;
    u64 main_offset;
    /// The function's first 16 words: the hook goes in only when all of them match.
    std::array<u32, 16> words;
};

constexpr std::array<Site, 2> OotSites{
    // zstd's decompressMultiFrame inlined: x0 the game's context, x1/x2 the output and its room,
    // x3/x4 the frames, x5/x6 a raw dictionary, x7 the game's DDict or 0.
    Site{Routine::ZstdFrames, 0xEB380,
         {0xD503237F, 0xA9BA7BFD, 0xA9016FFC, 0xA90267FA, 0x910003FD, 0xA9035FF8, 0xA90457F6,
          0xA9054FF4, 0xD11A43FF, 0xF9005BE2, 0xF90023E1, 0xAA0003EE, 0xF90037E7, 0xB5024E27,
          0xF100149F, 0x5401EBC3}},
    // The game's wrapper around zstd's buffer-less stream decode: x0 the stream object, x1/w2
    // one chunk of input, x3 the game's DDict or 0. Returns (w0) the bytes still to come.
    Site{Routine::ZstdStream, 0xF5C80,
         {0xD503237F, 0xD10243FF, 0xA9037BFD, 0xA9046FFC, 0xA90567FA, 0xA9065FF8, 0xA90757F6,
          0xA9084FF4, 0x9100C3FD, 0x39400008, 0x2A0203F7, 0xAA0103F4, 0xAA0003F3, 0x7100051F,
          0x54001F60, 0x71000D1F}},
};

constexpr u32 Pacibsp = 0xD503237F; // a hint: a no-op in the JIT, so a declined call skips it
constexpr u32 BrkBase = 0xD4200000;
constexpr u32 BrkImmBase = 0xF7A0; // BRK #0xF7A0 + routine

constexpr u32 BrkFor(Routine routine) {
    return BrkBase | ((BrkImmBase + static_cast<u32>(routine)) << 5);
}

/// FRITCHII_HOST_ZSTD (boot): bit 0 = whole-buffer decodes, bit 1 = stream decodes (3 = both).
u32 ZstdMode() {
    static const u32 mode = [] {
        const char* text = std::getenv("FRITCHII_HOST_ZSTD");
        const u32 value = text != nullptr ? static_cast<u32>(std::strtoul(text, nullptr, 10)) & 3 : 0;
        if (value != 0) {
            LOG_INFO(Core_ARM,
                     "FRITCHII_HOST_ZSTD={}: the game's zstd {} on the host's libzstd {}", value,
                     value == 3   ? "whole-buffer and stream decodes"
                     : value == 1 ? "whole-buffer decodes"
                                  : "stream decodes",
                     ZSTD_versionString());
        }
        return value;
    }();
    return mode;
}

bool RoutineOn(Routine routine) {
    switch (routine) {
    case Routine::ZstdFrames:
        return (ZstdMode() & 1) != 0;
    case Routine::ZstdStream:
        return (ZstdMode() & 2) != 0;
    default:
        return false;
    }
}

/// Per process: each site's verdict (checked once, at its first compile).
struct ProcessSites {
    std::mutex mutex;
    u64 process_id = ~u64{0};
    u64 main_base = 0;
    std::array<int, static_cast<size_t>(Routine::Count)> state{}; ///< 0 unchecked, 1 on, 2 off
};

ProcessSites& Verdicts() {
    static ProcessSites sites;
    return sites;
}

bool IsOot(Core::System& system, const Kernel::KProcess& process) {
    if (process.GetProgramId() != OotProgramId) {
        return false;
    }
    const auto& build_id = system.GetApplicationProcessBuildID();
    return std::equal(OotBuildId.begin(), OotBuildId.end(), build_id.begin());
}

// ------------------------------------------------------------------ counters and log line

struct Stats {
    std::atomic<u64> calls{};    ///< answered on the host
    std::atomic<u64> declined{}; ///< left to the game's code
    std::atomic<u64> out_bytes{};
    std::atomic<u64> host_ns{};
    std::atomic<u64> max_ns{};
    std::atomic<u64> errors{}; ///< answered with the decoder's error (as the game's would)
};
std::array<Stats, static_cast<size_t>(Routine::Count)> g_stats;
std::atomic<s64> g_last_log_ns{};

s64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void Add(std::atomic<u64>& counter, u64 value) {
    counter.fetch_add(value, std::memory_order_relaxed);
}

void NoteTime(Stats& stats, u64 ns) {
    Add(stats.host_ns, ns);
    u64 max = stats.max_ns.load(std::memory_order_relaxed);
    while (ns > max && !stats.max_ns.compare_exchange_weak(max, ns, std::memory_order_relaxed)) {
    }
}

/// "Host zstd" every 10 s at most, from whichever thread gets there, while anything ran.
void MaybeLog() {
    const s64 now = NowNs();
    s64 last = g_last_log_ns.load(std::memory_order_relaxed);
    if (now - last < 10'000'000'000LL ||
        !g_last_log_ns.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
        return;
    }
    const auto line = [](const char* name, Stats& s) {
        const u64 calls = s.calls.exchange(0, std::memory_order_relaxed);
        const u64 declined = s.declined.exchange(0, std::memory_order_relaxed);
        const u64 bytes = s.out_bytes.exchange(0, std::memory_order_relaxed);
        const u64 ns = s.host_ns.exchange(0, std::memory_order_relaxed);
        const u64 max_ns = s.max_ns.exchange(0, std::memory_order_relaxed);
        const u64 errors = s.errors.exchange(0, std::memory_order_relaxed);
        if (calls == 0 && declined == 0) {
            return;
        }
        LOG_INFO(Core_ARM,
                 "Host zstd (FRITCHII_HOST_ZSTD={}) {}: {} calls on the host, {:.1f} MiB out in "
                 "{:.1f} ms (longest call {:.2f} ms), {} errors as the game's, {} left to the "
                 "game's code (since the last line)",
                 ZstdMode(), name, calls, static_cast<double>(bytes) / (1024.0 * 1024.0),
                 static_cast<double>(ns) / 1e6, static_cast<double>(max_ns) / 1e6, errors,
                 declined);
    };
    line("whole-buffer", g_stats[static_cast<size_t>(Routine::ZstdFrames)]);
    line("stream", g_stats[static_cast<size_t>(Routine::ZstdStream)]);
}

/// Why a call went to the game's code: each reason logged once.
void NoteDecline(Routine routine, const char* why) {
    Add(g_stats[static_cast<size_t>(routine)].declined, 1);
    static std::mutex mutex;
    static std::array<const char*, 16> seen{};
    {
        std::scoped_lock lock{mutex};
        for (auto& entry : seen) {
            if (entry == why) {
                return;
            }
            if (entry == nullptr) {
                entry = why;
                break;
            }
        }
    }
    LOG_INFO(Core_ARM, "Host zstd: a {} decode runs as the game's code: {} (logged once)",
             routine == Routine::ZstdFrames ? "whole-buffer" : "stream", why);
}

// ------------------------------------------------------------------ the game's DDicts

// The game's zstd 1.4 DDict: +8 its content, +16 the size, +0x18..+0x6840 the entropy tables it
// built, +0x6840 the dictID, +0x6844 whether the tables are present (the code reads both).
constexpr u64 DictContent = 0x8;
constexpr u64 DictSize = 0x10;
constexpr u64 DictTables = 0x18;
constexpr u64 DictTablesEnd = 0x6840;
constexpr u64 DictId = 0x6840;
constexpr u64 DictEntropy = 0x6844;
constexpr u64 MaxDictBytes = 64ULL << 20;
constexpr size_t MaxDicts = 32;

struct HostDict {
    u64 guest = 0;
    u64 content = 0;
    u64 size = 0;
    u32 id = 0;
    u32 entropy = 0;
    std::vector<u8> tables; ///< the game's tables when it was made: another DDict built in place
                            ///< at the same address (same size and ID) has other tables
    ZSTD_DDict* dd = nullptr;

    HostDict() = default;
    HostDict(const HostDict&) = delete;
    HostDict& operator=(const HostDict&) = delete;
    ~HostDict() {
        ZSTD_freeDDict(dd);
    }
};

std::mutex g_dict_mutex;
std::vector<std::shared_ptr<HostDict>> g_dicts; ///< newest last

/// The game's DDict at `guest` as a libzstd one (its content copied), or null when libzstd does
/// not take it as the same dictionary.
std::shared_ptr<HostDict> DictFor(Core::Memory::Memory& memory, u64 guest) {
    const u64 content = memory.Read64(guest + DictContent);
    const u64 size = memory.Read64(guest + DictSize);
    const u32 id = memory.Read32(guest + DictId);
    const u32 entropy = memory.Read32(guest + DictEntropy);
    thread_local std::vector<u8> tables;
    tables.resize(DictTablesEnd - DictTables);
    memory.ReadBlock(guest + DictTables, tables.data(), tables.size());

    std::scoped_lock lock{g_dict_mutex};
    for (auto it = g_dicts.begin(); it != g_dicts.end(); ++it) {
        HostDict& dict = **it;
        if (dict.guest != guest) {
            continue;
        }
        if (dict.content == content && dict.size == size && dict.id == id &&
            dict.entropy == entropy && dict.tables == tables) {
            return *it;
        }
        // The game made another DDict here: streams still using the old one keep it.
        g_dicts.erase(it);
        break;
    }
    if (size == 0 || size > MaxDictBytes) {
        return nullptr;
    }
    std::vector<u8> bytes(size);
    memory.ReadBlock(content, bytes.data(), bytes.size());
    auto dict = std::make_shared<HostDict>();
    dict->dd = ZSTD_createDDict_advanced(bytes.data(), bytes.size(), ZSTD_dlm_byCopy,
                                         entropy != 0 ? ZSTD_dct_fullDict : ZSTD_dct_rawContent,
                                         ZSTD_defaultCMem);
    if (dict->dd == nullptr || ZSTD_getDictID_fromDDict(dict->dd) != id) {
        return nullptr;
    }
    dict->guest = guest;
    dict->content = content;
    dict->size = size;
    dict->id = id;
    dict->entropy = entropy;
    dict->tables = tables;
    if (g_dicts.size() >= MaxDicts) {
        g_dicts.erase(g_dicts.begin());
    }
    g_dicts.push_back(dict);
    LOG_INFO(Core_ARM, "Host zstd: the game's dictionary {:#x} (ID {:#x}, {} bytes{}) taken",
             guest, id, size, entropy != 0 ? ", with entropy tables" : ", raw content");
    return dict;
}

// ------------------------------------------------------------------ buffers

/// A per-thread buffer that grows without zero filling; given back when it got very large.
struct Scratch {
    std::unique_ptr<u8[]> data;
    size_t capacity = 0;

    u8* Get(size_t size) {
        if (size > capacity || data == nullptr) {
            data = std::make_unique_for_overwrite<u8[]>((std::max)(size, size_t{1}));
            capacity = (std::max)(size, size_t{1});
        }
        return data.get();
    }
    void Trim() {
        if (capacity > (64ULL << 20)) {
            data.reset();
            capacity = 0;
        }
    }
};

ZSTD_DCtx* ThreadDctx() {
    struct Holder {
        ZSTD_DCtx* z = ZSTD_createDCtx();
        ~Holder() {
            ZSTD_freeDCtx(z);
        }
    };
    thread_local Holder holder;
    return holder.z;
}

// ------------------------------------------------------------------ whole-buffer decode

constexpr u64 DctxFormat = 0x7110; // the game's context: frame format (0 zstd1, 1 magicless)
constexpr u64 MaxFrameInput = 256ULL << 20;
constexpr u64 MaxFrameOutput = 512ULL << 20;

/// main+0xEB380. The three callers set the context up in place, pass no raw dictionary and read
/// only the result: the content size, or zstd's error code.
bool ZstdFrames(Core::Memory::Memory& memory, Registers& regs, Stats& stats) {
    const u64 dctx = regs.Get(0);
    const u64 dst = regs.Get(1);
    const u64 room = regs.Get(2);
    const u64 src = regs.Get(3);
    const u64 size = regs.Get(4);
    const u64 ddict = regs.Get(7);
    if (regs.Get(5) != 0 || regs.Get(6) != 0) {
        NoteDecline(Routine::ZstdFrames, "a raw dictionary");
        return false;
    }
    if (memory.Read32(dctx + DctxFormat) != 0) {
        NoteDecline(Routine::ZstdFrames, "the context asks for another frame format");
        return false;
    }
    if (size > MaxFrameInput || room > MaxFrameOutput) {
        NoteDecline(Routine::ZstdFrames, "over 256 MiB in or 512 MiB out");
        return false;
    }
    std::shared_ptr<HostDict> dict;
    if (ddict != 0 && (dict = DictFor(memory, ddict)) == nullptr) {
        NoteDecline(Routine::ZstdFrames, "a dictionary libzstd does not take as the game's");
        return false;
    }
    ZSTD_DCtx* const z = ThreadDctx();
    if (z == nullptr) {
        NoteDecline(Routine::ZstdFrames, "no libzstd context");
        return false;
    }
    thread_local Scratch in_buffer;
    thread_local Scratch out_buffer;
    u8* const in = in_buffer.Get(size);
    u8* const out = out_buffer.Get(room);
    memory.ReadBlock(src, in, size);
    const size_t result =
        dict != nullptr ? ZSTD_decompress_usingDDict(z, out, room, in, size, dict->dd)
                        : ZSTD_decompress_usingDict(z, out, room, in, size, nullptr, 0);
    if (ZSTD_isError(result)) {
        // The game's decoder has written part of the output by then; its callers only test
        // the result.
        Add(stats.errors, 1);
    } else if (result != 0) {
        // WriteBlock marks the pages for the GPU caches as the game's stores would.
        memory.WriteBlock(dst, out, result);
        Add(stats.out_bytes, result);
    }
    in_buffer.Trim();
    out_buffer.Trim();
    regs.Set(0, result);
    return true;
}

// ------------------------------------------------------------------ stream decode

// The game's stream object (x0): its fields as the game's code reads and writes them.
constexpr u64 StState = 0;    // u8: 1 start, 2 decoding, 3 failed, 4 done
constexpr u64 StDst = 8;      // u64: where the next output goes
constexpr u64 StRoom = 16;    // s32: output room left
constexpr u64 StLeft = 20;    // u32: content still to come (the result)
constexpr u64 StWork = 24;    // u64: the context workspace
constexpr u64 StStageCap = 40; // s32: the staging buffer's size
constexpr u64 StStaged = 44;   // s32: bytes in the staging buffer
constexpr u64 StCtx = 48;     // u64: the context (the workspace, once started)
constexpr u64 MaxStreamContent = 512ULL << 20;
constexpr u64 MaxStreamBytes = 1ULL << 30; // all host streams' content buffers together
constexpr s32 MaxChunk = 64 << 20;
constexpr size_t MaxStreams = 64;

struct HostStream {
    ZSTD_DCtx* z = nullptr;
    std::shared_ptr<HostDict> dict;
    /// The frame's content as decoded so far: the window the next blocks copy from (the game's
    /// decoder reads it from its output; the bytes are the same).
    std::unique_ptr<u8[]> content;
    size_t content_size = 0;
    size_t produced = 0;
    /// The start of a piece that goes on in the next chunk (the game keeps it in its staging
    /// buffer; only this function reads that buffer, so the host keeps it here).
    std::vector<u8> staged;

    HostStream() = default;
    HostStream(const HostStream&) = delete;
    HostStream& operator=(const HostStream&) = delete;
    ~HostStream() {
        ZSTD_freeDCtx(z);
    }
};

std::mutex g_stream_mutex;
std::unordered_map<u64, std::unique_ptr<HostStream>> g_streams; ///< by the stream object's address
u64 g_stream_bytes = 0;
size_t g_stream_peak = 0;

/// One guest thread drives a stream object at a time (the game's job loop), so an entry is
/// used without the lock; only that stream's own calls make or free it.
HostStream* FindStream(u64 guest) {
    std::scoped_lock lock{g_stream_mutex};
    const auto it = g_streams.find(guest);
    return it != g_streams.end() ? it->second.get() : nullptr;
}

void DropStream(u64 guest) {
    std::scoped_lock lock{g_stream_mutex};
    const auto it = g_streams.find(guest);
    if (it != g_streams.end()) {
        g_stream_bytes -= it->second->content_size;
        g_streams.erase(it);
    }
}

/// A new frame on stream object `guest`: its entry made or reset, or null (the reason in *why).
HostStream* StartStream(u64 guest, u64 content_size, std::shared_ptr<HostDict> dict,
                        const char** why) {
    std::scoped_lock lock{g_stream_mutex};
    auto it = g_streams.find(guest);
    if (it != g_streams.end()) {
        g_stream_bytes -= it->second->content_size;
        g_streams.erase(it);
    }
    if (g_streams.size() >= MaxStreams) {
        *why = "64 host streams in use";
        return nullptr;
    }
    if (g_stream_bytes + content_size > MaxStreamBytes) {
        *why = "the host streams' buffers would pass 1 GiB";
        return nullptr;
    }
    auto stream = std::make_unique<HostStream>();
    stream->z = ZSTD_createDCtx();
    if (stream->z == nullptr) {
        *why = "no libzstd context";
        return nullptr;
    }
    stream->content =
        std::make_unique_for_overwrite<u8[]>((std::max)(content_size, u64{1}));
    stream->content_size = content_size;
    stream->dict = std::move(dict);
    HostStream* const result = stream.get();
    g_streams.emplace(guest, std::move(stream));
    g_stream_bytes += content_size;
    if (g_streams.size() > g_stream_peak) {
        g_stream_peak = g_streams.size();
        LOG_INFO(Core_ARM, "Host zstd: {} stream(s) decoding on the host at once", g_stream_peak);
    }
    return result;
}

/// One decompressContinue of `size` bytes, its output stored and counted as the game does.
size_t StreamContinue(Core::Memory::Memory& memory, HostStream& stream, u64 guest, const u8* src,
                      size_t size, Stats& stats) {
    const u64 dst = memory.Read64(guest + StDst);
    const u32 room32 = memory.Read32(guest + StRoom);
    const size_t room = static_cast<size_t>(static_cast<s64>(static_cast<s32>(room32)));
    const size_t capacity = (std::min)(room, stream.content_size - stream.produced);
    u8* const out = stream.content.get() + stream.produced;
    const size_t result = ZSTD_decompressContinue(stream.z, out, capacity, src, size);
    if (ZSTD_isError(result)) {
        return result;
    }
    if (result != 0) {
        memory.WriteBlock(dst, out, result);
        stream.produced += result;
        Add(stats.out_bytes, result);
    }
    memory.Write64(guest + StDst, dst + static_cast<u64>(static_cast<s64>(static_cast<s32>(result))));
    memory.Write32(guest + StRoom, room32 - static_cast<u32>(result));
    memory.Write32(guest + StLeft, memory.Read32(guest + StLeft) - static_cast<u32>(result));
    return result;
}

/// The game's loop over one chunk of a started stream: its result.
s32 StreamChunk(Core::Memory::Memory& memory, HostStream& stream, u64 guest, const u8* src, s32 n,
                Stats& stats) {
    const auto fail = [&](size_t result) -> s32 {
        memory.Write8(guest + StState, 3);
        Add(stats.errors, 1);
        return ZSTD_getErrorCode(result) == ZSTD_error_dstSize_tooSmall ? -2 : -1;
    };
    while (n >= 1) {
        const size_t want = ZSTD_nextSrcSizeToDecompress(stream.z);
        if (want == 0) {
            // Input after the frame's end.
            memory.Write8(guest + StState, 3);
            Add(stats.errors, 1);
            return -3;
        }
        const s32 w = static_cast<s32>(want);
        if (!stream.staged.empty()) {
            // A piece begun in an earlier chunk: filled up to its size.
            const s32 staged = static_cast<s32>(stream.staged.size());
            if (w > staged) {
                const s32 take = (std::min)(w - staged, n);
                stream.staged.insert(stream.staged.end(), src, src + take);
                src += take;
                n -= take;
            }
            if (static_cast<s32>(stream.staged.size()) < w) {
                continue;
            }
            const size_t result =
                StreamContinue(memory, stream, guest, stream.staged.data(), want, stats);
            if (ZSTD_isError(result)) {
                return fail(result);
            }
            stream.staged.erase(stream.staged.begin(), stream.staged.begin() + want);
            continue;
        }
        if (n < w) {
            // The piece goes on in the next chunk.
            if (static_cast<s32>(memory.Read32(guest + StStageCap)) < w) {
                memory.Write8(guest + StState, 3);
                Add(stats.errors, 1);
                return -1;
            }
            stream.staged.assign(src, src + n);
            break;
        }
        const size_t result = StreamContinue(memory, stream, guest, src, want, stats);
        if (ZSTD_isError(result)) {
            return fail(result);
        }
        src += want;
        n -= w;
    }
    if (ZSTD_nextSrcSizeToDecompress(stream.z) == 0) {
        memory.Write8(guest + StState, 4);
    }
    return static_cast<s32>(memory.Read32(guest + StLeft));
}

/// main+0xF5C80: the whole function for streams that started on the host; others (and every
/// state the game answers without its context) run as the game's code.
bool ZstdStream(Core::Memory::Memory& memory, Registers& regs, Stats& stats) {
    const u64 guest = regs.Get(0);
    const u64 src = regs.Get(1);
    const s32 n = static_cast<s32>(regs.Get(2));
    const u64 ddict = regs.Get(3);
    const u8 state = memory.Read8(guest + StState);
    if (state == 3 || state == 4) {
        DropStream(guest);
        return false; // (no decline count: the game's own answer, no decode)
    }
    if (n > MaxChunk) {
        DropStream(guest);
        NoteDecline(Routine::ZstdStream, "a chunk over 64 MiB");
        return false;
    }
    thread_local Scratch in_buffer;
    HostStream* stream = nullptr;
    const u8* in = nullptr;
    if (state == 1) {
        if (n <= 0) {
            DropStream(guest);
            NoteDecline(Routine::ZstdStream, "a first chunk with no bytes");
            return false;
        }
        if (memory.Read32(guest + StStaged) != 0) {
            // The game would begin this frame with those bytes: only its code does that.
            DropStream(guest);
            NoteDecline(Routine::ZstdStream, "staged bytes left from an earlier frame");
            return false;
        }
        u8* const bytes = in_buffer.Get(static_cast<size_t>(n));
        memory.ReadBlock(src, bytes, static_cast<size_t>(n));
        in = bytes;
        ZSTD_FrameHeader header{};
        const size_t header_result =
            ZSTD_getFrameHeader_advanced(&header, in, static_cast<size_t>(n), ZSTD_f_zstd1);
        if (header_result != 0) {
            // A bad or cut header: the game fails it the same way.
            DropStream(guest);
            return false;
        }
        const u64 content =
            header.frameType == ZSTD_skippableFrame ? 0 : header.frameContentSize;
        if (content == ZSTD_CONTENTSIZE_UNKNOWN || content == ZSTD_CONTENTSIZE_ERROR) {
            DropStream(guest);
            return false;
        }
        if ((memory.Read64(guest + StWork) & 7) != 0) {
            DropStream(guest);
            NoteDecline(Routine::ZstdStream, "the workspace is not 8-byte aligned");
            return false;
        }
        if (content > MaxStreamContent) {
            DropStream(guest);
            NoteDecline(Routine::ZstdStream, "content over 512 MiB");
            return false;
        }
        std::shared_ptr<HostDict> dict;
        if (ddict != 0 && (dict = DictFor(memory, ddict)) == nullptr) {
            DropStream(guest);
            NoteDecline(Routine::ZstdStream, "a dictionary libzstd does not take as the game's");
            return false;
        }
        const char* why = "";
        stream = StartStream(guest, content, std::move(dict), &why);
        if (stream == nullptr) {
            NoteDecline(Routine::ZstdStream, why);
            return false;
        }
        const size_t begin = ZSTD_decompressBegin_usingDDict(
            stream->z, stream->dict != nullptr ? stream->dict->dd : nullptr);
        if (ZSTD_isError(begin)) {
            DropStream(guest);
            NoteDecline(Routine::ZstdStream, "libzstd refused to begin");
            return false;
        }
        memory.Write32(guest + StLeft, static_cast<u32>(content));
        memory.Write64(guest + StCtx, memory.Read64(guest + StWork));
        memory.Write8(guest + StState, 2);
    } else {
        stream = FindStream(guest);
        if (stream == nullptr) {
            return false; // a stream that started as the game's code goes on there
        }
        if (n >= 1) {
            u8* const bytes = in_buffer.Get(static_cast<size_t>(n));
            memory.ReadBlock(src, bytes, static_cast<size_t>(n));
            in = bytes;
        }
    }
    const s32 result = StreamChunk(memory, *stream, guest, in, n, stats);
    const u8 after = memory.Read8(guest + StState);
    if (after == 3 || after == 4) {
        DropStream(guest);
    }
    in_buffer.Trim();
    regs.Set(0, static_cast<u32>(result));
    return true;
}

} // Anonymous namespace

Sites SitesFor(Core::System& system, const Kernel::KProcess& process) {
    if (ZstdMode() == 0 || !IsOot(system, process)) {
        return {};
    }
    const u64 main_base = GetInteger(process.GetEntryPoint()) + MainOffset;
    Sites sites{~u64{0}, 0};
    for (const Site& site : OotSites) {
        if (!RoutineOn(site.routine)) {
            continue;
        }
        sites.first = (std::min)(sites.first, main_base + site.main_offset);
        sites.last = (std::max)(sites.last, main_base + site.main_offset);
    }
    return sites.last != 0 ? sites : Sites{};
}

u32 InstructionFor(Core::System& system, Kernel::KProcess& process, Core::Memory::Memory& memory,
                   u64 vaddr, u32 instruction) {
    if (instruction != Pacibsp) {
        return instruction;
    }
    const u64 main_base = GetInteger(process.GetEntryPoint()) + MainOffset;
    for (size_t i = 0; i < OotSites.size(); ++i) {
        const Site& site = OotSites[i];
        if (vaddr != main_base + site.main_offset || !RoutineOn(site.routine)) {
            continue;
        }
        ProcessSites& verdicts = Verdicts();
        std::scoped_lock lock{verdicts.mutex};
        if (verdicts.process_id != process.GetProcessId() || verdicts.main_base != main_base) {
            verdicts.process_id = process.GetProcessId();
            verdicts.main_base = main_base;
            verdicts.state.fill(0);
        }
        int& state = verdicts.state[static_cast<size_t>(site.routine)];
        if (state == 0) {
            bool match = IsOot(system, process);
            for (size_t w = 0; match && w < site.words.size(); ++w) {
                match = memory.Read32(vaddr + w * 4) == site.words[w];
            }
            state = match ? 1 : 2;
            LOG_INFO(Core_ARM, "Host zstd: main+{:#x} ({}) {}", site.main_offset,
                     site.routine == Routine::ZstdFrames ? "whole-buffer decode" : "stream decode",
                     match ? "answered on the host" : "does not match this build's code: off");
        }
        return state == 1 ? BrkFor(site.routine) : instruction;
    }
    return instruction;
}

bool OnBreakpoint([[maybe_unused]] Core::System& system, Kernel::KProcess& process,
                  Core::Memory::Memory& memory, Registers& regs, u64 pc) {
    const u64 main_base = GetInteger(process.GetEntryPoint()) + MainOffset;
    for (const Site& site : OotSites) {
        if (pc != main_base + site.main_offset) {
            continue;
        }
        {
            // Only a BRK this file put there (the guest's own word is the pacibsp).
            ProcessSites& verdicts = Verdicts();
            std::scoped_lock lock{verdicts.mutex};
            if (verdicts.process_id != process.GetProcessId() ||
                verdicts.state[static_cast<size_t>(site.routine)] != 1) {
                return false;
            }
        }
        Stats& stats = g_stats[static_cast<size_t>(site.routine)];
        const s64 start = NowNs();
        const u64 return_address = regs.Get(30);
        bool answered = false;
        try {
            answered = site.routine == Routine::ZstdFrames ? ZstdFrames(memory, regs, stats)
                                                           : ZstdStream(memory, regs, stats);
        } catch (const std::bad_alloc&) {
            // Host memory ran out before the call wrote anything, except in a host stream
            // already under way: that one fails as a broken frame would (the game's code
            // cannot take it over).
            const u64 guest = regs.Get(0);
            const bool under_way = site.routine == Routine::ZstdStream &&
                                   memory.Read8(guest + StState) == 2 &&
                                   FindStream(guest) != nullptr;
            if (site.routine == Routine::ZstdStream) {
                DropStream(guest);
            }
            if (under_way) {
                memory.Write8(guest + StState, 3);
                regs.Set(0, static_cast<u32>(-1));
                answered = true;
            } else {
                NoteDecline(site.routine, "host memory ran out");
            }
        }
        if (answered) {
            regs.SetPC(return_address);
            Add(stats.calls, 1);
            NoteTime(stats, static_cast<u64>(NowNs() - start));
        } else {
            // The pacibsp the BRK stands for is a no-op here: the game's code goes on after it.
            regs.SetPC(pc + 4);
        }
        MaybeLog();
        return true;
    }
    return false;
}

} // namespace Core::HostRoutines
