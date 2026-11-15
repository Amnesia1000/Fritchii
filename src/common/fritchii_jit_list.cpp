// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// FRITCHII_JIT_BLOCK_LIST: see fritchii_jit_list.h.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_set>

#include <fmt/format.h>

#include "common/fritchii_jit_list.h"
#include "common/logging.h"
#include "common/thread.h"

namespace Common::FritchiiJitList {

namespace {

// An A64 location key: the pc in the low 56 bits, the FPCR bits and the single-step bit (57)
// above it.
constexpr u64 PcMask = (u64{1} << 56) - 1;
constexpr u64 SingleStepBit = u64{1} << 57;

constexpr std::array<char, 8> Magic{'F', 'R', 'J', 'I', 'T', 'B', 'L', '1'};
constexpr u32 Version = 1;
constexpr std::size_t HeaderSize = 64;
constexpr std::size_t ChunkHeaderSize = 16;
constexpr std::size_t MaxChunkEntries = 65536;
/// Blocks kept per title (8 bytes each). zortos measured ~646k for OoT.
constexpr std::size_t MaxEntries = 2'000'000;
/// Read no further than this: MaxEntries entries and their chunk headers fit.
constexpr std::uintmax_t MaxFileBytes = std::uintmax_t{64} << 20;
constexpr u32 MaxCores = 8;
constexpr auto WriteEvery = std::chrono::seconds{10};

struct State {
    std::mutex mutex;
    u64 base{};
    u64 size{};
    std::filesystem::path path;
    std::unordered_set<u64> known; ///< Entries listed or recorded
    std::vector<u64> list;         ///< Location keys to warm up, in list order
    std::array<std::size_t, MaxCores> cursor{};
    std::vector<u64> pending; ///< Entries to append to the file
    std::size_t loaded{};
    std::size_t recorded{};
    std::size_t written{};
    bool file_ok{};
    bool stopping{};
    std::condition_variable cv;
    std::thread writer;

    ~State() {
        // The process ends without End (it is called at shutdown): logging may be gone already,
        // so the writer is left to the exit rather than joined.
        if (writer.joinable()) {
            writer.detach();
        }
    }
};

State& GetState() {
    static State state;
    return state;
}

std::atomic<bool> g_recording{false};
std::atomic<bool> g_preloading{false};

template <typename T>
void Put(std::vector<u8>& out, std::size_t offset, T value) {
    std::memcpy(out.data() + offset, &value, sizeof(T));
}

template <typename T>
T Get(const std::vector<u8>& data, std::size_t offset) {
    T value;
    std::memcpy(&value, data.data() + offset, sizeof(T));
    return value;
}

u64 Checksum(const u8* data, std::size_t size) {
    u64 hash = 0xcbf29ce484222325ULL;
    for (std::size_t i = 0; i < size; ++i) {
        hash = (hash ^ data[i]) * 0x100000001b3ULL;
    }
    return hash;
}

std::vector<u8> MakeHeader(u64 program_id, std::span<const u8> build_id, u64 image_size) {
    std::vector<u8> header(HeaderSize, 0);
    std::memcpy(header.data(), Magic.data(), Magic.size());
    Put<u32>(header, 8, Version);
    Put<u64>(header, 16, program_id);
    Put<u64>(header, 24, image_size);
    std::memcpy(header.data() + 32, build_id.data(), (std::min)(build_id.size(), std::size_t{32}));
    return header;
}

std::vector<u8> MakeChunk(std::span<const u64> entries) {
    std::vector<u8> chunk(ChunkHeaderSize + entries.size() * sizeof(u64));
    std::memcpy(chunk.data() + ChunkHeaderSize, entries.data(), entries.size() * sizeof(u64));
    Put<u32>(chunk, 0, static_cast<u32>(entries.size()));
    Put<u64>(chunk, 8, Checksum(chunk.data() + ChunkHeaderSize, entries.size() * sizeof(u64)));
    return chunk;
}

/// Appends `entries` to the file as chunks. Called without the state's lock (the writer only).
bool AppendEntries(const std::filesystem::path& path, std::span<const u64> entries) {
    std::ofstream file{path, std::ios::binary | std::ios::app};
    if (!file) {
        return false;
    }
    for (std::size_t first = 0; first < entries.size(); first += MaxChunkEntries) {
        const std::size_t count = (std::min)(MaxChunkEntries, entries.size() - first);
        const std::vector<u8> chunk = MakeChunk(entries.subspan(first, count));
        file.write(reinterpret_cast<const char*>(chunk.data()),
                   static_cast<std::streamsize>(chunk.size()));
    }
    file.flush();
    return static_cast<bool>(file);
}

void WriterLoop() {
    Common::SetCurrentThreadName("FritchiiJitList");
    State& state = GetState();
    std::vector<u64> batch;
    bool stop = false;
    while (!stop) {
        std::filesystem::path path;
        std::size_t total = 0;
        std::array<std::size_t, MaxCores> cursor{};
        std::size_t list_size = 0;
        {
            std::unique_lock lock{state.mutex};
            state.cv.wait_for(lock, WriteEvery, [&state] { return state.stopping; });
            stop = state.stopping;
            if (!state.file_ok || state.pending.empty()) {
                continue;
            }
            batch.swap(state.pending);
            path = state.path;
            total = state.known.size();
            cursor = state.cursor;
            list_size = state.list.size();
        }
        const bool ok = AppendEntries(path, batch);
        {
            std::scoped_lock lock{state.mutex};
            if (ok) {
                state.written += batch.size();
            } else {
                state.file_ok = false;
            }
        }
        if (ok) {
            LOG_INFO(Core_ARM,
                     "JIT block list (FRITCHII_JIT_BLOCK_LIST={}): {} new blocks written, {} "
                     "listed; warm-up handed out {}/{}/{}/{} of {} loaded",
                     Mode(), batch.size(), total, cursor[0], cursor[1], cursor[2], cursor[3],
                     list_size);
        } else {
            LOG_WARNING(Core_ARM, "JIT block list: could not write {}, recording stops",
                        path.string());
        }
        batch.clear();
    }
}

std::string BuildIdText(std::span<const u8> build_id) {
    std::size_t length = (std::min)(build_id.size(), std::size_t{32});
    while (length > 8 && build_id[length - 1] == 0) {
        --length;
    }
    std::string text;
    for (std::size_t i = 0; i < length; ++i) {
        text += fmt::format("{:02X}", build_id[i]);
    }
    return text;
}

} // Anonymous namespace

int Mode() {
    static const int mode = [] {
        const char* const value = std::getenv("FRITCHII_JIT_BLOCK_LIST");
        return value != nullptr ? std::atoi(value) : 0;
    }();
    return mode;
}

void Begin(u64 program_id, std::span<const u8> build_id, u64 image_base, u64 image_size,
           const std::filesystem::path& cache_dir) {
    const int mode = Mode();
    if (mode == 0 || image_size == 0) {
        return;
    }
    End();
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    state.base = image_base;
    state.size = image_size;
    state.known.clear();
    state.list.clear();
    state.cursor = {};
    state.pending.clear();
    state.loaded = state.recorded = state.written = 0;
    state.stopping = false;
    state.file_ok = false;

    std::error_code error;
    const std::filesystem::path dir = cache_dir / "jit";
    std::filesystem::create_directories(dir, error);
    state.path = dir / fmt::format("{:016X}-{}.blocks", program_id, BuildIdText(build_id));
    const std::vector<u8> header = MakeHeader(program_id, build_id, image_size);

    // Read what earlier launches recorded.
    std::size_t valid_bytes = 0;
    std::vector<u8> data;
    const std::uintmax_t file_size = std::filesystem::file_size(state.path, error);
    if (!error && file_size >= HeaderSize && file_size <= MaxFileBytes) {
        std::ifstream file{state.path, std::ios::binary};
        data.resize(static_cast<std::size_t>(file_size));
        if (!file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()))) {
            data.clear();
        }
    }
    if (data.size() >= HeaderSize && std::memcmp(data.data(), header.data(), HeaderSize) == 0) {
        valid_bytes = HeaderSize;
        std::size_t offset = HeaderSize;
        while (offset + ChunkHeaderSize <= data.size() && state.known.size() < MaxEntries) {
            const u32 count = Get<u32>(data, offset);
            const u64 sum = Get<u64>(data, offset + 8);
            const std::size_t bytes = std::size_t{count} * sizeof(u64);
            if (count == 0 || count > MaxChunkEntries ||
                offset + ChunkHeaderSize + bytes > data.size() ||
                Checksum(data.data() + offset + ChunkHeaderSize, bytes) != sum) {
                break;
            }
            bool chunk_ok = true;
            for (u32 i = 0; i < count; ++i) {
                const u64 entry = Get<u64>(data, offset + ChunkHeaderSize + i * sizeof(u64));
                if ((entry & PcMask) >= image_size || (entry & SingleStepBit) != 0) {
                    chunk_ok = false;
                    break;
                }
            }
            if (!chunk_ok) {
                break;
            }
            for (u32 i = 0; i < count && state.known.size() < MaxEntries; ++i) {
                const u64 entry = Get<u64>(data, offset + ChunkHeaderSize + i * sizeof(u64));
                if (state.known.insert(entry).second) {
                    state.list.push_back(((image_base + (entry & PcMask)) & PcMask) |
                                         (entry & ~PcMask));
                }
            }
            offset += ChunkHeaderSize + bytes;
            valid_bytes = offset;
        }
    }
    state.loaded = state.list.size();
    if (valid_bytes == 0) {
        // No list for this image yet (or another build's): start one.
        std::ofstream file{state.path, std::ios::binary | std::ios::trunc};
        file.write(reinterpret_cast<const char*>(header.data()),
                   static_cast<std::streamsize>(header.size()));
        state.file_ok = static_cast<bool>(file);
    } else {
        if (valid_bytes < data.size()) {
            std::filesystem::resize_file(state.path, valid_bytes, error); // a torn tail
        }
        state.file_ok = !error;
    }
    if (mode != 1) {
        state.list.clear(); // record only
    }
    LOG_INFO(Core_ARM,
             "JIT block list (FRITCHII_JIT_BLOCK_LIST={}): {} blocks loaded from {} ({} bytes "
             "kept of {}), image {:#x} size {:#x}, {}",
             mode, state.loaded, state.path.string(), valid_bytes, data.size(), image_base,
             image_size,
             !state.file_ok ? "cannot write it, not recording"
                            : (mode == 1 ? "warming up from it and recording" : "recording only"));
    if (state.file_ok) {
        state.writer = std::thread{WriterLoop};
    }
    g_preloading.store(!state.list.empty(), std::memory_order_release);
    g_recording.store(state.file_ok, std::memory_order_release);
}

void End() {
    State& state = GetState();
    g_recording.store(false, std::memory_order_release);
    g_preloading.store(false, std::memory_order_release);
    std::thread writer;
    {
        std::scoped_lock lock{state.mutex};
        state.stopping = true;
        writer = std::move(state.writer);
    }
    state.cv.notify_all();
    if (writer.joinable()) {
        writer.join(); // it writes what is pending first
    }
    std::scoped_lock lock{state.mutex};
    state.known.clear();
    state.list.clear();
    state.list.shrink_to_fit();
    state.pending.clear();
}

void Note(u64 key) {
    if (!g_recording.load(std::memory_order_relaxed) || (key & SingleStepBit) != 0) {
        return;
    }
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    const u64 pc = key & PcMask;
    if (pc < state.base || pc - state.base >= state.size || state.known.size() >= MaxEntries) {
        return;
    }
    const u64 entry = (pc - state.base) | (key & ~PcMask);
    if (state.known.insert(entry).second) {
        state.pending.push_back(entry);
        ++state.recorded;
    }
}

std::size_t TakePreload(u32 core, std::vector<u64>& out, std::size_t max) {
    if (!g_preloading.load(std::memory_order_acquire) || core >= MaxCores) {
        return 0;
    }
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    std::size_t& cursor = state.cursor[core];
    const std::size_t count = (std::min)(max, state.list.size() - (std::min)(cursor, state.list.size()));
    out.insert(out.end(), state.list.begin() + static_cast<std::ptrdiff_t>(cursor),
               state.list.begin() + static_cast<std::ptrdiff_t>(cursor + count));
    cursor += count;
    return count;
}

bool Preloading() {
    return g_preloading.load(std::memory_order_acquire);
}

} // namespace Common::FritchiiJitList
