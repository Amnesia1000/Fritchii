// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "common/cityhash.h"
#include "common/fs/file.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/thread.h"
#include "common/zstd_compression.h"
#include "video_core/renderer_vulkan/vk_pipeline_binary.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan::PipelineBinary {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::array<char, 8> FileMagic{'F', 'R', 'P', 'B', 'I', 'N', '0', '1'};
constexpr std::array<char, 8> IndexMagic{'F', 'R', 'P', 'B', 'I', 'D', 'X', '1'};
constexpr u32 FormatVersion = 1;
constexpr u64 HeaderSize = 64;
constexpr u32 RecordMagic = 0x52504246; // "FBPR"
constexpr u64 RecordHeaderSize = 24;
constexpr u32 MaxPayload = 64U << 20;
/// More binaries for one pipeline are taken for a damaged entry
constexpr u32 MaxBinaries = 64;
/// Bytes queued for the writer before new pipelines are dropped (not saved this time)
constexpr u64 MaxQueuedBytes = 256ULL << 20;
constexpr u32 VerifyEvery = 16;
constexpr u8 FlagZstd = 1;
constexpr int ZstdLevel = 3;

constexpr std::string_view FileName = "pipeline_binaries.fritchii.bin";
constexpr std::string_view IndexName = "pipeline_binaries.fritchii.idx";
constexpr std::string_view LostName = "pipeline_binaries.fritchii.lost";

enum class Kind : u8 { Graphics = 0, Compute = 1 };

struct Identity {
    u32 vendor_id{};
    u32 device_id{};
    u32 driver_id{};
    u32 driver_version{};
    std::array<u8, VK_UUID_SIZE> uuid{};

    bool operator==(const Identity&) const = default;
};

struct Binary {
    std::vector<u8> key;
    std::vector<u8> data;
};

/// What one entry holds: the driver's key of the create info, and its binaries
struct Entry {
    std::vector<u8> pipeline_key;
    std::vector<Binary> binaries;
};

// Little helpers for the byte layout (the file is read on the PC that wrote it: native order)
template <typename T>
void Put(std::vector<u8>& out, const T& value) {
    const auto* bytes = reinterpret_cast<const u8*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
}

void PutBytes(std::vector<u8>& out, std::span<const u8> bytes) {
    out.insert(out.end(), bytes.begin(), bytes.end());
}

class Reader {
public:
    explicit Reader(std::span<const u8> bytes_) : bytes{bytes_} {}

    template <typename T>
    bool Get(T& value) {
        if (bytes.size() - pos < sizeof(T)) {
            return false;
        }
        std::memcpy(&value, bytes.data() + pos, sizeof(T));
        pos += sizeof(T);
        return true;
    }

    bool GetBytes(std::vector<u8>& out, size_t size) {
        if (bytes.size() - pos < size) {
            return false;
        }
        out.assign(bytes.begin() + pos, bytes.begin() + pos + size);
        pos += size;
        return true;
    }

    bool AtEnd() const {
        return pos == bytes.size();
    }

private:
    std::span<const u8> bytes;
    size_t pos{};
};

std::vector<u8> EncodeEntry(const Entry& entry) {
    std::vector<u8> out;
    size_t size = 8 + entry.pipeline_key.size();
    for (const Binary& binary : entry.binaries) {
        size += 12 + binary.key.size() + binary.data.size();
    }
    out.reserve(size);
    Put(out, static_cast<u32>(entry.pipeline_key.size()));
    PutBytes(out, entry.pipeline_key);
    Put(out, static_cast<u32>(entry.binaries.size()));
    for (const Binary& binary : entry.binaries) {
        Put(out, static_cast<u32>(binary.key.size()));
        PutBytes(out, binary.key);
        Put(out, static_cast<u64>(binary.data.size()));
        PutBytes(out, binary.data);
    }
    return out;
}

std::optional<Entry> DecodeEntry(std::span<const u8> body) {
    Reader reader{body};
    Entry entry;
    u32 key_size{};
    if (!reader.Get(key_size) || key_size == 0 || key_size > VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR ||
        !reader.GetBytes(entry.pipeline_key, key_size)) {
        return std::nullopt;
    }
    u32 count{};
    if (!reader.Get(count) || count == 0 || count > MaxBinaries) {
        return std::nullopt;
    }
    entry.binaries.resize(count);
    for (Binary& binary : entry.binaries) {
        u32 binary_key_size{};
        u64 data_size{};
        if (!reader.Get(binary_key_size) || binary_key_size == 0 ||
            binary_key_size > VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR ||
            !reader.GetBytes(binary.key, binary_key_size) || !reader.Get(data_size) ||
            data_size == 0 || data_size > MaxPayload || !reader.GetBytes(binary.data, data_size)) {
            return std::nullopt;
        }
    }
    if (!reader.AtEnd()) {
        return std::nullopt;
    }
    return entry;
}

std::vector<u8> EncodeHeader(const Identity& identity, u64 file_id) {
    std::vector<u8> out;
    out.reserve(HeaderSize);
    PutBytes(out, std::span{reinterpret_cast<const u8*>(FileMagic.data()), FileMagic.size()});
    Put(out, FormatVersion);
    Put(out, identity.vendor_id);
    Put(out, identity.device_id);
    Put(out, identity.driver_id);
    Put(out, identity.driver_version);
    PutBytes(out, identity.uuid);
    Put(out, file_id);
    Put(out, u32{0});
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(out.data()), out.size());
    Put(out, hash);
    return out;
}

enum class HeaderCheck { Ok, Damaged, OtherVersion, OtherIdentity };

HeaderCheck DecodeHeader(std::span<const u8> bytes, const Identity& identity, u64& file_id) {
    if (bytes.size() != HeaderSize) {
        return HeaderCheck::Damaged;
    }
    if (std::memcmp(bytes.data(), FileMagic.data(), FileMagic.size()) != 0) {
        return HeaderCheck::Damaged;
    }
    u64 hash{};
    std::memcpy(&hash, bytes.data() + HeaderSize - 8, 8);
    if (hash != Common::CityHash64(reinterpret_cast<const char*>(bytes.data()), HeaderSize - 8)) {
        return HeaderCheck::Damaged;
    }
    Reader reader{bytes.subspan(FileMagic.size())};
    u32 version{};
    Identity stored{};
    static_cast<void>(reader.Get(version));
    if (version != FormatVersion) {
        return HeaderCheck::OtherVersion;
    }
    static_cast<void>(reader.Get(stored.vendor_id));
    static_cast<void>(reader.Get(stored.device_id));
    static_cast<void>(reader.Get(stored.driver_id));
    static_cast<void>(reader.Get(stored.driver_version));
    static_cast<void>(reader.Get(stored.uuid));
    static_cast<void>(reader.Get(file_id));
    return stored == identity ? HeaderCheck::Ok : HeaderCheck::OtherIdentity;
}

u64 NewFileId() {
    std::random_device device;
    const u64 random = (u64{device()} << 32) ^ device();
    return random ^ static_cast<u64>(Clock::now().time_since_epoch().count());
}

u64 IndexKey(Kind kind, std::span<const u8> driver_key) {
    return Common::CityHash64(reinterpret_cast<const char*>(driver_key.data()), driver_key.size()) ^
           (u64{static_cast<u8>(kind) + 1U} * 0x9e3779b97f4a7c15ULL);
}

u64 MaxBytesSwitch() {
    if (const char* value = std::getenv("FRITCHII_PIPELINE_BINARY_MAX_MB"); value && *value) {
        const u64 mib = std::strtoull(value, nullptr, 10);
        if (mib != 0) {
            return mib << 20;
        }
    }
    return 6144ULL << 20;
}

bool HarvestSwitch() {
    const char* value = std::getenv("FRITCHII_PIPELINE_BINARY_HARVEST");
    return value == nullptr || *value != '0';
}

double Ms(u64 ns) {
    return static_cast<double>(ns) / 1e6;
}

/// The file and its index; any thread looks up, rejects and queues entries, the writer thread
/// alone appends to the file.
class Store {
public:
    struct Counts {
        u64 loaded{};
        u64 loaded_bytes{};
        u64 hits{};
        u64 hit_ns{};
        u64 harvested{};
        u64 misses{};
        u64 miss_ns{};
        u64 rejected{};
        u64 damaged{};
        u64 written{};
        u64 written_bytes{};
        u64 dropped{};
        u64 skipped{}; ///< Create infos the store leaves alone (libraries, key not given)
        u64 verified_same{};
        u64 verified_other{};
        u64 file_bytes{};
    };

    Store(std::filesystem::path dir, const Identity& identity_, int mode_, u64 max_bytes_)
        : file{dir / FileName}, index_file{dir / IndexName}, lost_file{dir / LostName},
          identity{identity_}, mode{mode_}, max_bytes{max_bytes_} {}

    ~Store() {
        Close();
    }

    Store(const Store&) = delete;
    Store& operator=(const Store&) = delete;

    /// False when the file can't be read or written: the store stays off.
    bool Open();

    bool Saves() const {
        return mode == 1 || mode == 3;
    }

    bool Verifies() const {
        return mode == 3;
    }

    u64 FileId() const {
        return file_id;
    }

    const std::filesystem::path& Path() const {
        return file;
    }

    /// The stored entry of this key, read from the file and checked. Empty when there is none, or
    /// it is damaged (then rejected).
    std::optional<Entry> Find(u64 key);

    void Hit(u64 ns) {
        hits.fetch_add(1, std::memory_order_relaxed);
        hit_ns.fetch_add(ns, std::memory_order_relaxed);
        used_any.store(true, std::memory_order_relaxed);
    }

    void Miss(u64 ns) {
        misses.fetch_add(1, std::memory_order_relaxed);
        miss_ns.fetch_add(ns, std::memory_order_relaxed);
    }

    void Harvested() {
        harvested.fetch_add(1, std::memory_order_relaxed);
    }

    void Skipped() {
        skipped.fetch_add(1, std::memory_order_relaxed);
    }

    void Verified(bool same) {
        (same ? verified_same : verified_other).fetch_add(1, std::memory_order_relaxed);
    }

    bool ShouldVerify() {
        return Verifies() &&
               verify_counter.fetch_add(1, std::memory_order_relaxed) % VerifyEvery == 0;
    }

    bool UsedAny() const {
        return used_any.load(std::memory_order_relaxed);
    }

    /// Not given again; a new build of it replaces it.
    void Reject(u64 key);

    /// Whether a new build of this key should be captured and saved
    bool Wants(u64 key);

    /// Queues an entry for the file (the writer compresses and appends it)
    void Save(u64 key, Kind kind, Entry entry);

    Counts GetCounts() const;

    void Close();

    const std::filesystem::path& LostFile() const {
        return lost_file;
    }

private:
    enum class State : u8 {
        Stored,   ///< In the file at `offset`
        Rejected, ///< In the file, not to be used
        Queued,   ///< Queued for the writer in this session
    };

    struct IndexEntry {
        u64 offset{};
        u32 size{}; ///< Payload bytes (after the record header)
        State state{};
    };

    struct Job {
        u64 key{};
        Kind kind{};
        Entry entry;
        u64 bytes{};
    };

    bool StartOver(std::string_view why);
    bool ReadIndex(u64 file_size, u64& covered);
    void Scan(u64 from, u64 file_size, u64& good_end);
    void WriterLoop();
    void WriteIndex();
    void Compact();

    const std::filesystem::path file;
    const std::filesystem::path index_file;
    const std::filesystem::path lost_file;
    const Identity identity;
    const int mode;
    const u64 max_bytes;
    u64 file_id{};

    mutable std::mutex mutex;
    std::unordered_map<u64, IndexEntry> index;
    Common::FS::IOFile reader;
    u64 file_bytes{};   ///< The file's size, with what is queued
    u64 dead_bytes{};   ///< Bytes compaction would drop (replaced and rejected entries)
    u64 queued_bytes{}; ///< Bytes waiting for the writer
    bool full_logged{};
    bool closed{};

    std::deque<Job> jobs;
    std::condition_variable jobs_cv;
    bool stop_writer{};
    std::thread writer;
    Common::FS::IOFile appender;
    u64 append_offset{};

    u64 loaded{};
    u64 loaded_bytes{};
    std::atomic<u64> hits{};
    std::atomic<u64> hit_ns{};
    std::atomic<u64> harvested{};
    std::atomic<u64> misses{};
    std::atomic<u64> miss_ns{};
    std::atomic<u64> rejected{};
    std::atomic<u64> damaged{};
    std::atomic<u64> written{};
    std::atomic<u64> written_bytes{};
    std::atomic<u64> dropped{};
    std::atomic<u64> skipped{};
    std::atomic<u64> verified_same{};
    std::atomic<u64> verified_other{};
    std::atomic<u32> verify_counter{};
    std::atomic<bool> used_any{};
};

bool Store::StartOver(std::string_view why) {
    if (!why.empty()) {
        LOG_INFO(Render_Vulkan, "Pipeline binaries: {}, starting the store over", why);
    }
    std::error_code ec;
    std::filesystem::remove(index_file, ec);
    file_id = NewFileId();
    const std::vector<u8> header = EncodeHeader(identity, file_id);
    Common::FS::IOFile out(file, Common::FS::FileAccessMode::Write,
                           Common::FS::FileType::BinaryFile, Common::FS::FileShareFlag::ShareNone);
    if (!out.IsOpen() || out.WriteSpan(std::span<const u8>{header}) != header.size() ||
        !out.Flush()) {
        LOG_WARNING(Render_Vulkan, "Pipeline binaries: can't write {}, store off",
                    Common::FS::PathToUTF8String(file));
        return false;
    }
    file_bytes = HeaderSize;
    return true;
}

bool Store::ReadIndex(u64 file_size, u64& covered) {
    std::ifstream in(index_file, std::ios::binary);
    if (!in) {
        return false;
    }
    const std::vector<u8> bytes{std::istreambuf_iterator<char>(in), {}};
    constexpr size_t Head = 8 + 8 + 8 + 8;
    constexpr size_t EntrySize = 8 + 8 + 4 + 4;
    if (bytes.size() < Head + 8 ||
        std::memcmp(bytes.data(), IndexMagic.data(), IndexMagic.size()) != 0) {
        return false;
    }
    u64 hash{};
    std::memcpy(&hash, bytes.data() + bytes.size() - 8, 8);
    if (hash != Common::CityHash64(reinterpret_cast<const char*>(bytes.data()), bytes.size() - 8)) {
        return false;
    }
    Reader reader{std::span{bytes}.subspan(8, bytes.size() - 16)};
    u64 id{}, count{};
    static_cast<void>(reader.Get(id));
    static_cast<void>(reader.Get(covered));
    static_cast<void>(reader.Get(count));
    if (id != file_id || covered < HeaderSize || covered > file_size ||
        count * EntrySize != bytes.size() - Head - 8) {
        return false;
    }
    index.reserve(count + count / 4);
    for (u64 i = 0; i < count; ++i) {
        u64 key{}, offset{};
        u32 size{}, reserved{};
        static_cast<void>(reader.Get(key));
        static_cast<void>(reader.Get(offset));
        static_cast<void>(reader.Get(size));
        static_cast<void>(reader.Get(reserved));
        if (offset < HeaderSize || offset + RecordHeaderSize + size > covered) {
            index.clear();
            return false;
        }
        index[key] = IndexEntry{.offset = offset, .size = size, .state = State::Stored};
    }
    return true;
}

void Store::Scan(u64 from, u64 file_size, u64& good_end) {
    Common::FS::IOFile in(file, Common::FS::FileAccessMode::Read,
                          Common::FS::FileType::BinaryFile,
                          Common::FS::FileShareFlag::ShareReadWrite);
    good_end = from;
    if (!in.IsOpen()) {
        return;
    }
    u64 offset = from;
    std::array<u8, RecordHeaderSize> head{};
    while (offset + RecordHeaderSize <= file_size) {
        if (!in.Seek(static_cast<s64>(offset)) ||
            in.ReadSpan(std::span<u8>{head}) != head.size()) {
            break;
        }
        u32 magic{}, size{};
        u64 key{};
        std::memcpy(&magic, head.data(), 4);
        std::memcpy(&size, head.data() + 4, 4);
        std::memcpy(&key, head.data() + 8, 8);
        if (magic != RecordMagic || size == 0 || size > MaxPayload ||
            offset + RecordHeaderSize + size > file_size) {
            break; // A record cut off by a crash (or damage): the file ends before it
        }
        if (const auto it = index.find(key); it != index.end()) {
            dead_bytes += RecordHeaderSize + it->second.size;
        }
        index[key] = IndexEntry{.offset = offset, .size = size, .state = State::Stored};
        offset += RecordHeaderSize + size;
    }
    good_end = offset;
}

bool Store::Open() {
    std::error_code ec;
    if (std::filesystem::exists(lost_file, ec)) {
        std::filesystem::remove(lost_file, ec);
        std::filesystem::remove(file, ec);
        std::filesystem::remove(index_file, ec);
        LOG_WARNING(Render_Vulkan, "Pipeline binaries: the last session that used them ended in a "
                                   "device loss, starting the store over");
    }
    u64 file_size = std::filesystem::file_size(file, ec);
    if (ec) {
        file_size = 0;
    }
    bool fresh = false;
    if (file_size < HeaderSize) {
        if (!Saves()) {
            LOG_INFO(Render_Vulkan, "Pipeline binaries: no store yet (load only: nothing saved)");
            return false;
        }
        if (!StartOver(file_size != 0 ? "the file is damaged" : "")) {
            return false;
        }
        fresh = true;
    } else {
        std::array<u8, HeaderSize> header{};
        {
            Common::FS::IOFile in(file, Common::FS::FileAccessMode::Read,
                                  Common::FS::FileType::BinaryFile,
                                  Common::FS::FileShareFlag::ShareReadWrite);
            if (!in.IsOpen() || in.ReadSpan(std::span<u8>{header}) != header.size()) {
                LOG_WARNING(Render_Vulkan, "Pipeline binaries: can't read {}, store off",
                            Common::FS::PathToUTF8String(file));
                return false;
            }
        }
        const HeaderCheck check = DecodeHeader(header, identity, file_id);
        if (check != HeaderCheck::Ok) {
            if (!Saves()) {
                LOG_INFO(Render_Vulkan, "Pipeline binaries: the store is for another device, "
                                        "driver or format (load only: left as it is)");
                return false;
            }
            const std::string_view why = check == HeaderCheck::OtherIdentity
                                             ? "the GPU or driver changed"
                                         : check == HeaderCheck::OtherVersion
                                             ? "the file is another format version"
                                             : "the file header is damaged";
            if (!StartOver(why)) {
                return false;
            }
            fresh = true;
        }
    }
    const auto begin = Clock::now();
    u64 good_end = HeaderSize;
    if (!fresh) {
        u64 covered = HeaderSize;
        if (!ReadIndex(file_size, covered)) {
            index.clear();
            covered = HeaderSize;
        }
        Scan(covered, file_size, good_end);
        if (good_end < file_size) {
            if (Saves()) {
                Common::FS::IOFile trim(file, Common::FS::FileAccessMode::ReadWrite,
                                        Common::FS::FileType::BinaryFile,
                                        Common::FS::FileShareFlag::ShareNone);
                if (!trim.IsOpen() || !trim.SetSize(good_end)) {
                    LOG_WARNING(Render_Vulkan, "Pipeline binaries: can't cut the damaged end of "
                                               "{}, store off",
                                Common::FS::PathToUTF8String(file));
                    return false;
                }
            }
            LOG_INFO(Render_Vulkan, "Pipeline binaries: {} bytes at the end of the file were cut "
                                    "off (a session that didn't close), dropped",
                     file_size - good_end);
        }
        file_bytes = good_end;
    }
    loaded = index.size();
    loaded_bytes = file_bytes;
    reader.Open(file, Common::FS::FileAccessMode::Read, Common::FS::FileType::BinaryFile,
                Common::FS::FileShareFlag::ShareReadWrite);
    if (!reader.IsOpen()) {
        LOG_WARNING(Render_Vulkan, "Pipeline binaries: can't read {}, store off",
                    Common::FS::PathToUTF8String(file));
        return false;
    }
    if (Saves()) {
        appender.Open(file, Common::FS::FileAccessMode::Append, Common::FS::FileType::BinaryFile,
                      Common::FS::FileShareFlag::ShareReadWrite);
        if (!appender.IsOpen()) {
            LOG_WARNING(Render_Vulkan, "Pipeline binaries: can't write {}, store off",
                        Common::FS::PathToUTF8String(file));
            reader.Close();
            return false;
        }
        append_offset = file_bytes;
        writer = std::thread([this] { WriterLoop(); });
    }
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    LOG_INFO(Render_Vulkan,
             "Pipeline binaries (FRITCHII_PIPELINE_BINARY={}, {}): {} pipelines stored for this "
             "GPU and driver, {} MiB, index read in {:.1f} ms; {} (up to {} MiB)",
             mode, mode == 2 ? "load only" : mode == 3 ? "load, save and verify" : "load and save",
             loaded, loaded_bytes >> 20, ms, Common::FS::PathToUTF8String(file), max_bytes >> 20);
    return true;
}

std::optional<Entry> Store::Find(u64 key) {
    std::vector<u8> record;
    {
        std::scoped_lock lock{mutex};
        if (closed) {
            return std::nullopt;
        }
        const auto it = index.find(key);
        if (it == index.end() || it->second.state != State::Stored) {
            return std::nullopt;
        }
        record.resize(RecordHeaderSize + it->second.size);
        if (!reader.Seek(static_cast<s64>(it->second.offset)) ||
            reader.ReadSpan(std::span<u8>{record}) != record.size()) {
            record.clear();
        }
    }
    const auto damaged_entry = [&] {
        damaged.fetch_add(1, std::memory_order_relaxed);
        Reject(key);
        return std::nullopt;
    };
    if (record.empty()) {
        return damaged_entry();
    }
    u32 magic{}, size{};
    u64 stored_key{}, hash{};
    std::memcpy(&magic, record.data(), 4);
    std::memcpy(&size, record.data() + 4, 4);
    std::memcpy(&stored_key, record.data() + 8, 8);
    std::memcpy(&hash, record.data() + 16, 8);
    const std::span<const u8> payload = std::span<const u8>{record}.subspan(RecordHeaderSize);
    if (magic != RecordMagic || stored_key != key || size != payload.size() || payload.size() < 8 ||
        hash != Common::CityHash64(reinterpret_cast<const char*>(payload.data()), payload.size())) {
        return damaged_entry();
    }
    const u8 flags = payload[0];
    u32 body_size{};
    std::memcpy(&body_size, payload.data() + 4, 4);
    const std::span<const u8> stored_body = payload.subspan(8);
    std::optional<Entry> entry;
    if ((flags & FlagZstd) != 0) {
        const auto body = Common::Compression::TryDecompressDataZSTD(stored_body, body_size);
        if (body && body->size() == body_size) {
            entry = DecodeEntry(*body);
        }
    } else if (stored_body.size() == body_size) {
        entry = DecodeEntry(stored_body);
    }
    if (!entry) {
        return damaged_entry();
    }
    return entry;
}

void Store::Reject(u64 key) {
    rejected.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{mutex};
    const auto it = index.find(key);
    if (it != index.end() && it->second.state == State::Stored) {
        it->second.state = State::Rejected;
        dead_bytes += RecordHeaderSize + it->second.size;
    }
}

bool Store::Wants(u64 key) {
    if (!Saves()) {
        return false;
    }
    std::scoped_lock lock{mutex};
    if (closed) {
        return false;
    }
    const auto it = index.find(key);
    if (it != index.end() && it->second.state != State::Rejected) {
        return false; // Stored (another thread built it meanwhile) or queued
    }
    if (file_bytes + queued_bytes >= max_bytes) {
        if (!full_logged) {
            full_logged = true;
            LOG_WARNING(Render_Vulkan,
                        "Pipeline binaries: the store reached {} MiB, nothing more is saved "
                        "(FRITCHII_PIPELINE_BINARY_MAX_MB)",
                        max_bytes >> 20);
        }
        return false;
    }
    return true;
}

void Store::Save(u64 key, Kind kind, Entry entry) {
    u64 bytes = entry.pipeline_key.size();
    for (const Binary& binary : entry.binaries) {
        bytes += binary.key.size() + binary.data.size();
    }
    {
        std::scoped_lock lock{mutex};
        if (closed || queued_bytes + bytes > MaxQueuedBytes) {
            dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        const auto it = index.find(key);
        if (it != index.end() && it->second.state != State::Rejected) {
            return;
        }
        // The replaced entry's bytes were counted dead when it was rejected
        index[key] = IndexEntry{.state = State::Queued};
        queued_bytes += bytes;
        jobs.push_back(Job{.key = key, .kind = kind, .entry = std::move(entry), .bytes = bytes});
    }
    jobs_cv.notify_one();
}

void Store::WriterLoop() {
    Common::SetCurrentThreadName("FritchiiPipeBin");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
    for (;;) {
        Job job;
        {
            std::unique_lock lock{mutex};
            jobs_cv.wait(lock, [this] { return stop_writer || !jobs.empty(); });
            if (jobs.empty()) {
                return; // Stopping, with everything written
            }
            job = std::move(jobs.front());
            jobs.pop_front();
        }
        const std::vector<u8> body = EncodeEntry(job.entry);
        std::vector<u8> compressed =
            Common::Compression::CompressDataZSTD(body.data(), body.size(), ZstdLevel);
        const bool use_zstd = !compressed.empty() && compressed.size() < body.size();
        const std::span<const u8> stored = use_zstd ? std::span<const u8>{compressed}
                                                    : std::span<const u8>{body};
        std::vector<u8> record;
        record.reserve(RecordHeaderSize + 8 + stored.size());
        const u32 payload_size = static_cast<u32>(8 + stored.size());
        Put(record, RecordMagic);
        Put(record, payload_size);
        Put(record, job.key);
        Put(record, u64{0}); // The hash, below
        Put(record, static_cast<u8>(use_zstd ? FlagZstd : 0));
        Put(record, static_cast<u8>(job.kind));
        Put(record, u16{0});
        Put(record, static_cast<u32>(body.size()));
        PutBytes(record, stored);
        const u64 hash = Common::CityHash64(
            reinterpret_cast<const char*>(record.data() + RecordHeaderSize), payload_size);
        std::memcpy(record.data() + 16, &hash, 8);
        const bool ok = payload_size <= MaxPayload &&
                        appender.WriteSpan(std::span<const u8>{record}) == record.size() &&
                        appender.Flush();
        std::scoped_lock lock{mutex};
        queued_bytes -= job.bytes;
        const auto it = index.find(job.key);
        if (!ok) {
            // The file may hold part of the record: nothing more is appended this session
            // (the next boot cuts the partial record off).
            if (it != index.end() && it->second.state == State::Queued) {
                index.erase(it);
            }
            LOG_WARNING(Render_Vulkan, "Pipeline binaries: writing {} failed, nothing more is "
                                       "saved this session",
                        Common::FS::PathToUTF8String(file));
            stop_writer = true;
            closed = true;
            jobs.clear();
            queued_bytes = 0;
            return;
        }
        if (it != index.end() && it->second.state == State::Queued) {
            it->second = IndexEntry{.offset = append_offset, .size = payload_size,
                                    .state = State::Stored};
        } else {
            dead_bytes += record.size();
        }
        append_offset += record.size();
        file_bytes = append_offset;
        written.fetch_add(1, std::memory_order_relaxed);
        written_bytes.fetch_add(record.size(), std::memory_order_relaxed);
    }
}

void Store::WriteIndex() {
    std::vector<u8> out;
    std::vector<std::pair<u64, IndexEntry>> live;
    live.reserve(index.size());
    for (const auto& [key, entry] : index) {
        if (entry.state == State::Stored) {
            live.emplace_back(key, entry);
        }
    }
    out.reserve(32 + live.size() * 24 + 8);
    PutBytes(out, std::span{reinterpret_cast<const u8*>(IndexMagic.data()), IndexMagic.size()});
    Put(out, file_id);
    Put(out, file_bytes);
    Put(out, static_cast<u64>(live.size()));
    for (const auto& [key, entry] : live) {
        Put(out, key);
        Put(out, entry.offset);
        Put(out, entry.size);
        Put(out, u32{0});
    }
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(out.data()), out.size());
    Put(out, hash);
    auto part = index_file;
    part += ".part";
    {
        std::ofstream stream(part, std::ios::binary | std::ios::trunc);
        stream.write(reinterpret_cast<const char*>(out.data()),
                     static_cast<std::streamsize>(out.size()));
        if (!stream) {
            return;
        }
    }
    std::error_code ec;
    std::filesystem::rename(part, index_file, ec);
}

void Store::Compact() {
    // Live entries in file order, copied to a new file with the same id (the preparation stamp
    // stays valid: the same pipelines are in it).
    std::vector<std::pair<u64, IndexEntry*>> live;
    for (auto& [key, entry] : index) {
        if (entry.state == State::Stored) {
            live.emplace_back(key, &entry);
        }
    }
    std::ranges::sort(live, [](const auto& a, const auto& b) {
        return a.second->offset < b.second->offset;
    });
    auto part = file;
    part += ".part";
    const auto begin = Clock::now();
    u64 offset = HeaderSize;
    std::vector<std::pair<IndexEntry*, u64>> moved;
    moved.reserve(live.size());
    {
        Common::FS::IOFile out(part, Common::FS::FileAccessMode::Write,
                               Common::FS::FileType::BinaryFile,
                               Common::FS::FileShareFlag::ShareNone);
        const std::vector<u8> header = EncodeHeader(identity, file_id);
        if (!out.IsOpen() || out.WriteSpan(std::span<const u8>{header}) != header.size()) {
            return;
        }
        std::vector<u8> record;
        for (const auto& [key, entry] : live) {
            record.resize(RecordHeaderSize + entry->size);
            if (!reader.Seek(static_cast<s64>(entry->offset)) ||
                reader.ReadSpan(std::span<u8>{record}) != record.size() ||
                out.WriteSpan(std::span<const u8>{record}) != record.size()) {
                return;
            }
            moved.emplace_back(entry, offset);
            offset += record.size();
        }
        if (!out.Flush()) {
            return;
        }
    }
    reader.Close();
    std::error_code ec;
    std::filesystem::rename(part, file, ec);
    if (ec) {
        std::filesystem::remove(part, ec);
        return;
    }
    for (const auto& [entry, new_offset] : moved) {
        entry->offset = new_offset;
    }
    for (auto it = index.begin(); it != index.end();) {
        it = it->second.state == State::Stored ? std::next(it) : index.erase(it);
    }
    LOG_INFO(Render_Vulkan, "Pipeline binaries: store compacted from {} to {} MiB in {:.1f} s",
             file_bytes >> 20, offset >> 20,
             std::chrono::duration<double>(Clock::now() - begin).count());
    file_bytes = offset;
    dead_bytes = 0;
}

void Store::Close() {
    {
        std::scoped_lock lock{mutex};
        if (closed && !writer.joinable() && !reader.IsOpen()) {
            return;
        }
        stop_writer = true;
    }
    jobs_cv.notify_all();
    if (writer.joinable()) {
        writer.join();
    }
    std::scoped_lock lock{mutex};
    closed = true;
    appender.Close();
    if (!reader.IsOpen()) {
        return;
    }
    if (Saves()) {
        if (dead_bytes > (64ULL << 20) && dead_bytes > file_bytes / 4) {
            Compact();
        }
        WriteIndex();
    }
    reader.Close();
}

Store::Counts Store::GetCounts() const {
    Counts counts{
        .loaded = loaded,
        .loaded_bytes = loaded_bytes,
        .hits = hits.load(std::memory_order_relaxed),
        .hit_ns = hit_ns.load(std::memory_order_relaxed),
        .harvested = harvested.load(std::memory_order_relaxed),
        .misses = misses.load(std::memory_order_relaxed),
        .miss_ns = miss_ns.load(std::memory_order_relaxed),
        .rejected = rejected.load(std::memory_order_relaxed),
        .damaged = damaged.load(std::memory_order_relaxed),
        .written = written.load(std::memory_order_relaxed),
        .written_bytes = written_bytes.load(std::memory_order_relaxed),
        .dropped = dropped.load(std::memory_order_relaxed),
        .skipped = skipped.load(std::memory_order_relaxed),
        .verified_same = verified_same.load(std::memory_order_relaxed),
        .verified_other = verified_other.load(std::memory_order_relaxed),
    };
    std::scoped_lock lock{mutex};
    counts.file_bytes = file_bytes;
    return counts;
}

// The open store, shared with the pipeline builds that use it
std::mutex g_store_mutex;
std::shared_ptr<Store> g_store;

std::shared_ptr<Store> CurrentStore() {
    std::scoped_lock lock{g_store_mutex};
    return g_store;
}

// Vulkan side

const VkBaseInStructure* FindNext(const void* next, VkStructureType type) {
    for (auto* item = static_cast<const VkBaseInStructure*>(next); item; item = item->pNext) {
        if (item->sType == type) {
            return item;
        }
    }
    return nullptr;
}

/// Libraries, and create infos that chain binaries, libraries or flags of their own, are left
/// to the normal path.
template <typename CreateInfo>
bool TakesBinaries(const CreateInfo& ci) {
    return (ci.flags & VK_PIPELINE_CREATE_LIBRARY_BIT_KHR) == 0 &&
           !FindNext(ci.pNext, VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR) &&
           !FindNext(ci.pNext, VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR) &&
           !FindNext(ci.pNext, VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR) &&
           !FindNext(ci.pNext, VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT);
}

VkResult CreateRaw(const vk::DeviceDispatch& dld, VkDevice device,
                   const VkGraphicsPipelineCreateInfo& ci, VkPipeline* pipeline) {
    return dld.vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, pipeline);
}

VkResult CreateRaw(const vk::DeviceDispatch& dld, VkDevice device,
                   const VkComputePipelineCreateInfo& ci, VkPipeline* pipeline) {
    return dld.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, pipeline);
}

vk::Pipeline CreateNormally(const vk::Device& logical, const VkGraphicsPipelineCreateInfo& ci,
                            VkPipelineCache cache) {
    return logical.CreateGraphicsPipeline(ci, cache);
}

vk::Pipeline CreateNormally(const vk::Device& logical, const VkComputePipelineCreateInfo& ci,
                            VkPipelineCache cache) {
    return logical.CreateComputePipeline(ci, cache);
}

std::optional<std::vector<u8>> KeyBytes(const VkPipelineBinaryKeyKHR& key) {
    if (key.keySize == 0 || key.keySize > VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR) {
        return std::nullopt;
    }
    return std::vector<u8>(key.key, key.key + key.keySize);
}

void DestroyBinaries(const vk::DeviceDispatch& dld, VkDevice device,
                     std::span<const VkPipelineBinaryKHR> binaries) {
    for (const VkPipelineBinaryKHR binary : binaries) {
        if (binary != VK_NULL_HANDLE) {
            dld.vkDestroyPipelineBinaryKHR(device, binary, nullptr);
        }
    }
}

/// The binaries a create info names: from a pipeline built with capture, or from the driver's
/// own cache (`key_info`). Empty when the driver gives none.
std::vector<VkPipelineBinaryKHR> MakeBinaries(const vk::DeviceDispatch& dld, VkDevice device,
                                              VkPipeline pipeline,
                                              const VkPipelineCreateInfoKHR* key_info) {
    const VkPipelineBinaryCreateInfoKHR binary_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR,
        .pNext = nullptr,
        .pKeysAndDataInfo = nullptr,
        .pipeline = pipeline,
        .pPipelineCreateInfo = key_info,
    };
    VkPipelineBinaryHandlesInfoKHR handles{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR,
        .pNext = nullptr,
        .pipelineBinaryCount = 0,
        .pPipelineBinaries = nullptr,
    };
    if (dld.vkCreatePipelineBinariesKHR(device, &binary_ci, nullptr, &handles) != VK_SUCCESS ||
        handles.pipelineBinaryCount == 0 || handles.pipelineBinaryCount > MaxBinaries) {
        return {};
    }
    std::vector<VkPipelineBinaryKHR> binaries(handles.pipelineBinaryCount, VK_NULL_HANDLE);
    handles.pPipelineBinaries = binaries.data();
    if (dld.vkCreatePipelineBinariesKHR(device, &binary_ci, nullptr, &handles) != VK_SUCCESS) {
        DestroyBinaries(dld, device, binaries);
        return {};
    }
    binaries.resize(handles.pipelineBinaryCount);
    return binaries;
}

/// The keys and data of binaries
std::optional<std::vector<Binary>> ReadBinaries(const vk::DeviceDispatch& dld, VkDevice device,
                                                std::span<const VkPipelineBinaryKHR> binaries) {
    std::vector<Binary> out;
    out.reserve(binaries.size());
    for (const VkPipelineBinaryKHR binary : binaries) {
        const VkPipelineBinaryDataInfoKHR data_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR,
            .pNext = nullptr,
            .pipelineBinary = binary,
        };
        VkPipelineBinaryKeyKHR binary_key{.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
        size_t size = 0;
        std::optional<std::vector<u8>> key_bytes;
        if (dld.vkGetPipelineBinaryDataKHR(device, &data_info, &binary_key, &size, nullptr) !=
                VK_SUCCESS ||
            size == 0 || size > MaxPayload || !(key_bytes = KeyBytes(binary_key))) {
            return std::nullopt;
        }
        std::vector<u8> data(size);
        if (dld.vkGetPipelineBinaryDataKHR(device, &data_info, &binary_key, &size, data.data()) !=
            VK_SUCCESS) {
            return std::nullopt;
        }
        data.resize(size);
        out.push_back({.key = std::move(*key_bytes), .data = std::move(data)});
    }
    return out;
}

/// The pipeline built from binary handles, or null when the driver refuses them
template <typename CreateInfo>
VkPipeline CreateWithBinaries(const vk::DeviceDispatch& dld, VkDevice device, const CreateInfo& ci,
                              std::span<const VkPipelineBinaryKHR> binaries) {
    const VkPipelineBinaryInfoKHR binary_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR,
        .pNext = ci.pNext,
        .binaryCount = static_cast<u32>(binaries.size()),
        .pPipelineBinaries = binaries.data(),
    };
    CreateInfo from_binaries = ci;
    from_binaries.pNext = &binary_info;
    VkPipeline pipeline = VK_NULL_HANDLE;
    // A pipeline built from binaries takes no pipeline cache (VUID-...-pNext-09616).
    if (CreateRaw(dld, device, from_binaries, &pipeline) != VK_SUCCESS) {
        if (pipeline != VK_NULL_HANDLE) {
            dld.vkDestroyPipeline(device, pipeline, nullptr);
        }
        return VK_NULL_HANDLE;
    }
    return pipeline;
}

/// The pipeline built from a stored entry's binaries, or null when the driver refuses them
template <typename CreateInfo>
VkPipeline CreateFromEntry(const vk::DeviceDispatch& dld, VkDevice device, const Entry& stored,
                           const CreateInfo& ci) {
    const u32 count = static_cast<u32>(stored.binaries.size());
    std::vector<VkPipelineBinaryKeyKHR> keys(count);
    std::vector<VkPipelineBinaryDataKHR> data(count);
    for (u32 i = 0; i < count; ++i) {
        const Binary& binary = stored.binaries[i];
        keys[i] = VkPipelineBinaryKeyKHR{.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR,
                                         .pNext = nullptr,
                                         .keySize = static_cast<u32>(binary.key.size()),
                                         .key = {}};
        std::copy(binary.key.begin(), binary.key.end(), keys[i].key);
        data[i] = VkPipelineBinaryDataKHR{.dataSize = binary.data.size(),
                                          .pData = const_cast<u8*>(binary.data.data())};
    }
    const VkPipelineBinaryKeysAndDataKHR keys_and_data{
        .binaryCount = count,
        .pPipelineBinaryKeys = keys.data(),
        .pPipelineBinaryData = data.data(),
    };
    const VkPipelineBinaryCreateInfoKHR binary_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR,
        .pNext = nullptr,
        .pKeysAndDataInfo = &keys_and_data,
        .pipeline = VK_NULL_HANDLE,
        .pPipelineCreateInfo = nullptr,
    };
    std::vector<VkPipelineBinaryKHR> binaries(count, VK_NULL_HANDLE);
    VkPipelineBinaryHandlesInfoKHR handles{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR,
        .pNext = nullptr,
        .pipelineBinaryCount = count,
        .pPipelineBinaries = binaries.data(),
    };
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (dld.vkCreatePipelineBinariesKHR(device, &binary_ci, nullptr, &handles) == VK_SUCCESS &&
        handles.pipelineBinaryCount == count) {
        pipeline = CreateWithBinaries(dld, device, ci, binaries);
    }
    DestroyBinaries(dld, device, binaries);
    return pipeline;
}

/// A pipeline compiled with VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR, and its binaries when the
/// driver gives them. The pipeline is null when the build failed.
template <typename CreateInfo>
std::pair<VkPipeline, std::optional<std::vector<Binary>>> CompileAndCapture(
    const vk::DeviceDispatch& dld, VkDevice device, const CreateInfo& ci) {
    const VkPipelineCreateFlags2CreateInfoKHR capture_flags{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR,
        .pNext = ci.pNext,
        .flags = static_cast<VkPipelineCreateFlags2KHR>(ci.flags) |
                 VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR,
    };
    CreateInfo capture_ci = ci;
    capture_ci.pNext = &capture_flags;
    VkPipeline pipeline = VK_NULL_HANDLE;
    // A capturing build takes no pipeline cache either (VUID-...-pNext-09617).
    if (CreateRaw(dld, device, capture_ci, &pipeline) != VK_SUCCESS) {
        if (pipeline != VK_NULL_HANDLE) {
            dld.vkDestroyPipeline(device, pipeline, nullptr);
        }
        return {VK_NULL_HANDLE, std::nullopt};
    }
    std::optional<std::vector<Binary>> captured;
    const std::vector<VkPipelineBinaryKHR> binaries = MakeBinaries(dld, device, pipeline, nullptr);
    if (!binaries.empty()) {
        captured = ReadBinaries(dld, device, binaries);
        DestroyBinaries(dld, device, binaries);
    }
    const VkReleaseCapturedPipelineDataInfoKHR release{
        .sType = VK_STRUCTURE_TYPE_RELEASE_CAPTURED_PIPELINE_DATA_INFO_KHR,
        .pNext = nullptr,
        .pipeline = pipeline,
    };
    dld.vkReleaseCapturedPipelineDataKHR(device, &release, nullptr);
    return {pipeline, std::move(captured)};
}

bool SameKeys(const std::vector<Binary>& a, const std::vector<Binary>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    std::vector<std::vector<u8>> keys_a, keys_b;
    for (const Binary& binary : a) {
        keys_a.push_back(binary.key);
    }
    for (const Binary& binary : b) {
        keys_b.push_back(binary.key);
    }
    std::ranges::sort(keys_a);
    std::ranges::sort(keys_b);
    return keys_a == keys_b;
}

u64 NsSince(Clock::time_point begin) {
    return static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count());
}

template <typename CreateInfo>
vk::Pipeline Create(const Device& device, const CreateInfo& ci, VkPipelineCache cache, Kind kind) {
    const vk::Device& logical = device.GetLogical();
    const std::shared_ptr<Store> store = CurrentStore();
    if (!store) {
        return CreateNormally(logical, ci, cache);
    }
    if (!TakesBinaries(ci)) {
        store->Skipped();
        return CreateNormally(logical, ci, cache);
    }
    const vk::DeviceDispatch& dld = device.GetDispatchLoader();
    const VkDevice handle = *logical;
    const VkPipelineCreateInfoKHR key_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_INFO_KHR,
        .pNext = const_cast<CreateInfo*>(&ci),
    };
    VkPipelineBinaryKeyKHR pipeline_key{.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
    std::optional<std::vector<u8>> driver_key;
    if (dld.vkGetPipelineKeyKHR(handle, &key_info, &pipeline_key) != VK_SUCCESS ||
        !(driver_key = KeyBytes(pipeline_key))) {
        store->Skipped();
        return CreateNormally(logical, ci, cache);
    }
    const u64 key = IndexKey(kind, *driver_key);
    const auto begin = Clock::now();
    if (const std::optional<Entry> stored = store->Find(key)) {
        if (stored->pipeline_key == *driver_key) {
            if (const VkPipeline pipeline = CreateFromEntry(dld, handle, *stored, ci)) {
                store->Hit(NsSince(begin));
                if (store->ShouldVerify()) {
                    // Compiled the normal way too: the driver's binaries of it must be the ones
                    // stored (the same keys).
                    auto [check, captured] = CompileAndCapture(dld, handle, ci);
                    if (check != VK_NULL_HANDLE) {
                        const bool same = captured && SameKeys(*captured, stored->binaries);
                        store->Verified(same);
                        if (!same) {
                            LOG_WARNING(Render_Vulkan,
                                        "Pipeline binaries verify: entry {:016x} differs from a "
                                        "normal compile ({} stored binaries, {} compiled)",
                                        key, stored->binaries.size(),
                                        captured ? captured->size() : 0);
                        }
                        dld.vkDestroyPipeline(handle, check, nullptr);
                    }
                }
                return vk::Pipeline(pipeline, handle, dld);
            }
        }
        // Another key under the same hash, or refused by the driver: built below, and replaced
        store->Reject(key);
    }
    if (!store->Wants(key)) {
        vk::Pipeline pipeline = CreateNormally(logical, ci, cache);
        store->Miss(NsSince(begin));
        return pipeline;
    }
    if (device.PipelineBinaryProperties().pipelineBinaryInternalCache && HarvestSwitch()) {
        // The driver's own cache may still hold it: taken from there, no compile.
        const std::vector<VkPipelineBinaryKHR> binaries =
            MakeBinaries(dld, handle, VK_NULL_HANDLE, &key_info);
        if (!binaries.empty()) {
            const VkPipeline pipeline = CreateWithBinaries(dld, handle, ci, binaries);
            std::optional<std::vector<Binary>> read;
            if (pipeline != VK_NULL_HANDLE) {
                read = ReadBinaries(dld, handle, binaries);
            }
            DestroyBinaries(dld, handle, binaries);
            if (pipeline != VK_NULL_HANDLE) {
                if (read) {
                    store->Save(key, kind,
                                Entry{.pipeline_key = std::move(*driver_key),
                                      .binaries = std::move(*read)});
                }
                store->Harvested();
                store->Miss(NsSince(begin));
                return vk::Pipeline(pipeline, handle, dld);
            }
        }
    }
    auto [pipeline, captured] = CompileAndCapture(dld, handle, ci);
    if (pipeline == VK_NULL_HANDLE) {
        // Built as without the store, which reports its own errors
        vk::Pipeline normal = CreateNormally(logical, ci, cache);
        store->Miss(NsSince(begin));
        return normal;
    }
    store->Miss(NsSince(begin));
    if (captured) {
        store->Save(key, kind,
                    Entry{.pipeline_key = std::move(*driver_key),
                          .binaries = std::move(*captured)});
    }
    return vk::Pipeline(pipeline, handle, dld);
}

void LogStoreCounts(const Store& store, std::string_view when) {
    const Store::Counts c = store.GetCounts();
    const double hit_ms = c.hits != 0 ? Ms(c.hit_ns) / static_cast<double>(c.hits) : 0.0;
    const double miss_ms = c.misses != 0 ? Ms(c.miss_ns) / static_cast<double>(c.misses) : 0.0;
    const double saved_s =
        c.misses != 0 && miss_ms > hit_ms ? (miss_ms - hit_ms) * static_cast<double>(c.hits) / 1e3
                                          : 0.0;
    LOG_INFO(Render_Vulkan,
             "Pipeline binaries {}: {} hits ({:.2f} ms each), {} misses ({:.1f} ms each, {} "
             "taken from the driver's cache), ~{:.0f} s of compiles saved; {} rejected ({} "
             "damaged), {} saved ({} MiB), {} not saved (queue full), {} left to the normal path; "
             "store {} MiB ({} pipelines at boot){}",
             when, c.hits, hit_ms, c.misses, miss_ms, c.harvested, saved_s, c.rejected, c.damaged,
             c.written, c.written_bytes >> 20, c.dropped, c.skipped, c.file_bytes >> 20, c.loaded,
             c.verified_same + c.verified_other != 0
                 ? fmt::format("; verify: {} same, {} different", c.verified_same,
                               c.verified_other)
                 : std::string{});
}

} // Anonymous namespace

int SwitchValue() {
    static const int value = [] {
        const char* text = std::getenv("FRITCHII_PIPELINE_BINARY");
        if (text == nullptr || text[0] < '0' || text[0] > '3' || text[1] != '\0') {
            return 0;
        }
        return text[0] - '0';
    }();
    return value;
}

void Open(const Device& device, const std::filesystem::path& title_dir) {
    Close("before another game's store opens");
    if (!device.IsKhrPipelineBinaryEnabled()) {
        return;
    }
    const vk::DeviceDispatch& dld = device.GetDispatchLoader();
    if (!dld.vkGetPipelineKeyKHR || !dld.vkCreatePipelineBinariesKHR ||
        !dld.vkDestroyPipelineBinaryKHR || !dld.vkGetPipelineBinaryDataKHR ||
        !dld.vkReleaseCapturedPipelineDataKHR) {
        LOG_WARNING(Render_Vulkan, "Pipeline binaries: the driver's functions are missing, off");
        return;
    }
    Identity identity{
        .vendor_id = device.GetVendorID(),
        .device_id = device.GetDeviceID(),
        .driver_id = static_cast<u32>(device.GetDriverID()),
        .driver_version = device.GetDriverVersion(),
        .uuid = {},
    };
    const auto& uuid = device.GetPipelineCacheUUID();
    std::copy(std::begin(uuid), std::end(uuid), identity.uuid.begin());
    // No global key (vkGetPipelineKeyKHR without a create info): one NVIDIA driver overwrites a
    // register its caller keeps there. The ids above name the driver, and each entry is checked
    // against the driver's key of its own create info.
    auto store = std::make_shared<Store>(title_dir, identity, SwitchValue(), MaxBytesSwitch());
    if (!store->Open()) {
        return;
    }
    std::scoped_lock lock{g_store_mutex};
    g_store = std::move(store);
}

void Close(std::string_view when) {
    std::shared_ptr<Store> store;
    {
        std::scoped_lock lock{g_store_mutex};
        store = std::move(g_store);
    }
    if (!store) {
        return;
    }
    LogStoreCounts(*store, when);
    // Written and closed here, or by the last build still holding it
    store->Close();
}

std::string StampSuffix() {
    const std::shared_ptr<Store> store = CurrentStore();
    if (!store || !store->Saves()) {
        return {};
    }
    return fmt::format(" pipebin-{:016x}", store->FileId());
}

vk::Pipeline CreateGraphicsPipeline(const Device& device, const VkGraphicsPipelineCreateInfo& ci,
                                    VkPipelineCache cache) {
    return Create(device, ci, cache, Kind::Graphics);
}

vk::Pipeline CreateComputePipeline(const Device& device, const VkComputePipelineCreateInfo& ci,
                                   VkPipelineCache cache) {
    return Create(device, ci, cache, Kind::Compute);
}

void ReportPeriodic() {
    static std::mutex mutex;
    static Clock::time_point last{};
    static u64 last_total{~0ULL};
    const std::shared_ptr<Store> store = CurrentStore();
    if (!store) {
        return;
    }
    std::scoped_lock lock{mutex};
    const auto now = Clock::now();
    if (now - last < std::chrono::seconds{10}) {
        return;
    }
    const Store::Counts c = store->GetCounts();
    const u64 total = c.hits + c.misses + c.rejected + c.written + c.dropped;
    if (total == last_total) {
        return;
    }
    last = now;
    last_total = total;
    LogStoreCounts(*store, "so far");
}

void LogCounts(std::string_view when) {
    if (const std::shared_ptr<Store> store = CurrentStore()) {
        LogStoreCounts(*store, when);
    }
}

void NoteDeviceLoss() {
    const std::shared_ptr<Store> store = CurrentStore();
    if (!store || !store->UsedAny()) {
        return;
    }
    std::ofstream{store->LostFile()} << "device loss\n";
    LOG_CRITICAL(Render_Vulkan, "Pipeline binaries: the device was lost in a session that built "
                                "pipelines from the store; the next boot starts it over");
}

} // namespace Vulkan::PipelineBinary
