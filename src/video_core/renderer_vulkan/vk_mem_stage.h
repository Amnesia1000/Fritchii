// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/buffer_cache/mem_read_log.h"
#include "video_core/nx2_profile.h"
#include "video_core/renderer_vulkan/vk_compute_pass.h"
#include "video_core/renderer_vulkan/vk_mem_stage_hook.h"
#include "video_core/renderer_vulkan/vk_seam_clock.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Tegra {
class GPU;
class MemoryManager;
}

namespace Vulkan {

struct BufferCacheParams;
class Device;
class MemoryAllocator;
class MemPlayer;
class Scheduler;

class MemStage {
public:
    enum class Cmd : u8 {
        UploadAhead,
        UploadBatched,
        Copy,
        PreCopyBarrier,
        PostCopyBarrier,
        Clear,
        QuadLutUpload,
        NullInit,
        BindIndex,
        BindIndexAlways,
        BindVertex,
        BindVertices,
        BindXfb,
        BindXfbs,
        Convert,
        Descriptors,
        Count,
    };

    enum Flag : u8 {
        OutsideRenderPass = 1,
        Barrier = 2,
        DynamicState = 4,
        ForgetOnly = 8,
        Played = 0x80, ///< Played early by a sync inside a held fill (SEAM = 2)
    };

    struct Entry {
        Cmd kind;
        u8 flags;
        u16 count;
        u32 bytes;
    };

    struct CopyPayload {
        VkBuffer src;
        VkBuffer dst;
        u32 count;
        u32 reserved;
    };
    struct ClearPayload {
        VkBuffer dst;
        u64 offset;
        u64 size;
        u32 value;
    };
    struct QuadLutPayload {
        VkBuffer src;
        VkDeviceSize src_offset;
        VkBuffer dst;
        VkDeviceSize size;
    };
    struct NullInitPayload {
        VkBuffer buffer;
    };
    struct IndexPayload {
        VkBuffer buffer;
        VkDeviceSize offset;
        VkIndexType type;
    };
    struct VertexOne {
        VkBuffer buffer;
        u32 index;
        u32 offset;
        u32 size;
        u32 stride;
    };
    struct VerticesPayload {
        u32 first;
        u32 count;
    };
    struct XfbOne {
        VkBuffer buffer;
        u32 index;
        u32 offset;
        u32 size;
    };
    struct XfbsPayload {
        u32 count;
        u32 reserved;
    };
    enum class Pass : u8 { Uint8, QuadIndexed, VertexRepack };
    struct ConvertPayload {
        Pass pass;
        ComputePassRun run;
    };
    struct DescriptorsPayload {
        u32 count;
        u32 reserved;
    };

    struct Span {
        u32 begin;
        u32 end;
    };
    struct BufferRef {
        VkBuffer buffer;
        u32 offset;
    };
    struct DrawPacket {
        u64 number;
        u32 flags;
        u32 repack_mask;
        u32 repack_dirty;
        Span geometry;
        std::array<Span, 5> stage;
        std::array<Span, 5> stage_descriptors;
        BufferRef gpu_count;
        BufferRef indirect;
        BufferRef indirect_count;
        Span reads;
        Span handle_values;
    };
    struct DispatchPacket {
        u64 number;
        u32 flags;
        Span before;
        Span descriptors;
        BufferRef indirect;
        Span after;
        Span reads;
        Span handle_values;
        std::array<u32, 64> qmd;
        std::array<u64, 8> null_slot_values;
        u32 null_slot_count;
    };
    struct OpPacket {
        u64 number;
        Span entries;
        BufferRef result;
    };

    enum class Site : u8 {
        Geometry,
        Stage,
        DispatchBefore,
        DispatchIndirect,
        DispatchAfter,
        DrawCount,
        DrawIndirect,
        ListWords,
        ListArgs,
        ListPages,
        Invalidation,
        Download,
        DownloadForRead,
        Frame,
        Dma,
        InlineUpload,
        Unmap,
        ImageCopy,
        Query,
        Fence,
        Draw,
        Dispatch,
        Count,
    };

    /// DrawPacket and DispatchPacket flags.
    enum PacketFlag : u32 {
        LateFill = 1,   ///< Texel buffers: filled where the calls are (as with SEAM = 1)
        Held = 2,       ///< Filled before the texture work, its sections played later
        RestPlayed = 4, ///< Configure did not play every section: PlayRest did
    };

    /// What FillDraw needs from the graphics pipeline (spec 4.2's StageFacts).
    struct DrawFacts {
        const std::array<u32, 5>* uniform_masks;
        const VideoCommon::UniformBufferSizes* uniform_sizes;
        const Shader::Info* stage_infos; ///< 5 of them
        u32 stages;                      ///< The stages Configure prepares, one bit each
        bool has_texel_buffers;
        bool uses_descriptor_buffer;
        std::array<u64, 2> hashes; ///< Vertex and fragment program, for the logs
    };

    /// What FillDispatch needs from the compute pipeline (spec 4.3's ComputeFacts).
    struct DispatchFacts {
        u32 uniform_mask;
        const VideoCommon::ComputeUniformBufferSizes* uniform_sizes;
        const Shader::Info* info;
        u64 shader_hash;
        bool has_texel_buffers;
        bool uses_descriptor_buffer;
    };

    using Cache = VideoCommon::BufferCache<BufferCacheParams>;

    enum class Sync : u8 {
        Finish,
        Wait,
        SubmitForHostRead,
        MultiRange,
        KeptWords,
        Count,
    };

    class Arena {
    public:
        explicit Arena(u32 initial_capacity);
        ~Arena();

        [[nodiscard]] u32 Size() const noexcept {
            return cursor;
        }
        void Reset() noexcept {
            cursor = 0;
        }
        /// Room for `bytes` more (a multiple of 8) at the end; returns its offset.
        u32 Reserve(u32 bytes);
        [[nodiscard]] u8* At(u32 offset) noexcept {
            return data.get() + offset;
        }
        [[nodiscard]] const u8* At(u32 offset) const noexcept {
            return data.get() + offset;
        }
        void Swap(Arena& other) noexcept {
            data.swap(other.data);
            std::swap(capacity, other.capacity);
            std::swap(cursor, other.cursor);
        }

    private:
        std::unique_ptr<u8[]> data;
        u32 capacity;
        u32 cursor{};
    };

    /// A fill bracket: with SEAM on, the entries pushed inside wait for a play.
    class Fill {
    public:
        Fill(MemStage& stage, Site site, u32 index = 0);
        ~Fill();
        Fill(const Fill&) = delete;
        Fill& operator=(const Fill&) = delete;

    private:
        MemStage& stage;
        Site site;
        u32 index;
        Site previous_site;
        u32 begin{};
        bool reads{}; ///< This bracket started an operation packet's read log
        std::optional<VideoCommon::Nx2Profile::Scope> zone;
    };

    /// A fill bracket that plays what it filled when it ends.
    class FillPlay {
    public:
        FillPlay(MemStage& stage_, Site site, u32 index = 0) : stage{stage_} {
            fill.emplace(stage_, site, index);
        }
        ~FillPlay();
        FillPlay(const FillPlay&) = delete;
        FillPlay& operator=(const FillPlay&) = delete;

    private:
        MemStage& stage;
        std::optional<Fill> fill;
    };

    /// Plays what is pending, then runs a sync: what the sync records is not counted.
    class SyncScope {
    public:
        SyncScope(MemStage* stage, Sync kind);
        ~SyncScope();
        SyncScope(const SyncScope&) = delete;
        SyncScope& operator=(const SyncScope&) = delete;
    };

    explicit MemStage(MemPlayer& player);
    ~MemStage();

    MemStage(const MemStage&) = delete;
    MemStage& operator=(const MemStage&) = delete;

    [[nodiscard]] MemPlayer& Player() noexcept {
        return player;
    }

    /// FRITCHII_MEMSTAGE_TICKS >= 1: the packet numbers and published ticks; null at 0.
    [[nodiscard]] SeamTicks* Ticks() const noexcept {
        return ticks.get();
    }

    /// Runs `fill` in a fill bracket, then plays: for the calls that return a buffer the caller
    /// records with.
    template <typename Func>
    decltype(auto) Ref(Site site, Func&& fill) {
        struct PlayAtEnd {
            MemStage& stage;
            ~PlayAtEnd() {
                stage.Play();
            }
        } play{*this};
        Fill bracket{*this, site};
        return fill();
    }

    /// Plays every pending entry this thread pushed.
    void Play();

    void BeginDraw(u32 repack_mask, u32 repack_dirty);
    void BeginDispatch();

    /// The cache whose ranges the read log asks about.
    void Attach(Cache& cache) noexcept;
    /// The read log's pending-mark question (GPU::ForEachPendingCpuWrite).
    void AttachGpu(Tegra::GPU& gpu) noexcept;

    /// The read log (Common::WriteRing::g_on): a gather of game writes applied
    /// (OnCacheInvalidations) or a FlushRegion (`which`: its CacheType bits), on this thread.
    enum class Gather : u8 { Apply, FlushRegion };
    void NoteGather(Gather kind, u32 which);

    /// The read log's classes of a difference (5.3).
    enum class DiffClass : u8 {
        Landing,       ///< an emulator write the ring names, a download landing
        EmulatorWrite, ///< an emulator write the ring names, anything else
        GameMarked,    ///< no emulator write; a game-side mark is pending (or the tracker holds it)
        MarkTaken,     ///< no emulator write, no pending mark: a store in flight at the gather
        BackGather,    ///< bound range: a gather ran on this thread between the fill and the play
        Unclassified,
        Count,
    };

    /// SEAM = 2: the draw's whole fill, before the texture work. Null when the draw is not held
    /// (another SEAM, texel buffers: LateFill, or the arena busy on another thread); Configure
    /// then fills where the calls are. A held packet is always played whole: RestAtEnd.
    const DrawPacket* FillDraw(Cache& cache, const DrawFacts& facts, bool is_indexed);
    /// SEAM = 2: the dispatch's `before` section, before Configure. Null as for FillDraw.
    const DispatchPacket* FillDispatch(Cache& cache, const DispatchFacts& facts);

    [[nodiscard]] const DrawPacket* HeldDraw() const noexcept {
        return held == HeldKind::Draw ? &draw : nullptr;
    }
    [[nodiscard]] const DispatchPacket* HeldDispatch() const noexcept {
        return held == HeldKind::Dispatch ? &dispatch : nullptr;
    }

    /// Plays the held packet up to the end of `span` (the old position of that section).
    void PlayHeld(Span span);
    /// Plays what is left of the held packet (descriptors left out) and ends it.
    void PlayRest();

    /// PlayRest when Configure ends: every return, true or false, and an exception.
    class RestAtEnd {
    public:
        explicit RestAtEnd(MemStage& stage_) noexcept : stage{stage_} {}
        ~RestAtEnd() {
            stage.PlayRest();
        }
        RestAtEnd(const RestAtEnd&) = delete;
        RestAtEnd& operator=(const RestAtEnd&) = delete;

    private:
        MemStage& stage;
    };

    /// The descriptor form the runtime's descriptor entries take: the held fill's pipeline's
    /// (it runs before Acquire), else the queue's.
    [[nodiscard]] bool DescriptorForm(bool queue_form) const noexcept {
        return fill_form < 0 ? queue_form : fill_form != 0;
    }

    enum class Value : u8 {
        Handle,
        Qmd,
        NullSlot,
        Shape,
        Count,
    };
    enum ValueFlag : u32 {
        HandleValuesFilled = 1U << 16,
        HandleNullPointers = 1U << 17,
        HandleValuesLooked = 1U << 18,
        HandleValuesRetired = 1U << 19,
        QmdByValue = 1U << 20,
    };
    enum VerifyBit : u32 {
        VerifyHandles = 1,
        VerifyQmd = 2,
        VerifyNullSlots = 4,
    };

    [[nodiscard]] bool ValuesByValue() const noexcept {
        return seam_level.load(std::memory_order_relaxed) >= 3;
    }
    [[nodiscard]] static u32 VerifyBits() noexcept;

    template <typename FillFn>
    void FillHandleValues(bool dispatch_side, bool null_pointer_handles, bool in_configure,
                          FillFn&& fill) {
        std::vector<u32>& values = dispatch_side ? dispatch_values : draw_values;
        values.clear();
        fill(values);
        CloseHandleValues(dispatch_side, null_pointer_handles, in_configure);
    }
    [[nodiscard]] const std::vector<u32>* HandleValues(bool dispatch_side,
                                                       bool& null_pointer_handles) const noexcept;
    void LookedAtHandleValues(bool dispatch_side, size_t taken);
    void NoteValueCheck(Value kind, bool same, u64 address, u64 shader, u64 packet_value,
                        u64 own_value);

    const std::array<u32, 64>& Qmd(Tegra::MemoryManager& gpu_memory, GPUVAddr address);
    u64 ReadPointerSlot(Tegra::MemoryManager& gpu_memory, GPUVAddr address);

    /// Once a frame on the GPU thread: reads SEAM (EDEN_NX2_EXP123) and logs the stats line.
    void TickFrame();

    void Push(Cmd kind, u8 flags = 0);
    void PushCopy(Cmd kind, u8 flags, VkBuffer src, VkBuffer dst,
                  std::span<const VkBufferCopy> copies);
    void PushClear(VkBuffer dst, u64 offset, u64 size, u32 value);
    void PushQuadLut(VkBuffer src, VkDeviceSize src_offset, VkBuffer dst, VkDeviceSize size);
    void PushNullInit(VkBuffer buffer);
    void PushIndex(Cmd kind, u8 flags, VkBuffer buffer, VkDeviceSize offset, VkIndexType type);
    void PushVertex(u8 flags, u32 index, VkBuffer buffer, u32 offset, u32 size, u32 stride);
    void PushVertices(u8 flags, u32 first, std::span<const VkBuffer> buffers,
                      std::span<const VkDeviceSize> offsets, std::span<const VkDeviceSize> sizes,
                      std::span<const VkDeviceSize> strides);
    void PushXfb(u32 index, VkBuffer buffer, u32 offset, u32 size);
    void PushXfbs(std::span<const VkBuffer> buffers, std::span<const VkDeviceSize> offsets,
                  std::span<const VkDeviceSize> sizes);
    void PushConvert(Pass pass, const ComputePassRun& run);
    /// `texel`: a texel buffer's descriptor (counted apart in the stats line).
    void PushDescriptor(const DescriptorUpdateEntry& entry, bool texel = false);

private:
    friend class SeamClock;
    friend void SeamHook::OnUse(SeamHook::Use use, const std::source_location& location);
    friend void SeamHook::OnTickEnd(u64 tick);
    friend void VideoCommon::MemReadLog::NoteBytes(VideoCommon::MemReadLog::Kind,
                                                   Tegra::MaxwellDeviceMemoryManager&, u64, u64);
    friend void VideoCommon::MemReadLog::NoteStaged(VideoCommon::MemReadLog::Kind,
                                                    Tegra::MaxwellDeviceMemoryManager&, u64,
                                                    const u8*, u64);
    friend void VideoCommon::MemReadLog::NoteValue(VideoCommon::MemReadLog::Kind,
                                                   Tegra::MemoryManager*, u64, u64, u32);
    friend void VideoCommon::MemReadLog::NoteRange(VideoCommon::MemReadLog::Kind, u64, u64);
    friend void VideoCommon::MemReadLog::NoteGuestWrite(Tegra::MemoryManager&, u64, u64);

    enum class HeldKind : u8 { None, Draw, Dispatch };

    /// One read of game memory a fill made (spec 5.3).
    struct ReadRecord {
        enum class How : u8 { Bytes, Staged, Value, Range };
        VideoCommon::MemReadLog::Kind kind;
        How how;
        u32 size;
        u64 address;
        u64 value;        ///< Value: the value read; Bytes: offset of the copy; Range: fill state
        const void* data; ///< Staged: the staged bytes
        void* memory;     ///< The device or GPU memory read
    };
    struct ReadLog {
        std::vector<ReadRecord> records;
        std::vector<u8> bytes;
        u64 skipped{}; ///< Not logged: past the byte budget
        u64 budget{};  ///< Bytes to read again at the compare
        /// Device ranges the packet uses: bound ranges, uniforms, values (over-read test).
        std::vector<std::pair<u64, u64>> used;
        u64 ring_seq{};      ///< Common::WriteRing's sequence when the fill began
        u64 gathers{};       ///< This thread's gathers applied when the fill ended
        u64 flush_regions{}; ///< This thread's FlushRegion calls when the fill ended
        void Clear() noexcept {
            records.clear();
            bytes.clear();
            skipped = 0;
            budget = 0;
            used.clear();
            ring_seq = 0;
            gathers = 0;
            flush_regions = 0;
        }
    };

    /// Emulator writes the ring named at a difference, by site.
    struct DiffSite {
        std::atomic<const char*> name{};
        std::atomic<bool> landing{};
        std::atomic<u64> count{};
    };
    static constexpr size_t DIFF_SITES = 32;

    inline static thread_local ReadLog* t_log = nullptr;

    /// A device range a held fill read (any kind), for the texture write check.
    struct ReadRange {
        u64 begin;
        u64 end;
        VideoCommon::MemReadLog::Kind kind;
    };
    inline static thread_local std::vector<ReadRange>* t_reads = nullptr;
    static void NoteRead(VideoCommon::MemReadLog::Kind kind, u64 address, u64 size) {
        if (t_reads != nullptr && size != 0) {
            t_reads->push_back({address, address + size, kind});
        }
    }

    struct Part {
        const void* data;
        u32 bytes;
    };

    template <typename T>
    static Part PartOf(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        return Part{&value, static_cast<u32>(sizeof(T))};
    }
    template <typename T>
    static Part PartOf(std::span<const T> values) {
        static_assert(std::is_trivially_copyable_v<T>);
        return Part{values.data(), static_cast<u32>(values.size_bytes())};
    }

    struct Counts {
        std::array<u64, static_cast<size_t>(Cmd::Count)> entries{};
        std::array<u64, static_cast<size_t>(Cmd::Count)> outside{};
        std::array<u64, static_cast<size_t>(Site::Count)> site_fills{};
        std::array<u64, static_cast<size_t>(Site::Count)> site_entries{};
        std::array<u64, static_cast<size_t>(Sync::Count)> syncs{};
        u64 draws{};
        u64 dispatches{};
        u64 deferred{};
        u64 plays{};
        u64 around{};
        u64 late{};
        u64 unclocked{};
        u64 texel{};
        u64 held_draws{};
        u64 late_draws{};
        u64 held_dispatches{};
        u64 late_dispatches{};
        u64 unheld{};
        u64 held_flushed{};
        u64 seam_clamped{};
        u64 texture_writes{};
        u64 texture_overlaps{};
        u64 rest_plays{};
        u64 rest_descriptors{};
        u64 form_mismatch{};
        u64 read_packets{};
        u64 read_records{};
        u64 read_skipped{};
        u64 fill_ns{};
        u64 play_ns{};
        std::array<u64, static_cast<size_t>(VideoCommon::MemReadLog::Kind::Count)> read_diffs{};
        std::array<u64, static_cast<size_t>(VideoCommon::MemReadLog::Kind::Count)>
            read_kind_records{};
        std::array<u64, static_cast<size_t>(DiffClass::Count)> diff_classes{};
        u64 diff_own{};
        u64 diff_over{};
        u64 ring_wrapped{};
        u64 gathers_held{};
        std::array<u64, 3> flush_regions_held{};
        std::array<u64, DIFF_SITES> diff_sites{};
        u64 diff_sites_other{};
    };

    void PushParts(Cmd kind, u8 flags, u32 count, std::initializer_list<Part> parts);
    void CountPush(Cmd kind) noexcept;
    [[nodiscard]] bool Deferring() const noexcept;
    [[nodiscard]] bool OwnedHere() const noexcept;
    void SetPending();
    void PlayPending(bool late);
    void CloseSection(Site site, u32 index, u32 begin);
    void EnterSync(Sync kind);
    void Note(SeamHook::Use use, const std::source_location& location, bool around);
    void LogStats();
    void Snapshot(Counts& out) const;
    void CloseHandleValues(bool dispatch_side, bool null_pointer_handles, bool in_configure);
    void RetireDrawValues();
    void RetireDispatchValues();
    void OpenDispatchByValue();
    void LogValueStats(double seconds, bool first);

    bool ClaimHeld();
    void UpdateUnplayed();
    void TakeHeld(HeldKind kind);
    struct HeldFillEnd;
    void EndHeldFill();
    void CheckHeld();
    void PlayHeldTo(u32 end, bool rest);
    bool StartReadLog(ReadLog& log, u64 number);
    void StopReadLog(ReadLog& log, bool held_packet);
    void NoteHeldFlush(SeamHook::Use use, const std::source_location& location);
    void CompareReads(ReadLog& log, const char* what, u64 number, u64 hash_a, u64 hash_b);
    void ClassifyDiff(const ReadLog& log, const ReadRecord& record, u64 first, u64 last,
                      const char* what, u64 number, u64 hash_a, u64 hash_b);
    void CountDiffSite(const char* name, bool landing);
    std::string GatherFrames() const;

    MemPlayer& player;
    Arena arena;
    std::atomic<uintptr_t> owner{0};
    std::atomic<bool> seam_on{false};
    std::atomic<int> seam_value{0};
    /// FRITCHII_MEMSTAGE_READ_LOG: every Nth packet's reads are compared, 0 off
    int read_log_every{};
    Cache* cache{};
    /// SEAM 2/3 asked for without packet ticks: run as SEAM 1 (logged once).
    bool clamp_logged{};

    HeldKind held{HeldKind::None};
    bool filling_held{};
    HeldKind filling_kind{HeldKind::None};
    /// The held packet saw a flush between its fill and its PlayRest (counted once).
    bool held_flush_seen{};
    s8 fill_form{-1};
    Arena held_arena;
    u32 held_cursor{};
    bool held_form{};
    std::array<u64, 2> held_hashes{};
    std::vector<ReadRange> held_reads;
    ReadLog op_log;
    ReadLog held_log;
    bool held_log_pending{};
    u64 op_fills{};
    std::vector<u8> compare_scratch;
    Tegra::GPU* gpu{};
    /// This stage turned Common::WriteRing on (FRITCHII_MEMSTAGE_READ_LOG).
    bool ring_owner{};
    std::array<std::atomic<u32>, static_cast<size_t>(DiffClass::Count)> class_logged{};
    /// The first gather on this thread after the held fill ended: where it came from.
    std::array<void*, 16> gather_frames{};
    u32 gather_frame_count{};
    std::mutex diff_sites_mutex;
    std::array<DiffSite, DIFF_SITES> diff_site_table{};
    bool claim_deferring{};
    bool pending{};
    u32 play_cursor{};
    u32 last_entry{~0U};

    u64 packet_number{};
    std::unique_ptr<SeamTicks> ticks;
    u64 op_number{};
    /// Packet ticks: the first number of the pending entries and of the held packet (SEAM >= 2).
    u64 pending_from{};
    u64 held_from{};
    DrawPacket draw{};
    DispatchPacket dispatch{};
    OpPacket op{};

    struct Counters {
        std::array<std::atomic<u64>, static_cast<size_t>(Cmd::Count)> entries{};
        std::array<std::atomic<u64>, static_cast<size_t>(Cmd::Count)> outside{};
        std::array<std::atomic<u64>, static_cast<size_t>(Site::Count)> site_fills{};
        std::array<std::atomic<u64>, static_cast<size_t>(Site::Count)> site_entries{};
        std::array<std::atomic<u64>, static_cast<size_t>(Sync::Count)> syncs{};
        std::atomic<u64> draws{};
        std::atomic<u64> dispatches{};
        std::atomic<u64> deferred{};
        std::atomic<u64> plays{};
        std::atomic<u64> around{};
        std::atomic<u64> late{};
        std::atomic<u64> unclocked{};
        std::atomic<u64> texel{};
        std::atomic<u64> held_draws{};
        std::atomic<u64> late_draws{};
        std::atomic<u64> held_dispatches{};
        std::atomic<u64> late_dispatches{};
        std::atomic<u64> unheld{};
        std::atomic<u64> held_flushed{};
        std::atomic<u64> seam_clamped{};
        std::atomic<u64> texture_writes{};
        std::atomic<u64> texture_overlaps{};
        std::atomic<u64> rest_plays{};
        std::atomic<u64> rest_descriptors{};
        std::atomic<u64> form_mismatch{};
        std::atomic<u64> read_packets{};
        std::atomic<u64> read_records{};
        std::atomic<u64> read_skipped{};
        std::atomic<u64> fill_ns{};
        std::atomic<u64> play_ns{};
        std::array<std::atomic<u64>, static_cast<size_t>(VideoCommon::MemReadLog::Kind::Count)>
            read_diffs{};
        std::array<std::atomic<u64>, static_cast<size_t>(VideoCommon::MemReadLog::Kind::Count)>
            read_kind_records{};
        std::array<std::atomic<u64>, static_cast<size_t>(DiffClass::Count)> diff_classes{};
        std::atomic<u64> diff_own{};
        std::atomic<u64> diff_over{};
        std::atomic<u64> ring_wrapped{};
        std::atomic<u64> gathers_held{};
        std::array<std::atomic<u64>, 3> flush_regions_held{};
        std::atomic<u64> diff_sites_other{};
    };
    Counters counters;
    Counts reported;
    u64 last_log_ns{};

    std::atomic<int> seam_level{0};
    bool dispatch_open{};
    std::vector<u32> draw_values;
    std::vector<u32> dispatch_values;

    struct ValueCounts {
        std::array<u64, 2> fills{};
        std::array<u64, 2> values{};
        std::array<u64, 2> in_configure{};
        std::array<u64, 2> skipped{};
        std::array<u64, 2> skipped_values{};
        u64 qmds{};
        u64 qmds_unused{};
        u64 null_slots{};
        u64 null_slots_unkept{};
        std::array<u64, static_cast<size_t>(Value::Count)> checks{};
        std::array<u64, static_cast<size_t>(Value::Count)> mismatches{};
    };
    struct ValueCounters {
        std::array<std::atomic<u64>, 2> fills{};
        std::array<std::atomic<u64>, 2> values{};
        std::array<std::atomic<u64>, 2> in_configure{};
        std::array<std::atomic<u64>, 2> skipped{};
        std::array<std::atomic<u64>, 2> skipped_values{};
        std::atomic<u64> qmds{};
        std::atomic<u64> qmds_unused{};
        std::atomic<u64> null_slots{};
        std::atomic<u64> null_slots_unkept{};
        std::array<std::atomic<u64>, static_cast<size_t>(Value::Count)> checks{};
        std::array<std::atomic<u64>, static_cast<size_t>(Value::Count)> mismatches{};
    };
    ValueCounters value_counters;
    ValueCounts reported_values;
    void SnapshotValues(ValueCounts& out) const;
};

/// Turns the memory stage's entries into scheduler and descriptor queue calls: the only copy of
/// that Vulkan code. Owns what the GPU thread recorded: the remembered bindings and the list
/// draw facility.
class MemPlayer {
public:
    struct ListDrawArgs {
        VkBuffer buffer{};
        VkDeviceSize offset{};
        bool mesh{}; ///< A task/mesh draw command (group counts)
    };

    explicit MemPlayer(const Device& device, MemoryAllocator& memory_allocator,
                       Scheduler& scheduler, GuestDescriptorQueue& guest_descriptor_queue,
                       ComputePassDescriptorQueue& compute_pass_descriptor_queue);
    ~MemPlayer();

    MemPlayer(const MemPlayer&) = delete;
    MemPlayer& operator=(const MemPlayer&) = delete;

    void SetPasses(Uint8Pass* uint8, QuadIndexedPass* quad_indexed,
                   VertexRepackPass* vertex_repack) noexcept;

    void Exec(const MemStage::Entry& entry);

    void AddDescriptors(const DescriptorUpdateEntry* entries, size_t count);

    /// The remembered index and vertex buffers are forgotten (SEAM changed).
    void ForgetBindings() noexcept;

    /// `keep_id`: the id the copy of the words is kept as (KeepListWords), 0 for none.
    void RequestListDrawArgs(u64 keep_id = 0) noexcept {
        list_draw_requested = true;
        list_draw_args = {};
        list_draw_keep_id = keep_id;
    }
    [[nodiscard]] bool ListDrawArgsRequested() const noexcept {
        return list_draw_requested;
    }
    /// Records the copy of the words at `offset` of `source`, outside the render pass, into a
    /// vertex draw command or (`mesh`) a task/mesh one.
    void RecordListDrawArgs(VkBuffer source, VkDeviceSize offset, bool mesh);
    /// The arguments recorded since the request (no buffer if none were), ending the request.
    [[nodiscard]] ListDrawArgs TakeListDrawArgs() noexcept {
        list_draw_requested = false;
        list_draw_keep_id = 0;
        return std::exchange(list_draw_args, {});
    }
    /// Records, outside the render pass, a copy of `count` (up to 3) words at `offset` of
    /// `source` as `keep_id`.
    void KeepListWords(u64 keep_id, VkBuffer source, VkDeviceSize offset, u32 count);
    /// Keeps words the host read itself as `keep_id`.
    void KeepListWordsRead(u64 keep_id, std::span<const u32> words);
    /// The words kept as `keep_id`, waiting for the host GPU to copy them if it must. False if
    /// none are (a later copy took the slot).
    [[nodiscard]] bool ReadKeptListWords(u64 keep_id, std::span<u32> words);

private:
    friend class MemStage;

    void EnsureListKeepBuffer();

    const Device& device;
    MemoryAllocator& memory_allocator;
    Scheduler& scheduler;
    GuestDescriptorQueue& guest_descriptor_queue;
    ComputePassDescriptorQueue& compute_pass_descriptor_queue;
    MemStage* stage{};

    Uint8Pass* uint8_pass{};
    QuadIndexedPass* quad_indexed_pass{};
    VertexRepackPass* vertex_repack_pass{};

    VkBuffer bound_index_buffer = VK_NULL_HANDLE;
    VkDeviceSize bound_index_offset = 0;
    VkIndexType bound_index_type = VK_INDEX_TYPE_UINT32;
    u64 bound_index_epoch = 0;
    bool bound_index_valid = false;

    struct BoundVertexBuffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        VkDeviceSize size = 0;
        VkDeviceSize stride = 0;
        u64 epoch = ~u64{0};
    };
    static constexpr size_t NUM_BOUND_VERTEX_BUFFERS = 32;
    std::array<BoundVertexBuffer, NUM_BOUND_VERTEX_BUFFERS> bound_vertex_buffers{};

    static constexpr u32 LIST_DRAW_SLOTS = 1024;
    static constexpr u32 LIST_DRAW_SLOT_BYTES = 32;
    vk::Buffer list_draw_buffer;
    u32 list_draw_cursor{};
    bool list_draw_requested{};
    ListDrawArgs list_draw_args{};
    u64 list_draw_keep_id{};

    static constexpr u32 LIST_KEEP_SLOTS = 1024;
    static constexpr u32 LIST_KEEP_SLOT_BYTES = 16;
    struct ListKeep {
        u64 keep_id{};
        u64 tick{}; ///< The submission that copies them, if on_gpu
        bool on_gpu{};
        u32 count{};
        std::array<u32, 3> words{}; ///< If not on_gpu
    };
    vk::Buffer list_keep_buffer;
    std::array<ListKeep, LIST_KEEP_SLOTS> list_keeps{};
};

} // namespace Vulkan
