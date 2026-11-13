// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <source_location>
#include <span>
#include <type_traits>
#include <utility>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_compute_pass.h"
#include "video_core/renderer_vulkan/vk_mem_stage_hook.h"
#include "video_core/renderer_vulkan/vk_seam_clock.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

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
        u64 fill_tick;
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
        Span handle_words;
    };
    struct DispatchPacket {
        u64 number;
        u64 fill_tick;
        u32 flags;
        Span before;
        Span descriptors;
        BufferRef indirect;
        Span after;
        Span reads;
        Span handle_words;
        std::array<u32, 64> qmd;
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
        Count,
    };

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

    MemPlayer& player;
    Arena arena;
    std::atomic<uintptr_t> owner{0};
    std::atomic<bool> seam_on{false};
    bool claim_deferring{};
    bool pending{};
    u32 play_cursor{};
    u32 last_entry{~0U};

    u64 packet_number{};
    std::unique_ptr<SeamTicks> ticks;
    u64 op_number{};
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
    };
    Counters counters;
    Counts reported;
    u64 last_log_ns{};
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
