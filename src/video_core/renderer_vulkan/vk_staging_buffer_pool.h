// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <climits>
#include <vector>

#include "common/common_types.h"

#include "video_core/buffer_churn_stats.h"
#include "video_core/renderer_vulkan/vk_buffer_recycler.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

struct StagingBufferRef {
    VkBuffer buffer;
    VkDeviceAddress device_address;
    VkDeviceSize offset;
    std::span<u8> mapped_span;
    MemoryUsage usage;
    u32 log2_level;
    u64 index;
};

class StagingBufferPool {
public:
    static constexpr size_t NUM_SYNCS = 16;

    /// `stream_divisor`: the stream buffer is its usual size divided by this.
    explicit StagingBufferPool(const Device& device, MemoryAllocator& memory_allocator,
                               Scheduler& scheduler, u32 stream_divisor = 1);
    ~StagingBufferPool();

    /// `caller`: who asks, counted when a buffer has to be created (EDEN_NX2_BUFFER_CHURN_STATS).
    StagingBufferRef Request(
        size_t size, MemoryUsage usage, bool deferred = false,
        VideoCommon::BufferChurnStats::StagingCaller caller =
            VideoCommon::BufferChurnStats::StagingCaller::Other);
    void FreeDeferred(StagingBufferRef& ref);

    /// The memory seam's tick guard (SEAM = 2): cached staging buffers stamped `from` get the
    /// current tick (the copy that reads them was recorded after a flush).
    void Retick(u64 from);

    [[nodiscard]] VkBuffer StreamBuf() const noexcept {
        return *stream_buffer;
    }

    void TickFrame();

    /// Bytes of the cached staging buffers of a kind, idle or in use (EDEN_NX2_MEM_TRACE).
    [[nodiscard]] u64 CachedBytes(MemoryUsage usage);

    /// The stream buffer's size (the "Memory owners" line).
    [[nodiscard]] u64 StreamBufferBytes() const noexcept {
        return stream_buffer_size;
    }

    /// EXP104's pool, shared with the buffer cache that uses this staging pool.
    [[nodiscard]] BufferRecycler& Recycler() noexcept {
        return recycler;
    }

    /// The buffer cache's clock for this pool and its recycler, before any request: with
    /// FRITCHII_MEMSTAGE_TICKS their stamps are packet numbers.
    void UseClock(const SeamClock& clock_) {
        clock = clock_;
        recycler.UseClock(clock_);
    }

private:
    struct StreamBufferCommit {
        size_t upper_bound;
        u64 tick;
    };

    struct StagingBuffer {
        vk::Buffer buffer;
        VkDeviceAddress device_address;
        std::span<u8> mapped_span;
        MemoryUsage usage;
        u32 log2_level;
        u64 index;
        u64 tick = 0;
        bool deferred{};

        StagingBufferRef Ref() const noexcept {
            return {
                .buffer = *buffer,
                .device_address = device_address,
                .offset = 0,
                .mapped_span = mapped_span,
                .usage = usage,
                .log2_level = log2_level,
                .index = index,
            };
        }
    };

    struct StagingBuffers {
        std::vector<StagingBuffer> entries;
        size_t delete_index = 0;
        size_t iterate_index = 0;
    };

    static constexpr size_t NUM_LEVELS = sizeof(size_t) * CHAR_BIT;
    using StagingBuffersCache = std::array<StagingBuffers, NUM_LEVELS>;

    StagingBufferRef GetStreamBuffer(size_t size,
                                     VideoCommon::BufferChurnStats::StagingCaller caller);

    bool AreRegionsActive(size_t region_begin, size_t region_end) const;

    StagingBufferRef GetStagingBuffer(size_t size, MemoryUsage usage, bool deferred,
                                      VideoCommon::BufferChurnStats::StagingCaller caller,
                                      bool stream_full = false);

    std::optional<StagingBufferRef> TryGetReservedBuffer(size_t size, MemoryUsage usage,
                                                         bool deferred);

    StagingBufferRef CreateStagingBuffer(size_t size, MemoryUsage usage, bool deferred,
                                         VideoCommon::BufferChurnStats::StagingCaller caller,
                                         bool stream_full);

    StagingBuffersCache& GetCache(MemoryUsage usage);

    void ReleaseCache(MemoryUsage usage);

    void ReleaseLevel(StagingBuffersCache& cache, size_t log2);

    /// EXP104: offers an idle staging buffer being released to the pool (left empty if taken).
    void Retire(StagingBuffer& entry);
    size_t Region(size_t iter) const noexcept {
        return iter / region_size;
    }

    const Device& device;
    MemoryAllocator& memory_allocator;
    SeamClock clock;
    BufferRecycler recycler;

    vk::Buffer stream_buffer;
    VkDeviceAddress stream_buffer_address{};
    std::span<u8> stream_pointer;
    VkDeviceSize stream_buffer_size;
    VkDeviceSize region_size;

    size_t iterator = 0;
    size_t used_iterator = 0;
    size_t free_iterator = 0;
    std::array<u64, NUM_SYNCS> sync_ticks{};

    StagingBuffersCache device_local_cache;
    StagingBuffersCache upload_cache;
    StagingBuffersCache download_cache;

    size_t current_delete_level = 0;
    u64 buffer_index = 0;
    u64 unique_ids{};
};

} // namespace Vulkan
