// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstdlib>
#include <array>
#include <bit>
#include <bitset>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/container/unordered_map.h"
#include <boost/unordered/unordered_node_map.hpp>
#include <boost/container/static_vector.hpp>
#include <boost/container/small_vector.hpp>

#include "common/common_types.h"
#include "common/spin_recursive_mutex.h"
#include "common/div_ceil.h"
#include "common/literals.h"
#include "common/lru_cache.h"
#include "common/range_sets.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/slot_vector.h"
#include "core/gpu_dirty_memory_manager.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/buffer_cache/buffer_base.h"
#include "video_core/buffer_cache/download_read_stats.h"
#include "video_core/buffer_cache/gpu_written_mirror.h"
#include "video_core/buffer_cache/read_trap_tracker.h"
#include "video_core/buffer_cache/virtual_range_cache.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/delayed_destruction_ring.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_runahead.h"
#include "video_core/nx2.h"
#include "video_core/nx2_async_readback.h"
#include "video_core/nx2_download_trace.h"
#include "video_core/rasterizer_download_area.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/types.h"

namespace VideoCommon {

using BufferId = Common::SlotId;

using VideoCore::Surface::PixelFormat;
using namespace Common::Literals;

#ifdef __APPLE__
constexpr u32 NUM_VERTEX_BUFFERS = 16;
#else
constexpr u32 NUM_VERTEX_BUFFERS = 32;
#endif
constexpr u32 NUM_TRANSFORM_FEEDBACK_BUFFERS = 4;
constexpr u32 NUM_GRAPHICS_UNIFORM_BUFFERS = 18;
constexpr u32 NUM_COMPUTE_UNIFORM_BUFFERS = 8;
// Native shaders can reference more than 16 storage buffers. Dropping a binding
// shifts all following descriptor payloads into the wrong descriptor types.
constexpr u32 NUM_STORAGE_BUFFERS = Shader::Info::MAX_SSBOS;
static_assert(NUM_STORAGE_BUFFERS <= 64, "storage binding masks must fit u64");
constexpr u32 NUM_TEXTURE_BUFFERS = 32;
constexpr u32 NUM_STAGES = 5;

static_assert(NUM_GRAPHICS_UNIFORM_BUFFERS <= 32, "fast bitmask must fit u32");

using UniformBufferSizes = std::array<std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES>;
using ComputeUniformBufferSizes = std::array<u32, NUM_COMPUTE_UNIFORM_BUFFERS>;

enum class ObtainBufferSynchronize : u32 {
    NoSynchronize = 0,
    FullSynchronize = 1,
    SynchronizeNoDirty = 2,
};

enum class ObtainBufferOperation : u32 {
    DoNothing = 0,
    MarkAsWritten = 1,
    DiscardWrite = 2,
    MarkQuery = 3,
};

static constexpr BufferId NULL_BUFFER_ID{0};
static constexpr u32 DEFAULT_SKIP_CACHE_SIZE = static_cast<u32>(4_KiB);

struct Binding {
    DAddr device_addr{};
    GPUVAddr gpu_addr{};
    u32 size{};
    BufferId buffer_id;
    u32 segment_first{};
    u32 segment_count{};
};

struct MultiRangeSegment {
    BufferId buffer_id;
    DAddr device_addr{};
    u32 size{};
};

struct TextureBufferBinding : Binding {
    PixelFormat format;
};

static constexpr Binding NULL_BINDING{
    .device_addr = 0,
    .size = 0,
    .buffer_id = NULL_BUFFER_ID,
};

template <typename Buffer>
struct HostBindings {
    boost::container::static_vector<Buffer*, NUM_VERTEX_BUFFERS> buffers;
    boost::container::static_vector<u64, NUM_VERTEX_BUFFERS> offsets;
    boost::container::static_vector<u64, NUM_VERTEX_BUFFERS> sizes;
    boost::container::static_vector<u64, NUM_VERTEX_BUFFERS> strides;
    u32 min_index{NUM_VERTEX_BUFFERS};
    u32 max_index{0};
};

class BufferCacheChannelInfo : public ChannelInfo {
public:
    BufferCacheChannelInfo() = delete;
    BufferCacheChannelInfo(Tegra::Control::ChannelState& state) noexcept : ChannelInfo(state) {}
    BufferCacheChannelInfo(const BufferCacheChannelInfo& state) = delete;
    BufferCacheChannelInfo& operator=(const BufferCacheChannelInfo&) = delete;

    Binding index_buffer;
    std::array<Binding, NUM_VERTEX_BUFFERS> vertex_buffers;
    std::array<std::array<Binding, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES> uniform_buffers;
    std::array<std::array<Binding, NUM_STORAGE_BUFFERS>, NUM_STAGES> storage_buffers;
    std::array<std::array<TextureBufferBinding, NUM_TEXTURE_BUFFERS>, NUM_STAGES> texture_buffers;
    std::array<Binding, NUM_TRANSFORM_FEEDBACK_BUFFERS> transform_feedback_buffers;
    Binding count_buffer_binding;
    Binding indirect_buffer_binding;

    std::array<Binding, NUM_COMPUTE_UNIFORM_BUFFERS> compute_uniform_buffers;
    std::array<Binding, NUM_STORAGE_BUFFERS> compute_storage_buffers;
    std::array<TextureBufferBinding, NUM_TEXTURE_BUFFERS> compute_texture_buffers;

    std::array<u32, NUM_STAGES> enabled_uniform_buffer_masks{};
    u32 enabled_compute_uniform_buffer_mask = 0;

    const UniformBufferSizes* uniform_buffer_sizes{};
    const ComputeUniformBufferSizes* compute_uniform_buffer_sizes{};

    std::array<u64, NUM_STAGES> enabled_storage_buffers{};
    std::array<u64, NUM_STAGES> written_storage_buffers{};
    u64 enabled_compute_storage_buffers = 0;
    u64 written_compute_storage_buffers = 0;
    // Bound by a native (Switch 2) compute program, whose pointers carry no size.
    u64 native_compute_storage_buffers = 0;
    // Bytes from each compute storage binding's start that a dispatch marks GPU-written.
    std::array<u32, NUM_STORAGE_BUFFERS> compute_storage_write_marks{};

    u32 total_graphics_storage_buffers = 0;
    u32 total_compute_storage_buffers = 0;

    std::array<u32, NUM_STAGES> enabled_texture_buffers{};
    std::array<u32, NUM_STAGES> written_texture_buffers{};
    std::array<u32, NUM_STAGES> image_texture_buffers{};
    u32 enabled_compute_texture_buffers = 0;
    u32 written_compute_texture_buffers = 0;
    u32 image_compute_texture_buffers = 0;

    std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS> uniform_cache_hits{};
    std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS> uniform_cache_shots{};

    u32 uniform_buffer_skip_cache_size = DEFAULT_SKIP_CACHE_SIZE;

    bool has_deleted_buffers = false;

    std::array<u32, NUM_STAGES> dirty_uniform_buffers{};
    std::array<u32, NUM_STAGES> fast_bound_uniform_buffers{};
    std::array<std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES>
        uniform_buffer_binding_sizes{};
};

template <class P>
class BufferCache : public VideoCommon::ChannelSetupCaches<BufferCacheChannelInfo> {
    // Page size for caching purposes.
    // This is unrelated to the CPU page size and it can be changed as it seems optimal.
#ifdef YUZU_LEGACY
    static constexpr u32 CACHING_PAGEBITS = 12;
#else
    static constexpr u32 CACHING_PAGEBITS = 16;
#endif
    static constexpr u64 CACHING_PAGESIZE = u64{1} << CACHING_PAGEBITS;

    static constexpr bool IS_OPENGL = P::IS_OPENGL;
    static constexpr bool HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS =
        P::HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS;
    static constexpr bool HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT =
        P::HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT;
    static constexpr bool NEEDS_BIND_UNIFORM_INDEX = P::NEEDS_BIND_UNIFORM_INDEX;
    static constexpr bool NEEDS_BIND_STORAGE_INDEX = P::NEEDS_BIND_STORAGE_INDEX;
    static constexpr bool USE_MEMORY_MAPS = P::USE_MEMORY_MAPS;
    static constexpr bool SEPARATE_IMAGE_BUFFERS_BINDINGS = P::SEPARATE_IMAGE_BUFFER_BINDINGS;
    static constexpr bool USE_MEMORY_MAPS_FOR_UPLOADS = P::USE_MEMORY_MAPS_FOR_UPLOADS;

#ifdef YUZU_LEGACY
    static constexpr s64 TARGET_THRESHOLD = 3_GiB;
#else
    static constexpr s64 TARGET_THRESHOLD = 4_GiB;
#endif

    static constexpr s64 DEFAULT_EXPECTED_MEMORY = 512_MiB;
    static constexpr s64 DEFAULT_CRITICAL_MEMORY = 1_GiB;

    // Debug Flags.

    static constexpr bool DISABLE_DOWNLOADS = true;

    using Maxwell = Tegra::Engines::Maxwell3D::Regs;

    using Runtime = typename P::Runtime;
    using Buffer = typename P::Buffer;
    using Async_Buffer = typename P::Async_Buffer;
    using MemoryTracker = typename P::MemoryTracker;

    struct OverlapResult {
        boost::container::small_vector<BufferId, 16> ids;
        DAddr begin;
        DAddr end;
        bool has_stream_leap = false;
    };

public:
    explicit BufferCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, Runtime& runtime_);

    ~BufferCache();

    void TickFrame();

    bool BindMultiRangeStorage(const Binding& binding, bool is_written,
                               std::span<const MultiRangeSegment> pool,
                               u64 write_mark_limit = ~u64{0});

    void ResolveMultiRangeStorage(Binding& binding, bool is_written,
                                  std::vector<MultiRangeSegment>& pool);

    void RefreshMultiRangeSegments(std::vector<MultiRangeSegment>& pool);

    void UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size);

    /// `keep_page_gpu_data`: download GPU-written bytes sharing the written pages first (false
    /// when the memory is going away).
    void WriteMemory(DAddr device_addr, u64 size, bool keep_page_gpu_data = true);

    /// EXP91 (fable52_answer_uniformkeeps stage 0, counting only): RasterizerVulkan::
    /// InnerInvalidation's ranges (FlushCaching's cached writes, EXP88's constant buffer data
    /// among them), noted first, before any early return. Drawing thread. Changes nothing bound.
    void NoteCachedWrites(std::span<const std::pair<DAddr, std::size_t>> sequences);

    /// EXP88 & 1 (zortos293/Lichii#1441 part c, fable52_answer_next70 3.1): why WriteMemory of
    /// [device_addr, device_addr + size) may change something, or None when it would only make
    /// its first three marks (NoteWriteUnlocked): no page ever left the CPU-modified state (so
    /// none is tracked, cached-written or GPU-modified), no GPU write ever marked one (Switch 2
    /// titles; so no gpu_modified_ranges there) and no enabled vertex binding overlaps the range
    /// (MarkVertexBuffersStale would act). Drawing thread only (v_buffer is its state), without
    /// the lock.
    enum class WriteReach : u8 { None, EverSynchronized, MayBeGpuWritten, VertexBinding };
    [[nodiscard]] WriteReach WriteReachesBuffers(DAddr device_addr, u64 size) const;

    /// EXP88: what WriteMemory does first, on a range WriteReachesBuffers answers None for: the
    /// command list note and the two page-flag marks the native bank check and the write watch
    /// read (g_buffer_write_pages, g_watch_write_pages). Atomics or their own mutex; no lock.
    void NoteWriteUnlocked(DAddr device_addr, u64 size);

    /// EXP88 & 4, under the lock: WriteMemory of the range would change nothing but its first
    /// three marks (no GPU-modified range on its pages, every page untouched, no vertex binding).
    [[nodiscard]] bool WriteChangesNothing(DAddr device_addr, u64 size);

    /// EXP88: whether ClaimCpuWrites may find anything in `ranges` (a page ever marked
    /// GPU-modified). Without the lock.
    [[nodiscard]] bool ClaimMayFindGpuWrites(std::span<const std::pair<DAddr, u64>> ranges) const;

    void CachedWriteMemory(DAddr device_addr, u64 size);

    /// Cancels pending downloads over the range (fences' and, with EDEN_NX2_EXP49, reads'), so
    /// none writes them over newer bytes. Those bytes reach guest memory later if at all (a GPU
    /// copy, a clear, an inline or a discarded write), so read downloads already taken land
    /// first: their reads get what a synchronous flush would have written.
    void ClearDownload(DAddr base_addr, u64 size);

    /// ClearDownload where guest memory gets the newer bytes before a read can find the range: a
    /// guest CPU write, a synchronous download landing under `mutex`, or a GPU-side write whose
    /// read downloads MemoryManager landed first. Read downloads leave the range rather than land,
    /// so this never waits for the host GPU.
    void ClearReplacedDownload(DAddr device_addr, u64 size);

    /// CPU writes collected but not yet applied (SetPendingCpuWrites).
    [[nodiscard]] Common::RangeSet<DAddr> PendingCpuWriteRanges();

    /// Writes downloaded bytes to guest memory, except over `cpu_written`. Callers take
    /// `cpu_written` inside a VideoCore::DownloadLanding over the bytes and write in it: the CPU
    /// writes collected after it store their bytes once the landing has ended.
    void WriteDownloadedMemory(const Common::RangeSet<DAddr>& cpu_written, DAddr device_addr,
                               const u8* data, u64 size);

    bool OnCPUWrite(DAddr device_addr, u64 size);

    void DownloadMemory(DAddr device_addr, u64 size);

    /// DownloadMemory for each range, waiting for the GPU once.
    void DownloadMemoryRanges(std::span<const std::pair<DAddr, u64>> ranges);

    /// EDEN_NX2_EXP49: DownloadMemory for a guest CPU read. Submits the copies without waiting
    /// for the host GPU, for FinishReadDownloads to write to guest memory. Their ranges stay
    /// GPU-modified until then, and another read of them waits for these copies (GetFlushArea:
    /// in_flight).
    void DownloadMemoryForRead(DAddr device_addr, u64 size);

    /// Writes the DownloadMemoryForRead copies over [device_addr, device_addr + size), and those
    /// submitted before them, to guest memory, waiting for the host GPU if it must. Any thread:
    /// the reading one before it reads, the GPU's side before it writes guest memory there. Takes
    /// `mutex` itself (a caller holding it keeps it held while the host GPU finishes). Returns how
    /// long it waited for the host GPU (at least 1 ns if it did), or nothing when no download was
    /// over the range.
    std::optional<u64> FinishReadDownloads(DAddr device_addr, u64 size);

    /// A GPU write into the buffers over [device_addr, device_addr + size), after the read
    /// downloads taken so far: they still write their older bytes to guest memory, as synchronous
    /// ones would have, but a later read takes the range again. MarkWrittenBuffer does this for
    /// the writes it records; the query cache's copies of its results call it themselves
    /// (`mutex` held).
    void OutdateReadDownloads(DAddr device_addr, u64 size);

    std::optional<VideoCore::RasterizerDownloadArea> GetFlushArea(DAddr device_addr, u64 size);

    /// Without downloads at every fence, a guest CPU read is the only way GPU-written buffer
    /// data reaches guest memory, so reads must ask the buffer cache (GetFlushArea).
    bool FlushesOnCpuRead() const {
        return !AsyncBufferDownloads();
    }

    bool InlineMemory(DAddr dest_address, size_t copy_size, std::span<const u8> inlined_buffer);

    void BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr, u32 size);

    void DisableGraphicsUniformBuffer(size_t stage, u32 index);

    void UpdateGraphicsBuffers(bool is_indexed);

    void UpdateComputeBuffers();

    void BindHostGeometryBuffers(bool is_indexed);

    void BindHostStageBuffers(size_t stage);

    void BindHostComputeBuffers();

    void SetUniformBuffersState(const std::array<u32, NUM_STAGES>& mask,
                                const UniformBufferSizes* sizes);

    void SetComputeUniformBufferState(u32 mask, const ComputeUniformBufferSizes* sizes);

    void UnbindGraphicsStorageBuffers(size_t stage);

    bool BindGraphicsStorageBuffer(size_t stage, size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                   bool is_written, u32 indirect_offset = ~0U);

    void UnbindGraphicsTextureBuffers(size_t stage);

    void BindGraphicsTextureBuffer(size_t stage, size_t tbo_index, GPUVAddr gpu_addr, u32 size,
                                   PixelFormat format, bool is_written, bool is_image);

    void UnbindComputeStorageBuffers();

    /// The compute program whose storage buffers are bound next (its write extents are checked
    /// per program, see ExtentTrust).
    void SetComputeShaderHash(u64 hash) noexcept {
        compute_shader_hash = hash;
    }

    /// Records the "after" copies of the write extent checks of the dispatch just recorded.
    void RecordComputeExtentProbes();

    /// write_extent: how far the program's stores reach past the pointer (native programs).
    void BindComputeStorageBuffer(size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                  bool is_written, u32 indirect_offset = ~0U,
                                  const Shader::StorageWriteExtent& write_extent = {});

    void UnbindComputeTextureBuffers();

    void BindComputeTextureBuffer(size_t tbo_index, GPUVAddr gpu_addr, u32 size, PixelFormat format,
                                  bool is_written, bool is_image);

    [[nodiscard]] std::pair<Buffer*, u32> ObtainBuffer(GPUVAddr gpu_addr, u32 size,
                                                       ObtainBufferSynchronize sync_info,
                                                       ObtainBufferOperation post_op);

    [[nodiscard]] std::pair<Buffer*, u32> ObtainCPUBuffer(DAddr gpu_addr, u32 size,
                                                          ObtainBufferSynchronize sync_info,
                                                          ObtainBufferOperation post_op);

    /// GPU work recorded now reads or writes these bytes, of a buffer obtained before (work set
    /// up once that keeps reading them, as conditional rendering does). Creates no buffer.
    void MarkUsed(DAddr device_addr, u64 size);
    void FlushCachedWrites();

    /// Where the CPU writes into GPU-written memory that were collected but not applied yet
    /// (WriteMemory) come from: downloads must not write the GPU's older copy over them.
    using PendingCpuWrites = std::function<void(const std::function<void(DAddr, size_t)>&)>;
    void SetPendingCpuWrites(PendingCpuWrites source) {
        pending_cpu_writes = std::move(source);
    }

    /// Takes CPU writes about to be applied one by one out of the GPU-written ranges first, so
    /// applying one doesn't download the GPU's copy over another (WriteMemory keeps the GPU data
    /// on the pages it writes).
    void ClaimCpuWrites(std::span<const std::pair<DAddr, u64>> ranges);

    /// EDEN_NX2_RUNAHEAD (nx2_runahead.h), back thread, mutex held: the guest CPU's bytes of
    /// [device_addr, device_addr + size) as a run-ahead chunk captured them (`bytes`). Registered
    /// memory gets them in its host buffer now, instead of a CPU-modified mark a later upload
    /// would read guest memory for (by then it may hold the next frame); the uniform bank
    /// snapshot of the page takes them too, so it keeps matching the host buffer. Memory no
    /// buffer holds is marked as before (a buffer made there later reads guest memory: a risk).
    void Nx2ApplyCpuBytes(DAddr device_addr, u64 size, const u8* bytes);

    /// Return true when there are uncommitted buffers to be downloaded
    [[nodiscard]] bool HasUncommittedFlushes() const noexcept;

    void AccumulateFlushes();

    /// Return true when the caller should wait for async downloads
    [[nodiscard]] bool ShouldWaitAsyncFlushes() const noexcept;

    /// Commit asynchronous downloads
    void CommitAsyncFlushes();
    void CommitAsyncFlushesHigh();

    /// Pop asynchronous downloads
    void PopAsyncFlushes();
    void PopAsyncBuffers();

    bool DMACopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount);

    bool DMAClear(GPUVAddr src_address, u64 amount, u32 value);

    /// Switch 2 GPU-written command lists (nx2.h, g_nx2_command_list_pages). While collecting (a
    /// compute dispatch or a DMA copy being recorded), writes to command list pages are noted;
    /// Nx2CopyListPages then records copies of those pages into host memory after the work.
    void Nx2CollectListCopies(bool collect) {
        nx2_collect_list_copies = collect;
        if (!collect) {
            nx2_list_copy_requests.clear();
        }
    }
    /// True when it recorded copies, which the GPU completes at `tick` (the caller submits).
    /// `writer`: the dispatch's shader hash, 0 for a DMA copy (EDEN_NX2_LIST_WAIT_TRACE).
    bool Nx2CopyListPages(u64 tick, u64 writer = 0);
    enum class Nx2ListRead { Unknown, Copied, Synced, NeedsSync };
    /// Front thread: Synced when guest memory holds the page `device_addr` lies in (flushed and
    /// written by no GPU work since), else Copied with `out` filled from the copy of its last
    /// write once the GPU is done with it. Unknown otherwise: the caller drains and flushes.
    /// Pages draws write are only Synced or Unknown, and NeedsSync until `past_sync`: the
    /// split's back ran the work before the last host-level wait (GpuSplit::WaitForSync).
    /// EXP40 & 1 (zortos293/Lichii#805 34656e8225): Synced too when no GPU work wrote the bytes
    /// [device_addr, device_addr + out.size()) since the page was flushed, or the host wrote them
    /// since; Copied reads the words GPU work wrote from the copy and, once the page was flushed,
    /// the others from guest memory.
    Nx2ListRead Nx2ReadListCopy(DAddr device_addr, std::span<u8> out, bool past_sync);
    /// Front thread, the split drained: the command list page was flushed to guest memory.
    void Nx2NoteListSynced(DAddr page);
    /// The split's back, the front waiting on it: downloads `pages` and every page draws wrote
    /// since it was last flushed with one GPU wait, and notes the latter synced. Each page's next
    /// lists then read guest memory instead of draining and flushing one page at a time.
    void Nx2FlushListPages(std::span<const DAddr> pages);
    /// CPU, inline and copy engine writes of bytes guest memory holds: they reach guest memory,
    /// so a synced page stays synced and the words they cover whole are guest memory's again, but
    /// the page's copy no longer holds the page.
    void Nx2NoteListHostWrite(DAddr device_addr, u64 size);

    /// Return true when a CPU region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(DAddr addr, size_t size);

    /// Visits the uniform and storage buffers bound for the last draw or dispatch with the cached
    /// host buffer holding each range, if one does, for EDEN_NX2_FRAME_DUMP_SAMPLED. Nothing is
    /// created or synchronized. Stage 5 is compute.
    template <typename Func>
    void ForEachBoundBuffer(bool compute, Func&& func) {
        const auto visit = [&](u32 stage, bool storage, u32 index, bool written,
                               const Binding& binding, u32 size) {
            if (binding.device_addr == 0 || size == 0) {
                return;
            }
            const Buffer* host = nullptr;
            u32 offset = 0;
            if (const BufferId id = page_table[binding.device_addr >> CACHING_PAGEBITS]) {
                const Buffer& buffer = slot_buffers[id];
                if (buffer.IsInBounds(binding.device_addr, size)) {
                    host = &buffer;
                    offset = buffer.Offset(binding.device_addr);
                }
            }
            func(stage, storage, index, written, binding, size, host, offset);
        };
        if (compute) {
            const auto* const sizes = channel_state->compute_uniform_buffer_sizes;
            ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
                const Binding& binding = channel_state->compute_uniform_buffers[index];
                visit(5, false, index, false, binding,
                      sizes ? std::min(binding.size, (*sizes)[index]) : binding.size);
            });
            ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
                const Binding& binding = channel_state->compute_storage_buffers[index];
                visit(5, true, index, ((channel_state->written_compute_storage_buffers >> index) & 1) != 0,
                      binding, binding.size);
            });
            return;
        }
        const auto* const sizes = channel_state->uniform_buffer_sizes;
        for (u32 stage = 0; stage < NUM_STAGES; ++stage) {
            ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
                const Binding& binding = channel_state->uniform_buffers[stage][index];
                visit(stage, false, index, false, binding,
                      sizes ? std::min(binding.size, (*sizes)[stage][index]) : binding.size);
            });
            ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
                const Binding& binding = channel_state->storage_buffers[stage][index];
                visit(stage, true, index,
                      ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0, binding,
                      binding.size);
            });
        }
    }

    /// Return true when a region is registered on the cache
    [[nodiscard]] bool IsRegionRegistered(DAddr addr, size_t size);

    /// Return true when a CPU region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(DAddr addr, size_t size);

    void SetDrawIndirect(
        const Tegra::Engines::Maxwell3D::DrawManager::IndirectParams* current_draw_indirect_) {
        current_draw_indirect = current_draw_indirect_;
    }

    [[nodiscard]] std::pair<Buffer*, u32> GetDrawIndirectCount();

    [[nodiscard]] std::pair<Buffer*, u32> GetDrawIndirectBuffer();

    template <typename Func>
    void BufferOperations(Func&& func) {
        do {
            channel_state->has_deleted_buffers = false;
            func();
        } while (channel_state->has_deleted_buffers);
    }

    Common::SpinRecursiveMutex mutex;
    /// EXP86 (fable52_answer_gpuwritten): per-page mirrors of gpu_modified_ranges, the tracker's
    /// Preflushable bits and read_trapped_ranges, all changed under `mutex`, with a sequence
    /// counter that is odd from a hold's first change to its release (the mutex's release hook).
    GpuWrittenMirror gpu_written_mirror{mutex};

    /// EXP86: what the locked GetFlushArea would do for a guest read of [addr, addr + size), from
    /// the mirrors and without the lock. Clean: it would answer the preemptive page-aligned area
    /// and change nothing (EXP81's stamps aside). Anything else: take the locked path.
    [[nodiscard]] MirrorAnswer MirrorCleanForRead(DAddr addr, u64 size) const {
        if (!PageBitmap::InRange(Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
                                 Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE) -
                                     Common::AlignDown(addr, Core::DEVICE_PAGESIZE))) {
            return MirrorAnswer::Broken;
        }
        const u64 seq = gpu_written_mirror.seq.Load();
        if ((seq & 1) != 0) {
            return MirrorAnswer::Busy;
        }
        const DAddr begin = Common::AlignDown(addr, Core::DEVICE_PAGESIZE);
        const u64 length = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE) - begin;
        const bool modified = gpu_written_mirror.gpu_modified_pages.Any(begin, length);
        // "Any page", as IsRegionPreflushable asks.
        const bool marked = gpu_written_mirror.preflush_pages.Any(begin, length);
        const bool armed = gpu_written_mirror.trap_pages.Any(begin, length);
        if (gpu_written_mirror.seq.Load() != seq) {
            return MirrorAnswer::Busy;
        }
        if (gpu_written_mirror.seq.Broken()) {
            return MirrorAnswer::Broken;
        }
        if (modified) {
            return MirrorAnswer::Modified;
        }
        if (armed && !marked) {
            return MirrorAnswer::ArmedUnmarked;
        }
        // F2 counter: a read answered here skips GetFlushArea (verify mode's calls are reads too).
        if (DownloadReadStats::g_enabled) [[unlikely]] {
            DownloadReadStats::Read(begin, begin + length);
        }
        return MirrorAnswer::Clean;
    }

    /// EXP86 verify (`mutex` held, before GetFlushArea): whether GetFlushArea leaves the read
    /// traps alone for this read: a Preflushable mark on its pages, or nothing armed there.
    [[nodiscard]] bool ReadLeavesTrapsAlone(DAddr addr, u64 size) {
        const DAddr begin = Common::AlignDown(addr, Core::DEVICE_PAGESIZE);
        const DAddr end = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE);
        return memory_tracker.IsRegionPreflushable(addr, size) ||
               !read_trapped_ranges.Intersects(begin, end - begin);
    }

    Runtime& runtime;
    bool any_buffer_uploaded = false;
    /// The guest words of each page of native programs' indirectly read banks when last checked:
    /// unchanged pages keep their tracking (marking them CPU-modified costs two mprotect calls
    /// per page).
    struct NativeBankPage {
        std::array<u8, Core::DEVICE_PAGESIZE> words;
        /// EDEN_NX2_EXP4 & 2: the bytes of the page compared during draw `checked_draw`.
        u64 checked_draw = 0;
        u32 checked_begin = 0;
        u32 checked_end = 0;
        /// EDEN_NX2_EXP9: frame_tick of the page's last compare, and whether the next bind
        /// compares it again (a flagged write's bytes may land after the compare that took it).
        u64 checked_frame = ~u64{0};
        /// frame_tick when the page was last seen changed: such pages are compared on every bind.
        u64 changed_frame = ~u64{0};
        bool recheck = false;
    };
    /// Node map: NoteNativeBankPageChanged keeps pointers to earlier pages' words until the bind's
    /// uploads are flushed, so entries must not move (std::unordered_map's lookups were ~2% of the
    /// split's back at Castle Town, 600k a second).
    boost::unordered_node_map<DAddr, NativeBankPage> native_indirect_page_snapshots;
    /// EDEN_NX2_BANK_PAGE_TABLE (live, default 0): the snapshots in a page-indexed table instead
    /// of the node map (the shape of Hookshot (NotRexed/Hookshot) bf13340's write-back pages: 2 MiB
    /// regions of 512 page slots under a flat top level, the last region looked up kept). Pages
    /// are allocated one by one, so their words stay where they are, as in the node map.
    static constexpr u64 NativeBankRegionBits = 21;
    static constexpr u64 NativeBankRegionPages = u64{1}
                                                 << (NativeBankRegionBits - Core::DEVICE_PAGEBITS);
    static constexpr u64 NativeBankAddressBits = Core::DevicePageFlags::AddressBits;
    struct NativeBankRegion {
        std::array<std::unique_ptr<NativeBankPage>, NativeBankRegionPages> pages;
        /// A bit per page slot holding a snapshot.
        std::array<u64, NativeBankRegionPages / 64> present{};
        /// A bit per page whose NativeBankPage::recheck is set (kept with it).
        std::array<u64, NativeBankRegionPages / 64> recheck{};
    };
    using NativeBankRegionTable =
        std::array<std::unique_ptr<NativeBankRegion>,
                   u64{1} << (NativeBankAddressBits - NativeBankRegionBits)>;
    std::unique_ptr<NativeBankRegionTable> native_bank_regions;
    /// The regions allocated in native_bank_regions, to clear them.
    std::vector<u32> native_bank_region_list;
    size_t native_bank_table_pages = 0;
    u64 native_bank_last_region_index = ~u64{0};
    NativeBankRegion* native_bank_last_region = nullptr;
    /// Which store holds the snapshots: the switch as last seen (a change empties both).
    bool native_bank_table_on = false;
    /// A page's snapshot, and the region holding it with the table (null with the node map).
    struct NativeBankSnapshotRef {
        NativeBankPage* page;
        NativeBankRegion* region;
        bool inserted;
    };
    /// Counts graphics draws (UpdateGraphicsBuffers) for EDEN_NX2_EXP4 & 2.
    u64 native_bank_draw = 0;
    /// EDEN_NX2_EXP4 != 0: bank check cost, logged every 5 s.
    struct NativeBankCheckStats {
        u64 binds = 0;
        u64 bytes_compared = 0;
        u64 bytes_trimmed = 0;
        u64 pages_compared = 0;
        u64 pages_skipped = 0;
        u64 pages_changed = 0;
        std::chrono::steady_clock::time_point last_log{};
    } native_bank_stats;
    /// EDEN_NX2_EXP9 != 0: why bank pages were compared and which of them had changed, by the
    /// first reason that applies (NativeBankReason), logged every 5 s.
    /// EDEN_NX2_EXP9: a page that changed within this many frames is compared on every bind.
    static constexpr u64 HotFrames = 120;
    enum NativeBankReason : size_t {
        BankCollected, ///< Collected CPU writes (Core::g_collected_write_pages)
        BankWritten,   ///< Writes the buffer cache saw (Core::g_buffer_write_pages)
        BankRecheck,   ///< Flagged at the previous bind
        BankCpuDirty,  ///< CPU-modified in the tracker
        BankHot,       ///< Changed within the last HotFrames frames (per-draw data)
        BankFrame,     ///< Not compared yet this frame
        BankNone,      ///< No reason: EXP9=2 compares these only to verify
        BankReasons,
    };
    struct NativeBankTrackedStats {
        u64 binds = 0;
        u64 pages_new = 0;
        u64 pages_skipped = 0;
        std::array<u64, BankReasons> compared{};
        std::array<u64, BankReasons> changed{};
        /// EDEN_NX2_BANK_SLOTS: binds whose page loop was skipped whole, and pages of those
        /// slots the hot rule would have compared.
        u64 banks_skipped = 0;
        u64 hot_left = 0;
        /// EDEN_NX2_BANK_SKIP_VERIFY: pages compared anyway, those found changed, and pages of a
        /// skipped bank the loop had a reason for.
        u64 skip_verified = 0;
        u64 skip_missed = 0;
        u64 skip_disagreed = 0;
        std::chrono::steady_clock::time_point last_log{};
    } native_bank_tracked_stats;
    /// EDEN_NX2_BANK_SLOT_STATS: the tracked check's counts per constant buffer index (c0..c17),
    /// logged and cleared with native_bank_tracked_stats.
    struct NativeBankSlotStats {
        u64 binds = 0;
        u64 pages = 0;     ///< pages the binds covered (mapped)
        u64 pages_new = 0; ///< pages first seen (no snapshot yet)
        u32 stages = 0;    ///< stages seen binding the index (bit per stage)
        u64 banks_skipped = 0; ///< EDEN_NX2_BANK_SLOTS: page loops skipped whole
        u64 hot_left = 0;      ///< EDEN_NX2_BANK_SLOTS: hot pages not compared
        std::array<u64, BankReasons> compared{};
        std::array<u64, BankReasons> changed{};
    };
    std::array<NativeBankSlotStats, Tegra::Engines::Maxwell3D::Regs::MaxConstBuffers>
        native_bank_slot_stats{};
    /// EDEN_NX2_EXP26 & 4: changed bank pages of one bind, copied to the host buffer from their
    /// snapshots (FlushNativeBankUploads) instead of being marked CPU-modified. A 64 KiB bank
    /// spans at most 17 pages.
    struct NativeBankUpload {
        DAddr page;
        const u8* words;
    };
    static constexpr size_t NativeBankUploadCapacity = 32;
    struct NativeBankUploads {
        /// Only the first `count` are set (left uninitialized: one of these per bank bind).
        std::array<NativeBankUpload, NativeBankUploadCapacity> pages;
        size_t count = 0;
    };

private:
    template <typename Mask, typename Func>
    static void ForEachEnabledBit(Mask enabled_mask, Func&& func) {
        for (u32 index = 0; enabled_mask != 0; ++index, enabled_mask >>= 1) {
            const int disabled_bits = std::countr_zero(enabled_mask);
            index += disabled_bits;
            enabled_mask >>= disabled_bits;
            func(index);
        }
    }

    template <typename Func>
    void ForEachBufferInRange(DAddr device_addr, u64 size, Func&& func) {
        const u64 page_end = Common::DivCeil(device_addr + size, CACHING_PAGESIZE);
        for (u64 page = device_addr >> CACHING_PAGEBITS; page < page_end;) {
            const BufferId buffer_id = page_table[page];
            if (!buffer_id) {
                ++page;
                continue;
            }
            Buffer& buffer = slot_buffers[buffer_id];
            func(buffer_id, buffer);

            const DAddr end_addr = buffer.CpuAddr() + buffer.SizeBytes();
            page = Common::DivCeil(end_addr, CACHING_PAGESIZE);
        }
    }

    static bool IsRangeGranular(DAddr device_addr, size_t size) {
        return (device_addr & ~Core::DEVICE_PAGEMASK) ==
               ((device_addr + size) & ~Core::DEVICE_PAGEMASK);
    }

    void RunGarbageCollector(bool aggressive_gc);

    void BindHostIndexBuffer();

    void BindHostVertexBuffers();

    void BindHostDrawIndirectBuffers();

    void BindHostGraphicsUniformBuffers(size_t stage);

    void BindHostGraphicsUniformBuffer(size_t stage, u32 index, u32 binding_index, bool needs_bind);

    /// EDEN_NX2_EXP9: the native uniform bank check of a bind of [device_addr, device_addr + size).
    /// EDEN_NX2_EXP26 (bits, see Experiment in nx2_debug_control.h): `trims`; with & 4 changed
    /// pages that lie whole in `buffer` are uploaded from their snapshots.
    /// `stage` and `index`: the binding's shader stage and constant buffer index
    /// (EDEN_NX2_BANK_SLOT_STATS).
    void CheckNativeBankPagesTracked(DAddr device_addr, u32 size, int mode, int trims,
                                     Buffer& buffer, size_t stage, u32 index);

    /// A native bank page whose snapshot was just (re)taken: marked CPU-modified, or with
    /// EXP26 & 4 (`upload_to` not null) queued in `uploads` when it lies whole in that buffer and
    /// holds no GPU-written data.
    void NoteNativeBankPageChanged(DAddr page, const u8* words, Buffer* upload_to,
                                   NativeBankUploads& uploads);

    /// EDEN_NX2_EXP26 & 4: copies the queued pages' snapshots into `buffer`.
    void FlushNativeBankUploads(Buffer& buffer, const NativeBankUploads& uploads);

    /// EDEN_NX2_BANK_PAGE_TABLE: picks the snapshot store from the switch (emptying both when it
    /// changed) and empties it once it holds 16384 pages, as before each bank check.
    void SelectNativeBankStore();
    /// The snapshot of `page` (page aligned), made (words left unset) when there is none.
    [[nodiscard]] NativeBankSnapshotRef NativeBankSnapshot(DAddr page);
    /// The snapshot of `page` (page aligned), or null.
    [[nodiscard]] NativeBankPage* FindNativeBankSnapshot(DAddr page);
    void ClearNativeBankSnapshots();
    /// EDEN_NX2_BANK_SLOTS, with the table only: whether every page of [first_page, end) (page
    /// aligned) has a snapshot and none waits for a recheck. False with the node map.
    [[nodiscard]] bool NativeBankPagesSettled(DAddr first_page, DAddr end);
    /// The tracked check's 5 s report (every 1024 binds it looks at the clock).
    void LogNativeBankTrackedStats(int mode);
    /// Sets the page's recheck flag (and its region bit with the table).
    static void SetNativeBankRecheck(const NativeBankSnapshotRef& ref, DAddr page, bool value) {
        ref.page->recheck = value;
        if (ref.region != nullptr) {
            const u64 slot = (page >> Core::DEVICE_PAGEBITS) & (NativeBankRegionPages - 1);
            const u64 bit = u64{1} << (slot % 64);
            if (value) {
                ref.region->recheck[slot / 64] |= bit;
            } else {
                ref.region->recheck[slot / 64] &= ~bit;
            }
        }
    }

    void BindHostGraphicsStorageBuffers(size_t stage);

    void BindHostGraphicsTextureBuffers(size_t stage);

    void BindHostTransformFeedbackBuffers();

    void BindHostVertexBuffer(u32 index, Buffer& buffer, u32 offset, u32 size, u32 stride);

    void BindHostComputeUniformBuffers();

    void BindHostComputeStorageBuffers();

    void BindHostComputeTextureBuffers();

    void DoUpdateGraphicsBuffers(bool is_indexed);

    void DoUpdateComputeBuffers();

    void UpdateIndexBuffer();

    void UpdateVertexBuffers();

    void UpdateVertexBuffer(u32 index);

    void UpdateDrawIndirect();

    void UpdateUniformBuffers(size_t stage);

    void UpdateStorageBuffers(size_t stage);

    void UpdateTextureBuffers(size_t stage);

    void UpdateTransformFeedbackBuffers();

    void UpdateTransformFeedbackBuffer(u32 index);

    void UpdateComputeUniformBuffers();

    void UpdateComputeStorageBuffers();

    void UpdateComputeTextureBuffers();

    /// `capped`: `size` is EDEN_NX2_WRITTEN_SSBO_MARK_KIB's cap, below the binding's size, so the
    /// GPU may write past it (Nx2NoteListWrite).
    void MarkWrittenBuffer(BufferId buffer_id, DAddr device_addr, u32 size, bool needs_sync = false,
                           bool capped = false);

    /// Counts a GPU write to the command list pages in the range (invalidating their copies),
    /// notes the words it may write (Nx2ListPage::gpu_words) and, while collecting, asks for new
    /// copies. `to_page_end`: the GPU may write past the range (a capped mark), so the words up to
    /// the end of its last page count as written too.
    void Nx2NoteListWrite(BufferId buffer_id, DAddr device_addr, u64 size,
                          bool to_page_end = false);

    [[nodiscard]] BufferId FindBuffer(DAddr device_addr, u32 size, bool sparse_compatible);

    void SynchronizeBufferWrites(Buffer& buffer);

    [[nodiscard]] OverlapResult ResolveOverlaps(DAddr device_addr, u32 wanted_size);

    /// Returns whether it copied (false: skipped as wholly CPU-modified, EDEN_NX2_JOIN_SKIP_CPU).
    bool JoinOverlap(BufferId new_buffer_id, BufferId overlap_id, bool accumulate_stream_score);
    /// The range's host contents are all stale and nothing else reads them (JoinOverlap).
    [[nodiscard]] bool JoinCopyIsDead(DAddr device_addr, u64 size);

    [[nodiscard]] BufferId CreateBuffer(DAddr device_addr, u32 wanted_size,
                                        bool sparse_compatible);

    void Register(BufferId buffer_id);

    void Unregister(BufferId buffer_id);

    template <bool insert>
    void ChangeRegister(BufferId buffer_id);

    void TouchBuffer(Buffer& buffer, BufferId buffer_id) noexcept;

    bool SynchronizeBuffer(Buffer& buffer, DAddr device_addr, u32 size);

    /// EDEN_NX2_EXP59 & 8: before the full walk, counts and logs what the dirty-word walk would
    /// have left out. Returns whether the dirty-word walk would have walked anything.
    bool VerifyComputeDirtyWalk(DAddr device_addr, u32 size);

    /// EDEN_NX2_STORAGE_SYNC_SCAN=2: before the full walk, counts and logs (every 10 s) the CPU
    /// and unprotected pages the scanned walk would have left out.
    void VerifyStorageSyncScan(DAddr device_addr, u32 size);

    /// EDEN_NX2_STORAGE_SYNC_MEMO: SynchronizeBuffer, skipped when the range's memo still holds;
    /// stores the memo when the synchronize left the range clean. `verify` (=2): a hit checks
    /// the range and synchronizes anyway. Returns what SynchronizeBuffer returns (true: nothing
    /// uploaded).
    bool SynchronizeStorageMemo(Buffer& buffer, DAddr device_addr, u32 size, bool verify);

    void UploadMemory(Buffer& buffer, u64 total_size_bytes, u64 largest_copy,
                      std::span<BufferCopy> copies);

    void ImmediateUploadMemory(Buffer& buffer, u64 largest_copy,
                               std::span<const BufferCopy> copies);

    void MappedUploadMemory(Buffer& buffer, u64 total_size_bytes, std::span<BufferCopy> copies);

    void DownloadBufferMemory(Buffer& buffer_id);

    void DownloadBufferMemory(Buffer& buffer_id, DAddr device_addr, u64 size);

    /// Downloads the GPU-written data of several buffers with one wait for the GPU.
    void DownloadBuffersMemory(std::span<const BufferId> buffer_ids);

    void DownloadRangesMemory(std::span<const std::pair<DAddr, u64>> ranges);

    /// ClearDownload, for the downloads taken at fences only.
    void ClearFenceDownloads(DAddr device_addr, u64 size);

    /// Whether every GPU-modified byte in the range is in a read download's clean_ranges.
    [[nodiscard]] bool ReadDownloadsCover(DAddr device_addr, u64 size) const;

    /// Writes the read downloads up to `tick` to guest memory, oldest first (`mutex` held, the
    /// host GPU past `tick`).
    void WriteReadDownloads(u64 tick);

    /// WriteReadDownloads for those the host GPU has completed: nothing waits.
    void WriteCompletedReadDownloads();

    /// WriteReadDownloads up to the newest read download over the range, if the host GPU ran it.
    /// Returns whether it did.
    bool WriteFinishedReadDownloads(DAddr device_addr, u64 size);

    /// Records and submits a read download of the GPU-modified bytes over `ranges` that no read
    /// download brings yet.
    void RecordReadDownload(std::span<const std::pair<DAddr, u64>> ranges);

    void DeleteBuffer(BufferId buffer_id, bool do_not_mark = false);

    [[nodiscard]] Binding StorageBufferBinding(GPUVAddr ssbo_addr, u32 cbuf_index, bool is_written,
                                               bool native_pointer = false,
                                               bool table_pointer = false,
                                               bool compute = false) const;

    void TraceTablePointer(bool compute, size_t stage, size_t ssbo_index, u32 cbuf_index,
                           u32 cbuf_offset, u32 indirect_offset, GPUVAddr holder,
                           const Binding& binding);

    [[nodiscard]] TextureBufferBinding GetTextureBufferBinding(GPUVAddr gpu_addr, u32 size,
                                                               PixelFormat format);

    [[nodiscard]] std::span<const u8> ImmediateBufferWithData(DAddr device_addr, size_t size);

    [[nodiscard]] std::span<u8> ImmediateBuffer(size_t wanted_capacity);

    [[nodiscard]] bool HasFastUniformBufferBound(size_t stage, u32 binding_index) const noexcept;

    [[nodiscard]] Binding& VertexBufferSlot(u32 index);

    [[nodiscard]] const Binding& VertexBufferSlot(u32 index) const;

    void UpdateVertexBufferSlot(u32 index, const Binding& binding);

    /// Flags the vertex buffers overlapping a written range so host copies derived from them are
    /// rebuilt.
    void MarkVertexBuffersStale(DAddr device_addr, u64 size);

    /// Re-dirties repacked vertex buffers whose source data was written since they were bound.
    void RefreshRepackedVertexBuffers();

    void InlineMemoryImplementation(DAddr dest_address, size_t copy_size,
                                    std::span<const u8> inlined_buffer);

    Tegra::MaxwellDeviceMemoryManager& device_memory;

    Common::SlotVector<Buffer> slot_buffers;
#ifdef YUZU_LEGACY
    static constexpr size_t TICKS_TO_DESTROY = 6;
#else
    static constexpr size_t TICKS_TO_DESTROY = 8;
#endif
    DelayedDestructionRing<Buffer, TICKS_TO_DESTROY> delayed_destruction_ring;

    const Tegra::Engines::Maxwell3D::DrawManager::IndirectParams* current_draw_indirect{};

    u32 last_index_count = 0;

    u32 enabled_vertex_buffers_mask = 0;
    u32 stale_vertex_buffers_mask = 0;
    u64 vertex_buffers_serial = 0;
    std::array<Binding, 32> v_buffer{};

    boost::container::small_vector<BufferCopy, 4> upload_copies;

    MemoryTracker memory_tracker;
    Common::RangeSet<DAddr> uncommitted_gpu_modified_ranges;

    /// Whether GPU-written buffer ranges are copied back to guest memory at the next fence;
    /// otherwise only when something flushes them (as upstream yuzu below High GPU accuracy).
    /// Not for Switch 2 titles: title 1 had 250-330 MB/s copied back (a fence per compute
    /// dispatch, the fencing thread holding the buffer cache while it wrote them), while its CPU
    /// read GPU-written memory at most 64 times in 200 s (counted with reactive flushing) and
    /// the native MME's DMA reads never touched those pages. Without them a fence costs ~1 us
    /// instead of ~5 and draw setup ~7% less. EDEN_NX2_BUFFER_DOWNLOADS=0/1 forces it.
    bool AsyncBufferDownloads() const {
        static const int forced = [] {
            const char* value = std::getenv("EDEN_NX2_BUFFER_DOWNLOADS");
            return value == nullptr ? -1 : (value[0] == '0' ? 0 : 1);
        }();
        if (forced >= 0) {
            return forced == 1;
        }
        return gpu_memory == nullptr || !gpu_memory->UsesNativeTiling();
    }

    /// Experiment, EDEN_NX2_WRITTEN_SSBO_MARK_KIB=<n>: marks at most n KiB of a written Switch 2
    /// storage buffer binding GPU-written. Native pointers carry no size, so a written binding
    /// spans up to 8 MiB of its mapping and arms read traps and fence downloads over CPU data
    /// the shader never writes. Writes past the cap are not tracked, so this only measures how
    /// much of the marked range the GPU really writes.
    u32 WrittenStorageMarkSize(u32 size) const {
        static const u64 cap = [] {
            const char* value = std::getenv("EDEN_NX2_WRITTEN_SSBO_MARK_KIB");
            return value == nullptr ? u64{0} : std::strtoull(value, nullptr, 10) * 1024;
        }();
        if (cap == 0 || gpu_memory == nullptr || !gpu_memory->UsesNativeTiling()) {
            return size;
        }
        return static_cast<u32>((std::min<u64>)(size, cap));
    }

    /// Write extent self-check (fable_answer_25; Ocarina of Time's glowing triangles,
    /// 2026-11-09). The store analysis bounds how far a native compute program writes, and only
    /// that much is marked GPU-written; one program writes past it, and the CPU then read stale
    /// guest bytes back (particle positions turned into one right vertex and garbage). Each program
    /// is marked in full until a dispatch at its dims was checked: its tail past the proof, as
    /// the GPU left it after that one dispatch, is compared with guest memory from right before
    /// it (known GPU writes and unapplied CPU writes left out). The same: the proof holds at these
    /// dims (and at smaller ones). Different: the program is marked in full for the session and
    /// logged, naming it. EDEN_NX2_EXP45=1 (live) skips the check (the proof is trusted as before).
    struct ExtentTrust {
        std::array<u32, 3> block{};
        std::array<u32, 3> grid{};
        bool distrusted{};
        bool probe_pending{};
    };
    struct ExtentProbeRequest {
        bool active{};
        u64 proven{};
        u64 pointer{};
        u64 pad{};
        std::array<u32, 3> block{};
        std::array<u32, 3> grid{};
    };
    struct ExtentProbe {
        u64 shader_hash{};
        BufferId buffer_id{};
        DAddr device_addr{}; ///< The tail's start (in buffer_id)
        u64 binding_size{};
        ExtentProbeRequest request{};
        u64 span{};
        /// Guest memory of the tail at bind (fable_answer_26). Under the full mark the CPU's reads
        /// bring the GPU's bytes into guest memory, so only a copy made before the dispatch shows
        /// what the CPU would have read without the mark.
        std::vector<u8> guest;
        /// Tail offsets [begin, end) left out of the compare: known GPU writes and CPU writes not
        /// applied to the buffer yet.
        std::vector<std::pair<u64, u64>> skip;
        std::optional<Async_Buffer> after;
        u64 tick{};
    };
    u64 compute_shader_hash{};
    std::unordered_map<u64, ExtentTrust> extent_trust;
    std::array<ExtentProbeRequest, NUM_STORAGE_BUFFERS> extent_probe_requests{};
    std::vector<ExtentProbe> extent_probes;
    u64 extent_probes_trusted{};
    u64 extent_probes_distrusted{};
    u64 extent_full_marks{};

    void CheckExtentProbes();
    void ProbeExtentTail(u32 index, const Binding& binding);

    /// Whether a GPU write to this range is copied back at the next fence. Ranges the guest CPU
    /// has read (Preflushable, set by the first reactive flush) always are: later reads skip the
    /// flush and rely on these downloads, e.g. collision meshes skinned by compute shaders.
    bool DownloadsAtFence(DAddr device_addr, u64 size) noexcept {
        return AsyncBufferDownloads() || memory_tracker.IsRegionPreflushable(device_addr, size);
    }

    /// Queues a GPU write for download at the next fence where DownloadsAtFence holds. With only
    /// Preflushable marks, just the marked pages are queued: a write that covers one marked page
    /// of a large buffer would otherwise copy the whole buffer back at every fence.
    void QueueFenceDownload(DAddr device_addr, u64 size) {
        static const bool exact = [] {
            const char* value = std::getenv("EDEN_NX2_EXACT_PREFLUSH");
            return value == nullptr || value[0] != '0';
        }();
        if (AsyncBufferDownloads()) {
            uncommitted_gpu_modified_ranges.Add(device_addr, size);
            return;
        }
        // Preflushed writes arm the read traps too: their data reaches guest memory only when
        // the fence's download lands, and a read before that saw stale words (Ocarina of Time's
        // whole screen flashed white at the sword boulder, ~180 times in 30 s). The trapped read
        // flushes on demand only while the data is still in flight (GetFlushArea).
        if (!exact) {
            if (memory_tracker.IsRegionPreflushable(device_addr, size)) {
                uncommitted_gpu_modified_ranges.Add(device_addr, size);
            }
            ArmReadTraps(device_addr, size);
            return;
        }
        const DAddr end = device_addr + size;
        u64 queued = 0;
        memory_tracker.ForEachPreflushableRange(device_addr, size, [&](DAddr begin, u64 length) {
            const DAddr lower = (std::max)(begin, device_addr);
            const DAddr upper = (std::min)(begin + length, end);
            if (lower < upper) {
                uncommitted_gpu_modified_ranges.Add(lower, upper - lower);
                queued += upper - lower;
            }
        });
        ArmReadTraps(device_addr, size);
        if (Nx2DownloadTrace::g_enabled && queued != 0) [[unlikely]] {
            Nx2DownloadTrace::Queued(device_addr, size, queued);
        }
    }

    /// GPU writes that are not downloaded at fences: trap the guest CPU's next read of their
    /// pages, which flushes them on demand (or marks them Preflushable with preflushing enabled).
    void ArmReadTraps(DAddr device_addr, u64 size) {
        const DAddr begin = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
        const DAddr end = Common::AlignUp(device_addr + size, Core::DEVICE_PAGESIZE);
        if (!read_trapped_ranges.Arm(begin, end - begin)) {
            return;
        }
        device_memory.ArmReadTraps(begin, end - begin);
    }

    ReadTrapTracker read_trapped_ranges;
    /// Ranges the guest CPU read after GPU writes. A read that must flush downloads the GPU
    /// writes in all of them, so the reads that usually follow it find nothing left to flush.
    Common::RangeSet<DAddr> cpu_read_ranges;
    /// EXP69: the ranges GetFlushArea marked Preflushable since the last reset.
    Common::RangeSet<DAddr> preflush_marked_ranges;
    u64 preflush_reset_bytes{};
    /// EXP81: frame_tick of each device page's last CPU read (GetFlushArea), and bytes the aged
    /// reset kept marked because they were read within the last EXP81 frames.
    std::unordered_map<u64, u64> preflush_read_frames;
    u64 preflush_kept_bytes{};
    /// fable52_answer_bosswave 3 (counter before the change): preflushed reads that flush
    /// although every GPU-modified byte of theirs is in a fence download already on its way
    /// (none uncommitted), and how many fences behind the newest that download was.
    std::atomic<u64> preflush_reads_flushed{};
    std::atomic<u64> preflush_reads_in_flight{};
    std::atomic<u64> preflush_reads_in_flight_depth{};
    /// EDEN_NX2_EXP72: fence downloads with copies, numbered from 1 in commit order; landed counts
    /// the ones PopAsyncBuffers wrote to guest memory (pending_download_ranges[i] is landed + i + 1).
    std::atomic<u64> fence_downloads_landed{};
    std::mutex fence_landed_mutex;
    std::condition_variable fence_landed_cv;
    std::atomic<u64> fence_waits{};
    std::atomic<u64> fence_wait_ns{};
    std::atomic<u64> fence_wait_timeouts{};

public:
    /// EDEN_NX2_EXP72 (fable52_answer_bosswave 3): a guest read whose GPU-written bytes are all in
    /// a submitted fence download waits for that download to land instead of a new copy queued
    /// behind everything since (11-17 fences at the night market). No cache lock held.
    bool WaitFenceDownload(u64 sequence, std::chrono::milliseconds timeout) {
        const auto start = std::chrono::steady_clock::now();
        std::unique_lock lock{fence_landed_mutex};
        const bool done = fence_landed_cv.wait_for(lock, timeout, [&] {
            return fence_downloads_landed.load(std::memory_order_acquire) >= sequence;
        });
        lock.unlock();
        fence_waits.fetch_add(1, std::memory_order_relaxed);
        fence_wait_ns.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                     std::chrono::steady_clock::now() - start)
                                                     .count()),
                                std::memory_order_relaxed);
        if (!done) {
            fence_wait_timeouts.fetch_add(1, std::memory_order_relaxed);
        }
        return done;
    }

private:
    u64 preflush_resets{};
    bool download_cpu_read_ranges{};
    /// Keeps gpu_written_mirror.gpu_modified_pages (EXP86) at every Add, Subtract and Clear.
    MirroredRangeSet<DAddr> gpu_modified_ranges;
    std::deque<Common::RangeSet<DAddr>> committed_gpu_modified_ranges;

    // Async Buffers
    std::deque<std::optional<Async_Buffer>> async_buffers;
    std::deque<boost::container::small_vector<BufferCopy, 4>> pending_downloads;
    /// For each entry of pending_downloads: the ranges it still writes to guest memory when its
    /// fence pops. A write to a range after a download was taken removes it from that download
    /// (and every other pending one): its bytes are older than the write.
    std::deque<Common::RangeSet<DAddr>> pending_download_ranges;
    PendingCpuWrites pending_cpu_writes;
    std::optional<Async_Buffer> current_buffer;

    /// EDEN_NX2_EXP49: a DownloadMemoryForRead, its copies, and what is left of them to do once
    /// the host GPU has run them.
    struct ReadDownload {
        u64 tick{}; ///< The submission with the copies
        Async_Buffer staging{};
        /// src_offset: the device address copied; dst_offset: where it is in `staging`.
        boost::container::small_vector<BufferCopy, 4> copies;
        /// Still to write to guest memory: a newer value there (ClearReplacedDownload) takes its
        /// bytes out. A newer GPU write doesn't: until a download of its own lands, guest memory
        /// is to hold what a synchronous flush would have left.
        Common::RangeSet<DAddr> write_ranges;
        /// No longer GPU-modified once written: a newer GPU write (OutdateReadDownloads) or value
        /// in guest memory takes its bytes out.
        Common::RangeSet<DAddr> clean_ranges;
    };
    /// Oldest first, and written to guest memory in that order: none lands over newer bytes.
    std::deque<ReadDownload> read_downloads;
    /// All the read downloads' clean_ranges (they don't overlap): GPU-modified bytes already on
    /// their way, which later reads wait for rather than download again.
    Common::RangeSet<DAddr> read_download_ranges;

    std::deque<Async_Buffer> async_buffers_death_ring;

    size_t immediate_buffer_capacity = 0;
    Common::ScratchBuffer<u8> immediate_buffer_alloc;
    std::vector<u8> guest_uint8_indices;
    /// Guest memory large uploads came from since the last cold hint (device address, size).
    std::vector<std::pair<DAddr, u64>> uploaded_buffer_ranges;

    struct LRUItemParams {
        using ObjectType = BufferId;
        using TickType = u64;
    };
    Common::LeastRecentlyUsedCache<LRUItemParams> lru_cache;
    VirtualRangeCache virtual_ranges;
    std::vector<MultiRangeSegment> graphics_segments;
    std::vector<MultiRangeSegment> compute_segments;
    /// EDEN_NX2_GPU_PROFILE: SynchronizeBuffer counts its uploads as compute storage uploads
    /// (set in BindHostComputeStorageBuffers).
    bool count_compute_sync = false;
    /// EDEN_NX2_EXP59 bits 4 and 8 while BindHostComputeStorageBuffers runs (0 otherwise):
    /// SynchronizeBuffer walks only the dirty tracker words (4), or walks all and checks (8).
    int compute_dirty_walk = 0;
    /// EXP59 & 8: synchronizes checked, and those where the dirty-word walk would have differed,
    /// since the last totals line (VerifyComputeDirtyWalk).
    struct DirtyWalkVerify {
        u64 checked = 0;
        u64 mismatches = 0;
        u64 cpu_pages = 0;
        u64 untracked_pages = 0;
        u64 total_checked = 0;
        u64 total_mismatches = 0;
        u32 logged = 0;
        std::chrono::steady_clock::time_point last_report{};
    } dirty_walk_verify;
    /// EDEN_NX2_STORAGE_SYNC_SCAN while a storage bind runs (BindHostGraphicsStorageBuffers, and
    /// BindHostComputeStorageBuffers without EXP59), 0 otherwise: 1 = SynchronizeBuffer walks the
    /// scanned windows only, 2 = it walks all after counting what the scan would leave out.
    int storage_sync_scan = 0;
    /// EDEN_NX2_STORAGE_SYNC_SCAN=2 totals (VerifyStorageSyncScan).
    DirtyWalkVerify storage_scan_verify;
    /// EDEN_NX2_STORAGE_SYNC_MEMO: a range SynchronizeBuffer left clean, and the dirty generation
    /// sum of its tracker managers then (MemoryTrackerBase::DirtyGenerationSum). size 0 = empty.
    struct StorageSyncMemo {
        DAddr device_addr{};
        u32 size{};
        u64 gen_sum{};
    };
    static constexpr u32 STORAGE_SYNC_MEMO_BITS = 9;
    std::array<StorageSyncMemo, size_t{1} << STORAGE_SYNC_MEMO_BITS> storage_sync_memo{};
    /// EDEN_NX2_STORAGE_SYNC_MEMO=2 totals: hits checked, hits whose range was not clean
    /// (cpu_pages / untracked_pages unused).
    DirtyWalkVerify storage_memo_verify;
    u64 frame_tick = 0;

    /// EXP91 (fable52_answer_uniformkeeps stage 0): counts how many of today's slot rebinds
    /// (BindSmallUniformReusing: same bytes, same tick) a keep could have made without reading
    /// guest memory. Drawing thread only; nothing here changes what is bound.
    struct UniformKeepCount {
        static constexpr size_t RingSize = 128;
        static constexpr size_t NumStages = 5;
        static constexpr size_t NumSlots = 18;
        struct Range {
            DAddr begin{};
            DAddr end{};
        };
        /// The last RingSize cached write ranges; range number `seq` sits at seq % RingSize.
        std::array<Range, RingSize> ring{};
        u64 write_seq{};
        struct Slot {
            DAddr addr{};
            u32 size{};
            bool valid{};
            u64 seen_seq{}; ///< write_seq at the slot's last bind
        };
        std::array<std::array<Slot, NumSlots>, NumStages> slots{};
        /// Since the last log line. A slot rebind lands in exactly one of kept .. pending.
        struct Totals {
            u64 binds{};        ///< BindSmallUniformReusing calls
            u64 rebinds{};      ///< every rebind (slot and EXP17 range reuse)
            u64 slot_rebinds{}; ///< today's rebind of the slot's own copy
            u64 kept{};         ///< K: no noted range touched it since the slot's last bind
            u64 no_keep{};      ///< the slot was not streamed since counting (re)started
            u64 other_range{};  ///< the slot's last bind was another range
            u64 overflow{};     ///< more than RingSize ranges noted since the slot's last bind
            u64 overlap{};      ///< a range noted since then overlaps
            u64 pending{};      ///< MemoryManager holds unflushed cached writes at the bind
        };
        Totals totals{};
        u64 calls{};
        u64 last_frame{};
        u64 log_write_seq{}; ///< write_seq at the last log line
        std::chrono::steady_clock::time_point last_log{};
    };
    UniformKeepCount uniform_keep_count;

    /// EXP91 != 0: BindSmallUniformReusing as the stream branch calls it, plus the counting.
    bool BindSmallUniformCountingKeeps(size_t stage, u32 index, u32 binding_index,
                                       DAddr device_addr, u32 size, bool range_reuse);

    /// EXP92 (fable52_answer_uniformkeeps 3 and 4; idea zortos293/Lichii#1441 part d): uniform
    /// keeps. Drawing thread only; the ring is UniformKeepCount's (tier 1), the page generations
    /// and watches Common::UniformKeepPages (tier 2).
    struct UniformKeeps {
        static constexpr size_t NumStages = UniformKeepCount::NumStages;
        static constexpr size_t NumSlots = UniformKeepCount::NumSlots;
        /// Every 256th kept bind runs the normal path as a check with EXP92 = 1.
        static constexpr u64 TripwireInterval = 256;
        static constexpr u32 MaxStaleLogs = 32;
        struct Keep {
            DAddr addr{};
            u32 size{};
            u32 copy{};        ///< the stream reuse copy the slot bound
            u32 copy_serial{}; ///< its serial then
            std::array<u32, 2> page_gens{}; ///< first and last page's generation before the read
            u32 global_gen{};
            u64 seen_seq{};    ///< ring sequence the range is known clean up to
            u64 translation{}; ///< device memory translation generation before the read
            u64 made_frame{};
            bool valid{};
        };
        std::array<std::array<Keep, NumSlots>, NumStages> slots{};
        int mode{}; ///< EXP92 as the last bind saw it (a change drops every keep)
        bool tripped{}; ///< a tripwire check found a stale keep: keeps off for the session
        /// Why a keepable-looking bind was not kept (one per bind that was not kept).
        enum Reason : u8 {
            NoKeep,       ///< nothing kept for the slot yet (or the last bind made none)
            OtherRange,   ///< the slot's last bind was another range
            Ineligible,   ///< native bank bind or run-ahead frontier set
            Tick,         ///< the slot's copy is gone or from an older command buffer
            Translation,  ///< the device memory's translation changed
            PageGen,      ///< a write was noted on one of the range's pages
            RingOverflow, ///< more than RingSize cached writes since
            RangeWritten, ///< a cached write overlapped the range
            Pending,      ///< MemoryManager holds cached writes not flushed yet
            NumReasons,
        };
        struct Totals {
            u64 binds{};    ///< small binds through the stream branch while EXP92 is on
            u64 kept{};     ///< bound from the keep (verified ones included)
            u64 verified{}; ///< keepable binds that ran the normal path as a check
            u64 stale{};    ///< of those, the normal path made a fresh copy
            u64 denied{};   ///< not kept: a page of the range is denied (no keep made)
            std::array<u64, NumReasons> not_kept{};
        };
        Totals totals{};
        u64 kept_count{}; ///< kept binds this session (the tripwire's interval)
        u64 stale_total{};
        u32 stale_logged{};
        u64 calls{};
        /// A check's copy of the kept bytes (the normal path may overwrite the copy).
        std::array<u8, 4096> verify_bytes{};
        std::array<u8, 4096> fresh_bytes{};
        u64 log_guest_stores{}; ///< UniformKeepPages guest_stores at the last log line
        u64 log_pages_denied{};
        std::chrono::steady_clock::time_point last_log{};
    };
    UniformKeeps uniform_keeps;

    /// EXP92 != 0: the stream branch's small uniform bind with keeps (see UniformKeeps).
    bool BindSmallUniformKeeping(size_t stage, u32 index, u32 binding_index, DAddr device_addr,
                                 u32 size, bool range_reuse, bool eligible, int mode);
    void LogUniformKeeps(int mode);

    /// EDEN_NX2_EXP18 (zortos293/Lichii#641): resolved bindings are kept while the GPU mappings
    /// and the set of buffers stay the same. Bumped whenever a buffer is created, joined or
    /// deleted (ChangeRegister) and when the inline index buffer is replaced.
    u64 binding_epoch = 1;

    /// The last BindGraphicsUniformBuffer of a (stage, slot) and what it translated to, valid
    /// while that binding is the one it set (EXP18 & 1).
    struct UniformBindMemo {
        u64 epoch{}; ///< binding_epoch when stored; 0 never matches
        u64 map_serial{};
        size_t as_id{};
        const void* channel{};
        GPUVAddr gpu_addr{};
        DAddr device_addr{};
        u32 size{};
    };
    std::array<std::array<UniformBindMemo, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES>
        uniform_bind_memo{};

    /// One storage binding's last lookup when it resolved to a single buffer (no segments)
    /// (EXP18 & 2). Index NUM_STAGES is compute.
    struct StorageBindingMemo {
        u64 epoch{}; ///< binding_epoch when stored; 0 never matches
        u64 map_serial{};
        size_t as_id{};
        const void* channel{};
        DAddr device_addr{};
        GPUVAddr gpu_addr{};
        u32 size{};
        bool is_written{};
        BufferId buffer_id{};
    };
    std::array<std::array<StorageBindingMemo, NUM_STORAGE_BUFFERS>, NUM_STAGES + 1>
        storage_binding_memo{};

    /// FindBuffer and ResolveMultiRangeStorage for one storage binding, from its memo when
    /// nothing they depend on changed.
    void ResolveStorageBinding(size_t memo_stage, u32 index, Binding& binding, bool is_written,
                               std::vector<MultiRangeSegment>& pool);

    /// EDEN_NX2_TABLE_POINTER_TRACE: last table entry and size bound per table pointer slot.
    std::unordered_map<u64, std::pair<GPUVAddr, u32>> traced_table_pointers;
    u64 total_used_memory = 0;
    u64 buffer_bytes = 0; ///< The buffers' own size (total_used_memory may be the device's)
    u64 minimum_memory = 0;
    u64 critical_memory = 0;
    BufferId inline_buffer_id;
#ifdef YUZU_LEGACY
    bool immediately_free = false;
#endif

    // Covers the device address space (MaxwellDeviceTraits::device_virtual_bits).
    std::array<BufferId, ((1ULL << 36) >> CACHING_PAGEBITS)> page_table;
    Common::ScratchBuffer<u8> tmp_buffer;

    /// A command list page (Nx2CopyListPages): the GPU writes noted to it and the last copy.
    using Nx2Staging = decltype(std::declval<Runtime&>().DownloadStagingBuffer(size_t{}, true));
    static constexpr size_t Nx2ListPageWords = (size_t{1} << Tegra::Nx2WrittenPageBits) / 4;
    struct Nx2ListPage {
        u64 writes{};      ///< GPU and host writes noted to the page
        u64 copy_writes{}; ///< `writes` when the copy was made: the copy is current while equal
        u64 copy_tick{};   ///< GPU tick that completes the copy
        /// `writes` when the front last flushed the page: guest memory holds it while equal
        u64 synced_writes{~u64{0}};
        u32 copy_begin{};  ///< Bytes of the page the copy holds (a buffer can end inside it)
        u32 copy_end{};
        u64 copy_writer{}; ///< What recorded the copy (Nx2CopyListPages)
        /// Written outside a compute dispatch or a DMA copy (a draw's storage writes) at least
        /// once: the front can't tell from what it sent which draws write the page, so it reads
        /// no copy. A draw's writes reach a pushbuffer fetch only after a host-level wait, so
        /// the page's lists read guest memory once the back ran the work before the last such
        /// wait and no write came since the front flushed the page (Nx2ReadListCopy).
        bool draw_written{};
        /// The page's 4-byte words GPU work wrote that guest memory does not hold: set by the GPU
        /// writes noted to the page (the whole range a dispatch or copy may write), cleared for
        /// the words the host writes whole and for every word once the front flushed the page.
        /// Once it did (`flushed`), guest memory holds the other words, so a list on none of
        /// these reads it directly (EXP40 & 1). Before, GPU work may have written the page before
        /// it became a command list page, unnoted.
        std::bitset<Nx2ListPageWords> gpu_words;
        bool flushed{};
        std::optional<Nx2Staging> copy;
    };
    /// Sets (`set`) or clears the words of `entry.gpu_words` within [begin, end) bytes of its
    /// page: every word the bytes touch when setting, only the words they cover when clearing.
    static void Nx2MarkListWords(Nx2ListPage& entry, u64 begin, u64 end, bool set);
    /// Whether GPU work wrote any word of [begin, end) bytes of the page that guest memory lacks.
    [[nodiscard]] static bool Nx2HasListWords(const Nx2ListPage& entry, u64 begin, u64 end);
    /// Fills `out` with the page's bytes from `offset` (`device_addr`) on: the words of
    /// `entry.gpu_words` from its copy, the others from guest memory once the page was flushed
    /// (EXP40 & 1); else, or for EDEN_NX2_LIST_VERIFY's read as before, all from the copy.
    void Nx2ReadListWords(const Nx2ListPage& entry, DAddr device_addr, u64 offset,
                          std::span<u8> out);
    std::mutex nx2_list_mutex; ///< nx2_list_pages: the split's front reads the copies
    std::unordered_map<DAddr, Nx2ListPage> nx2_list_pages;
    /// Copies replaced while the GPU may still write them: freed once it is past their tick.
    std::vector<std::pair<u64, Nx2Staging>> nx2_retired_list_copies;
    boost::container::small_vector<std::pair<BufferId, DAddr>, 8> nx2_list_copy_requests;
    bool nx2_collect_list_copies{};
};

} // namespace VideoCommon
