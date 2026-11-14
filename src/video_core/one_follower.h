// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// One follower (work\fable_b_one_follower.md): the renderer's front end split behind the engine.
//
//   F, the GPU thread (GpuSplit's front): command lists and macros, unchanged.
//   E, the engine thread = GpuSplit's back thread: GpuSplit::Execute on the real engines, game
//      memory writes from the stream, the buffer cache, each draw's buffer fill (by value,
//      MemStage::FillRingDraw into ring B).
//   R, the render thread (new, this file): the pipeline cache, texture cache, query cache, fences,
//      state tracker and the Vulkan scheduler. It has no engine: it plays packets and keeps a
//      view of the registers it reads, fed by deltas E sends.
//
// Everything R cannot do from a packet yet runs "lent": E pushes a lent packet and parks; R plays
// every packet before it, runs the operation reading E's engine directly, then releases E. That is
// today's serial order run on two threads, so every step below keeps the picture.
//
// Steps (section 8 / 8.1 of the design), one switch each, nested (each needs the one before and
// refuses loudly without it; the effective step is what Step() returns):
//   1 FRITCHII_ONEF=1 (boot): R exists, every draw and every R-owned operation lent.
//   2 FRITCHII_ONEF_VIEW=1 (live): E packs the delta spans; R applies them to its view and, at
//     every lent draw, compares the view with the engine (Check::View*).
//   3 FRITCHII_ONEF_BUILD=1 (live): R also builds each lent draw from the packet and its view and
//     compares that with what the lent path used (Check::Build*).
//   4 FRITCHII_ONEF_UNLEND=<mask> (live): the draw kinds in the mask run on R from the packet
//     alone while E goes on (UnlendKind). Every other draw and op stays lent, and every lent draw
//     counts its reason (LentReason). FRITCHII_ONEF_VERIFY_EVERY=N forces every Nth unlent draw
//     lent and runs step 3's compare on it: step 4's own must-be-0.
//   FRITCHII_ONEF_AHEAD (live): E's window in packets R has not played (1, 8, 0 = the ring).
//   FRITCHII_ONEF_PIN (boot, default 1): R gets PinnedRole::GpuMem's physical core.
//
// Ownership (section 5): state has one owner. E never touches an R-owned object (scheduler,
// texture, pipeline and query caches, fence manager, state tracker) unless R is parked behind a
// lent packet or drained; R never touches the buffer cache, the engines or game memory writes
// outside a lent packet. Check::OwnerOnE / OwnerOnR count violations (must be 0).
//
// Out of this branch (step 5): reports, counter resets, clears, fences with served landings,
// compute dispatches and 0xe2c on R. They stay lent here.

#include <array>
#include <atomic>
#include <cstddef>
#include <span>

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/one_follower_hooks.h"

namespace Core {
class System;
}

namespace VideoCommon {
class PageFilter;
}

namespace VideoCommon::OneFollower {

using Maxwell3D = Tegra::Engines::Maxwell3D;

// --- Switches ------------------------------------------------------------------------------------
// Stored in VideoCore::Nx2Debug (nx2_debug_control.h): g_onef (boot), g_onef_view, g_onef_build,
// g_onef_unlend, g_onef_verify_every, g_onef_ahead (live), g_onef_pin (boot).

/// The steps, as Step() reports them after nesting.
enum class StepLevel : u8 {
    Off = 0,    ///< FRITCHII_ONEF=0 at boot: today's code path, untouched
    Lent = 1,   ///< R exists, everything lent
    View = 2,   ///< + the register view and its delta, verify only
    Build = 3,  ///< + draws built from the packet, verify only
    Unlend = 4, ///< + the draw kinds of FRITCHII_ONEF_UNLEND unlent
};

/// FRITCHII_ONEF_UNLEND's bits: the draw kinds that run on R from the packet alone at step 4.
/// Plain draws first; every other kind starts lent and gets its own bit.
enum UnlendKind : u32 {
    PlainIndexed = 1u << 0, ///< draw.end, indexed, one instance or a plain instanced draw
    PlainArrays = 1u << 1,  ///< the same, non-indexed
    InstanceLoop = 1u << 2, ///< a deferred instance loop (DrawDeferred, folded first instance)
    Indirect = 1u << 3,     ///< DrawIndirect (counts and arguments in GPU memory)
    List = 1u << 4,         ///< list draws (nx2_list_draw.h: two words in GPU memory)
    Mesh = 1u << 5,         ///< task/mesh pipelines, emulated or native
    GpuCount = 1u << 6,     ///< a vertex count GPU work wrote (nx2_indirect_grid.h)
    InlineIndex = 1u << 7,  ///< inline index draws (the indexes live in a vector on E)
    AllKinds = 0xff,
};

/// FRITCHII_ONEF_VERIFY_EVERY's default: every 64th unlent draw is forced lent and compared.
inline constexpr int DefaultVerifyEvery = 64;
/// FRITCHII_ONEF_AHEAD's default (packets R has not played that E may run ahead by; 0 = ring).
inline constexpr int DefaultAhead = 1;

// --- Why something ran lent (counted per reason; not a must-be-0) ---------------------------------

/// Lent draws by reason. A draw counts exactly one: the first that applies, in this order.
enum class LentReason : u8 {
    StepBelowUnlend, ///< step 1 to 3: every draw is lent
    KindNotInMask,   ///< its UnlendKind bit is off (counted per kind as well, LentKind)
    PipelineMiss,    ///< R's lookup on its view found no ready pipeline
    Verify,          ///< FRITCHII_ONEF_VERIFY_EVERY: forced lent to compare
    StreamWrite,     ///< the stream-write guard hit since the last packet (Check::StreamWrite)
    Diagnostic,      ///< a frame dump, render trace, probe or capture reads E's state
    Transient,       ///< state R's view cannot carry yet (channel switch pending, view not seeded)
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(LentReason::Count)> lent_reason_names{
    "step below 4", "kind not in mask", "pipeline miss", "verify", "stream write", "diagnostic",
    "transient",
};

/// Lent operations other than draws (step 5's kinds and the rest): E parks for each.
enum class LentOp : u8 {
    Draw,         ///< a draw (its reason: LentReason)
    Clear,        ///< RasterizerVulkan::Clear
    Dispatch,     ///< DispatchCompute (its recording; the fill stays E's)
    Query,        ///< Query, ResetCounter, report and counter work
    Fence,        ///< SignalFence, SyncOperation*, SignalSyncPoint, SignalReference, ReleaseFences
    ReadBack,     ///< FlushRegion*, FinishRead, GetFlushArea, WaitForIdle, TickWork's texture half
    Invalidation, ///< InvalidateRegion, OnCacheInvalidation(s), OnCPUWrite, UnmapMemory, gathers
    Copy,         ///< accelerated DMA, surface copies, inline uploads onto images
    FrameEnd,     ///< TickFrame, present, AccelerateDisplay, FlushCommands
    Channel,      ///< a channel bind (the view is seeded again from the new channel's engine)
    Other,        ///< any other call into an R-owned object
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(LentOp::Count)> lent_op_names{
    "draw", "clear", "dispatch", "query", "fence", "read back", "invalidation", "copy",
    "frame end", "channel", "other",
};

// --- Must-be-0 checks ------------------------------------------------------------------------------
// Logged every 10 s on one line memstage_gate.py reads, in its pair form:
//   "One follower checks (must be 0; this window / session): view words 0 / 0, view flags 0 / 0, ..."
// with the names below.

enum class Check : u8 {
    // Step 2, at every lent draw: R's view against E's engine.
    ViewWords,     ///< a watched register word differs
    ViewFlags,     ///< an R-owned or shared dirty flag differs
    ViewDraw,      ///< the packet's DrawValues differ from the engine's draw manager
    ViewPrograms,  ///< the six program hashes differ
    ViewPools,     ///< the TIC or TSC pool differs
    // Step 3 (and step 4's forced-lent draws): the draw built from the packet and the view against
    // what the lent path used.
    BuildPipeline,    ///< the pipeline (key) differs
    BuildImageViews,  ///< the image view ids differ
    BuildDynamic,     ///< the dynamic state differs
    BuildDescriptors, ///< the descriptors (image/sampler entries) differ
    BuildRenderTargets, ///< the render targets differ
    // Section 7: E's game memory write from the stream onto a page R's texture reads still need
    // (a page the texture cache tracks, while R has unplayed packets). A hit forces that write
    // lent (step 4); counted at every step.
    StreamWrite,
    // Ownership.
    OwnerOnE, ///< E used an R-owned object while R was neither parked nor drained
    OwnerOnR, ///< R used the buffer cache or an engine outside a lent packet
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Check::Count)> check_names{
    "view words",    "view flags",       "view draw",         "view programs",
    "view pools",    "build pipeline",   "build image views", "build dynamic",
    "build descriptors", "build render targets", "stream write", "owner on E", "owner on R",
};

// --- The packet (E to R) -----------------------------------------------------------------------------
// One ring, one writer (E), one reader (R). Every packet starts with a PacketHeader; its spans
// follow in this order, each present when its bit is in `spans`, each padded to 8 bytes:
//   RegPair[delta_count], FlagBits, DrawValues, Programs, Pools.
// The buffer half of a draw is MemStage's DrawPacket in ring B (vk_mem_stage.h), filled by E with
// FillRingDraw and taken by R with TakeRingDraw; `buffer_number` names it. Ring B keeps its format.

enum class PacketKind : u8 {
    Draw = 0,  ///< an unlent draw: R plays it from the packet and its view (step 4)
    Lent = 1,  ///< E is parked: R plays what is before, runs `call`, releases E
    Delta = 2, ///< spans only (the delta before a lent op, so R's view is current when it runs)
};

enum SpanBit : u8 {
    SpanDelta = 1,    ///< RegPair[delta_count]
    SpanFlags = 2,    ///< FlagBits
    SpanDraw = 4,     ///< DrawValues
    SpanPrograms = 8, ///< Programs
    SpanPools = 16,   ///< Pools
};

enum PacketFlag : u8 {
    /// Step 2/3: R compares its view (and at 3 its build) with the engine while E is parked.
    PacketVerify = 1,
    /// The view is seeded from the engine in full before the spans (channel bind, step change,
    /// first packet): R copies regs and flags from E's engine while E is parked (lent only).
    PacketSeed = 2,
};

/// One changed register word R watches.
struct RegPair {
    u32 index;
    u32 value;
};
static_assert(sizeof(RegPair) == 8);

/// The R-owned and shared dirty flags E raised since the last packet (Maxwell3D::DirtyState).
using FlagBits = Maxwell3D::DirtyState::Flags;

/// The draw manager's state for one draw, by value (no inline index vector: those draws stay lent).
struct DrawValues {
    Maxwell3D::Regs::PrimitiveTopology topology;
    u32 draw_mode; ///< Maxwell3D::DrawManager::DrawMode
    u32 base_index;
    u32 base_instance;
    u32 instance_count;
    u32 drawn_instances;
    u32 draw_element_count;
    Maxwell3D::Regs::VertexBuffer vertex_buffer;
    Maxwell3D::Regs::IndexBuffer index_buffer;
    u32 kind;      ///< the draw's UnlendKind (one bit)
    u8 indexed;    ///< RasterizerVulkan::Draw's is_indexed
    u8 indirect;   ///< DrawIndirect
    u8 draw_indexed; ///< the draw manager's own draw_indexed
    u8 reserved;
    u32 instance_arg; ///< RasterizerVulkan::Draw's instance_count argument
};

/// The six program hashes RefreshStages gives (ShaderCache::RefreshStages's unique_hashes).
struct Programs {
    std::array<u64, 6> hashes;
};

/// The texture and sampler pools (regs.tex_header, regs.tex_sampler and the linked-TSC mode).
struct Pools {
    GPUVAddr tic_address;
    u32 tic_limit;
    u32 tsc_linked; ///< regs.sampler_binding (1: the TSC index comes from the TIC handle)
    GPUVAddr tsc_address;
    u32 tsc_limit;
    u32 reserved;
};

/// A lent operation: run on R with E parked. `function(argument)` is E's code, unchanged.
struct LentCall {
    void (*function)(void*);
    void* argument;
};

struct PacketHeader {
    u32 bytes;        ///< the whole packet with its spans, a multiple of 8
    PacketKind kind;
    u8 spans;         ///< SpanBit
    u8 flags;         ///< PacketFlag
    u8 reason;        ///< LentReason (a lent draw) or 0
    u8 op;            ///< LentOp (a lent packet)
    u8 reserved[3];
    u32 delta_count;  ///< RegPair count (SpanDelta)
    u64 number;       ///< packets since R started, E's count (1 = the first)
    u64 position;     ///< ring A position of E's entry (GpuSplit::BackEntryPosition)
    u64 draw_seq;     ///< E's draw count since R started
    u64 buffer_number; ///< the draw's MemStage ring B packet number, 0 = none (lent draws)
    LentCall call;    ///< PacketKind::Lent
    s32 channel;      ///< the channel bound at the packet
    u32 reserved2;
};
static_assert(sizeof(PacketHeader) % 8 == 0);

// --- The register view (R) -----------------------------------------------------------------------

/// R's view: only packets write it (or a seed from E's engine under a lent packet). R's code reads
/// `regs` and `flags` the way it reads the engine's today.
struct RegisterView {
    Maxwell3D::Regs regs{};
    FlagBits flags{};
    DrawValues draw{};
    Programs programs{};
    Pools pools{};
    u64 packets{}; ///< packets applied
    bool seeded{};
};

/// The words E sends in deltas and R's verify compares: every register word except those R never
/// reads (constant buffer data words, macro upload and firmware words, the shadow scratch). Built
/// once (one_follower_view.cpp).
[[nodiscard]] bool Watched(u32 index) noexcept;

/// Which side owns a dirty flag (section 2): flags only the buffer side reads stay on E; flags
/// only R reads are forwarded; flags both read exist on both, raised on both, cleared by each.
enum class FlagOwner : u8 { E, R, Both };
[[nodiscard]] FlagOwner OwnerOf(u8 flag) noexcept;

// --- Lifetime and the thread (one_follower.cpp) ----------------------------------------------------

/// Reads the switches (GpuSplit::MaybeStart, before the back thread exists). Boot ones stick.
void ReadSwitches();
/// FRITCHII_ONEF=1 at boot (R exists).
[[nodiscard]] bool On() noexcept;
/// The effective step after nesting (StepLevel), refreshed at frame ends and lent packets.
[[nodiscard]] StepLevel Step() noexcept;
/// True on R.
[[nodiscard]] bool OnRenderThread() noexcept;
/// R runs a lent call now (true on R inside a lent packet).
[[nodiscard]] bool InLent() noexcept;

/// E: run `function(argument)` on R with E parked. R first plays every packet before it. Counted
/// by `op` (and `reason` for a draw); the round trip is timed (the hop's price, step 1's number).
void Lend(LentOp op, LentReason reason, void (*function)(void*), void* argument);
template <typename F>
void Lend(LentOp op, LentReason reason, F& function) {
    Lend(op, reason, [](void* argument) { (*static_cast<F*>(argument))(); }, &function);
}
template <typename F>
void Lend(LentOp op, F& function) {
    Lend(op, LentReason::StepBelowUnlend, function);
}

/// E: wait until R has played every packet pushed (E is about to touch an R-owned object itself).
void Drain();
/// E: packets R has not played.
[[nodiscard]] u64 Behind() noexcept;

/// Owner asserts: an R-owned object is used here (scheduler, texture/pipeline/query caches, fence
/// manager, state tracker); counted as Check::OwnerOnE unless on R, or on E with R parked or
/// drained. `where` names the object for the first reports.
void CheckRenderOwned(const char* where) noexcept;
/// The buffer cache or an engine is used here: Check::OwnerOnR when on R outside a lent call.
void CheckEngineOwned(const char* where) noexcept;

/// Counts a must-be-0 event (any thread). `what` names it in the first reports.
void Note(Check check, const char* what) noexcept;

// --- The engine side of the delta (part b, one_follower_view.cpp) ---------------------------------

/// Maxwell3D's write paths: register word `index` took a new value on E. Off: one load and a branch.
void NoteRegWrite(u32 index) noexcept;
/// E raised dirty flag `flag` outside the method write path (the dirty tables cover that path).
void NoteFlagRaised(u8 flag) noexcept;

/// E, at a draw or before a lent op (step 2 on): appends the spans that changed since the last
/// packet to `header` (SpanDelta, SpanFlags, SpanDraw, SpanPrograms, SpanPools) at `out`;
/// returns the bytes written.
std::size_t PackSpans(const Maxwell3D& engine, PacketHeader& header, std::span<u8> out);

/// R: applies a packet's spans to the view.
void ApplySpans(RegisterView& view, const PacketHeader& header, const u8* spans);

/// R, at a lent draw with PacketVerify (step 2): compares the view with E's engine (E parked);
/// counts Check::View*.
void VerifyView(const RegisterView& view, const Maxwell3D& engine);

/// R's view (R only, or E while R is parked).
[[nodiscard]] RegisterView& View() noexcept;

// --- Section 7: the stream-write guard -----------------------------------------------------------

/// E writes game memory from the stream (constant buffer words, inline uploads, copies, reports):
/// true when it lands on a page the texture cache tracks while R has unplayed packets. The caller
/// then makes that write lent (Check::StreamWrite counted). GuardStreamWrite below does both: it
/// drains R before the write, which is the write run lent; no draw is lent for it, so
/// LentReason::StreamWrite stays 0 in this branch.
[[nodiscard]] bool StreamWriteHits(DAddr address, u64 size) noexcept;

// --- Added with part (a): the thread, the ring, the lend (one_follower.cpp) ------------------------

// Detail::t_engine and Detail::t_render (ShouldLend's one load): one_follower_hooks.h.

/// E's rasterizer entry points: true on E with one follower on; the call is then lent to R (or
/// pushed, step 4). On R and on every other thread the call runs where it is, as today.
[[nodiscard]] inline bool ShouldLend() noexcept {
    return Detail::t_engine;
}
/// GpuSplit::BackLoop at its start: this thread is E (no-op with FRITCHII_ONEF=0).
void MarkEngineThread() noexcept;
[[nodiscard]] inline bool OnEngineThread() noexcept {
    return Detail::t_engine;
}

/// GpuSplit::MaybeStart, before the back thread starts / GpuSplit::Stop, after it joined.
void Start(Core::System& system);
void Stop();

/// Step 4, part (c): R plays an unlent draw packet with this (the rasterizer registers it).
using DrawPlayer = void (*)(void* context, const PacketHeader& header, const u8* spans);
void SetDrawPlayer(DrawPlayer player, void* context) noexcept;
[[nodiscard]] bool HasDrawPlayer() noexcept;
/// Step 4, part (c): E's side of an unlent draw (the buffer fill into ring B, PackSpans, Push).
/// Returns false when it could not; the caller then lends the draw (LentReason::Transient).
using DrawPusher = bool (*)(void* context, bool is_indexed, u32 instance_count);
void SetDrawPusher(DrawPusher pusher, void* context) noexcept;
[[nodiscard]] bool PushDraw(bool is_indexed, u32 instance_count);

/// E, at each draw (RasterizerVulkan::OneFollowerDraw): LentReason::Count when the draw of
/// UnlendKind `kind` may run unlent, else why it is lent (the first reason that applies).
/// `diagnostic`: a frame dump, trace or probe wants E's state at this draw.
[[nodiscard]] LentReason DecideDraw(u32 kind, bool diagnostic) noexcept;
/// R (part c) at an unlent draw it could not play from the packet (no ready pipeline, ...):
/// counted under that reason; the draw is still R's to finish (E has moved on).
void NoteUnlentFallback(LentReason reason) noexcept;
/// Those fallbacks, by reason (not lends: a test can tell them from LentReason counts).
[[nodiscard]] u64 UnlentFallbacks(LentReason reason) noexcept;

/// E, before Lend or PushDraw of a draw: what the draw is (DrawValues' kind, indexed, indirect
/// and the instance count argument). Lend puts the kind in the header (reserved[0], the UnlendKind
/// bit) for the counters, PackSpans puts all of it in the draw span. Consumed by the next packet.
void SetNextDraw(u32 kind, bool indexed, bool indirect, u32 instance_arg) noexcept;

/// E: pushes a packet R plays while E goes on (PacketKind::Draw or Delta), within the window
/// (FRITCHII_ONEF_AHEAD) and the ring's room. Fills number, position and draw_seq.
void Push(PacketHeader& header, std::span<const u8> spans);

/// The engine R's view mirrors (the bound channel's Maxwell3D): set by the channel bind (on R
/// under a lent packet, or on E). PackSpans reads it on E, the seed and VerifyView on R.
void SetEngine(Maxwell3D* engine) noexcept;
[[nodiscard]] Maxwell3D* Engine() noexcept;

/// GpuSplit's two positions (8.2): E stores its own `consumed`, then calls this, which raises
/// GpuSplit's `played` (what every front wait reads) to that position clamped to the oldest packet
/// R has not played; R raises `played` again after each packet it plays (step 4).
void EngineConsumed(u64 position) noexcept;

/// Live switch values (refreshed with Step()).
[[nodiscard]] u32 UnlendMask() noexcept;
[[nodiscard]] int VerifyEvery() noexcept;
[[nodiscard]] int Ahead() noexcept;

// The counters and the 10 s lines are part (d)'s: one_follower_stats.h (Note() is defined there).

// --- Added with part (b): the engine side's write tracking (one_follower_view.cpp) ---------------

// g_track_writes and NoteRegWrite: one_follower_hooks.h.

/// E: the view's spans need a full seed first (tracking just started, a step change or a channel
/// bind): the next lent packet carries PacketSeed. Unlent draws wait for it (LentReason::Transient).
[[nodiscard]] bool ViewReady() noexcept;

/// The texture cache's "pages that ever held an image" filter (set by the rasterizer at creation):
/// StreamWriteHits looks there.
void SetTexturePages(const PageFilter* filter) noexcept;
/// E's stream writes (Maxwell3D's constant buffer words, memory_manager.cpp's cached writes): on a
/// texture page they are counted (StreamWritesOnTexturePages, informational at every step), and
/// with R behind (step 4) Check::StreamWrite is counted and E drains R before the write lands.
void GuardStreamWrite(DAddr address, u64 size) noexcept;
/// Writes that landed on texture pages, any step (section 7's run 4 number; not a must-be-0).
[[nodiscard]] u64 StreamWritesOnTexturePages() noexcept;

/// R, under a lent packet with PacketSeed (or a channel bind): the view takes the engine's
/// registers, flags, draw state and pools in full.
void SeedView(RegisterView& view, const Maxwell3D& engine);
/// R, around a lent call (step 2+): before, at step 4 the engine's forwarded flags take the
/// view's (R's truth); after, the view takes what the lent code left in the engine.
void BeforeLentCall(RegisterView& view, Maxwell3D& engine);
void AfterLentCall(RegisterView& view, Maxwell3D& engine);

/// E, step 4 (part c's pusher), right after its buffer fill of an unlent draw: the forwarded flags
/// the fill left are E's baseline for the next packet's flag span (the fill clears E's copies of
/// the shared flags; their raises were packed before it).
void NoteEngineFlags(const Maxwell3D& engine) noexcept;

namespace Detail {
/// Between one_follower.cpp and one_follower_view.cpp.
/// The draw SetNextDraw described, or null (E; PackSpans's draw span).
[[nodiscard]] const DrawValues* NextDraw() noexcept;
/// E's write tracking stops (step below 2) or its engine changed: the next lent packet seeds the
/// view in full. On E, or on R under a lent packet (E parked).
void ViewTrackingStopped() noexcept;
} // namespace Detail

} // namespace VideoCommon::OneFollower
