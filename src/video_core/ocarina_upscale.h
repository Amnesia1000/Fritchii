// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>
#include <string_view>

#include "common/settings_enums.h"

namespace VideoCommon {

enum class OcarinaUpscaleBackend { Native, Fsr, Dlss };

constexpr OcarinaUpscaleBackend SelectOcarinaUpscaler(bool fsr_enabled, bool dlss_enabled,
                                                    bool dlss_available,
                                                    std::string_view override) {
    if (override == "native") {
        return OcarinaUpscaleBackend::Native;
    }
    if (override == "fsr") {
        return OcarinaUpscaleBackend::Fsr;
    }
    if ((dlss_enabled || override == "dlss") && dlss_available) {
        return OcarinaUpscaleBackend::Dlss;
    }
    return fsr_enabled ? OcarinaUpscaleBackend::Fsr : OcarinaUpscaleBackend::Native;
}

/// Whether the "OoT anti-aliasing" choice asks for AMD FSR (3.1, or 4 with FSR 3.1 as fallback).
constexpr bool OcarinaFsrChosen(Settings::OotAntiAliasing choice) {
    return choice == Settings::OotAntiAliasing::Fsr3 || choice == Settings::OotAntiAliasing::Fsr4;
}

/// The backend for the "OoT anti-aliasing" choice and the NVIDIA DLSS setting. An FSR choice wins
/// over DLSS; with Game or No sharpening, DLSS replaces the game's pass when it's on and available
/// (as before FSR came in).
constexpr OcarinaUpscaleBackend SelectOcarinaUpscaler(Settings::OotAntiAliasing choice,
                                                    bool dlss_enabled, bool dlss_available,
                                                    std::string_view override) {
    const bool fsr = OcarinaFsrChosen(choice);
    return SelectOcarinaUpscaler(fsr, dlss_enabled && !fsr, dlss_available, override);
}

/// EDEN_NX2_DLSS_JITTER (boot, 0 = off): DLSS gets the camera jitter and the texture LOD bias
/// the FSR path uses (a player's request; DLSS is made for jittered input). 1 = the jitter as FSR
/// gets it, 2 = with its vertical sign flipped (Streamline's convention is unverified here).
inline int OcarinaDlssJitterMode() {
    static const int mode = [] {
        const char* value = std::getenv("EDEN_NX2_DLSS_JITTER");
        return value != nullptr ? std::atoi(value) : 0;
    }();
    return mode;
}

/// FRITCHII_DLSS_SCALED (boot, 0 = off): with DLSS or FSR picked for OoT, keep the player's
/// resolution scale instead of setting it to 1x for the session. The upscaler then takes the
/// scene at the scaled size (2x: four times the pixels, 0.5x: a quarter) and picks its mode from
/// the real render and output sizes. Off: the scale goes to 1x as before.
inline bool OcarinaScaledUpscale() {
    static const bool on = [] {
        const char* value = std::getenv("FRITCHII_DLSS_SCALED");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return on;
}

struct OcarinaExtent {
    unsigned width{};
    unsigned height{};
    constexpr bool operator==(const OcarinaExtent&) const = default;
};

/// What the temporal upscaler writes for one frame with the scene at the resolution scale.
/// `render`: the scene's real size (the scaled size when the scaler scaled it). `target`: the real
/// size of the game's upscale target. `native_aa`: the DLSS quality "Native" (the output keeps the
/// render size). `want_4k`: the 1440p pass with the 4K output. DLSS and FSR only enlarge, so a
/// scene larger than the target is upscaled to its own size (native AA) and the output pass
/// shrinks the result into the target: supersampling.
struct OcarinaScaledOutput {
    OcarinaExtent output;
    bool upscale_4k{};
};

constexpr OcarinaScaledOutput SelectOcarinaScaledOutput(OcarinaExtent render, OcarinaExtent target,
                                                        bool native_aa, bool want_4k) {
    const auto fits = [render](OcarinaExtent output) {
        return render.width <= output.width && render.height <= output.height;
    };
    if (native_aa) {
        return {render, false};
    }
    constexpr OcarinaExtent uhd{3840, 2160};
    if (want_4k && fits(uhd)) {
        return {uhd, true};
    }
    return {fits(target) ? target : render, false};
}

/// Texture LOD bias added to Ocarina's mipmapped samplers while FSR upscales. AMD recommends
/// log2(render / output) - 1: -1.58 for docked 1440p -> 4K. With EDEN_NX2_DLSS_JITTER, DLSS gets
/// NVIDIA's log2(render / output): -0.58.
inline float OcarinaFsrLodBias(OcarinaUpscaleBackend backend) {
    if (backend == OcarinaUpscaleBackend::Fsr) {
        return -1.58f;
    }
    return backend == OcarinaUpscaleBackend::Dlss && OcarinaDlssJitterMode() != 0 ? -0.58f : 0.0f;
}

/// Whether a sampler with that bias keeps it for an image: only docked (handheld isn't measured),
/// not for images the GPU wrote (render targets would read a mip too sharp), and not once the
/// upscaler failed this session (the game's own pass runs again).
constexpr bool KeepOcarinaFsrLodBias(bool docked, bool upscaler_failed, bool gpu_written) {
    return docked && !upscaler_failed && !gpu_written;
}

} // namespace VideoCommon
