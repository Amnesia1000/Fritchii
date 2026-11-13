// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <source_location>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"

namespace Vulkan {

class MemStage;
class Scheduler;

// The buffer cache side's only way to the scheduler's clock and syncs (vk_mem_stage.cpp).
class SeamClock {
public:
    explicit SeamClock(Scheduler& scheduler_, MemStage* stage_ = nullptr);

    [[nodiscard]] u64 Now() const noexcept {
        return master->CurrentTick();
    }

    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return master->IsFree(tick);
    }

    [[nodiscard]] u64 KnownGpu() const noexcept {
        return master->KnownGpuTick();
    }

    [[nodiscard]] Scheduler& GetScheduler() const noexcept {
        return *scheduler;
    }

    void Finish(std::source_location location = std::source_location::current());

    void Wait(u64 tick);

    u64 SubmitForHostRead(std::source_location location = std::source_location::current());

private:
    Scheduler* scheduler;
    MasterSemaphore* master;
    MemStage* stage;
};

} // namespace Vulkan
