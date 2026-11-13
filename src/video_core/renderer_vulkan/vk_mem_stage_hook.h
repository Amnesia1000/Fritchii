// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <source_location>

#include "common/common_types.h"

// The memory stage's watch on the scheduler and the descriptor queues: vk_mem_stage.cpp.
namespace Vulkan::SeamHook {

enum class Use : u8 {
    Record,
    RecordUpload,
    OutsideRenderPass,
    RenderPass,
    Flush,
    Finish,
    DispatchWork,
    InvalidateState,
    DescriptorAdd,
    DescriptorAcquire,
};

inline std::atomic<u32> g_watch{0};

void OnUse(Use use, const std::source_location& location);

inline void Watch(Use use,
                  const std::source_location& location = std::source_location::current()) {
    if (g_watch.load(std::memory_order_relaxed) != 0) [[unlikely]] {
        OnUse(use, location);
    }
}

} // namespace Vulkan::SeamHook
