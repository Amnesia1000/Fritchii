// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <unordered_map>
#include <vector>

#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/program.h"

namespace Shader::Optimization {

/// The block of each instruction and the immediate dominator of each block of a program (Cooper,
/// Harvey and Kennedy, over post order numbers). Built from the program as it is: instructions
/// added later are not known to it (Dominates says false for them).
class Dominance {
public:
    explicit Dominance(const IR::Program& program) : order{program.post_order_blocks} {
        for (size_t number = 0; number < order.size(); ++number) {
            numbers.emplace(order[number], number);
            for (const IR::Inst& inst : order[number]->Instructions()) {
                blocks.emplace(&inst, order[number]);
            }
        }
        idom.assign(order.size(), NONE);
        if (order.empty()) {
            return;
        }
        const size_t entry{order.size() - 1};
        idom[entry] = entry;
        const auto intersect = [this](size_t lhs, size_t rhs) {
            while (lhs != rhs) {
                while (lhs < rhs) lhs = idom[lhs];
                while (rhs < lhs) rhs = idom[rhs];
            }
            return lhs;
        };
        for (bool changed = true; changed;) {
            changed = false;
            for (size_t number = entry; number-- > 0;) {
                size_t dominator{NONE};
                for (const IR::Block* const predecessor : order[number]->ImmPredecessors()) {
                    const auto it{numbers.find(predecessor)};
                    if (it == numbers.end() || idom[it->second] == NONE) {
                        continue;
                    }
                    dominator = dominator == NONE ? it->second : intersect(it->second, dominator);
                }
                if (dominator != idom[number]) {
                    idom[number] = dominator;
                    changed = true;
                }
            }
        }
    }

    /// Whether def runs before use on every path to use.
    [[nodiscard]] bool Dominates(const IR::Inst* def, const IR::Inst* use) const {
        const auto def_block{blocks.find(def)};
        const auto use_block{blocks.find(use)};
        if (def_block == blocks.end() || use_block == blocks.end()) {
            return false;
        }
        if (def_block->second == use_block->second) {
            for (const IR::Inst& inst : def_block->second->Instructions()) {
                if (&inst == def) return true;
                if (&inst == use) return false;
            }
            return false;
        }
        const size_t target{numbers.at(def_block->second)};
        for (size_t number = numbers.at(use_block->second); idom[number] != NONE;) {
            if (idom[number] == number) {
                return false; // the entry
            }
            number = idom[number];
            if (number == target) {
                return true;
            }
        }
        return false;
    }

private:
    static constexpr size_t NONE{~size_t{0}};
    const IR::BlockList& order;
    std::unordered_map<const IR::Block*, size_t> numbers;
    std::unordered_map<const IR::Inst*, const IR::Block*> blocks;
    std::vector<size_t> idom;
};

} // namespace Shader::Optimization
