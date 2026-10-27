/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/functional.hpp>

#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// Do not CSE-merge instructions whose (single-output) result is very large. Merging
// structurally-identical large buffers -- notably the per-layer INT4 weight
// dequant tiles in deep LLMs -- makes their outputs share one producer and stay
// co-live, defeating per-layer scratch reuse in memory_coloring and blowing up
// peak GPU memory (O(num_layers) scratch). Small common ops are unaffected, so
// normal CSE benefit is preserved.
static bool cse_output_too_large(instruction_ref ins)
{
    static const std::size_t threshold = 64UL * 1024 * 1024; // 64 MB
    const auto& s = ins->get_shape();
    // Only consider standard (non-tuple) shapes; tuple/sub-shape ops are left to CSE.
    if(not s.sub_shapes().empty())
        return false;
    return s.bytes() > threshold;
}

// Outputs always follow their inputs, so by the time an instruction is visited its inputs have
// already been merged and a single forward pass finds every common subexpression.
void eliminate_common_subexpression::apply(module& m) const
{
    std::unordered_multimap<std::string, instruction_ref> instructions;
    for(auto ins : iterator_for(m))
    {
        // Skip dead instructions
        if(ins->outputs().empty())
            continue;

        // Skip merging very large buffers (see cse_output_too_large): prevents the
        // per-layer weight-dequant tiles from being co-materialized, which would
        // otherwise cause an O(depth) scratch blow-up / GPU OOM on deep models.
        if(cse_output_too_large(ins))
            continue;

        // Find instruction with the same name
        auto found_instructions = range(instructions.equal_range(ins->name()));
        auto eq = find_if(found_instructions, [&](const auto& pp) { return *pp.second == *ins; });
        if(eq != found_instructions.end())
        {
            m.replace_instruction(ins, eq->second);
            continue;
        }
        instructions.emplace(ins->name(), ins);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
