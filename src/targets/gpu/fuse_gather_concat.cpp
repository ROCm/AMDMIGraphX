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
#include <migraphx/gpu/fuse_gather_concat.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/op/gather.hpp>
#include <migraphx/op/concat.hpp>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_GATHER_CONCAT_FUSION)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_GATHER_CONCAT_FUSION)

namespace {

/**
 * Checks if all gather operations are compatible for fusion
 */
bool are_gathers_compatible(const std::vector<instruction_ref>& gathers)
{
    if(gathers.empty())
        return false;
    
    // Get reference gather properties
    auto ref_op = any_cast<op::gather>(gathers[0]->get_operator());
    auto ref_axis = ref_op.axis;
    
    // All gathers must have the same axis and be single-use
    for(const auto& gather_ins : gathers)
    {
        if(gather_ins->name() != "gather")
            return false;
        
        auto op = any_cast<op::gather>(gather_ins->get_operator());
        if(op.axis != ref_axis)
            return false;
        
        // Each gather should be used only by the concat
        if(gather_ins->outputs().size() != 1)
            return false;
    }
    
    return true;
}

/**
 * Matcher for multiple gathers feeding into a single concat
 */
struct find_gather_concat
{
    auto matcher() const
    {
        return match::name("concat")(match::any_of[match::inputs()](match::name("gather")));
    }
    
    void apply(module& m, const match::matcher_result& r) const
    {
        if(enabled(MIGRAPHX_DISABLE_GATHER_CONCAT_FUSION{}))
            return;
        
        auto concat_ins = r.result;
        auto concat_op = any_cast<op::concat>(concat_ins->get_operator());
        
        // Get all inputs to concat
        auto concat_inputs = concat_ins->inputs();
        
        // Find which inputs are gathers
        std::vector<instruction_ref> gather_inputs;
        std::vector<instruction_ref> non_gather_inputs;
        std::vector<std::size_t> gather_positions;
        
        for(std::size_t i = 0; i < concat_inputs.size(); ++i)
        {
            auto input = concat_inputs[i];
            if(input->name() == "gather")
            {
                gather_inputs.push_back(input);
                gather_positions.push_back(i);
            }
            else
            {
                non_gather_inputs.push_back(input);
            }
        }
        
        // Need at least 2 gathers to be worth fusing
        if(gather_inputs.size() < 2)
            return;
        
        // Check if all gathers are compatible
        if(not are_gathers_compatible(gather_inputs))
            return;
        
        // Don't fuse if there are non-gather inputs mixed in
        // (makes fusion more complex and less beneficial)
        if(not non_gather_inputs.empty())
            return;
        
        // Get gather axis
        auto gather_op = any_cast<op::gather>(gather_inputs[0]->get_operator());
        auto gather_axis = gather_op.axis;
        
        // Trace output
        if(enabled(MIGRAPHX_TRACE_GATHER_CONCAT_FUSION{}))
        {
            std::cout << "Fusing Gather-Concat Pattern:\n";
            std::cout << "  Number of gathers: " << gather_inputs.size() << "\n";
            std::cout << "  Gather axis: " << gather_axis << "\n";
            std::cout << "  Concat axis: " << concat_op.axis << "\n";
            std::cout << "  Output shape: " << concat_ins->get_shape() << "\n";
        }
        
        // Build input list for fused operation:
        // [data0, indices0, data1, indices1, ..., dataN, indicesN]
        std::vector<instruction_ref> fused_inputs;
        for(const auto& gather_ins : gather_inputs)
        {
            auto gather_input_refs = gather_ins->inputs();
            fused_inputs.push_back(gather_input_refs[0]); // data
            fused_inputs.push_back(gather_input_refs[1]); // indices
        }
        
        // Create fused operation with metadata
        auto fused_op = make_op("gpu::fused_gather_concat",
                               {{"gather_axis", gather_axis},
                                {"concat_axis", concat_op.axis},
                                {"num_gathers", gather_inputs.size()}});
        
        // Replace concat with fused operation
        m.replace_instruction(concat_ins, fused_op, fused_inputs);
        
        if(enabled(MIGRAPHX_TRACE_GATHER_CONCAT_FUSION{}))
        {
            std::cout << "  Fusion successful!\n\n";
        }
    }
};

} // anonymous namespace

void fuse_gather_concat::apply(module& m) const
{
    match::find_matches(m, find_gather_concat{});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

