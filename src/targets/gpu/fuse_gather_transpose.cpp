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
#include <migraphx/gpu/fuse_gather_transpose.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/op/gather.hpp>
#include <migraphx/op/transpose.hpp>
#include <migraphx/op/concat.hpp>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_GATHER_TRANSPOSE_FUSION)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_GATHER_TRANSPOSE_FUSION)

namespace {

/**
 * Pattern 1: Single gather followed by transpose
 */
struct find_gather_transpose
{
    auto matcher() const
    {
        return match::name("transpose")(
            match::arg(0)(match::name("gather").bind("gather")));
    }
    
    void apply(module& m, const match::matcher_result& r) const
    {
        if(enabled(MIGRAPHX_DISABLE_GATHER_TRANSPOSE_FUSION{}))
            return;
        
        auto transpose_ins = r.result;
        auto gather_ins = r.instructions["gather"];
        
        // Only fuse if gather is single-use
        if(gather_ins->outputs().size() != 1)
            return;
        
        auto gather_op = any_cast<op::gather>(gather_ins->get_operator());
        auto transpose_op = any_cast<op::transpose>(transpose_ins->get_operator());
        
        if(enabled(MIGRAPHX_TRACE_GATHER_TRANSPOSE_FUSION{}))
        {
            std::cout << "Fusing Gather-Transpose Pattern:\n";
            std::cout << "  Gather axis: " << gather_op.axis << "\n";
            std::cout << "  Transpose permutation: [";
            for(size_t i = 0; i < transpose_op.dims.size(); ++i)
            {
                std::cout << transpose_op.dims[i];
                if(i < transpose_op.dims.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
            std::cout << "  Output shape: " << transpose_ins->get_shape() << "\n";
        }
        
        // Create fused operation
        auto fused_op = make_op("gpu::fused_gather_transpose",
                               {{"gather_axis", gather_op.axis},
                                {"permutation", transpose_op.dims}});
        
        // Replace transpose with fused operation, using gather's inputs
        m.replace_instruction(transpose_ins, fused_op, gather_ins->inputs());
        
        if(enabled(MIGRAPHX_TRACE_GATHER_TRANSPOSE_FUSION{}))
        {
            std::cout << "  Fusion successful!\n\n";
        }
    }
};

/**
 * Pattern 2: Multiple gather+transpose operations feeding into concat
 */
struct find_gather_transpose_concat
{
    auto matcher() const
    {
        return match::name("concat")(
            match::any_of[match::inputs()](
                match::name("transpose")(
                    match::arg(0)(match::name("gather")))));
    }
    
    void apply(module& m, const match::matcher_result& r) const
    {
        if(enabled(MIGRAPHX_DISABLE_GATHER_TRANSPOSE_FUSION{}))
            return;
        
        auto concat_ins = r.result;
        auto concat_op = any_cast<op::concat>(concat_ins->get_operator());
        
        // Check if all inputs are transpose(gather)
        auto concat_inputs = concat_ins->inputs();
        std::vector<instruction_ref> gather_transpose_pairs;
        bool all_gather_transpose = true;
        
        for(const auto& input : concat_inputs)
        {
            if(input->name() == "transpose")
            {
                auto transpose_input = input->inputs()[0];
                if(transpose_input->name() == "gather" && 
                   transpose_input->outputs().size() == 1 &&
                   input->outputs().size() == 1)
                {
                    gather_transpose_pairs.push_back(transpose_input);  // gather
                    gather_transpose_pairs.push_back(input);            // transpose
                }
                else
                {
                    all_gather_transpose = false;
                    break;
                }
            }
            else
            {
                all_gather_transpose = false;
                break;
            }
        }
        
        // Need at least 2 gather+transpose pairs
        if(!all_gather_transpose || gather_transpose_pairs.size() < 4)  // 2 pairs minimum
            return;
        
        // Verify all gathers have same axis and all transposes have same permutation
        auto ref_gather = gather_transpose_pairs[0];
        auto ref_transpose = gather_transpose_pairs[1];
        auto ref_gather_op = any_cast<op::gather>(ref_gather->get_operator());
        auto ref_transpose_op = any_cast<op::transpose>(ref_transpose->get_operator());
        
        for(size_t i = 2; i < gather_transpose_pairs.size(); i += 2)
        {
            auto gather_op = any_cast<op::gather>(gather_transpose_pairs[i]->get_operator());
            auto transpose_op = any_cast<op::transpose>(gather_transpose_pairs[i+1]->get_operator());
            
            if(gather_op.axis != ref_gather_op.axis)
                return;
            
            if(transpose_op.dims != ref_transpose_op.dims)
                return;
        }
        
        if(enabled(MIGRAPHX_TRACE_GATHER_TRANSPOSE_FUSION{}))
        {
            std::cout << "Fusing Gather-Transpose-Concat Pattern:\n";
            std::cout << "  Number of gather+transpose pairs: " << gather_transpose_pairs.size() / 2 << "\n";
            std::cout << "  Gather axis: " << ref_gather_op.axis << "\n";
            std::cout << "  Transpose permutation: [";
            for(size_t i = 0; i < ref_transpose_op.dims.size(); ++i)
            {
                std::cout << ref_transpose_op.dims[i];
                if(i < ref_transpose_op.dims.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
            std::cout << "  Concat axis: " << concat_op.axis << "\n";
        }
        
        // Build input list: [data0, indices0, data1, indices1, ...]
        std::vector<instruction_ref> fused_inputs;
        for(size_t i = 0; i < gather_transpose_pairs.size(); i += 2)
        {
            auto gather_ins = gather_transpose_pairs[i];
            auto gather_inputs = gather_ins->inputs();
            fused_inputs.push_back(gather_inputs[0]); // data
            fused_inputs.push_back(gather_inputs[1]); // indices
        }
        
        // Create fused operation
        auto fused_op = make_op("gpu::fused_gather_transpose_concat",
                               {{"gather_axis", ref_gather_op.axis},
                                {"permutation", ref_transpose_op.dims},
                                {"concat_axis", concat_op.axis},
                                {"num_gathers", gather_transpose_pairs.size() / 2}});
        
        m.replace_instruction(concat_ins, fused_op, fused_inputs);
        
        if(enabled(MIGRAPHX_TRACE_GATHER_TRANSPOSE_FUSION{}))
        {
            std::cout << "  Fusion successful!\n\n";
        }
    }
};

} // anonymous namespace

void fuse_gather_transpose::apply(module& m) const
{
    // First fuse parallel gather+transpose+concat patterns
    match::find_matches(m, find_gather_transpose_concat{});
    
    // Then fuse remaining single gather+transpose patterns
    match::find_matches(m, find_gather_transpose{});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

