/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/quantization.hpp>

// Reduced from the action-masking head of a reinforcement learning agent model, which failed to
// compile for the GPU with --fp16. Each branch slices a shared mask and consumes that slice
// twice, once through `1 - mask` and once through `logits * mask`. Because two branches slice the
// same tensor, fp16 quantization turns the shared convert into a pointwise that
// `split_pointwise_through_slices` clones per branch; common subexpression elimination then folds
// the cloned slices back together, leaving a single instruction wired into two operand slots of
// the pointwise fused into the reduce. Reduce code generation has to collapse those slots into one
// lambda parameter, otherwise it emits a duplicate parameter name and HIP compilation fails.
struct test_fp32_fp16_masked_reduce : verify_program<test_fp32_fp16_masked_reduce>
{
    migraphx::program create_program() const
    {
        const std::size_t branch_size = 2;
        const std::size_t branches    = 2;

        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape mask_shape{migraphx::shape::float_type, {1, branch_size * branches}};
        migraphx::shape branch_shape{migraphx::shape::float_type, {1, branch_size}};

        auto mask = mm->add_parameter("mask", mask_shape);
        auto one  = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {1.0f}});

        std::vector<migraphx::instruction_ref> results;
        for(std::size_t i = 0; i < branches; i++)
        {
            auto logits = mm->add_parameter("logits" + std::to_string(i), branch_shape);
            auto slice = mm->add_instruction(migraphx::make_op("slice",
                                                               {{"axes", {1}},
                                                                {"starts", {i * branch_size}},
                                                                {"ends", {(i + 1) * branch_size}}}),
                                             mask);
            // The literal is broadcast per branch so that each branch starts from a distinct
            // instruction, matching what the onnx parser produces.
            auto bone = mm->add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", branch_shape.lens()}}), one);
            auto inverted = mm->add_instruction(migraphx::make_op("sub"), bone, slice);
            auto masked   = mm->add_instruction(migraphx::make_op("mul"), logits, slice);
            auto scores   = mm->add_instruction(migraphx::make_op("sub"), masked, inverted);
            results.push_back(
                mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), scores));
        }
        mm->add_return(results);
        migraphx::quantize_fp16(p);

        return p;
    };

    std::string section() const { return "reduce"; }
};
