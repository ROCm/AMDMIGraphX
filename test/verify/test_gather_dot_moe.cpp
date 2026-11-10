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
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>

// A MoE layer with as many rows as experts: each of the 4 tokens runs through
// 2 of the 4 experts, so the gathered dots are computed for every expert
struct test_gather_dot_moe : verify_program<test_gather_dot_moe>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto x       = mm->add_parameter("x", {migraphx::shape::float_type, {4, 16}});
        auto w1      = mm->add_parameter("w1", {migraphx::shape::float_type, {4, 16, 12}});
        auto b1      = mm->add_parameter("b1", {migraphx::shape::float_type, {4, 12}});
        auto w2      = mm->add_parameter("w2", {migraphx::shape::float_type, {4, 12, 16}});
        auto weights = mm->add_parameter("weights", {migraphx::shape::float_type, {8}});
        auto indices = mm->add_literal(migraphx::literal{
            migraphx::shape{migraphx::shape::int32_type, {8}}, {0, 2, 1, 3, 3, 0, 2, 1}});
        auto xu      = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), x);
        auto xb      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {4, 2, 16}}}), xu);
        auto xr  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {8, 1, 16}}}), xb);
        auto w1g = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w1, indices);
        auto h1  = mm->add_instruction(migraphx::make_op("dot"), xr, w1g);
        auto b1g = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), b1, indices);
        auto b1u = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), b1g);
        auto a1  = mm->add_instruction(migraphx::make_op("add"), h1, b1u);
        auto act = mm->add_instruction(migraphx::make_op("relu"), a1);
        auto w2g = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w2, indices);
        auto h2  = mm->add_instruction(migraphx::make_op("dot"), act, w2g);
        auto wr = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {8, 1, 1}}}), weights);
        auto wb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {8, 1, 16}}}), wr);
        auto scaled = mm->add_instruction(migraphx::make_op("mul"), h2, wb);
        auto experts =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, 2, 16}}}), scaled);
        auto out = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), experts);
        mm->add_return({out});
        return p;
    }
    std::string section() const { return "gemm"; }
};
