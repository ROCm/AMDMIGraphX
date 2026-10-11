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

// The matvecs of the experts selected by the indices, as in a MoE layer:
// the gathers of the weights and bias fuse into the reduce reading the
// selected rows in place
struct test_gather_reduce_moe : verify_program<test_gather_reduce_moe>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto x       = mm->add_parameter("x", {migraphx::shape::half_type, {1, 1, 32}});
        auto w       = mm->add_parameter("w", {migraphx::shape::half_type, {8, 16, 32}});
        auto bias    = mm->add_parameter("bias", {migraphx::shape::half_type, {8, 16}});
        auto indices = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {2}}, {5, 2}});
        auto wg = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, indices);
        auto bg = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), bias, indices);
        auto xb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 16, 32}}}), x);
        auto mul  = mm->add_instruction(migraphx::make_op("mul"), xb, wg);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), mul);
        auto rs   = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), rsum);
        auto out  = mm->add_instruction(migraphx::make_op("add"), rs, bg);
        mm->add_return({out});
        return p;
    }
    std::string section() const { return "reduce"; }
};
