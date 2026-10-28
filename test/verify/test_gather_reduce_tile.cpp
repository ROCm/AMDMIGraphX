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
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>

// The rows of w selected by the indices are scaled and reduced in one fused
// kernel. The many short reductions are tiled along axis 1, past the gather
// axis, which only the block_tile algorithm can read through the gather view.
struct test_gather_reduce_tile : verify_program<test_gather_reduce_tile>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {3, 8192, 128}});
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {2, 1, 128}});
        auto idx = mm->add_literal(
            migraphx::literal{{migraphx::shape::int32_type, {2}}, std::vector<int32_t>{2, 0}});
        auto g  = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx);
        auto xb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 8192, 128}}}), x);
        auto mul  = mm->add_instruction(migraphx::make_op("mul"), g, xb);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), mul);
        mm->add_return({rsum});
        return p;
    }
};
