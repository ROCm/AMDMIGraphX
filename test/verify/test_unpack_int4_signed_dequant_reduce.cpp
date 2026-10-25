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

// Signed int4 variant of the dequantized reduction, checking the
// sign-extending unpack in the fused kernel
struct test_unpack_int4_signed_dequant_reduce
    : verify_program<test_unpack_int4_signed_dequant_reduce>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm    = p.get_main_module();
        auto x      = mm->add_parameter("x", {migraphx::shape::half_type, {1, 1, 32}});
        auto scales = mm->add_parameter("scales", {migraphx::shape::half_type, {16}});
        auto packed = mm->add_parameter("wp", {migraphx::shape::int8_type, {4, 16}});
        auto up     = mm->add_instruction(migraphx::make_op("unpack_int4"), packed);
        auto scales_reshape =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, 4, 1}}}), scales);
        auto scales_bcast = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {4, 4, 8}}}), scales_reshape);
        auto scales_flat =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, 32}}}), scales_bcast);
        auto dq  = mm->add_instruction(migraphx::make_op("dequantizelinear"), up, scales_flat);
        auto xb  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), x);
        auto dqb = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 1}}}), dq);
        auto xbb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 4, 32}}}), xb);
        auto mul  = mm->add_instruction(migraphx::make_op("mul"), xbb, dqb);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), mul);
        auto out  = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), rsum);
        mm->add_return({out});
        return p;
    }

    std::string section() const { return "reduce"; }
};
