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

// The gathered dot of a quantized MoE layer with as many rows as experts: the
// packed weights of every expert are dequantized with float scales and
// converted to the type of the dot
struct test_gather_dot_unpack_int4 : verify_program<test_gather_dot_unpack_int4>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto x       = mm->add_parameter("x", {migraphx::shape::half_type, {4, 16}});
        auto w       = mm->add_parameter("w", {migraphx::shape::uint8_type, {4, 12, 8}});
        auto scales  = mm->add_parameter("scales", {migraphx::shape::float_type, {4, 12}});
        auto indices = mm->add_literal(migraphx::literal{
            migraphx::shape{migraphx::shape::int32_type, {8}}, {0, 2, 1, 3, 3, 0, 2, 1}});
        auto zero    = mm->add_literal(migraphx::literal{{migraphx::shape::uint8_type, {1}}, {8}});
        auto xu      = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), x);
        auto xb      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {4, 2, 16}}}), xu);
        auto xr = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {8, 1, 16}}}), xb);
        auto wg = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, indices);
        auto wu = mm->add_instruction(migraphx::make_op("unpack_int4"), wg);
        auto sg = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), scales, indices);
        auto su = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), sg);
        auto sb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {8, 12, 16}}}), su);
        auto zb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {8, 12, 16}}}), zero);
        auto dq = mm->add_instruction(migraphx::make_op("dequantizelinear"), wu, sb, zb);
        auto wh = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), dq);
        auto wt =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), wh);
        auto out = mm->add_instruction(migraphx::make_op("dot"), xr, wt);
        mm->add_return({out});
        return p;
    }
    std::string section() const { return "gemm"; }
};
