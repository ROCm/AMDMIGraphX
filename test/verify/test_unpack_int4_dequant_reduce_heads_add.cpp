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

// An int4 dequantized matvec whose output is split into heads and transposed
// before the bias add, as a QKV projection is. The reduce is rebuilt at the
// head dims so the add fuses into it.
struct test_unpack_int4_dequant_reduce_heads_add
    : verify_program<test_unpack_int4_dequant_reduce_heads_add>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape ps{migraphx::shape::uint8_type, {256, 32}};
        migraphx::shape ss{migraphx::shape::half_type, {256, 2, 1}};
        migraphx::shape bs{migraphx::shape::half_type, {1, 4, 1, 64}};
        auto x      = mm->add_parameter("x", {migraphx::shape::half_type, {1, 1, 64}});
        auto packed = mm->add_parameter("wp", ps);
        auto scales = mm->add_parameter("scales", ss);
        auto bias   = mm->add_parameter("bias", bs);
        auto zp     = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::uint8_type, {1}}, {8}});
        auto up           = mm->add_instruction(migraphx::make_op("unpack_int4"), packed);
        auto scales_bcast = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {256, 2, 32}}}), scales);
        auto scales_flat =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {256, 64}}}), scales_bcast);
        auto zp_bcast =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {256, 64}}}), zp);
        auto dq =
            mm->add_instruction(migraphx::make_op("dequantizelinear"), up, scales_flat, zp_bcast);
        auto xu  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), x);
        auto dqu = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 1}}}), dq);
        auto xb  = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 256, 64}}}), xu);
        auto mul  = mm->add_instruction(migraphx::make_op("mul"), xb, dqu);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), mul);
        auto heads =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 4, 64, 1}}}), rsum);
        auto headst = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3, 4}}}), heads);
        auto headss = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {4}}}), headst);
        auto add    = mm->add_instruction(migraphx::make_op("add"), headss, bias);
        mm->add_return({add});
        return p;
    }
    std::string section() const { return "reduce"; }
};
