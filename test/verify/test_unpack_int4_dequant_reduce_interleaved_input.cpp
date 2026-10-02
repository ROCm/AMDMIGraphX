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

// Swiglu over the interleaved gate and up halves of an input feeding an int4
// dequantized matvec written as a reduction: the swiglu fuses as a prologue
// of the reduce reading its inputs with a stride of two along the reduced
// axis
struct test_unpack_int4_dequant_reduce_interleaved_input
    : verify_program<test_unpack_int4_dequant_reduce_interleaved_input>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm    = p.get_main_module();
        auto x      = mm->add_parameter("x", {migraphx::shape::half_type, {1, 1, 128}});
        auto packed = mm->add_parameter("wp", {migraphx::shape::uint8_type, {64, 32}});
        auto scales = mm->add_parameter("scales", {migraphx::shape::half_type, {64, 2, 1}});
        auto zp     = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::uint8_type, {1}}, {8}});
        // The gate and up values alternate along the input
        auto pairs =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 64, 2}}}), x);
        auto gate = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {1}}}), pairs);
        auto up = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {1}}, {"ends", {2}}}), pairs);
        auto sigmoid      = mm->add_instruction(migraphx::make_op("sigmoid"), gate);
        auto silu         = mm->add_instruction(migraphx::make_op("mul"), gate, sigmoid);
        auto h            = mm->add_instruction(migraphx::make_op("mul"), silu, up);
        auto hs           = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), h);
        auto unpack       = mm->add_instruction(migraphx::make_op("unpack_int4"), packed);
        auto scales_bcast = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {64, 2, 32}}}), scales);
        auto scales_flat =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {64, 64}}}), scales_bcast);
        auto zp_bcast =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {64, 64}}}), zp);
        auto dq = mm->add_instruction(
            migraphx::make_op("dequantizelinear"), unpack, scales_flat, zp_bcast);
        auto hu  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), hs);
        auto dqu = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 1}}}), dq);
        auto hb  = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 64, 64}}}), hu);
        auto mul  = mm->add_instruction(migraphx::make_op("mul"), hb, dqu);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), mul);
        mm->add_return({rsum});
        return p;
    }
    std::string section() const { return "reduce"; }
};
