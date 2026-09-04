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

// Swiglu over the two halves of a single int4 dequantized matvec written as
// a reduction, feeding a second matvec: the sliced reduce splits into a
// fused gate/up kernel with the swiglu epilogue
struct test_unpack_int4_dequant_reduce_slice_swiglu
    : verify_program<test_unpack_int4_dequant_reduce_slice_swiglu>
{
    static migraphx::instruction_ref gemv_as_reduce(migraphx::module& mm,
                                                    migraphx::instruction_ref x,
                                                    migraphx::instruction_ref packed,
                                                    migraphx::instruction_ref scales,
                                                    migraphx::instruction_ref zp)
    {
        auto n      = packed->get_shape().lens().front();
        auto k      = 2 * packed->get_shape().lens().back();
        auto blocks = scales->get_shape().lens().at(1);
        auto up     = mm.add_instruction(migraphx::make_op("unpack_int4"), packed);
        auto scales_bcast = mm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {n, blocks, k / blocks}}}), scales);
        auto scales_flat =
            mm.add_instruction(migraphx::make_op("reshape", {{"dims", {n, k}}}), scales_bcast);
        auto zp_bcast =
            mm.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {n, k}}}), zp);
        auto dq =
            mm.add_instruction(migraphx::make_op("dequantizelinear"), up, scales_flat, zp_bcast);
        auto xu  = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), x);
        auto dqu = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 1}}}), dq);
        auto xb  = mm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, n, k}}}), xu);
        auto mul  = mm.add_instruction(migraphx::make_op("mul"), xb, dqu);
        auto rsum = mm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), mul);
        return mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), rsum);
    }

    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::half_type, {1, 1, 32}});
        auto packed1 =
            mm->add_parameter("wp1", {migraphx::shape::uint8_type, {16, 16}});
        auto scales1 = mm->add_parameter("scales1", {migraphx::shape::half_type, {16, 4, 1}});
        auto packed2 = mm->add_parameter("wp2", {migraphx::shape::uint8_type, {4, 4}});
        auto scales2 = mm->add_parameter("scales2", {migraphx::shape::half_type, {4, 1, 1}});
        auto zp      = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::uint8_type, {1}}, {8}});
        auto gate_up = gemv_as_reduce(*mm, x, packed1, scales1, zp);
        auto gate    = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {8}}}), gate_up);
        auto up = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {8}}, {"ends", {16}}}), gate_up);
        auto sigmoid = mm->add_instruction(migraphx::make_op("sigmoid"), gate);
        auto silu    = mm->add_instruction(migraphx::make_op("mul"), gate, sigmoid);
        auto h       = mm->add_instruction(migraphx::make_op("mul"), silu, up);
        auto out     = gemv_as_reduce(*mm, h, packed2, scales2, zp);
        mm->add_return({out});
        return p;
    }
    std::string section() const { return "reduce"; }
};
