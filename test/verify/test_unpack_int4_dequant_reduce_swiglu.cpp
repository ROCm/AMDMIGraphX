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

// Swiglu over two int4 dequantized matvecs written as reductions: the
// activation is scaled by a pointwise mul before both branches and both
// unpacks fuse into a single reduce kernel
struct test_unpack_int4_dequant_reduce_swiglu
    : verify_program<test_unpack_int4_dequant_reduce_swiglu>
{
    static migraphx::instruction_ref gemv_as_reduce(migraphx::module& mm,
                                                    migraphx::instruction_ref x,
                                                    migraphx::instruction_ref packed,
                                                    migraphx::instruction_ref scales,
                                                    migraphx::instruction_ref zp)
    {
        auto up           = mm.add_instruction(migraphx::make_op("unpack_int4"), packed);
        auto scales_bcast = mm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {8, 4, 8}}}), scales);
        auto scales_flat =
            mm.add_instruction(migraphx::make_op("reshape", {{"dims", {8, 32}}}), scales_bcast);
        auto zp_bcast =
            mm.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8, 32}}}), zp);
        auto dq =
            mm.add_instruction(migraphx::make_op("dequantizelinear"), up, scales_flat, zp_bcast);
        auto xu  = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), x);
        auto dqu = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 1}}}), dq);
        auto xb  = mm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 8, 32}}}), xu);
        auto mul  = mm.add_instruction(migraphx::make_op("mul"), xb, dqu);
        auto rsum = mm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), mul);
        return mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), rsum);
    }

    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape ps{migraphx::shape::uint8_type, {8, 16}};
        migraphx::shape ss{migraphx::shape::half_type, {8, 4, 1}};
        auto x       = mm->add_parameter("x", {migraphx::shape::half_type, {1, 1, 32}});
        auto xscale  = mm->add_parameter("xscale", {migraphx::shape::half_type, {32}});
        auto packed1 = mm->add_parameter("wp1", ps);
        auto scales1 = mm->add_parameter("scales1", ss);
        auto packed2 = mm->add_parameter("wp2", ps);
        auto scales2 = mm->add_parameter("scales2", ss);
        auto zp      = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::uint8_type, {1}}, {8}});
        auto xscale_bcast = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 32}}}), xscale);
        auto xs      = mm->add_instruction(migraphx::make_op("mul"), x, xscale_bcast);
        auto r1      = gemv_as_reduce(*mm, xs, packed1, scales1, zp);
        auto r2      = gemv_as_reduce(*mm, xs, packed2, scales2, zp);
        auto sigmoid = mm->add_instruction(migraphx::make_op("sigmoid"), r1);
        auto silu    = mm->add_instruction(migraphx::make_op("mul"), r1, sigmoid);
        auto out     = mm->add_instruction(migraphx::make_op("mul"), silu, r2);
        mm->add_return({out});
        return p;
    }

    std::string section() const { return "reduce"; }
};
