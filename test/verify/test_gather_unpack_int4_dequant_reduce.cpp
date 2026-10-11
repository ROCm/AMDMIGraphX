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

// Int4 dequantized matvecs of the experts selected by the indices: the
// gathers of the packed weights and of the per-block scales broadcast over
// the blocks fuse into the reduce with the unpack
struct test_gather_unpack_int4_dequant_reduce
    : verify_program<test_gather_unpack_int4_dequant_reduce>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto x       = mm->add_parameter("x", {migraphx::shape::half_type, {1, 1, 64}});
        auto packed  = mm->add_parameter("wp", {migraphx::shape::uint8_type, {8, 16, 32}});
        auto scales  = mm->add_parameter("scales", {migraphx::shape::half_type, {8, 16, 2, 1}});
        auto indices = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {2}}, {7, 0}});
        auto zp = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::uint8_type, {1}}, {8}});
        auto pg = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), packed, indices);
        auto sg = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), scales, indices);
        auto up = mm->add_instruction(migraphx::make_op("unpack_int4"), pg);
        auto scales_bcast = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 16, 2, 32}}}), sg);
        auto scales_flat = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {2, 16, 64}}}), scales_bcast);
        auto zp_bcast = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 16, 64}}}), zp);
        auto dq =
            mm->add_instruction(migraphx::make_op("dequantizelinear"), up, scales_flat, zp_bcast);
        auto xb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 16, 64}}}), x);
        auto mul  = mm->add_instruction(migraphx::make_op("mul"), xb, dq);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), mul);
        mm->add_return({rsum});
        return p;
    }
    std::string section() const { return "reduce"; }
};
