/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

instruction_ref add_dyn_scale_calc(instruction_ref input)
{
    // TODO
}

struct test_mxfp4_gemm : verify_program<test_mxfp4_gemm>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::module_ref mmain = p.get_main_module();
        // TODO these scale literals need to be E8M0 values
        auto x_0 = mmain->add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1, 1000}}, 0));
        auto x_1 = mmain->add_literal(migraphx::abs(migraphx::generate_literal(
            migraphx::shape{migraphx::shape::float_type, {64, 1, 1000}, {1, 1, 64}}, 1)));
        auto x_2 = mmain->add_literal(migraphx::generate_literal(
            migraphx::shape{migraphx::shape::fp4x2_type, {1000, 1024}}, 2));
        auto p_x3 =
            mmain->add_parameter("x3", migraphx::shape{migraphx::shape::float_type, {1, 64, 1}});
        auto p_x1 =
            mmain->add_parameter("x1", migraphx::shape{migraphx::shape::fp4x2_type, {1, 1024}});
        auto x_5 = mmain->add_instruction(migraphx::make_op("unpack_fp4", {{"axis", 1}}), p_x1);
        auto x_6 = mmain->add_instruction(migraphx::make_op("unpack_fp4", {{"axis", 1}}), x_2);
        auto x_7 =
            mmain->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), x_6);
        auto x_8 = mmain->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 64, 32}}}), p_x3);
        auto x_9 = mmain->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2048}}}), x_8);
        auto x_10 = mmain->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {64, 32, 1000}}}), x_1);
        auto x_11 =
            mmain->add_instruction(migraphx::make_op("reshape", {{"dims", {2048, 1000}}}), x_10);
        auto x_12 = mmain->add_instruction(migraphx::make_op("quant_dot"), x_5, x_7, x_9, x_11);
        auto x_13 = mmain->add_instruction(migraphx::make_op("add"), x_12, x_0);
        mmain->add_return({x_13});
    }
    std::string section() const { return "gemm"; }
};
