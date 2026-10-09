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

struct test_softmax_quantizelinear_tune : verify_program<test_softmax_quantizelinear_tune>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto x    = mm->add_parameter("x", {migraphx::shape::float_type, {1, 2, 2, 32769}});
        auto mask = mm->add_parameter("mask", {migraphx::shape::half_type, {1, 1, 2, 32769}});

        auto x_scale =
            mm->add_literal({migraphx::shape{migraphx::shape::float_type, {1}}, {0.25f}});
        auto x_scale_b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 32769}}}), x_scale);
        auto dq = mm->add_instruction(migraphx::make_op("dequantizelinear"), x, x_scale_b);

        auto alpha = mm->add_literal({migraphx::shape{migraphx::shape::float_type, {1}}, {0.125f}});
        auto alpha_b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 32769}}}), alpha);
        auto mul = mm->add_instruction(migraphx::make_op("mul"), dq, alpha_b);

        auto mask_b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 32769}}}), mask);
        auto mask_f = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), mask_b);
        auto add = mm->add_instruction(migraphx::make_op("add"), mul, mask_f);

        auto add_h = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), add);
        auto softmax   = mm->add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), add_h);
        auto softmax_f = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), softmax);

        auto y_scale =
            mm->add_literal({migraphx::shape{migraphx::shape::float_type, {1}}, {1.0f / 448}});
        auto y_scale_b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 32769}}}), y_scale);
        mm->add_instruction(
            migraphx::make_op("quantizelinear", {{"out_type", migraphx::shape::fp8e4m3fn_type}}),
            softmax_f,
            y_scale_b);
        return p;
    }

    std::string section() const { return "reduce"; }

    std::size_t get_tolerance() const { return 1; }
};
