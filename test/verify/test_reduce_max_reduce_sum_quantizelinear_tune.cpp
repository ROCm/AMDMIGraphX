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

struct test_reduce_max_reduce_sum_quantizelinear_tune
    : verify_program<test_reduce_max_reduce_sum_quantizelinear_tune>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        // The odd reduction length can't be vectorized, so exhaustive tuning compiles block
        // sizes where each lane holds enough elements that the generated reduce isn't inlined
        auto x      = mm->add_parameter("x", {migraphx::shape::half_type, {1, 2, 2, 32769}});
        auto mask   = mm->add_parameter("mask", {migraphx::shape::half_type, {1, 1, 2, 32769}});
        auto mask_b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 32769}}}), mask);
        auto add = mm->add_instruction(migraphx::make_op("add"), x, mask_b);

        auto rmax   = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), add);
        auto rmax_b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 32769}}}), rmax);
        auto sub    = mm->add_instruction(migraphx::make_op("sub"), add, rmax_b);
        auto exp    = mm->add_instruction(migraphx::make_op("exp"), sub);
        auto rsum   = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
        auto rsum_b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 32769}}}), rsum);
        auto div = mm->add_instruction(migraphx::make_op("div"), exp, rsum_b);

        auto div_f = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), div);
        auto scale =
            mm->add_literal({migraphx::shape{migraphx::shape::float_type, {1}}, {1.0f / 448}});
        auto scale_b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 32769}}}), scale);
        mm->add_instruction(
            migraphx::make_op("quantizelinear", {{"out_type", migraphx::shape::fp8e4m3fn_type}}),
            div_f,
            scale_b);
        return p;
    }

    std::string section() const { return "reduce"; }

    migraphx::compile_options get_compile_options() const
    {
        return migraphx::compile_options{.exhaustive_tune = true};
    }

    std::size_t get_tolerance() const { return 1; }
};
