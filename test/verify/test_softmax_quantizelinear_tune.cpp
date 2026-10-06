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
        auto* mm = p.get_main_module();
        migraphx::shape xs{migraphx::shape::float_type, {1, 12, 256, 256}};
        migraphx::shape ms{migraphx::shape::half_type, {1, 1, 256, 256}};
        auto x    = mm->add_parameter("x", xs);
        auto mask = mm->add_parameter("mask", ms);

        auto broadcast = [&](migraphx::instruction_ref ins) {
            return mm->add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", xs.lens()}}), ins);
        };
        auto scale = [&](float s) {
            return broadcast(mm->add_literal(
                migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {s}}));
        };
        auto convert = [&](migraphx::instruction_ref ins, migraphx::shape::type_t t) {
            return mm->add_instruction(migraphx::make_op("convert", {{"target_type", t}}), ins);
        };

        auto dq  = mm->add_instruction(migraphx::make_op("dequantizelinear"), x, scale(0.25f));
        auto mul = mm->add_instruction(migraphx::make_op("mul"), dq, scale(0.125f));
        auto add = mm->add_instruction(
            migraphx::make_op("add"), mul, convert(broadcast(mask), migraphx::shape::float_type));
        auto softmax = mm->add_instruction(migraphx::make_op("softmax", {{"axis", 3}}),
                                           convert(add, migraphx::shape::half_type));
        auto q       = mm->add_instruction(
            migraphx::make_op("quantizelinear", {{"out_type", migraphx::shape::fp8e4m3fn_type}}),
            convert(softmax, migraphx::shape::float_type),
            scale(1.0f / 448));
        mm->add_return({q});
        return p;
    }

    std::string section() const { return "reduce"; }

    migraphx::compile_options get_compile_options() const
    {
        return migraphx::compile_options{.exhaustive_tune = true};
    }
};
