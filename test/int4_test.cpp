/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-24 Advanced Micro Devices, Inc. All rights reserved.
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

#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/propagate_constant.hpp>

#include <migraphx/quantize_int4.hpp>

#include <migraphx/generate.hpp>
#include <test.hpp>

void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m,
                         {migraphx::quantize_int4_pass{},
                          migraphx::propagate_constant{},
                          migraphx::dead_code_elimination{}});
}

TEST_CASE(int4_test)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {16, 8, 6, 6}}));
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        m1.add_instruction(migraphx::make_op("relu"), conv);
    }
    migraphx::run_passes(m1,
                         {migraphx::quantize_int4_pass{},
                          migraphx::propagate_constant{},
                          migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w = m2.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {16, 8, 6, 6}}));
        auto conv = m2.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        m2.add_instruction(migraphx::make_op("relu"), conv);
    }
    migraphx::run_passes(m2, {migraphx::quantize_int4_pass{}, migraphx::dead_code_elimination{}});

    // EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
