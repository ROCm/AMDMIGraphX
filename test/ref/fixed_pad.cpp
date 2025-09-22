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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(fixed_pad_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 3}, {3, 3}}};
    auto x = mm->add_parameter("x", s);
    mm->add_instruction(migraphx::make_op("fixed_pad"), x);
    p.compile(migraphx::make_target("ref"));
    std::vector<float> data = {-3, -2, -1, 0, 1, 2};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3}};
    migraphx::argument arg(s2, data.data());
    auto result = p.eval({{"x", arg}}).back();
    std::vector<float> results_vector(9);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-3, -2, -1, 0, 1, 2, 0, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(fixed_pad_same_shape_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 2}, {3, 3}}};
    auto x = mm->add_parameter("x", s);
    mm->add_instruction(migraphx::make_op("fixed_pad"), x);
    p.compile(migraphx::make_target("ref"));
    std::vector<float> data = {-3, -2, -1, 0, 1, 2};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3}};
    migraphx::argument arg(s2, data.data());
    auto result = p.eval({{"x", arg}}).back();
    std::vector<float> results_vector(6);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-3, -2, -1, 0, 1, 2};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
