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
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>

#include <test.hpp>

TEST_CASE(eval_expr_input_shape)
{
    using dd = migraphx::shape::dynamic_dimension;
    auto n   = migraphx::sym::var("N", {1, 16});
    auto h   = migraphx::sym::var("H", {1, 32});
    auto w   = migraphx::sym::var("W", {1, 32});

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x",
                               migraphx::shape{migraphx::shape::float_type,
                                                 {dd{n}, dd{migraphx::sym::lit(3)}, dd{h}, dd{w}}});
    mm->add_instruction(migraphx::make_op("eval_expr",
                                          {{"expressions",
                                            migraphx::value::array{
                                                migraphx::to_value(n),
                                                migraphx::to_value(h / migraphx::sym::lit(2)),
                                                migraphx::to_value(w / migraphx::sym::lit(2))}}}),
                        x);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_shape{migraphx::shape::float_type, {7, 3, 10, 12}};
    std::vector<float> data(input_shape.elements());
    auto result = p.eval({{"x", migraphx::argument{input_shape, data.data()}}}).back();

    std::vector<int64_t> values;
    result.visit([&](auto output) { values.assign(output.begin(), output.end()); });
    EXPECT(result.get_shape() == migraphx::shape{migraphx::shape::int64_type, {3}});
    EXPECT(values == std::vector<int64_t>{7, 5, 6});
}

TEST_CASE(eval_expr_multi_symbol)
{
    using dd = migraphx::shape::dynamic_dimension;
    auto m   = migraphx::sym::var("M", {1, 16});
    auto n   = migraphx::sym::var("N", {1, 16});

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {dd{m}, dd{n}}});
    mm->add_instruction(
        migraphx::make_op("eval_expr",
                          {{"expressions", migraphx::value::array{migraphx::to_value(m + n)}}}),
        x);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_shape{migraphx::shape::float_type, {3, 4}};
    std::vector<float> data(input_shape.elements());
    auto result = p.eval({{"x", migraphx::argument{input_shape, data.data()}}}).back();

    EXPECT(result.get_shape() == migraphx::shape{migraphx::shape::int64_type, {1}});
    EXPECT(result.at<int64_t>() == 7);
}
