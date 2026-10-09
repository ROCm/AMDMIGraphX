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
#include <onnx_test.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>

TEST_CASE(concat_symbolic_test)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;

    const auto a = var("a", {0, 4});
    const auto b = var("b", {0, 6});

    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto x       = mm->add_parameter("x", {migraphx::shape::float_type, sym_dims({a, lit(4)})});
    auto y       = mm->add_parameter("y", {migraphx::shape::float_type, sym_dims({b, lit(4)})});
    auto x_count = mm->add_instruction(
        migraphx::make_op(
            "eval_expr_from_shape",
            {{"expressions", migraphx::to_value(std::vector<migraphx::sym::expr>{a})}}),
        x);
    auto y_count = mm->add_instruction(
        migraphx::make_op(
            "eval_expr_from_shape",
            {{"expressions", migraphx::to_value(std::vector<migraphx::sym::expr>{b})}}),
        y);
    auto concat =
        mm->add_instruction(migraphx::make_op("dyn_concat", {{"axis", 0}}), x, y, x_count, y_count);
    auto buffer = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), concat);
    auto total  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), concat);
    const auto concat_count = var("main_Concat_2", {0, 10});
    auto starts = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto result = mm->add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {0}},
                           {"starts", {0}},
                           {"ends", migraphx::value::array{migraphx::to_value(concat_count)}},
                           {"always_leq", true}}),
        buffer,
        starts,
        total);
    mm->add_return({result});

    migraphx::onnx_options options;
    options.use_symbolic_shapes = true;
    options.dim_params          = {{"a", migraphx::shape::dynamic_dimension{0, 4}},
                                   {"b", migraphx::shape::dynamic_dimension{0, 6}},
                                   {"c", migraphx::shape::dynamic_dimension{0, 10}}};
    auto prog                   = read_onnx("concat_symbolic_test.onnx", options);
    EXPECT(p == prog);
}
