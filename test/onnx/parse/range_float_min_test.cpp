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

TEST_CASE(range_float_min_test)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;
    const auto n = var("N", {1, 10});
    const migraphx::shape x_shape{migraphx::shape::float_type, sym_dims({n})};

    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto zero = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto one  = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {1}});
    auto limit =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {5.0f}});
    auto start =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {0}});
    auto delta =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {1}});
    auto x = mm->add_parameter("x", x_shape);

    auto dims = mm->add_instruction(migraphx::make_op("dimensions_of", {{"end", 1}}), x);
    dims      = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), dims);
    auto count = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), dims);
    count = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), count);
    count = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), count);
    count = mm->add_instruction(migraphx::make_op("min"), limit, count);
    count = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), count);
    // The count resolves to min(5, N) through the float round trip, so the range has that length
    const migraphx::shape::dynamic_dimension length{migraphx::sym::min(lit(5), n)};
    auto ret = mm->add_instruction(
        migraphx::make_op("dynamic_range", {{"output_dim", migraphx::to_value(length)}}),
        start,
        count,
        delta);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["x"] = x_shape.dyn_dims();
    options.use_symbolic_shapes     = true;
    EXPECT(p == read_onnx("range_float_min_test.onnx", options));
}
