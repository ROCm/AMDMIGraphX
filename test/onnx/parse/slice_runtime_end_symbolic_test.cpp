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

TEST_CASE(slice_runtime_end_symbolic_test)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;
    const auto n = var("N", {1, 16});
    const migraphx::shape x_shape{migraphx::shape::float_type, sym_dims({n, lit(4)})};

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto half =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {0.5f}});
    mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto zero = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto x    = mm->add_parameter("x", x_shape);
    auto y    = mm->add_parameter("y", {migraphx::shape::float_type, {8}});

    half = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8}}}), half);
    auto above = mm->add_instruction(migraphx::make_op("greater"), y, half);
    above      = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), above);
    above = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int64_type}}), above);
    auto count = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), above);
    // The end is only known at run time, so it is a symbol bounded by the length of x
    auto end = var("main_Slice_10_end", {0, 16});
    auto ret = mm->add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {0}},
                           {"starts", {0}},
                           {"ends", migraphx::value::array{migraphx::to_value(end)}}}),
        x,
        zero,
        count);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["x"] = x_shape.dyn_dims();
    options.use_symbolic_shapes     = true;
    EXPECT(p == read_onnx("slice_runtime_end_symbolic_test.onnx", options));
}
