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

TEST_CASE(slice_static_data_symbolic_end_test)
{
    auto sequence_length = migraphx::sym::var("sequence_length", {1, 64}, {1});

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto data =
        mm->add_parameter("data", migraphx::shape{migraphx::shape::float_type, {64, 128}});
    auto length_source = mm->add_parameter(
        "length_source",
        migraphx::shape{migraphx::shape::float_type,
                        {migraphx::shape::dynamic_dimension{sequence_length}}});
    auto starts =
        mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {1}});
    auto ends = mm->add_instruction(
        migraphx::make_op("dimensions_of", {{"start", 0}, {"end", 1}}), length_source);
    auto output = mm->add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {0}},
                           {"starts", {0}},
                           {"ends",
                            migraphx::value::array{migraphx::to_value(sequence_length)}}}),
        data,
        starts,
        ends);
    mm->add_return({output});

    migraphx::onnx_options options;
    options.use_symbolic_shapes           = true;
    options.dim_params["sequence_length"] = {1, 64, {1}};
    auto prog = read_onnx("slice_static_data_symbolic_end_test.onnx", options);

    EXPECT(p == prog);
}
