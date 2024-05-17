/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(if_then_else_multi_output_shapes_test)
{
    migraphx::program p = read_onnx("if_then_else_multi_output_shapes_test.onnx");
    p.compile(migraphx::make_target("ref"));
    migraphx::shape s_data{migraphx::shape::float_type, {2, 3, 1}};
    std::vector<float> data = {0.0625, 0.75, -0.0625, 0.125, -0.125, -0.5625};
    migraphx::shape bool_data{migraphx::shape::bool_type, {1}};
    bool b_data = true;

    migraphx::parameter_map pp;
    pp["x"]    = migraphx::argument(s_data, data.data());
    pp["y"]    = migraphx::argument(s_data, data.data());
    pp["cond"] = migraphx::argument(bool_data, &b_data);

    auto result_args = p.eval(pp);
    auto result      = result_args.front();
    auto result_b    = result_args.back();

    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> result_vector_back;
    result_b.visit([&](auto output) { result_vector_back.assign(output.begin(), output.end()); });

    result_vector.insert(result_vector.end(), result_vector_back.begin(), result_vector_back.end());

    std::vector<float> gold = {
        1.0625, 1.75, 0.9375, 1.125, 0.875, 0.4375, 0.125, 1.50, -0.125, 0.250, -0.250, -1.125};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
