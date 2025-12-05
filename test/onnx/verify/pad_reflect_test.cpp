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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(pad_reflect_test)
{
    migraphx::program p = read_onnx("pad_reflect_test.onnx");
    p.compile(migraphx::make_target("ref"));

    auto input_type = migraphx::shape::float_type;
    migraphx::shape data_shape{input_type, {2, 2}};
    std::vector<float> data = {1, 2, 3, 4};
    migraphx::parameter_map pp;
    pp["0"]    = migraphx::argument(data_shape, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    // gold values generated using numpy:
    // >>> import numpy as np
    // >>> original_array = np.array([[1, 2], [3, 4]])
    // >>> pad_widths = ((0, 0), (2, 1))
    // >>> padded_array = np.pad(original_array, pad_width=pad_widths, mode='reflect')

    std::vector<float> gold = {1, 2, 1, 2, 1, 3, 4, 3, 4, 3};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(pad_reflect_3l2r_test)
{
    migraphx::program p = read_onnx("pad_reflect_3l2r_test.onnx");
    p.compile(migraphx::make_target("ref"));

    auto input_type = migraphx::shape::float_type;
    migraphx::shape data_shape{input_type, {2, 2}};
    std::vector<float> data = {1, 2, 3, 4};
    migraphx::parameter_map pp;
    pp["0"]    = migraphx::argument(data_shape, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    // gold values generated using numpy:
    // >>> import numpy as np
    // >>> original_array = np.array([[1, 2], [3, 4]])
    // >>> pad_widths = ((0, 0), (3, 2))
    // >>> padded_array = np.pad(original_array, pad_width=pad_widths, mode='reflect')
    std::vector<float> gold = {2, 1, 2, 1, 2, 1, 2, 4, 3, 4, 3, 4, 3, 4};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(pad_reflect_2l2r_test)
{
    migraphx::program p = read_onnx("pad_reflect_2l2r_test.onnx");
    p.compile(migraphx::make_target("ref"));

    auto input_type = migraphx::shape::float_type;
    migraphx::shape data_shape{input_type, {4, 4}};
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    migraphx::parameter_map pp;
    pp["0"]    = migraphx::argument(data_shape, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    // gold values generated using numpy:
    // >>> import numpy as np
    // >>> original_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    // >>> pad_widths = ((0, 0), (2, 2))
    // >>> padded_array = np.pad(original_array, pad_width=pad_widths, mode='reflect')
    std::vector<float> gold = {3,  2,  1,  2,  3,  4,  3,  2, 
        7,  6,  5,  6,  7,  8,  7,  6,
        11, 10, 9, 10, 11, 12, 11, 10,
        15, 14, 13, 14, 15, 16, 15, 14};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
TEST_CASE(pad_reflect_multiaxis_test)
{
    migraphx::program p = read_onnx("pad_reflect_multiaxis_test.onnx");
    p.compile(migraphx::make_target("ref"));

    auto input_type = migraphx::shape::float_type;
    migraphx::shape data_shape{input_type, {2, 3}};
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    migraphx::parameter_map pp;
    pp["0"]    = migraphx::argument(data_shape, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    // gold values generated using numpy:
    // >>> import numpy as np
    // >>> original_array = np.array([[1, 2, 3], [4, 5, 6]])
    // >>> pad_widths = ((0, 2), (2, 0))
    // >>> padded_array = np.pad(original_array, pad_width=pad_widths, mode='reflect')
    std::vector<float> gold = {3, 2, 1, 2, 3, 6, 5, 4, 5, 6, 3, 2, 1, 2, 3, 6, 5, 4, 5, 6};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
