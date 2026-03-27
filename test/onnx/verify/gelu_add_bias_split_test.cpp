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

TEST_CASE(gelu_add_bias_split_test)
{
    migraphx::program p = read_onnx("gelu_add_bias_split_test.onnx");
    p.compile(migraphx::make_target("ref"));

    auto input_type = migraphx::shape::float_type;
    migraphx::shape data_shape{input_type, {2, 4, 6}};
    migraphx::shape bias_shape{input_type, {6}};
    std::vector<float> data(48);
    std::iota(data.begin(), data.end(), -24);
    std::vector<float> bias = {-3, -2, -1, 0, 1, 2};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(data_shape, data.data());
    pp["y"] = migraphx::argument(bias_shape, bias.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    // gold values generated using numpy:
    // >>> import numpy as np
    // >>> import math
    // >>> x = np.arange(-24, 24, dtype=np.float32)
    // >>> x = np.reshape(x, [2, 4, 6])
    // >>> y = np.array([-3, -2, -1, 0, 1, 2]).astype(np.float32)
    // >>> sum = x + y
    // >>> sum_left, sum_right = np.split(sum, 2, axis=-1)
    // >>> gelu_out = 0.5 * sum_right * (1 + np.vectorize(math.erf)(sum_right / np.sqrt(2)))
    // >>> np.set_printoptions(suppress=True)
    // >>> np.ndarray.flatten(sum_left * gelu_out)
    std::vector<float> gold = {
        0,          0,          0,          0,           0,           0,           0,   0,
        0.00001577, 0.03644725, 1.11058678, -4.20672373, -8.98785092, -4.99999857, 7,   27,
        55,         91,         135,        187,         247,         315,         391, 475};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
