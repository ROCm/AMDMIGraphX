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

TEST_CASE(biasadd_test)
{
    migraphx::program p = read_onnx("biasadd_test.onnx");
    p.compile(migraphx::make_target("ref"));

    auto input_type = migraphx::shape::float_type;
    migraphx::shape data_shape{input_type, {2, 3, 4}};
    migraphx::shape bias_shape{input_type, {4}};
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), -12); // range from -12 to 11
    std::vector<float> bias = {-2, -1, 0, 1};
    std::vector<float> skip(24);
    std::iota(skip.begin(), skip.end(), -11); // range from -11 to 12

    migraphx::parameter_map pp;
    pp["x"]    = migraphx::argument(data_shape, data.data());
    pp["bias"] = migraphx::argument(bias_shape, bias.data());
    pp["skip"] = migraphx::argument(data_shape, skip.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    // gold values generated using numpy:
    // >>> import numpy as np
    // >>> x = np.arange(-12, 12, dtype=np.float32)
    // >>> x = np.reshape(x, [2, 3, 4])
    // >>> bias = np.array([-2, -1, 0, 1]).astype(np.float32)
    // >>> skip = np.arange(-11, 13, dtype=np.float32)
    // >>> skip = np.reshape(skip, [2, 3, 4])
    // >>> x_plus_bias = x + bias
    // >>> np.ndarray.flatten(x_plus_bias + skip)
    std::vector<float> gold = {-25., -22., -19., -16., -17., -14., -11., -8., -9., -6., -3., 0.,
                               -1.,  2.,   5.,   8.,   7.,   10.,  13.,  16., 15., 18., 21., 24.};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
