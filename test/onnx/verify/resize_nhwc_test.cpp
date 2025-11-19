/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

TEST_CASE(resize_nhwc_test)
{
    migraphx::program p = read_onnx("resize_nhwc_test.onnx");
    p.compile(migraphx::make_target("ref"));

    // Input shape: [1, 3, 2, 2] (NCHW) - using smaller size for easier verification
    migraphx::shape sx{migraphx::shape::float_type, {1, 3, 2, 2}};
    // clang-format off
    std::vector<float> dx = 
    {
        0.0f, 1.0f, 2.0f, 3.0f,    // Channel 0
        4.0f, 5.0f, 6.0f, 7.0f,    // Channel 1
        8.0f, 9.0f, 10.0f, 11.0f   // Channel 2
    };
    // clang-format on
    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sx, dx.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // The output should be [1, 3, 4, 4] after:
    // 1. Transpose to NHWC [1, 2, 2, 3]
    // 2. Resize with scales [1.0, 2.0, 2.0, 1.0] -> [1, 4, 4, 3]
    // 3. Transpose back to NCHW [1, 3, 4, 4]

    // Expected output size
    EXPECT(result_vector.size() == 1 * 3 * 4 * 4);

    // Verify output shape
    EXPECT(result.get_shape() == migraphx::shape{migraphx::shape::float_type, {1, 3, 4, 4}});

    // Expected golden values for resize with asymmetric coordinate transformation
    // and linear interpolation mode
    // clang-format off
    std::vector<float> gold = 
    {
        // Channel 0
        0.0f,  0.5f,  1.0f,  1.0f,
        1.0f,  1.5f,  2.0f,  2.0f,
        2.0f,  2.5f,  3.0f,  3.0f,
        2.0f,  2.5f,  3.0f,  3.0f,
        // Channel 1
        4.0f,  4.5f,  5.0f,  5.0f,
        5.0f,  5.5f,  6.0f,  6.0f,
        6.0f,  6.5f,  7.0f,  7.0f,
        6.0f,  6.5f,  7.0f,  7.0f,
        // Channel 2
        8.0f,  8.5f,  9.0f,  9.0f,
        9.0f,  9.5f,  10.0f, 10.0f,
        10.0f, 10.5f, 11.0f, 11.0f,
        10.0f, 10.5f, 11.0f, 11.0f
    };
    // clang-format on
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
