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

TEST_CASE(qlinear_mat_mul_2d_per_channel_test)
{
    migraphx::program p = read_onnx("QLinearMatMul_2d_per_channel.onnx");

    p.compile(migraphx::make_target("ref"));

    // Generate test data for A
    migraphx::shape a_shape{migraphx::shape::uint8_type, {1, 2048}};
    std::vector<uint8_t> data_a(2048);
    for(std::size_t i = 0; i < data_a.size(); ++i)
    {
        data_a[i] = static_cast<uint8_t>(i % 256);
    }

    migraphx::parameter_map pp;
    pp["A"] = migraphx::argument(a_shape, data_a.data());

    auto result = p.eval(pp).back();

    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Verify output shape
    EXPECT(result.get_shape().lens() == std::vector<std::size_t>{1, 1000});
    EXPECT(result.get_shape().type() == migraphx::shape::uint8_type);

    // Gold values computed using numpy reference implementation
    std::vector<uint8_t> gold = {
        0, 255, 255, 0, 0, 0, 11, 53, 0, 255, 255, 255, 0, 0, 255, 255, 255, 157, 139, 0, 0, 255, 187, 255, 0,
        255, 0, 255, 0, 184, 0, 255, 0, 122, 0, 0, 0, 255, 0, 0, 234, 0, 255, 0, 255, 0, 255, 255, 0, 255,
        0, 255, 0, 70, 0, 34, 0, 255, 95, 255, 0, 0, 255, 0, 255, 0, 0, 255, 255, 255, 247, 0, 0, 255, 0,
        0, 255, 0, 0, 0, 255, 255, 0, 0, 255, 255, 255, 0, 255, 0, 0, 255, 255, 0, 255, 0, 0, 0, 0, 255,
        0, 0, 0, 255, 0, 0, 255, 157, 255, 255, 255, 255, 0, 86, 255, 255, 243, 255, 0, 0, 0, 0, 255, 255, 255,
        215, 0, 255, 255, 255, 178, 0, 0, 0, 0, 0, 255, 147, 0, 0, 255, 68, 52, 193, 255, 255, 0, 255, 46, 0,
        0, 106, 0, 0, 83, 0, 255, 0, 0, 0, 255, 255, 255, 255, 244, 255, 255, 255, 85, 255, 0, 45, 66, 255, 0,
        255, 0, 123, 0, 0, 255, 255, 255, 255, 0, 255, 0, 0, 255, 0, 31, 46, 255, 255, 0, 0, 255, 0, 255, 0,
        255, 0, 0, 0, 255, 255, 255, 0, 218, 0, 255, 255, 0, 0, 255, 170, 0, 255, 0, 0, 169, 255, 0, 0, 255,
        44, 0, 237, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 255, 0, 255, 255, 0, 255, 0, 255, 0, 0,
        255, 65, 255, 0, 0, 0, 88, 171, 0, 0, 255, 255, 0, 255, 0, 255, 255, 255, 0, 255, 175, 255, 255, 0, 255,
        0, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 236, 0, 255, 83, 0, 0, 255, 0, 61, 255, 0, 130,
        0, 0, 255, 0, 0, 255, 255, 0, 255, 0, 255, 0, 0, 0, 0, 97, 255, 255, 0, 0, 0, 0, 0, 255, 0,
        164, 255, 255, 255, 255, 255, 120, 107, 255, 98, 0, 255, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0,
        255, 215, 255, 155, 255, 255, 255, 0, 0, 0, 0, 255, 0, 255, 0, 255, 0, 0, 68, 0, 255, 132, 0, 255, 255,
        255, 75, 255, 62, 0, 0, 255, 0, 255, 255, 0, 0, 255, 255, 0, 255, 0, 241, 0, 0, 0, 0, 0, 255, 0,
        228, 0, 0, 0, 0, 0, 0, 45, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 255, 255, 0, 255, 0, 176, 102,
        0, 0, 255, 0, 188, 0, 0, 255, 255, 255, 0, 0, 255, 33, 0, 255, 0, 0, 0, 255, 0, 61, 0, 255, 0,
        255, 64, 0, 255, 0, 0, 255, 255, 0, 255, 255, 255, 0, 255, 0, 255, 255, 255, 255, 0, 0, 66, 255, 0, 0,
        255, 0, 43, 255, 255, 0, 230, 255, 255, 255, 0, 0, 0, 198, 0, 0, 77, 129, 0, 0, 255, 39, 0, 255, 0,
        177, 77, 0, 0, 0, 0, 255, 255, 200, 255, 201, 255, 0, 0, 255, 0, 255, 255, 0, 0, 255, 0, 255, 255, 0,
        255, 0, 255, 0, 255, 255, 0, 255, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 185, 24, 99, 39, 135, 0, 0,
        0, 255, 0, 157, 0, 0, 72, 6, 0, 0, 255, 0, 255, 255, 0, 255, 0, 19, 255, 0, 255, 0, 255, 0, 0,
        255, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 255, 255, 255, 255, 0, 232, 89, 0, 235, 0, 255, 255, 255, 255,
        0, 255, 255, 255, 0, 87, 0, 255, 0, 0, 255, 97, 44, 0, 0, 0, 0, 90, 75, 255, 255, 0, 255, 255, 0,
        255, 255, 0, 0, 0, 255, 255, 0, 255, 0, 102, 255, 49, 147, 0, 0, 152, 0, 255, 0, 0, 255, 43, 0, 255,
        0, 0, 72, 0, 157, 111, 255, 255, 0, 0, 0, 0, 255, 255, 0, 106, 255, 0, 0, 255, 255, 0, 255, 255, 0,
        255, 255, 255, 255, 0, 96, 0, 0, 0, 255, 0, 255, 0, 255, 51, 0, 255, 255, 255, 32, 255, 79, 0, 0, 255,
        0, 0, 0, 0, 255, 0, 0, 0, 98, 255, 255, 0, 255, 0, 139, 0, 0, 255, 0, 255, 153, 255, 255, 255, 255,
        0, 255, 0, 0, 255, 255, 255, 255, 255, 231, 0, 218, 255, 0, 0, 82, 255, 0, 255, 83, 87, 0, 255, 0, 178,
        255, 255, 0, 0, 0, 0, 255, 0, 0, 255, 183, 255, 0, 0, 0, 255, 0, 0, 179, 130, 255, 255, 0, 0, 0,
        255, 255, 0, 255, 212, 0, 175, 0, 0, 162, 0, 0, 2, 0, 0, 0, 61, 255, 0, 0, 0, 255, 0, 0, 0,
        0, 0, 255, 255, 0, 255, 0, 255, 0, 0, 255, 0, 255, 255, 0, 0, 0, 0, 255, 0, 139, 255, 255, 255, 255,
        255, 58, 0, 255, 0, 255, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 195, 0, 0, 255, 0, 0, 217, 255, 0,
        0, 46, 255, 253, 255, 80, 0, 158, 0, 0, 0, 0, 255, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0, 131, 255,
        255, 255, 0, 0, 255, 12, 0, 0, 0, 98, 0, 255, 255, 191, 0, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0,
        0, 255, 255, 149, 0, 0, 255, 63, 98, 90, 0, 98, 255, 0, 0, 255, 255, 255, 165, 255, 255, 0, 0, 174, 255,
        0, 255, 255, 0, 0, 143, 0, 0, 255, 0, 222, 0, 0, 255, 255, 0, 77, 255, 255, 0, 255, 255, 255, 255, 255,
        0, 0, 34, 255, 0, 255, 0, 255, 0, 255, 0, 0, 0, 0, 255, 255, 0, 255, 0, 0, 0, 255, 0, 0, 255,
        255, 0, 167, 0, 0, 255, 0, 0, 255, 0, 255, 255, 0, 255, 255, 0, 0, 0, 139, 0, 255, 0, 255, 96, 255
    };

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
