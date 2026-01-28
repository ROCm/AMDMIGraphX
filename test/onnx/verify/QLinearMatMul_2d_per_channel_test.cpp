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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(qlinear_mat_mul_2d_per_channel_test)
{
    migraphx::program p = read_onnx("qlinearmatmul_2D_perchannel_test.onnx");

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
        255, 198, 255, 255, 66,  255, 255, 94,  0,   60,  20,  255, 255, 0,   0,   106, 175, 58,
        201, 49,  0,   0,   255, 0,   0,   255, 255, 255, 255, 138, 111, 255, 255, 0,   0,   255,
        0,   0,   0,   170, 255, 0,   0,   0,   0,   255, 0,   22,  0,   255, 179, 0,   0,   144,
        255, 133, 0,   0,   0,   0,   0,   255, 161, 133, 0,   0,   255, 108, 0,   142, 255, 0,
        0,   0,   0,   255, 0,   255, 0,   202, 0,   255, 0,   0,   255, 0,   32,  255, 0,   8,
        0,   0,   0,   0,   255, 255, 108, 255, 255, 92,  0,   0,   169, 0,   91,  255, 255, 0,
        0,   225, 255, 255, 255, 0,   255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   255,
        0,   0,   152, 255, 0,   0,   255, 0,   255, 0,   0,   255, 114, 255, 255, 255, 0,   255,
        45,  255, 0,   164, 255, 154, 0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 0,   255,
        0,   255, 0,   255, 0,   0,   255, 0,   0,   255, 0,   255, 255, 0,   255, 0,   0,   0,
        255, 163, 190, 0,   45,  0,   0,   91,  0,   255, 255, 255, 0,   255, 188, 0,   0,   70,
        255, 255, 255, 255, 0,   0,   18,  0,   255, 255, 245, 0,   0,   203, 0,   255, 255, 0,
        54,  0,   0,   0,   255, 255, 255, 255, 0,   0,   255, 0,   0,   255, 0,   0,   0,   84,
        255, 0,   0,   0,   255, 0,   255, 0,   255, 0,   0,   255, 255, 0,   255, 255, 0,   255,
        100, 0,   255, 255, 0,   0,   255, 255, 255, 62,  0,   33,  255, 0,   255, 12,  0,   0,
        255, 0,   0,   0,   255, 255, 0,   162, 255, 255, 0,   255, 255, 255, 0,   0,   255, 255,
        255, 255, 255, 0,   184, 255, 0,   255, 73,  0,   0,   0,   255, 255, 197, 0,   255, 255,
        96,  60,  246, 255, 255, 0,   95,  255, 0,   255, 65,  0,   255, 0,   0,   0,   0,   0,
        0,   255, 0,   0,   255, 0,   0,   0,   183, 0,   255, 23,  0,   255, 0,   255, 0,   255,
        59,  255, 237, 0,   255, 255, 57,  70,  255, 255, 0,   0,   255, 255, 0,   255, 0,   255,
        0,   0,   0,   255, 255, 255, 0,   0,   0,   255, 0,   0,   255, 0,   255, 0,   255, 0,
        218, 255, 255, 255, 0,   255, 0,   48,  255, 58,  0,   0,   255, 255, 0,   0,   0,   0,
        0,   0,   0,   255, 44,  0,   0,   255, 0,   255, 255, 0,   255, 255, 255, 110, 0,   0,
        255, 130, 255, 0,   255, 0,   0,   0,   0,   255, 159, 0,   0,   115, 255, 255, 255, 106,
        100, 0,   0,   18,  255, 0,   0,   0,   255, 0,   255, 255, 0,   0,   0,   255, 0,   0,
        157, 255, 0,   255, 212, 0,   247, 0,   255, 255, 0,   0,   118, 0,   255, 255, 0,   0,
        0,   255, 255, 0,   0,   0,   255, 0,   255, 0,   0,   0,   0,   255, 0,   0,   0,   255,
        0,   0,   255, 255, 0,   0,   20,  0,   0,   255, 255, 0,   255, 0,   255, 0,   255, 0,
        184, 0,   0,   255, 255, 255, 255, 255, 0,   66,  0,   240, 255, 244, 0,   255, 255, 0,
        148, 0,   0,   255, 255, 149, 255, 58,  70,  0,   255, 255, 255, 230, 255, 0,   255, 0,
        66,  255, 0,   67,  0,   0,   0,   255, 0,   255, 255, 0,   0,   0,   0,   0,   255, 0,
        0,   255, 0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 0,   223, 8,   255, 255, 255,
        0,   255, 0,   0,   255, 255, 255, 0,   151, 255, 66,  0,   0,   0,   255, 0,   0,   0,
        0,   255, 0,   255, 0,   0,   255, 204, 225, 188, 255, 255, 149, 0,   255, 0,   255, 0,
        0,   0,   0,   255, 255, 255, 70,  0,   34,  0,   0,   0,   0,   190, 255, 255, 124, 0,
        0,   0,   0,   0,   20,  0,   0,   0,   0,   99,  0,   0,   0,   59,  255, 0,   255, 0,
        255, 255, 0,   0,   0,   255, 7,   255, 0,   0,   255, 255, 0,   0,   0,   0,   0,   255,
        0,   0,   0,   0,   255, 255, 125, 0,   0,   132, 87,  255, 255, 127, 219, 255, 0,   0,
        255, 0,   0,   0,   0,   112, 255, 255, 0,   0,   0,   0,   255, 0,   255, 0,   255, 255,
        255, 255, 255, 0,   0,   0,   0,   117, 0,   122, 0,   255, 255, 255, 0,   0,   181, 0,
        255, 255, 103, 255, 0,   255, 0,   255, 75,  0,   255, 255, 255, 0,   0,   255, 0,   0,
        255, 80,  0,   75,  0,   0,   0,   255, 0,   255, 255, 164, 114, 255, 0,   0,   0,   0,
        0,   0,   255, 255, 84,  255, 0,   89,  0,   255, 0,   255, 0,   0,   255, 255, 0,   255,
        0,   255, 183, 16,  255, 0,   255, 255, 0,   0,   255, 255, 167, 255, 0,   104, 0,   255,
        0,   216, 0,   255, 255, 255, 255, 255, 0,   0,   0,   255, 255, 7,   255, 0,   0,   255,
        255, 255, 255, 255, 255, 255, 0,   0,   255, 0,   255, 255, 255, 255, 0,   243, 255, 255,
        151, 0,   255, 0,   0,   0,   255, 255, 238, 255, 0,   0,   0,   0,   255, 255, 0,   0,
        0,   0,   0,   0,   0,   255, 0,   0,   0,   0,   0,   255, 23,  0,   255, 214, 59,  255,
        0,   45,  255, 0,   0,   0,   0,   255, 255, 0,   255, 0,   3,   0,   0,   39,  23,  0,
        255, 119, 0,   255, 0,   255, 147, 0,   0,   0,   255, 76,  163, 0,   255, 0,   255, 0,
        255, 0,   0,   255, 255, 17,  0,   255, 0,   0,   0,   255, 0,   255, 255, 0,   0,   0,
        0,   255, 0,   0,   255, 0,   41,  0,   0,   255, 255, 255, 255, 255, 255, 0,   0,   0,
        0,   255, 255, 0,   255, 0,   255, 255, 0,   0,   0,   0,   0,   255, 0,   0,   255, 0,
        255, 255, 255, 255, 0,   0,   0,   255, 0,   0,   255, 255, 0,   0,   0,   255, 255, 0,
        255, 255, 255, 255, 0,   255, 255, 255, 255, 255, 255, 239, 0,   255, 255, 0,   255, 0,
        0,   0,   255, 0,   255, 0,   255, 0,   255, 255
    };

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
