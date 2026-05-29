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

TEST_CASE(fused_matmul_bcast_verify_test)
{
    // Exercises batch broadcasting combined with alpha: A [1, 3, 4] is broadcast to [2, 3, 4]
    // and multiplied with B [2, 4, 5], scaled by alpha=0.25.
    migraphx::program p = read_onnx("fused_matmul_bcast_verify_test.onnx");
    p.compile(migraphx::make_target("ref"));

    auto input_type = migraphx::shape::float_type;
    migraphx::shape a_shape{input_type, {1, 3, 4}};
    migraphx::shape b_shape{input_type, {2, 4, 5}};

    std::vector<float> a_data(1 * 3 * 4);
    std::iota(a_data.begin(), a_data.end(), 0); // 0..11
    std::vector<float> b_data(2 * 4 * 5);
    std::iota(b_data.begin(), b_data.end(), 0); // 0..39

    migraphx::parameter_map pp;
    pp["1"] = migraphx::argument(a_shape, a_data.data());
    pp["2"] = migraphx::argument(b_shape, b_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Gold values generated with numpy:
    // >>> A = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    // >>> B = np.arange(40, dtype=np.float32).reshape(2, 4, 5)
    // >>> Y = 0.25 * (A @ B)                            # [2, 3, 5]
    std::vector<float> gold = {17.5f,  19.0f,  20.5f,  22.0f,  23.5f,  47.5f,  53.0f,  58.5f,
                               64.0f,  69.5f,  77.5f,  87.0f,  96.5f,  106.0f, 115.5f, 47.5f,
                               49.0f,  50.5f,  52.0f,  53.5f,  157.5f, 163.0f, 168.5f, 174.0f,
                               179.5f, 267.5f, 277.0f, 286.5f, 296.0f, 305.5f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
