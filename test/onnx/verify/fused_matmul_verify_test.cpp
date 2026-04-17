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

TEST_CASE(fused_matmul_verify_test)
{
    migraphx::program p = read_onnx("fused_matmul_verify_test.onnx");
    p.compile(migraphx::make_target("ref"));

    auto input_type = migraphx::shape::float_type;
    migraphx::shape a_shape{input_type, {2, 3, 4}};
    migraphx::shape b_shape{input_type, {2, 5, 4}};

    std::vector<float> a_data(2 * 3 * 4);
    std::iota(a_data.begin(), a_data.end(), 0); // 0..23
    std::vector<float> b_data(2 * 5 * 4);
    std::iota(b_data.begin(), b_data.end(), 0); // 0..39

    migraphx::parameter_map pp;
    pp["1"] = migraphx::argument(a_shape, a_data.data());
    pp["2"] = migraphx::argument(b_shape, b_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Gold values generated with numpy:
    // >>> A = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    // >>> B = np.arange(40, dtype=np.float32).reshape(2, 5, 4)
    // >>> Y = 0.5 * A @ B.transpose(0, 2, 1)
    std::vector<float> gold = {
        7.0f,   19.0f,   31.0f,   43.0f,   55.0f,   19.0f,   63.0f,   107.0f,
        151.0f, 195.0f,  31.0f,   107.0f,  183.0f,  259.0f,  335.0f,  583.0f,
        691.0f, 799.0f,  907.0f,  1015.0f, 755.0f,  895.0f,  1035.0f, 1175.0f,
        1315.0f, 927.0f, 1099.0f, 1271.0f, 1443.0f, 1615.0f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
