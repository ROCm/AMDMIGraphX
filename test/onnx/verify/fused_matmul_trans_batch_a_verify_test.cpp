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

TEST_CASE(fused_matmul_trans_batch_a_verify_test)
{
    // Exercises transBatchA: A [3, 2, 4] is permuted with [1, 0, 2] to [2, 3, 4] before matmul
    // with B [2, 4, 5]. Matches ORT's MatMulComputeHelper: dim-0 is moved to dim-(rank-2).
    migraphx::program p = read_onnx("fused_matmul_trans_batch_a_verify_test.onnx");
    p.compile(migraphx::make_target("ref"));

    auto input_type = migraphx::shape::float_type;
    migraphx::shape a_shape{input_type, {3, 2, 4}};
    migraphx::shape b_shape{input_type, {2, 4, 5}};

    std::vector<float> a_data(3 * 2 * 4);
    std::iota(a_data.begin(), a_data.end(), 0); // 0..23
    std::vector<float> b_data(2 * 4 * 5);
    std::iota(b_data.begin(), b_data.end(), 0); // 0..39

    migraphx::parameter_map pp;
    pp["1"] = migraphx::argument(a_shape, a_data.data());
    pp["2"] = migraphx::argument(b_shape, b_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Gold values generated with numpy:
    // >>> A = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    // >>> A_perm = np.transpose(A, (1, 0, 2))           # [2, 3, 4]
    // >>> B = np.arange(40, dtype=np.float32).reshape(2, 4, 5)
    // >>> Y = A_perm @ B                                # [2, 3, 5]
    std::vector<float> gold = {
        70.0f,   76.0f,   82.0f,   88.0f,   94.0f,   310.0f,  348.0f,  386.0f,  424.0f,  462.0f,
        550.0f,  620.0f,  690.0f,  760.0f,  830.0f,  630.0f,  652.0f,  674.0f,  696.0f,  718.0f,
        1510.0f, 1564.0f, 1618.0f, 1672.0f, 1726.0f, 2390.0f, 2476.0f, 2562.0f, 2648.0f, 2734.0f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
