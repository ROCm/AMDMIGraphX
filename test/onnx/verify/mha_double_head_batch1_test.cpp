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

#include <migraphx/bf16.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(mha_double_head_biased_batch1_test)
{
    auto p = optimize_onnx("mha_double_head_bias_batch1_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape q_shape{migraphx::shape::float_type, {1, 2, 4}};
    // Taken from attention_op_test.cc from Onnxruntime repo
    std::vector<float> q_data = {0.03, -0.11, 0.44, 1.6, 0.12, -0.08, 1.19, 0.52};

    migraphx::shape k_shape{migraphx::shape::float_type, {1, 2, 4}};
    // Taken from attention_op_test.cc from Onnxruntime repo
    std::vector<float> k_data = {2.78, 2.54, 2.3, 3.96, -0.21, -1.1, -0.72, -2.2};

    migraphx::shape v_shape{migraphx::shape::float_type, {1, 2, 4}};
    // Taken from attention_op_test.cc from Onnxruntime repo
    std::vector<float> v_data = {8.19, -0.53, 3.95, 4.45, -4.59, 0.02, -0.41, -0.63};

    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data = {
        -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

    migraphx::literal query{q_shape, q_data};
    migraphx::literal key{k_shape, k_data};
    migraphx::literal value{v_shape, v_data};
    migraphx::literal bias{bias_shape, bias_data};

    pp["q"]    = query.get_argument();
    pp["k"]    = key.get_argument();
    pp["v"]    = value.get_argument();
    pp["bias"] = bias.get_argument();

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Gold data from AttentionNoMaskIndex from attention_op_test.cc from Onnxruntime
    std::vector<float> gold = {3.1495983600616455f,
                               0.10843668878078461f,
                               4.25f,
                               5.6499996185302734f,
                               3.9696791172027588f,
                               0.073143675923347473f,
                               4.2499995231628418f,
                               5.6499991416931152f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
