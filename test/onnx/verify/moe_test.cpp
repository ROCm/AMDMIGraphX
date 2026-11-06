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

TEST_CASE(moe_test)
{
    // Gold values computed with a float64 numpy implementation of the
    // com.microsoft MoE specification: relu, k=2, normalized routing weights
    auto p = optimize_onnx("moe_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> x_data = {1.0f, 2.0f, -1.0f, 0.5f, 0.25f, -0.5f, 1.5f, 2.0f};

    migraphx::shape router_shape{migraphx::shape::float_type, {2, 3}};
    std::vector<float> router_data = {2.0f, 1.0f, 0.0f, 0.0f, 1.0f, 2.0f};

    migraphx::shape w1_shape{migraphx::shape::float_type, {3, 2, 4}};
    std::vector<float> w1_data = {0.0f,  0.0f,  -0.5f, 0.0f, -0.5f, -0.5f, 1.0f, -1.0f,
                                  -1.0f, -1.0f, 1.0f,  0.5f, 1.0f,  -0.5f, 0.5f, -0.5f,
                                  0.0f,  -1.0f, 0.5f,  1.0f, 0.5f,  -1.0f, 0.0f, -0.5f};

    migraphx::shape b1_shape{migraphx::shape::float_type, {3, 2}};
    std::vector<float> b1_data = {0.5f, 0.5f, -0.5f, 0.0f, 0.5f, 0.5f};

    migraphx::shape w2_shape{migraphx::shape::float_type, {3, 4, 2}};
    std::vector<float> w2_data = {0.5f,  0.5f,  -1.0f, 0.5f,  0.0f, -1.0f, -0.5f, 0.0f,
                                  0.5f,  0.0f,  0.5f,  1.0f,  0.5f, -0.5f, 0.5f,  0.0f,
                                  -1.0f, -1.0f, 0.5f,  -0.5f, 0.0f, -0.5f, -0.5f, -1.0f};

    migraphx::shape b2_shape{migraphx::shape::float_type, {3, 4}};
    std::vector<float> b2_data = {
        -0.25f, 0.5f, 0.0f, -0.25f, 0.0f, 0.5f, -0.25f, 0.0f, 0.0f, 0.25f, 0.0f, 0.25f};

    migraphx::parameter_map pp;
    pp["input"]               = migraphx::argument(x_shape, x_data.data());
    pp["router_probs"]        = migraphx::argument(router_shape, router_data.data());
    pp["fc1_experts_weights"] = migraphx::argument(w1_shape, w1_data.data());
    pp["fc1_experts_bias"]    = migraphx::argument(b1_shape, b1_data.data());
    pp["fc2_experts_weights"] = migraphx::argument(w2_shape, w2_data.data());
    pp["fc2_experts_bias"]    = migraphx::argument(b2_shape, b2_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.182764645f,
                               -0.231058579f,
                               -0.0672353553f,
                               -0.548293934f,
                               -2.53029289f,
                               2.01207348f,
                               0.156014905f,
                               -0.976793414f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
