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

#include <migraphx/half.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(qmoe_test)
{
    // 4-bit block quantized (block=2) mixture of experts with zero points and
    // fused interleaved swiglu. Gold values computed with a float64 numpy
    // implementation of the com.microsoft QMoE specification
    auto p = optimize_onnx("qmoe_test.onnx");
    p.compile(migraphx::make_target("ref"));

    using migraphx::half;
    migraphx::shape x_shape{migraphx::shape::half_type, {1, 2, 4}};
    std::vector<half> x_data{half{1.0f},
                             half{-1.0f},
                             half{2.0f},
                             half{0.5f},
                             half{-0.5f},
                             half{1.5f},
                             half{0.25f},
                             half{-2.0f}};

    migraphx::shape router_shape{migraphx::shape::half_type, {2, 2}};
    std::vector<half> router_data{half{1.0f}, half{-1.0f}, half{-0.5f}, half{0.5f}};

    migraphx::shape w1_shape{migraphx::shape::uint8_type, {2, 4, 2}};
    std::vector<uint8_t> w1_data = {
        72, 254, 74, 160, 105, 162, 203, 234, 246, 144, 85, 201, 36, 214, 42, 58};

    migraphx::shape s1_shape{migraphx::shape::half_type, {2, 4, 2}};
    std::vector<float> s1_float = {0.25f,
                                   0.25f,
                                   0.25f,
                                   0.5f,
                                   0.5f,
                                   0.25f,
                                   0.125f,
                                   0.5f,
                                   0.25f,
                                   0.5f,
                                   0.25f,
                                   0.125f,
                                   0.5f,
                                   0.25f,
                                   0.125f,
                                   0.25f};
    std::vector<half> s1_data{s1_float.begin(), s1_float.end()};

    migraphx::shape z1_shape{migraphx::shape::uint8_type, {2, 4, 1}};
    std::vector<uint8_t> z1_data = {63, 178, 33, 67, 201, 248, 133, 127};

    migraphx::shape b1_shape{migraphx::shape::half_type, {2, 4}};
    std::vector<float> b1_float = {0.0f, 0.0f, 0.25f, 0.0f, 0.0f, 1.0f, 0.75f, 0.75f};
    std::vector<half> b1_data{b1_float.begin(), b1_float.end()};

    migraphx::shape w2_shape{migraphx::shape::uint8_type, {2, 4, 1}};
    std::vector<uint8_t> w2_data = {43, 180, 37, 148, 118, 84, 112, 242};

    migraphx::shape s2_shape{migraphx::shape::half_type, {2, 4, 1}};
    std::vector<float> s2_float = {0.25f, 0.25f, 0.125f, 0.25f, 0.125f, 0.125f, 0.25f, 0.25f};
    std::vector<half> s2_data{s2_float.begin(), s2_float.end()};

    migraphx::shape z2_shape{migraphx::shape::uint8_type, {2, 4, 1}};
    std::vector<uint8_t> z2_data = {0, 11, 13, 7, 11, 5, 2, 0};

    migraphx::shape b2_shape{migraphx::shape::half_type, {2, 4}};
    std::vector<float> b2_float = {-0.75f, 0.0f, 1.0f, 0.0f, 0.75f, 1.0f, 0.75f, 0.25f};
    std::vector<half> b2_data{b2_float.begin(), b2_float.end()};

    migraphx::parameter_map pp;
    pp["input"]               = migraphx::argument(x_shape, x_data.data());
    pp["router_probs"]        = migraphx::argument(router_shape, router_data.data());
    pp["fc1_experts_weights"] = migraphx::argument(w1_shape, w1_data.data());
    pp["fc1_scales"]          = migraphx::argument(s1_shape, s1_data.data());
    pp["fc1_experts_bias"]    = migraphx::argument(b1_shape, b1_data.data());
    pp["fc2_experts_weights"] = migraphx::argument(w2_shape, w2_data.data());
    pp["fc2_scales"]          = migraphx::argument(s2_shape, s2_data.data());
    pp["fc2_experts_bias"]    = migraphx::argument(b2_shape, b2_data.data());
    pp["fc1_zero_points"]     = migraphx::argument(z1_shape, z1_data.data());
    pp["fc2_zero_points"]     = migraphx::argument(z2_shape, z2_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {-92.9831645f,
                               64.8573547f,
                               12.2649636f,
                               39.4768881f,
                               -3.0110766f,
                               0.0484433247f,
                               -2.06474728f,
                               2.93885893f};
    // Half precision evaluation against a float64 reference
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold, 4096));
}
