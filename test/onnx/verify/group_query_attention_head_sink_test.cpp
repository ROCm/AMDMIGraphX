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

TEST_CASE(group_query_attention_head_sink_decode_test)
{
    auto p = read_onnx("group_query_attention_head_sink_decode_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape qkv_shape{migraphx::shape::half_type, {1, 1, 96}};
    std::vector<float> qkv_data = {
        6.153,  7.545,  -0.804, -6.835, -6.139, 2.956,  6.716,  6.893,  7.473,  -9.716, -3.429,
        -4.660, 7.343,  -3.562, 9.919,  -7.503, 4.383,  -2.274, 1.762,  -6.985, 4.702,  -5.070,
        5.009,  6.450,  -2.873, 0.363,  0.052,  -9.278, 1.541,  -2.714, 0.147,  -5.890, -5.202,
        -9.477, 1.640,  9.588,  8.967,  -2.795, -8.801, 7.888,  -7.699, -6.706, -7.154, 6.284,
        -5.744, -5.343, 4.492,  -8.902, 1.595,  0.696,  5.202,  1.360,  -0.066, -1.406, -5.225,
        -4.940, 6.140,  2.266,  -6.849, -7.607, 0.914,  0.885,  9.477,  -7.357, 8.032,  9.065,
        -5.225, 6.465,  0.300,  -9.999, -0.089, 6.549,  -8.623, -7.224, 7.020,  -5.164, -8.470,
        -9.049, 0.766,  -8.397, 2.805,  7.043,  2.467,  8.405,  0.738,  -3.961, -4.948, 7.460,
        2.534,  -4.354, -5.608, 2.411,  -7.487, -0.264, -7.888, 0.128};

    migraphx::shape key_value_shape{migraphx::shape::float_type, {1}};
    std::vector<float> key_value_data = {0};

    migraphx::shape past_key_values_shape{migraphx::shape::half_type, {1, 2, 10, 16}};
    std::vector<float> past_key_values_data(past_key_values_shape.elements(), 1);

    migraphx::shape slk_shape{migraphx::shape::int32_type, {1, 1}};
    std::vector<int> slk_data = {8};

    migraphx::literal qkv{qkv_shape, qkv_data};
    migraphx::literal key{key_value_shape, key_value_data};
    migraphx::literal value{key_value_shape, key_value_data};
    migraphx::literal past_key_values_key{past_key_values_shape, past_key_values_data};
    migraphx::literal past_key_values_value{past_key_values_shape, past_key_values_data};
    migraphx::literal seqlens_k{slk_shape, slk_data};

    migraphx::parameter_map pp;
    pp["qkv"]                   = qkv.get_argument();
    pp["key"]                   = key.get_argument();
    pp["value"]                 = value.get_argument();
    pp["past_key_values_key"]   = past_key_values_key.get_argument();
    pp["past_key_values_value"] = past_key_values_value.get_argument();
    pp["seqlens_k"]             = seqlens_k.get_argument();

    auto outputs       = p.eval(pp);
    const auto& result = outputs.front();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // numpy gold: no rotary; softmax denominator includes the per-head sink logit {0.5, -1},
    // so head 0 (whose visible values are all ones) drops below 1 by the sink probability mass
    std::vector<float> gold = {
        0.990677788, 0.990677788, 0.990677788,  0.990677788, 0.990677788, 0.990677788,
        0.990677788, 0.990677788, 0.990677788,  0.990677788, 0.990677788, 0.990677788,
        0.990677788, 0.990677788, 0.990677788,  0.990677788, 2.80468159,  7.04295147,
        2.46679187,  8.40622907,  0.737792602,  -3.96092527, -4.94920387, 7.4609191,
        2.53319794,  -4.35545546, -5.60935835,  2.4101514,   -7.48825956, -0.263913696,
        -7.88669599, 0.128053026};

    EXPECT(migraphx::verify::verify_range_with_tolerance(result_vector,
                                                         migraphx::verify::expected{gold}));
}
