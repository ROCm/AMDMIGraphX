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

TEST_CASE(mha_double_head_bias_asym_mask_scale_test)
{
    auto p = optimize_onnx("mha_bias_asym_mask_2d_scale_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape q_shape{migraphx::shape::float_type, {2, 2, 4}};
    std::vector<float> query_data = {1.0444,   1.46694, 0.647463, 1.97097, 0.782412, 1.29233,
                                     0.539048, 1.36031, 0.905094, 1.48508, 0.632634, 1.96646,
                                     0.707612, 1.19037, 0.5344,   1.20187, 0.776302, 1.42483,
                                     0.624828, 1.89007, 0.796374, 1.46134, 0.635375, 1.97777};

    migraphx::shape k_shape{migraphx::shape::float_type, {2, 2, 4}};
    std::vector<float> key_data = {2.3318,   1.64208,  1.60944,  1.77893, 1.25478, 0.718364,
                                   0.786262, 1.17953,  1.74936,  1.202,   1.06744, 1.43902,
                                   1.23419,  0.816938, 0.995569, 1.24722, 1.5753,  1.09366,
                                   1.19299,  1.37392,  1.60332,  1.11632, 1.15639, 1.37755};

    migraphx::shape value_shape{migraphx::shape::float_type, {2, 2, 4}};
    std::vector<float> value_data = {1.13264, 1.71478, 2.75432, 1.90153, 0.997459, 1.2426,
                                     1.7832,  1.33311, 1.19222, 1.79091, 2.3413,   1.72189,
                                     1.02915, 1.14124, 1.63861, 1.35184, 1.36728,  1.80437,
                                     2.07335, 1.7597,  1.37682, 1.87957, 2.1393,   1.78558};

    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data{0.3945137f,
                                 0.959043f,
                                 0.4768401f,
                                 0.644274f,
                                 0.1149479f,
                                 0.2881487f,
                                 0.142187f,
                                 0.797775f,
                                 0.816875f,
                                 0.774307f,
                                 0.664817f,
                                 0.818296f};

    migraphx::shape mask_shape{migraphx::shape::int32_type, {2, 3}};
    std::vector<float> mask_data = {0, 0, 1, 1, 1, 1};

    migraphx::literal query{q_shape, query_data};
    migraphx::literal key{k_shape, key_data};
    migraphx::literal value{value_shape, value_data};
    migraphx::literal bias{bias_shape, bias_data};
    migraphx::literal mask{mask_shape, mask_data};

    pp["q"]                = query.get_argument();
    pp["k"]                = key.get_argument();
    pp["v"]                = value.get_argument();
    pp["bias"]             = bias.get_argument();
    pp["key_padding_mask"] = mask.get_argument();

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Gold data from AttentionNoMaskIndex from attention_op_test.cc from Onnxruntime
    std::vector<float> gold = {1.19222f,  1.790912f, 2.341303f, 1.721886f, 1.19222f,  1.790912f,
                               2.341303f, 1.721886f, 1.19222f,  1.790912f, 2.341303f, 1.7218862f,
                               1.263277f, 1.619741f, 1.953595f, 1.635238f, 1.264119f, 1.621471f,
                               1.954919f, 1.636430f, 1.264281f, 1.62180f,  1.955085f, 1.6365795f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
