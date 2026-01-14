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

    migraphx::shape q_shape{migraphx::shape::float_type, {2, 3, 4}};
    std::vector<float> query_data = {
        0.649887f, 0.507894f, 0.170623f, 1.3267f,  0.387899f, 0.333282f, 0.0622077f, 0.716032f,
        0.51058f,  0.526034f, 0.155793f, 1.32218f, 0.313098f, 0.231324f, 0.0575602f, 0.557592f,
        0.381788f, 0.465788f, 0.147988f, 1.24579f, 0.40186f,  0.502293f, 0.158535f,  1.33349f};

    migraphx::shape k_shape{migraphx::shape::float_type, {2, 3, 4}};
    std::vector<float> key_data = {
        2.21686f, 1.35393f,  1.46725f,  0.981156f, 1.13984f, 0.430216f, 0.644075f, 0.38176f,
        1.63441f, 0.913853f, 0.925256f, 0.641248f, 1.11924f, 0.528789f, 0.853382f, 0.449449f,
        1.46035f, 0.805509f, 1.0508f,   0.576144f, 1.48837f, 0.828169f, 1.0142f,   0.579774f};

    migraphx::shape value_shape{migraphx::shape::float_type, {2, 3, 4}};
    std::vector<float> value_data = {
        0.315761f, 0.940478f, 2.08951f, 1.08323f,  0.180584f, 0.468298f, 1.11838f,  0.514818f,
        0.375347f, 1.0166f,   1.67648f, 0.903589f, 0.212276f, 0.36693f,  0.973797f, 0.533547f,
        0.550404f, 1.03006f,  1.40853f, 0.941403f, 0.559943f, 1.10526f,  1.47448f,  0.96728f};

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
    std::vector<int32_t> mask_data = {0, 0, 1, 1, 1, 1};

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
