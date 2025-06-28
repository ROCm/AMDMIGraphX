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

TEST_CASE(attention_double_head_bias_asym_mask_scale_test)
{
    auto p = optimize_onnx("attention_double_head_bias_asym_mask_scale_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape input_shape{migraphx::shape::float_type, {2, 3, 4}};
    // Taken from attention_op_test.cc from Onnxruntime repo
    std::vector<float> input_data = {
        0.0029872f, 0.880535f, 0.766311f, 0.9520083f, 0.09028f,  0.754762f,  0.2297131f, 0.26980f,
        0.3421925f, 0.738659f, 0.661078f, 0.3622329f, 0.024412f, 0.5054599f, 0.1434262f, 0.62552f,
        0.553510f,  0.441841f, 0.437590f, 0.587748f,  0.595663f, 0.473892f,  0.504267f,  0.505959f,
    };

    migraphx::shape weight_shape{migraphx::shape::float_type, {4, 12}};
    std::vector<float> weight_data = {
        0.051867f,  0.348283f,  0.088763f,  0.9173639f, 0.38668f,   0.0204705f, 0.349724f,
        0.0139212f, 0.6570288f, 0.901269f,  0.4489016f, 0.5793316f, 0.388074f,  0.3070760f,
        0.0078492f, 0.4900729f, 0.924945f,  0.0638553f, 0.3836499f, 0.1902723f, 0.0636417f,
        0.2222438f, 0.9386154f, 0.2971302f, 0.2408046f, 0.2437261f, 0.155429f,  0.8300618f,
        0.773222f,  0.9903515f, 0.2512911f, 0.504405f,  0.0147009f, 0.708178f,  0.9458579f,
        0.4706725f, 0.1297135f, 0.0521970f, 0.0465745f, 0.2692676f, 0.849494f,  0.565883f,
        0.982996f,  0.44857f,   0.2589197f, 0.2094598f, 0.5639232f, 0.48233464f};

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

    migraphx::literal input{input_shape, input_data};
    migraphx::literal weights{weight_shape, weight_data};
    migraphx::literal bias{bias_shape, bias_data};
    migraphx::literal mask{mask_shape, mask_data};

    pp["input"]      = input.get_argument();
    pp["weights"]    = weights.get_argument();
    pp["bias"]       = bias.get_argument();
    pp["mask_index"] = mask.get_argument();

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
