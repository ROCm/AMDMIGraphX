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

TEST_CASE(attention_double_head_bias_asym_mask_filter_val_test)
{
    auto p = optimize_onnx("attention_double_head_bias_asym_mask_filter_val_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape input_shape{migraphx::shape::float_type, {2, 3, 4}};
    // Taken from attention_op_test.cc from Onnxruntime repo
    std::vector<float> input_data = {0.900670f,  0.448478f,  0.603542f,  0.5393899f, 0.4175282f,
                                     0.98499f,   0.3315366f, 0.671495f,  0.4700915f, 0.371528f,
                                     0.547780f,  0.935466f,  0.539521f,  0.4525090f, 0.722473f,
                                     0.0401732f, 0.4163277f, 0.748433f,  0.837911f,  0.0789781f,
                                     0.6254225f, 0.621954f,  0.3697444f, 0.13887641f};

    migraphx::shape weight_shape{migraphx::shape::float_type, {4, 12}};
    std::vector<float> weight_data = {
        0.7571687f, 0.2359257f, 0.3641960f, 0.7332285f, 0.2369418f, 0.7827402f, 0.0390439f,
        0.732264f,  0.549397f,  0.1185614f, 0.8941f,    0.844365f,  0.9183342f, 0.2338227f,
        0.1152721f, 0.0582466f, 0.629299f,  0.608058f,  0.8386450f, 0.2993283f, 0.369621f,
        0.510179f,  0.951667f,  0.5053690f, 0.1303317f, 0.0906144f, 0.771531f,  0.705696f,
        0.1723916f, 0.5167796f, 0.0805798f, 0.1879250f, 0.6520960f, 0.3181830f, 0.948946f,
        0.59911f,   0.621624f,  0.1569286f, 0.1357425f, 0.5584053f, 0.5210206f, 0.5264289f,
        0.795218f,  0.1389356f, 0.729620f,  0.716727f,  0.5536462f, 0.992480f};

    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data{0.846411f,
                                 0.1964233f,
                                 0.719782f,
                                 0.1685972f,
                                 0.5629161f,
                                 0.669332f,
                                 0.755575f,
                                 0.4544251f,
                                 0.4017672f,
                                 0.7687284f,
                                 0.0021288f,
                                 0.534924f};

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
    std::vector<float> gold = {1.837100f, 1.858778f, 1.813764f, 2.376228f, 1.837100f, 1.858778f,
                               1.813764f, 2.376228f, 1.837100f, 1.858778f, 1.813764f, 2.3762286f,
                               1.405702f, 1.417828f, 1.722106f, 1.765331f, 1.406624f, 1.419057f,
                               1.722792f, 1.765720f, 1.40663f,  1.41908f,  1.720432f, 1.7643524f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
