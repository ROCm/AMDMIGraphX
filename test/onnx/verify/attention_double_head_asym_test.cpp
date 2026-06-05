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

TEST_CASE(attention_double_head_bias_mask_asym_test1)
{
    auto p = optimize_onnx("attention_double_head_bias_asym_mask_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape input_shape{migraphx::shape::float_type, {2, 3, 4}};
    // Taken from attention_op_test.cc from Onnxruntime repo
    std::vector<float> input_data = {0.424442f,  0.391239f,  0.895206f,  0.329658f,  0.806407f,
                                     0.841954f,  0.166734f,  0.3947177f, 0.769718f,  0.1397568f,
                                     0.1470478f, 0.8689365f, 0.0794171f, 0.3475261f, 0.32916f,
                                     0.8890806f, 0.2026632f, 0.9164701f, 0.381084f,  0.3943821f,
                                     0.1804103f, 0.3630361f, 0.217396f,  0.71882147f};

    migraphx::shape weight_shape{migraphx::shape::float_type, {4, 12}};
    std::vector<float> weight_data = {
        0.5328653f, 0.990889f,  0.3436559f, 0.0316787f, 0.710585f,  0.2747402f, 0.9005202f,
        0.1460364f, 0.0912429f, 0.2431252f, 0.583230f,  0.1790725f, 0.436488f,  0.857843f,
        0.9507031f, 0.7371746f, 0.4792084f, 0.3621650f, 0.266098f,  0.0429294f, 0.8168309f,
        0.878928f,  0.777824f,  0.2294480f, 0.619469f,  0.8848825f, 0.5321201f, 0.261897f,
        0.0185219f, 0.1786854f, 0.5919701f, 0.636265f,  0.047953f,  0.895432f,  0.4190886f,
        0.870233f,  0.0969924f, 0.718856f,  0.4210773f, 0.885644f,  0.5691978f, 0.626694f,
        0.255162f,  0.7583785f, 0.638615f,  0.810659f,  0.573155f,  0.73166305f};

    // Make Bias all zero since ONNX-RT doesn't have non-biased working (CPU/ROCm EP results in
    // segfault) using zero bias allows us to ensure the scale dot attention head is correct before
    // testing with batching/multiple heads/additional parameters
    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data = {0.64068f,
                                    0.740358f,
                                    0.4191080f,
                                    0.4902010f,
                                    0.41921f,
                                    0.306204f,
                                    0.5357299f,
                                    0.276288f,
                                    0.311639f,
                                    0.271845f,
                                    0.982978f,
                                    0.998660f};

    migraphx::shape mask_shape{migraphx::shape::int32_type, {2, 3}};
    std::vector<float> mask_data = {0, 1, 0, 0, 1, 1};

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

    // Gold data from Accuracy checker with CPU and ROCm EP
    std::vector<float> gold = {1.333021f, 1.677202f, 2.40430f,  1.770149f, 1.333021f, 1.677202f,
                               2.40430f,  1.770149f, 1.333021f, 1.677202f, 2.40430f,  1.770149f,
                               1.229508f, 1.6116f,   2.03985f,  1.847749f, 1.23127f,  1.614291f,
                               2.044070f, 1.848215f, 1.229239f, 1.61129f,  2.040097f, 1.8477759f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(attention_double_head_asym_mask_test2)
{

    auto p = optimize_onnx("attention_double_head_bias_asym_mask_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape input_shape{migraphx::shape::float_type, {2, 3, 4}};
    // Inputs generated via Accuracy checker tool with CPU/ ROCm EP
    std::vector<float> input_data = {0.563641f,  0.596089f,  0.6873016f, 0.2283578f, 0.586944f,
                                     0.3032279f, 0.0856082f, 0.2830461f, 0.4797580f, 0.5876359f,
                                     0.1328294f, 0.657682f,  0.1587376f, 0.1097770f, 0.9658276f,
                                     0.3364122f, 0.936998f,  0.902808f,  0.4177504f, 0.9092307f,
                                     0.4622655f, 0.631753f,  0.0453660f, 0.990014f};

    migraphx::shape weight_shape{migraphx::shape::float_type, {4, 12}};
    std::vector<float> weight_data = {
        0.550476f,  0.2990797f, 0.3499473f, 0.502457f,  0.8914552f, 0.6897412f, 0.350504f,
        0.838558f,  0.966813f,  0.8194670f, 0.3466599f, 0.10317f,   0.0214707f, 0.3867859f,
        0.991419f,  0.963052f,  0.640068f,  0.5958870f, 0.2184076f, 0.7763474f, 0.0521373f,
        0.0766689f, 0.815702f,  0.3086021f, 0.2658478f, 0.954776f,  0.5995340f, 0.1700607f,
        0.0902024f, 0.4917013f, 0.6761719f, 0.1209063f, 0.887309f,  0.1192176f, 0.4385506f,
        0.313657f,  0.0626774f, 0.3990616f, 0.5720665f, 0.5780725f, 0.0506778f, 0.4353910f,
        0.463869f,  0.449338f,  0.673546f,  0.9808122f, 0.977306f,  0.76408f};

    // Make Bias all zero since ONNX-RT doesn't have non-biased working (CPU/ROCm EP results in
    // segfault) using zero bias allows us to ensure the scale dot attention head is correct before
    // testing with batching/multiple heads/additional parameters
    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data{0.2705385f,
                                 0.253350f,
                                 0.4282024f,
                                 0.8318862f,
                                 0.957445f,
                                 0.1495394f,
                                 0.05389f,
                                 0.6314289f,
                                 0.5862025f,
                                 0.820197f,
                                 0.3228435f,
                                 0.15375982f};

    migraphx::shape mask_shape{migraphx::shape::int32_type, {2, 3}};
    std::vector<float> mask_data = {0, 1, 1, 0, 0, 1};

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

    // Gold data accuracy checker run against ROCm and CPU EP
    std::vector<float> gold = {1.556113f, 1.791649f, 1.477053f, 0.8038040f, 1.548904f, 1.780873f,
                               1.45414f,  0.788925f, 1.551593f, 1.784892f,  1.477318f, 0.803976f,
                               1.773139f, 2.22387f,  1.985858f, 1.16709f,   1.773139f, 2.22387f,
                               1.985858f, 1.16709f,  1.773139f, 2.22387f,   1.985858f, 1.16709f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
