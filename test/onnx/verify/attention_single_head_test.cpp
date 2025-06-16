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

TEST_CASE(attention_single_head_batch1_test)
{
    auto p = optimize_onnx("attention_single_head_batch1_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape input_shape{migraphx::shape::float_type, {1, 2, 4}};
    // Taken from attention_op_test.cc from Onnxruntime repo
    std::vector<float> input_data = {
        0.552612f, 0.0190766f, 0.1843486f, 0.785303f,
        0.3964166f, 0.8213836f, 0.3575855f, 0.553267f};

    migraphx::shape weight_shape{migraphx::shape::float_type, {4, 12}};
    std::vector<float> weight_data = {
        0.2094991f, 0.1440841f, 0.7808530f, 0.9267818f, 
        0.5590372f, 0.307970f,  0.5298086f, 0.926752f, 
        0.5599791f, 0.3038375f, 0.756784f, 0.1927229f, 
        0.773385f, 0.713124f, 0.1607708f, 0.612250f, 
        0.797840f, 0.7062700f,  0.1150899f, 0.459506f, 
        0.5240009f, 0.750125f, 0.4189021f, 0.9421159f,
        0.487010f, 0.714613f, 0.1044489f, 0.720409f, 
        0.3246490f, 0.8285f,  0.5867274f, 0.3617129f, 
        0.0103818f, 0.3994490f, 0.789325f, 0.0984387f,
        0.601747f, 0.3892322f, 0.723926f, 0.2909479f, 
        0.0136847f, 0.7251270f,  0.4336005f, 0.4485420f,
        0.4666978f, 0.9596705f, 0.9241839f, 0.973919f};

    // Make Bias all zero since ONNX-RT doesn't have non-biased working (CPU/ROCm EP results in segfault)
    // using zero bias allows us to ensure the scale dot attention head is correct before
    // testing with batching/multiple heads/additional parameters
    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data(12, 0.0f);

    migraphx::literal input{input_shape, input_data};
    migraphx::literal weights{weight_shape,  weight_data};
    migraphx::literal bias{bias_shape, bias_data};

    pp["input"]   = input.get_argument();
    pp["weights"] = weights.get_argument();
    pp["bias"]    = bias.get_argument();

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Gold data from Accuracy checker with CPU and ROCm EP
    std::vector<float> gold = {0.824904f, 1.252098f, 1.382309f, 1.220220f,  0.8425703f, 1.283372f, 1.393244f, 1.260538f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(attention_single_head_batch2_test)
{
    auto p = optimize_onnx("attention_single_head_batch2_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape input_shape{migraphx::shape::float_type, {2, 2, 4}};
    // Inputs generated via Accuracy checker tool with CPU/ ROCm EP
    std::vector<float> input_data = {
        0.0492283f, 0.4500561f, 0.494455f, 0.740587f,  
        0.806324f, 0.1756662f, 0.270381f,  0.7099001f, 
        0.7952192f, 0.669329f, 0.1112365f, 0.9587710f,  
        0.673276f, 0.638647f, 0.902916f, 0.788577f, 
        0.9669319f, 0.776140f, 0.1396422f, 0.5938265f,
        0.6022728f, 0.1756347f, 0.8688715f, 0.7493034f,
        0.1602717f, 0.1328538f, 0.785192f, 0.3228947f, 
        0.703402f, 0.885987f, 0.79808f,  0.34035727f};

    migraphx::shape weight_shape{migraphx::shape::float_type, {4, 12}};
    std::vector<float> weight_data = {
        0.0919332f, 0.2026623f, 0.7237189f, 0.276620f,
        0.9985840f, 0.0523637f,  0.9158346f, 0.3989335f, 
        0.5585241f, 0.328731f, 0.2656563f, 0.2488507f, 
        0.1137267f, 0.395782f, 0.2316500f, 0.905320f, 
        0.484163f, 0.746900f,  0.383682f, 0.947402f, 
        0.2134336f, 0.2400222f, 0.9916923f, 0.2402641f,
        0.3755372f, 0.383770f, 0.9288193f, 0.7472737f, 
        0.964018f, 0.3360211f,  0.4654952f, 0.943254f, 
        0.137201f, 0.1918403f, 0.1057573f, 0.0732027f, 
        0.4662406f, 0.6562635f, 0.9587680f, 0.0033893f, 
        0.4242725f, 0.1530022f,  0.505921f, 0.5462211f, 
        0.1485172f, 0.4703321f, 0.3025542f, 0.516489f};

    // Make Bias all zero since ONNX-RT doesn't have non-biased working (CPU/ROCm EP results in segfault)
    // using zero bias allows us to ensure the scale dot attention head is correct before
    // testing with batching/multiple heads/additional parameters
    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data(12, 0.0f);

    migraphx::literal input{input_shape, input_data};
    migraphx::literal weights{weight_shape,  weight_data};
    migraphx::literal bias{bias_shape, bias_data};

    pp["input"]   = input.get_argument();
    pp["weights"] = weights.get_argument();
    pp["bias"]    = bias.get_argument();

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Gold data accuracy checker run against ROCm and CPU EP
    std::vector<float> gold = {
        0.678256f, 0.835921f, 1.026668f, 0.7562694f,
        0.677548f, 0.833964f, 1.022365f, 0.755472f,  
        0.691058f, 0.8484984f, 1.046081f, 0.7651155f,  
        0.71016f,  0.868482f, 1.077081f, 0.7776254f,  
        0.691978f, 0.74249f,  1.033656f, 0.633824f,  
        0.700076f, 0.748571f, 1.049427f, 0.6394321f,  
        0.676378f, 0.731741f, 0.998651f, 0.6247614f, 
        0.7023505f, 0.749478f, 1.059474f, 0.63880485f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
