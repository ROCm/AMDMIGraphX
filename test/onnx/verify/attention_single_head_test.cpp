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
    std::vector<float> input_data = {0.552612f,
                                     0.0190766f,
                                     0.1843486f,
                                     0.785303f,
                                     0.3964166f,
                                     0.8213836f,
                                     0.3575855f,
                                     0.553267f};

    migraphx::shape weight_shape{migraphx::shape::float_type, {4, 12}};
    std::vector<float> weight_data = {
        0.2094991f, 0.1440841f, 0.7808530f, 0.9267818f, 0.5590372f, 0.307970f,  0.5298086f,
        0.926752f,  0.5599791f, 0.3038375f, 0.756784f,  0.1927229f, 0.773385f,  0.713124f,
        0.1607708f, 0.612250f,  0.797840f,  0.7062700f, 0.1150899f, 0.459506f,  0.5240009f,
        0.750125f,  0.4189021f, 0.9421159f, 0.487010f,  0.714613f,  0.1044489f, 0.720409f,
        0.3246490f, 0.8285f,    0.5867274f, 0.3617129f, 0.0103818f, 0.3994490f, 0.789325f,
        0.0984387f, 0.601747f,  0.3892322f, 0.723926f,  0.2909479f, 0.0136847f, 0.7251270f,
        0.4336005f, 0.4485420f, 0.4666978f, 0.9596705f, 0.9241839f, 0.973919f};

    // Make Bias all zero since ONNX-RT doesn't have non-biased working (CPU/ROCm EP results in
    // segfault) using zero bias allows us to ensure the scale dot attention head is correct before
    // testing with batching/multiple heads/additional parameters
    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data(12, 0.0f);

    migraphx::literal input{input_shape, input_data};
    migraphx::literal weights{weight_shape, weight_data};
    migraphx::literal bias{bias_shape, bias_data};

    pp["input"]   = input.get_argument();
    pp["weights"] = weights.get_argument();
    pp["bias"]    = bias.get_argument();

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Gold data from Accuracy checker with CPU and ROCm EP
    std::vector<float> gold = {
        0.824904f, 1.252098f, 1.382309f, 1.220220f, 0.8425703f, 1.283372f, 1.393244f, 1.260538f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(attention_single_head_batch2_test)
{
    auto p = optimize_onnx("attention_single_head_batch2_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape input_shape{migraphx::shape::float_type, {2, 2, 4}};
    // Inputs generated via Accuracy checker tool with CPU/ ROCm EP
    std::vector<float> input_data = {0.801922f,
                                     0.3853472f,
                                     0.370015f,
                                     0.860284f,
                                     0.263503f,
                                     0.795997f,
                                     0.5618871f,
                                     0.3625675f,
                                     0.779154f,
                                     0.2929783f,
                                     0.894270f,
                                     0.0289014f,
                                     0.554448f,
                                     0.006574f,
                                     0.9899347f,
                                     0.788727f};

    migraphx::shape weight_shape{migraphx::shape::float_type, {4, 12}};
    std::vector<float> weight_data = {
        0.0310612f, 0.2073076f, 0.422496f,  0.5100093f, 0.445159f,  0.863975f,  0.5272242f,
        0.520853f,  0.670335f,  0.322454f,  0.1155668f, 0.1083195f, 0.228852f,  0.661951f,
        0.524143f,  0.9668951f, 0.1439316f, 0.499265f,  0.114529f,  0.3760084f, 0.0013527f,
        0.0608055f, 0.0439942f, 0.5202921f, 0.71258f,   0.811452f,  0.470202f,  0.778556f,
        0.824370f,  0.2653890f, 0.0469467f, 0.1800754f, 0.5910899f, 0.3669454f, 0.8135282f,
        0.2441214f, 0.6245076f, 0.3581591f, 0.647997f,  0.989607f,  0.4919585f, 0.3192114f,
        0.3586453f, 0.8427194f, 0.2083097f, 0.738242f,  0.866889f,  0.50372523f};

    // Make Bias all zero since ONNX-RT doesn't have non-biased working (CPU/ROCm EP results in
    // segfault) using zero bias allows us to ensure the scale dot attention head is correct before
    // testing with batching/multiple heads/additional parameters
    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data(12, 0.0f);

    migraphx::literal input{input_shape, input_data};
    migraphx::literal weights{weight_shape, weight_data};
    migraphx::literal bias{bias_shape, bias_data};

    pp["input"]   = input.get_argument();
    pp["weights"] = weights.get_argument();
    pp["bias"]    = bias.get_argument();

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Gold data accuracy checker run against ROCm and CPU EP
    std::vector<float> gold = {0.8458339f,
                               0.938286f,
                               1.074253f,
                               0.798552f,
                               0.8394189f,
                               0.9301322f,
                               1.068407f,
                               0.7976641f,
                               1.096127f,
                               0.926494f,
                               1.280131f,
                               0.6113027f,
                               1.098830f,
                               0.9479323f,
                               1.309692f,
                               0.621160f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
