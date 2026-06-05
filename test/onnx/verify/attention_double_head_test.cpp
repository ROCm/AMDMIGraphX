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

TEST_CASE(attention_double_head_batch2_test)
{
    auto p = optimize_onnx("attention_double_head_bias_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape input_shape{migraphx::shape::float_type, {2, 2, 4}};
    // Taken from attention_op_test.cc from Onnxruntime repo
    std::vector<float> input_data = {0.09382452f,
                                     0.91307306f,
                                     0.0450552f,
                                     0.4325229f,
                                     0.6958981f,
                                     0.08963848f,
                                     0.78016704f,
                                     0.6676332f,
                                     0.640169f,
                                     0.79590225f,
                                     0.96877795f,
                                     0.61622876f,
                                     0.93758523f,
                                     0.03348486f,
                                     0.04337639f,
                                     0.7014834f};

    migraphx::shape weight_shape{migraphx::shape::float_type, {4, 12}};
    std::vector<float> weight_data = {
        0.03996459f, 0.47680363f, 0.9262067f,  0.3939682f,  0.5910101f,  0.9245723f,  0.8277026f,
        0.8104753f,  0.6631966f,  0.9758513f,  0.9751434f,  0.17177974f, 0.38831252f, 0.8211175f,
        0.43048874f, 0.78231025f, 0.17060293f, 0.38664082f, 0.7114742f,  0.54639995f, 0.07888859f,
        0.85774803f, 0.00584858f, 0.96129435f, 0.04915032f, 0.25172415f, 0.50097096f, 0.27563626f,
        0.8513271f,  0.4637983f,  0.5921937f,  0.35282055f, 0.49395546f, 0.81918603f, 0.10324866f,
        0.46248904f, 0.77152f,    0.24577223f, 0.2999466f,  0.64094317f, 0.5150664f,  0.4463948f,
        0.8547919f,  0.17810917f, 0.0822628f,  0.61511374f, 0.44340402f, 0.34248054f};

    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data = {
        0.28379261f,
        0.2949806f,
        0.5498075f,
        0.4487797f,
        0.6147562f,
        0.01953102f,
        0.11069784f,
        0.8256148f,
        0.4151909f,
        0.9846566f,
        0.61103475f,
        0.9475537f,
    };

    migraphx::literal input{input_shape, input_data};
    migraphx::literal weights{weight_shape, weight_data};
    migraphx::literal bias{bias_shape, bias_data};

    pp["input"]   = input.get_argument();
    pp["weights"] = weights.get_argument();
    pp["bias"]    = bias.get_argument();

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Gold data from AttentionNoMaskIndex from attention_op_test.cc from Onnxruntime
    std::vector<float> gold = {1.1643721f,
                               2.6504831f,
                               1.4329824f,
                               1.8247899f,
                               1.1484636f,
                               2.636544f,
                               1.4694388f,
                               1.8119928f,
                               1.350189f,
                               3.1860104f,
                               1.6439152f,
                               2.3360693f,
                               1.3286103f,
                               3.1122856f,
                               1.6598269f,
                               2.2605965f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
