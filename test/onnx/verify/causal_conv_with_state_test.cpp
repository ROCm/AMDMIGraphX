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

// Gold values follow the ONNXRuntime reference: a depthwise convolution over past_state followed
// by input, with present_state the last (kernel_size - 1) * dilation positions of that sequence.

// Runs the model on the ref target and returns every output as float
static std::vector<std::vector<float>>
run_causal_conv(const std::string& model,
                const std::vector<std::pair<std::string, migraphx::literal>>& inputs)
{
    auto p = read_onnx(model);
    p.compile(migraphx::make_target("ref"));
    migraphx::parameter_map pm;
    for(const auto& [name, data] : inputs)
        pm[name] = data.get_argument();
    auto results = p.eval(pm);
    std::vector<std::vector<float>> outputs;
    std::transform(
        results.begin(), results.end(), std::back_inserter(outputs), [](const auto& result) {
            std::vector<float> output;
            result.visit([&](auto v) { output.assign(v.begin(), v.end()); });
            return output;
        });
    return outputs;
}

TEST_CASE(causal_conv_with_state_verify_test)
{
    migraphx::shape::type_t t = migraphx::shape::float_type;
    std::vector<float> input  = {-0.5f, 0.25f, 1.0f, -1.5f, 2.0f, 0.75f, -0.25f, 1.25f};
    std::vector<float> weight = {0.5f, -1.0f, 2.0f, 1.5f, 0.25f, -0.75f};
    std::vector<float> bias   = {0.1f, -0.2f};
    std::vector<float> past   = {1.0f, -2.0f, 0.5f, 3.0f};

    auto outputs = run_causal_conv("causal_conv_with_state_test.onnx",
                                   {{"input", {{t, {1, 2, 4}}, input}},
                                    {"weight", {{t, {2, 1, 3}}, weight}},
                                    {"bias", {{t, {2}}, bias}},
                                    {"past_state", {{t, {1, 2, 2}}, past}}});

    std::vector<float> gold_output = {
        1.33123f, 0.0524979f, 1.33123f, -0.084646f, -0.0900332f, 4.17717f, 3.04763f, -0.0360944f};
    std::vector<float> gold_present = {1.0f, -1.5f, -0.25f, 1.25f};
    EXPECT(migraphx::verify::verify_rms_range(outputs[0], gold_output));
    EXPECT(migraphx::verify::verify_rms_range(outputs[1], gold_present));
}

TEST_CASE(causal_conv_with_state_no_state_verify_test)
{
    migraphx::shape::type_t t = migraphx::shape::float_type;
    std::vector<float> input  = {-0.5f, 0.25f, 1.0f, -1.5f, 2.0f, 0.75f, -0.25f, 1.25f};
    std::vector<float> weight = {0.5f, -1.0f, 2.0f, 1.5f, 0.25f, -0.75f};

    auto outputs =
        run_causal_conv("causal_conv_with_state_no_state_test.onnx",
                        {{"input", {{t, {1, 2, 4}}, input}}, {"weight", {{t, {2, 1, 3}}, weight}}});

    std::vector<float> gold_output = {-1.0f, 1.0f, 1.5f, -3.875f, -1.5f, -0.0625f, 3.375f, 0.125f};
    EXPECT(outputs.size() == 1);
    EXPECT(migraphx::verify::verify_rms_range(outputs[0], gold_output));
}

TEST_CASE(causal_conv_with_state_dilation_verify_test)
{
    migraphx::shape::type_t t = migraphx::shape::half_type;
    std::vector<float> input  = {
        0.5f, -1.0f, 1.5f, 2.0f, -0.5f, 1.0f, -2.0f, 0.25f, 0.75f, -1.25f, 1.75f, -0.75f};
    std::vector<float> weight = {1.0f, -0.5f, 0.25f, -1.0f, 2.0f, 0.5f};
    std::vector<float> bias   = {0.5f, -1.0f};
    std::vector<float> past   = {1.0f, 2.0f, -1.0f, -2.0f, 0.5f, -0.5f, 1.5f, -1.5f};

    auto outputs = run_causal_conv("causal_conv_with_state_dilation_test.onnx",
                                   {{"input", {{t, {1, 2, 6}}, input}},
                                    {"weight", {{t, {2, 1, 3}}, weight}},
                                    {"bias", {{t, {2}}, bias}},
                                    {"past_state", {{t, {1, 2, 4}}, past}}});

    std::vector<float> gold_output  = {2.125f,
                                       3.25f,
                                       -0.375f,
                                       -0.5f,
                                       0.125f,
                                       -1.25f,
                                       0.5f,
                                       -3.375f,
                                       -6.125f,
                                       0.375f,
                                       3.375f,
                                       -4.125f};
    std::vector<float> gold_present = {1.5f, 2.0f, -0.5f, 1.0f, 0.75f, -1.25f, 1.75f, -0.75f};
    EXPECT(migraphx::verify::verify_rms_range(outputs[0], gold_output));
    EXPECT(migraphx::verify::verify_rms_range(outputs[1], gold_present));
}

TEST_CASE(causal_conv_with_state_channels_last_verify_test)
{
    migraphx::shape::type_t t = migraphx::shape::float_type;
    // (batch, seq_len, 2, 2): the two trailing axes are the four channels
    std::vector<float> input = {
        0.5f, -1.0f, 1.5f, 2.0f, -0.5f, 1.0f, -2.0f, 0.25f, 0.75f, -1.25f, 1.75f, -0.75f};
    std::vector<float> weight = {
        1.0f, -0.5f, 0.25f, -1.0f, 2.0f, 0.5f, 0.5f, 0.5f, -1.0f, 2.0f, -1.5f, 1.0f};
    std::vector<float> bias = {0.1f, 0.2f, -0.3f, 0.4f};
    std::vector<float> past = {1.0f, 2.0f, -1.0f, -2.0f, 0.5f, -0.5f, 1.5f, -1.5f};

    auto outputs = run_causal_conv("causal_conv_with_state_channels_last_test.onnx",
                                   {{"input", {{t, {1, 3, 2, 2}}, input}},
                                    {"weight", {{t, {4, 1, 3}}, weight}},
                                    {"bias", {{t, {4}}, bias}},
                                    {"past_state", {{t, {1, 2, 2, 2}}, past}}});

    std::vector<float> gold_output  = {0.707962f,
                                       -0.117385f,
                                       -0.271384f,
                                       0.427057f,
                                       0.125103f,
                                       -0.24802f,
                                       3.07467f,
                                       -0.0252826f,
                                       0.766056f,
                                       2.39278f,
                                       -0.209583f,
                                       3.15566f};
    std::vector<float> gold_present = {-0.5f, 1.0f, -2.0f, 0.25f, 0.75f, -1.25f, 1.75f, -0.75f};
    EXPECT(migraphx::verify::verify_rms_range(outputs[0], gold_output));
    EXPECT(migraphx::verify::verify_rms_range(outputs[1], gold_present));
}

TEST_CASE(causal_conv_with_state_window_verify_test)
{
    migraphx::shape::type_t t = migraphx::shape::float_type;
    std::vector<float> input  = {0.5f, -1.0f, 1.5f, 2.0f};
    std::vector<float> weight = {0.5f, -1.0f, 2.0f, 1.5f, 0.25f, -0.75f};
    // Only the last slot of past_state is read; the others hold junk
    std::vector<float> past = {
        100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 1.0f, -2.0f, 0.5f, 3.0f};

    auto outputs = run_causal_conv("causal_conv_with_state_window_test.onnx",
                                   {{"input", {{t, {1, 2, 2}}, input}},
                                    {"weight", {{t, {2, 1, 3}}, weight}},
                                    {"past_state", {{t, {3, 1, 2, 2}}, past}}});

    std::vector<float> gold_output = {3.5f, -3.5f, 0.375f, 3.375f};
    // Slot 0 predates the two input positions, slot 1 follows the first, slot 2 the second
    std::vector<float> gold_present = {
        0.0f, 0.0f, 0.0f, 0.0f, -2.0f, 0.5f, 3.0f, 1.5f, 0.5f, -1.0f, 1.5f, 2.0f};
    EXPECT(migraphx::verify::verify_rms_range(outputs[0], gold_output));
    EXPECT(migraphx::verify::verify_rms_range(outputs[1], gold_present));
}
