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

#include <onnx_test.hpp>

// external_data_test.onnx stores conv.bias as raw data and conv.weight in an external file, both
// filled with 1s. The caller provided values differ, so the result shows which source was used.
TEST_CASE(external_initializer_test)
{
    std::vector<float> weight_data(1210, 2);
    std::vector<float> bias_data(10, 3);

    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto bias = mm->add_literal(migraphx::literal({migraphx::shape::float_type, {10}}, bias_data));
    auto weights = mm->add_literal(
        migraphx::literal({migraphx::shape::float_type, {10, 1, 11, 11}}, weight_data));
    auto param = mm->add_parameter("input", {migraphx::shape::float_type, {1, 1, 224, 224}});
    auto conv  = mm->add_instruction(
        migraphx::make_op("convolution", {{"padding", {0, 0, 0, 0}}}), param, weights);
    auto bias_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 10, 214, 214}}}), bias);
    auto add = mm->add_instruction(migraphx::make_op("add"), conv, bias_bcast);
    mm->add_return({add});

    migraphx::onnx_options options;
    options.external_initializers["conv.weight"] = {weight_data.data(),
                                                    weight_data.size() * sizeof(float)};
    options.external_initializers["conv.bias"]   = {bias_data.data(),
                                                    bias_data.size() * sizeof(float)};
    auto prog                                    = read_onnx("external_data_test.onnx", options);
    EXPECT(p == prog);
}

TEST_CASE(external_initializer_partial_test)
{
    std::vector<float> weight_data(1210, 2);
    std::vector<float> bias_data(10, 1);

    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto bias = mm->add_literal(migraphx::literal({migraphx::shape::float_type, {10}}, bias_data));
    auto weights = mm->add_literal(
        migraphx::literal({migraphx::shape::float_type, {10, 1, 11, 11}}, weight_data));
    auto param = mm->add_parameter("input", {migraphx::shape::float_type, {1, 1, 224, 224}});
    auto conv  = mm->add_instruction(
        migraphx::make_op("convolution", {{"padding", {0, 0, 0, 0}}}), param, weights);
    auto bias_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 10, 214, 214}}}), bias);
    auto add = mm->add_instruction(migraphx::make_op("add"), conv, bias_bcast);
    mm->add_return({add});

    migraphx::onnx_options options;
    options.external_initializers["conv.weight"] = {weight_data.data(),
                                                    weight_data.size() * sizeof(float)};
    auto prog                                    = read_onnx("external_data_test.onnx", options);
    EXPECT(p == prog);
}

TEST_CASE(external_initializer_scalar_test)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {2, 8, {3}};
    auto p                        = read_onnx("clip_dyn_min_only_test.onnx", options);

    float min_val                        = 0.0f;
    options.external_initializers["min"] = {&min_val, sizeof(min_val)};
    auto prog                            = read_onnx("clip_dyn_min_only_test.onnx", options);
    EXPECT(p == prog);
}

TEST_CASE(external_initializer_size_mismatch_test)
{
    std::vector<float> weight_data(1209, 2);
    migraphx::onnx_options options;
    options.external_initializers["conv.weight"] = {weight_data.data(),
                                                    weight_data.size() * sizeof(float)};
    EXPECT(test::throws([&] { read_onnx("external_data_test.onnx", options); }));
}

TEST_CASE(external_initializer_null_data_test)
{
    migraphx::onnx_options options;
    options.external_initializers["conv.weight"] = {nullptr, 1210 * sizeof(float)};
    EXPECT(test::throws([&] { read_onnx("external_data_test.onnx", options); }));
}
