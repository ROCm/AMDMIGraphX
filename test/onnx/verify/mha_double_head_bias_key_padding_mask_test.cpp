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

// Took tests from the Attention OP contrib variant and reused gold data + extracted q, k ,v tensors from the resulting
// input/weights as same input should give us the same output.

TEST_CASE(mha_double_head_bias_mask_batch1_passthrough_mask_test)
{
    auto p = optimize_onnx("mha_double_head_bias_mask_batch1_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape q_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> query_data = {1.63389, 1.35173, 1.3877, 1.52187, 0.901106, 0.730935, 0.639857, 0.926961};

    migraphx::shape k_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> key_data = {1.61202, 1.33192, 1.03611, 1.09385, 0.776542, 0.692002, 0.544437, 0.687849};

    migraphx::shape value_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> value_data = {1.50825, 1.11683, 0.704501, 0.897266, 0.761568, 0.450901, 0.389297, 0.527515};

    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data(12, 0.0f);

    migraphx::shape mask_shape{migraphx::shape::int32_type, {1, 2}};
    std::vector<float> mask_data = {1, 1};

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
    std::vector<float> gold = {
        1.380390f, 1.002805f, 0.6146411f, 0.791859f, 1.28671f, 0.919257f, 0.5846517f, 0.756680f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(mha_double_head_bias_mask_batch1_last_mask_test)
{
    auto p = optimize_onnx("mha_double_head_bias_mask_batch1_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape q_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> query_data = {1.42192, 1.42665, 1.21449, 0.959491, 1.18012, 1.21737, 0.973761, 0.659853};

    migraphx::shape k_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> key_data = {1.30758, 1.16746, 1.10112, 1.77517, 1.26305, 1.09709, 0.983969, 1.63047};

    migraphx::shape value_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> value_data = {0.7423, 1.56581, 1.2923, 1.38868, 0.546948, 1.41101, 1.24849, 1.23179};


    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data = {0.561730f,
                                    0.3377643f,
                                    0.2519498f,
                                    0.2072306f,
                                    0.5650865f,
                                    0.347968f,
                                    0.1985114f,
                                    0.6656775f,
                                    0.3533229f,
                                    0.5744964f,
                                    0.5860762f,
                                    0.76156753f};

    // 0 = mask,1  = pass through
    migraphx::shape mask_shape{migraphx::shape::int32_type, {1, 2}};
    std::vector<float> mask_data = {1, 0};

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
    std::vector<float> gold = {
        0.7423008f, 1.565812f, 1.292300f, 1.388675f, 0.7423008f, 1.565812f, 1.292300f, 1.388675f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(mha_double_head_bias_mask_batch1_first_mask_test)
{
    auto p = optimize_onnx("mha_double_head_bias_mask_batch1_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape q_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> query_data = {1.16918, 1.28233, 0.941054, 1.00275, 1.74152, 1.74252, 1.14048, 1.14762};

    migraphx::shape k_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> key_data = {0.993029, 1.62161, 1.56195, 0.45243, 1.06582, 2.15747, 2.22264, 0.966104};

    migraphx::shape value_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> value_data = {1.28486, 1.35648, 1.34486, 1.52285, 1.66392, 1.54765, 1.69621, 1.65823};

    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data = {0.71639f,
                                    0.6984057f,
                                    0.641569f,
                                    0.5584603f,
                                    0.807863f,
                                    0.9522143f,
                                    0.964991f,
                                    0.0536422f,
                                    0.6487126f,
                                    0.924860f,
                                    0.73642f,
                                    0.85529f};

    migraphx::shape mask_shape{migraphx::shape::int32_type, {1, 2}};
    std::vector<float> mask_data = {0, 1};

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
    std::vector<float> gold = {
        1.6639f, 1.547649f, 1.696211f, 1.658239f, 1.6639f, 1.547649f, 1.696211f, 1.6582391f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(mha_double_head_bias_mask_batch1_all_mask_test)
{
    auto p = optimize_onnx("mha_double_head_bias_mask_batch1_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape q_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> query_data = {2.21325, 2.02489, 1.72693, 1.08797, 2.42754, 1.86212, 1.88447, 1.08512};

    migraphx::shape k_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> key_data = {2.07036, 2.64231, 1.94134, 2.36241, 1.98163, 2.67183, 1.85018, 2.60435};

    migraphx::shape value_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> value_data = {1.10809, 1.5582, 1.5385, 2.20247, 1.22999, 1.75193, 1.28318, 2.40112};

    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data = {0.751496f,
                                    0.557292f,
                                    0.6720010f,
                                    0.1879267f,
                                    0.352546f,
                                    0.600021f,
                                    0.0552079f,
                                    0.5959239f,
                                    0.0404032f,
                                    0.1882552f,
                                    0.2718655f,
                                    0.84921235f};

    migraphx::shape mask_shape{migraphx::shape::int32_type, {1, 2}};
    std::vector<float> mask_data = {0, 0};

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
    std::vector<float> gold = {
        1.166096f, 1.650388f, 1.406106f, 2.30548f, 1.165592f, 1.649586f, 1.406728f, 2.304997};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
