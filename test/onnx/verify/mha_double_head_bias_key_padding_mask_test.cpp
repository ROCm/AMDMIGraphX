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

// Took tests from the Attention OP contrib variant and reused gold data + extracted q, k ,v tensors
// from the resulting input/weights as same input should give us the same output.

TEST_CASE(mha_double_head_bias_mask_batch1_passthrough_mask_test)
{
    auto p = optimize_onnx("mha_double_head_bias_mask_batch1_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape q_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> query_data = {
        1.63389, 1.35173, 1.3877, 1.52187, 0.901106, 0.730935, 0.639857, 0.926961};

    migraphx::shape k_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> key_data = {
        1.61202, 1.33192, 1.03611, 1.09385, 0.776542, 0.692002, 0.544437, 0.687849};

    migraphx::shape value_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> value_data = {
        1.50825, 1.11683, 0.704501, 0.897266, 0.761568, 0.450901, 0.389297, 0.527515};

    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data(12, 0.0f);

    migraphx::shape mask_shape{migraphx::shape::int32_type, {1, 2}};
    std::vector<int32_t> mask_data = {1, 1};

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
    std::vector<float> query_data = {
        0.860191, 1.08888, 0.962536, 0.75226, 0.618388, 0.879605, 0.721811, 0.452622};

    migraphx::shape k_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> key_data = {
        0.742495, 0.819491, 0.90261, 1.1095, 0.697966, 0.749119, 0.785458, 0.964789};

    migraphx::shape value_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> value_data = {
        0.388977, 0.991311, 0.706223, 0.627108, 0.193625, 0.836511, 0.662411, 0.470219};

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
    std::vector<int32_t> mask_data = {1, 0};

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
    std::vector<float> query_data = {
        0.45279, 0.583924, 0.299485, 0.444288, 1.02513, 1.04411, 0.498912, 0.589157};

    migraphx::shape k_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> key_data = {
        0.185166, 0.669398, 0.596961, 0.398788, 0.257954, 1.20526, 1.25765, 0.912461};

    migraphx::shape value_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> value_data = {
        0.63615, 0.431616, 0.608435, 0.667559, 1.01521, 0.622788, 0.959786, 0.802939};

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
    std::vector<int32_t> mask_data = {0, 1};

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
    std::vector<float> query_data = {
        1.46175, 1.4676, 1.05493, 0.900047, 1.67605, 1.30483, 1.21247, 0.897198};

    migraphx::shape k_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> key_data = {
        1.71781, 2.04228, 1.88613, 1.76649, 1.62908, 2.07181, 1.79497, 2.00843};

    migraphx::shape value_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> value_data = {
        1.06769, 1.36994, 1.26663, 1.35326, 1.18959, 1.56367, 1.01132, 1.55191};

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
    std::vector<int32_t> mask_data = {0, 0};

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

TEST_CASE(mha_double_head_bias_mask_batch2_right_mask_test)
{
    auto p = optimize_onnx("mha_double_head_bias_mask_right_batch2_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape q_shape{migraphx::shape::float_type, {2, 2, 4}};
    std::vector<float> query_data = {1.46175,
                                     1.4676,
                                     1.05493,
                                     0.900047,
                                     1.67605,
                                     1.30483,
                                     1.21247,
                                     0.897198,
                                     1.46175,
                                     1.4676,
                                     1.05493,
                                     0.900047,
                                     1.67605,
                                     1.30483,
                                     1.21247,
                                     0.897198};

    migraphx::shape k_shape{migraphx::shape::float_type, {2, 2, 4}};
    std::vector<float> key_data = {1.71781,
                                   2.04228,
                                   1.88613,
                                   1.76649,
                                   1.62908,
                                   2.07181,
                                   1.79497,
                                   2.00843,
                                   1.71781,
                                   2.04228,
                                   1.88613,
                                   1.76649,
                                   1.62908,
                                   2.07181,
                                   1.79497,
                                   2.00843};

    migraphx::shape value_shape{migraphx::shape::float_type, {2, 2, 4}};
    std::vector<float> value_data = {1.06769,
                                     1.36994,
                                     1.26663,
                                     1.35326,
                                     1.18959,
                                     1.56367,
                                     1.01132,
                                     1.55191,
                                     1.06769,
                                     1.36994,
                                     1.26663,
                                     1.35326,
                                     1.18959,
                                     1.56367,
                                     1.01132,
                                     1.55191};

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

    migraphx::shape mask_shape{migraphx::shape::int32_type, {2}};
    std::vector<int32_t> mask_data = {2, 1};

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
    std::vector<float> gold = {1.10809f,
                               1.5582f,
                               1.5385f,
                               2.20247f,
                               1.10809f,
                               1.5582f,
                               1.5385f,
                               2.20247f,
                               1.10809f,
                               1.5582f,
                               1.5385f,
                               2.20247f,
                               1.10809f,
                               1.5582f,
                               1.5385f,
                               2.20247f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
