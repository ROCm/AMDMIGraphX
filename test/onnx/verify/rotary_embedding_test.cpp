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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(rotary_embedding_verify_test)
{
    std::size_t batch_size = 1;
    std::size_t sequence_length = 2;
    std::size_t num_heads = 3;
    std::size_t head_size = 6;
    std::size_t rotary_embedding_dim = 0;
    std::size_t max_sequence_length = 4;
    bool interleaved = false;

    migraphx::program p = read_onnx("rotary_embedding_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_s{migraphx::shape::half_type, {batch_size, sequence_length, num_heads * head_size}};
    migraphx::shape pos_ids_s{migraphx::shape::int32_type, {batch_size, sequence_length}};
    migraphx::shape cache_s{migraphx::shape::half_type, {max_sequence_length, head_size / 2}};

    std::vector<float> input = {
      -1.0408f, 0.9166f, -1.3042f, -1.1097f, -1.2188f, 1.1676f, 1.0076f, -0.7529f,
      -0.2250f, -0.4327f, -1.5071f, -0.4586f, -0.8663f, -0.2656f, 0.1665f, 0.7911f,
      -0.9320f, -0.8579f, -1.0574f, -0.1188f, -0.9078f, 0.3452f, -0.5713f, -0.2351f,
      -0.8480f, 0.5266f, -1.2944f, -0.0243f, -0.2354f, -0.7087f, -0.9647f, -0.0991f,
      -0.2994f, -0.0650f, -1.5720f, -1.3211f};
    std::vector<migraphx::half> input_data{input.cbegin(), input.cend()};

    std::vector<int> pos_ids_data = {0, 1};

    std::vector<float> cos_cache = {
      1.0000f, 1.0000f, 1.0000f, 0.5403f, 0.9989f, 1.0000f, -0.4161f, 0.9957f,
      1.0000f, -0.9900f, 0.9903f, 1.0000f};
    std::vector<migraphx::half> cos_cache_data{cos_cache.cbegin(), cos_cache.cend()};

    std::vector<float> sin_cache = {
      0.0000f, 0.0000f, 0.0000f, 0.8415f, 0.0464f, 0.0022f, 0.9093f, 0.0927f, 0.0043f,
      0.1411f, 0.1388f, 0.0065f};
    std::vector<migraphx::half> sin_cache_data{sin_cache.cbegin(), sin_cache.cend()};

    migraphx::parameter_map param_map;
    param_map["input"] = migraphx::argument(input_s, input_data.data());
    param_map["pos_ids"] = migraphx::argument(pos_ids_s, pos_ids_data.data());
    param_map["cos_cache"] = migraphx::argument(cache_s, cos_cache_data.data());
    param_map["sin_cache"] = migraphx::argument(cache_s, sin_cache_data.data());

    auto result = p.eval(param_map).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
      -1.0408f, 0.9166f, -1.3042f, -1.1097f, -1.2188f, 1.1676f, 1.0076f, -0.7529f,
      -0.2250f, -0.4327f, -1.5071f, -0.4586f, -0.8663f, -0.2656f, 0.1665f, 0.7911f,
      -0.9320f, -0.8579f, -0.8618f, -0.0922f, -0.9073f, -0.7032f, -0.5762f, -0.2371f,
      -0.4377f, 0.5370f, -1.2929f, -0.7267f, -0.2107f, -0.7115f, -0.4666f, -0.0261f,
      -0.2965f, -0.8469f, -1.5749f, -1.3217f};

    for(auto i = 0; i < result_vector.size(); ++i)
    {
        std::cout << i << ": " << result_vector[i] << " | " << gold[i] << std::endl;;
    }
    std::cout << std::endl;

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


TEST_CASE(rotary_embedding_interleaved_verify_test)
{
    std::size_t batch_size = 1;
    std::size_t sequence_length = 3;
    std::size_t num_heads = 2;
    std::size_t head_size = 4;
    std::size_t rotary_embedding_dim = 0;
    std::size_t max_sequence_length = 8;
    bool interleaved = true;

    migraphx::program p = read_onnx("rotary_embedding_interleaved_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_s{migraphx::shape::half_type, {batch_size, sequence_length, num_heads * head_size}};
    migraphx::shape pos_ids_s{migraphx::shape::int32_type, {1}};
    migraphx::shape cache_s{migraphx::shape::half_type, {max_sequence_length, head_size / 2}};

    std::vector<float> input = {
      -1.0408f, 0.9166f, -1.3042f, -1.1097f, -0.1320f, -0.2751f, -0.2350f, 0.0937f,
      -1.2188f, 1.1676f, -1.0574f, -0.1188f, -0.7396f, -1.2425f, -0.1752f, 0.6990f,
      -0.8110f, 0.6737f, -1.1233f, -0.0919f, -0.6861f, 0.7202f, 0.1963f, 0.6142f};
    std::vector<migraphx::half> input_data{input.cbegin(), input.cend()};

    std::vector<int> pos_ids_data = {0};

    std::vector<float> cos_cache = {
      1.0000f, 1.0000f, 0.5403f, 0.9999f, -0.4161f, 0.9998f, -0.9900f, 0.9996f,
      -0.6536f, 0.9992f, 0.2837f, 0.9988f, 0.9602f, 0.9982f, 0.7539f, 0.9976f};
    std::vector<migraphx::half> cos_cache_data{cos_cache.cbegin(), cos_cache.cend()};

    std::vector<float> sin_cache = {
      0.0000f, 0.0000f, 0.8415f, 0.0100f, 0.9093f, 0.0200f, 0.1411f, 0.0300f,
      -0.7568f, 0.0400f, -0.9589f, 0.0500f, -0.2794f, 0.0600f, 0.6570f, 0.0699f};
    std::vector<migraphx::half> sin_cache_data{sin_cache.cbegin(), sin_cache.cend()};

    migraphx::parameter_map param_map;
    param_map["input"] = migraphx::argument(input_s, input_data.data());
    param_map["pos_ids"] = migraphx::argument(pos_ids_s, pos_ids_data.data());
    param_map["cos_cache"] = migraphx::argument(cache_s, cos_cache_data.data());
    param_map["sin_cache"] = migraphx::argument(cache_s, sin_cache_data.data());

    auto result = p.eval(param_map).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
      -1.0408f, 0.9166f, -1.3042f, -1.1097f, -0.1320f, -0.2751f, -0.2350f, 0.0937f,
      -1.6411f, -0.3948f, -1.0561f, -0.1294f, 0.6460f, -1.2937f, -0.1822f, 0.6972f,
      -0.2751f, -1.0178f, -1.1212f, -0.1143f, -0.3694f, -0.9235f, 0.1840f, 0.6180f};

    for(auto i = 0; i < result_vector.size(); ++i)
    {
        std::cout << i << ": " << result_vector[i] << " | " << gold[i] << std::endl;;
    }
    std::cout << std::endl;

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


TEST_CASE(rotary_embedding_float_verify_test)
{
    std::size_t batch_size = 1;
    std::size_t sequence_length = 2;
    std::size_t num_heads = 3;
    std::size_t head_size = 6;
    std::size_t rotary_embedding_dim = 0;
    std::size_t max_sequence_length = 4;
    bool interleaved = false;

    migraphx::program p = read_onnx("rotary_embedding_float_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_s{migraphx::shape::float_type, {batch_size, sequence_length, num_heads * head_size}};
    migraphx::shape pos_ids_s{migraphx::shape::int32_type, {batch_size, sequence_length}};
    migraphx::shape cache_s{migraphx::shape::float_type, {max_sequence_length, head_size / 2}};

    std::vector<float> input_data = {
      -1.0408f, 0.9166f, -1.3042f, -1.1097f, -1.2188f, 1.1676f, 1.0076f, -0.7529f,
      -0.2250f, -0.4327f, -1.5071f, -0.4586f, -0.8663f, -0.2656f, 0.1665f, 0.7911f,
      -0.9320f, -0.8579f, -1.0574f, -0.1188f, -0.9078f, 0.3452f, -0.5713f, -0.2351f,
      -0.8480f, 0.5266f, -1.2944f, -0.0243f, -0.2354f, -0.7087f, -0.9647f, -0.0991f,
      -0.2994f, -0.0650f, -1.5720f, -1.3211f};
    // std::vector<migraphx::half> input_data{input.cbegin(), input.cend()};

    std::vector<int> pos_ids_data = {0, 1};

    std::vector<float> cos_cache_data = {
      1.0000f, 1.0000f, 1.0000f, 0.5403f, 0.9989f, 1.0000f, -0.4161f, 0.9957f,
      1.0000f, -0.9900f, 0.9903f, 1.0000f};
    // std::vector<migraphx::half> cos_cache_data{cos_cache.cbegin(), cos_cache.cend()};

    std::vector<float> sin_cache_data = {
      0.0000f, 0.0000f, 0.0000f, 0.8415f, 0.0464f, 0.0022f, 0.9093f, 0.0927f, 0.0043f,
      0.1411f, 0.1388f, 0.0065f};
    // std::vector<migraphx::half> sin_cache_data{sin_cache.cbegin(), sin_cache.cend()};

    migraphx::parameter_map param_map;
    param_map["input"] = migraphx::argument(input_s, input_data.data());
    param_map["pos_ids"] = migraphx::argument(pos_ids_s, pos_ids_data.data());
    param_map["cos_cache"] = migraphx::argument(cache_s, cos_cache_data.data());
    param_map["sin_cache"] = migraphx::argument(cache_s, sin_cache_data.data());

    auto result = p.eval(param_map).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
      -1.0408f, 0.9166f, -1.3042f, -1.1097f, -1.2188f, 1.1676f, 1.0076f, -0.7529f,
      -0.2250f, -0.4327f, -1.5071f, -0.4586f, -0.8663f, -0.2656f, 0.1665f, 0.7911f,
      -0.9320f, -0.8579f, -0.8618f, -0.0922f, -0.9073f, -0.7032f, -0.5762f, -0.2371f,
      -0.4377f, 0.5370f, -1.2929f, -0.7267f, -0.2107f, -0.7115f, -0.4666f, -0.0261f,
      -0.2965f, -0.8469f, -1.5749f, -1.3217f};

    for(auto i = 0; i < result_vector.size(); ++i)
    {
        std::cout << i << ": " << result_vector[i] << " | " << gold[i] << std::endl;;
    }
    std::cout << std::endl;

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}