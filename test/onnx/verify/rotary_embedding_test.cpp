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
    std::size_t batch_size          = 1;
    std::size_t sequence_length     = 2;
    std::size_t num_heads           = 3;
    std::size_t head_size           = 6;
    std::size_t max_sequence_length = 4;

    migraphx::program p = read_onnx("rotary_embedding_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_s{migraphx::shape::half_type,
                            {batch_size, sequence_length, num_heads * head_size}};
    migraphx::shape pos_ids_s{migraphx::shape::int32_type, {batch_size, sequence_length}};
    migraphx::shape cache_s{migraphx::shape::half_type, {max_sequence_length, head_size / 2}};

    std::vector<float> input = {
        -1.0408f, 0.9166f,  -1.3042f, -1.1097f, -1.2188f, 1.1676f,  1.0076f,  -0.7529f, -0.2250f,
        -0.4327f, -1.5071f, -0.4586f, -0.8663f, -0.2656f, 0.1665f,  0.7911f,  -0.9320f, -0.8579f,
        -1.0574f, -0.1188f, -0.9078f, 0.3452f,  -0.5713f, -0.2351f, -0.8480f, 0.5266f,  -1.2944f,
        -0.0243f, -0.2354f, -0.7087f, -0.9647f, -0.0991f, -0.2994f, -0.0650f, -1.5720f, -1.3211f};
    std::vector<migraphx::half> input_data{input.cbegin(), input.cend()};

    std::vector<int> pos_ids_data = {0, 1};

    std::vector<float> cos_cache = {1.0000f,
                                    1.0000f,
                                    1.0000f,
                                    0.5403f,
                                    0.9989f,
                                    1.0000f,
                                    -0.4161f,
                                    0.9957f,
                                    1.0000f,
                                    -0.9900f,
                                    0.9903f,
                                    1.0000f};
    std::vector<migraphx::half> cos_cache_data{cos_cache.cbegin(), cos_cache.cend()};

    std::vector<float> sin_cache = {0.0000f,
                                    0.0000f,
                                    0.0000f,
                                    0.8415f,
                                    0.0464f,
                                    0.0022f,
                                    0.9093f,
                                    0.0927f,
                                    0.0043f,
                                    0.1411f,
                                    0.1388f,
                                    0.0065f};
    std::vector<migraphx::half> sin_cache_data{sin_cache.cbegin(), sin_cache.cend()};

    migraphx::parameter_map param_map;
    param_map["input"]     = migraphx::argument(input_s, input_data.data());
    param_map["pos_ids"]   = migraphx::argument(pos_ids_s, pos_ids_data.data());
    param_map["cos_cache"] = migraphx::argument(cache_s, cos_cache_data.data());
    param_map["sin_cache"] = migraphx::argument(cache_s, sin_cache_data.data());

    auto result = p.eval(param_map).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        -1.0410f, 0.9165f,  -1.3047f, -1.1094f, -1.2188f, 1.1680f,  0.9082f,  -0.6816f, -0.2239f,
        0.6138f,  -1.5391f, -0.4590f, -0.8662f, -0.2656f, 0.1665f,  0.7910f,  -0.9321f, -0.8579f,
        -0.8613f, -0.0922f, -0.9067f, -0.7031f, -0.5757f, -0.2371f, -0.8481f, 0.5264f,  -1.2939f,
        -0.0243f, -0.2354f, -0.7085f, -0.4668f, -0.0261f, -0.2964f, -0.8462f, -1.5742f, -1.3213f};

    EXPECT(migraphx::verify::verify_range_with_tolerance(
        result_vector, migraphx::verify::expected{gold}, migraphx::verify::tolerance{1e-3}));
}

TEST_CASE(rotary_embedding_float_verify_test)
{
    std::size_t batch_size          = 1;
    std::size_t sequence_length     = 2;
    std::size_t num_heads           = 3;
    std::size_t head_size           = 6;
    std::size_t max_sequence_length = 4;

    migraphx::program p = read_onnx("rotary_embedding_float_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_s{migraphx::shape::float_type,
                            {batch_size, num_heads, sequence_length, head_size}};
    migraphx::shape pos_ids_s{migraphx::shape::int32_type, {batch_size, sequence_length}};
    migraphx::shape cache_s{migraphx::shape::float_type, {max_sequence_length, head_size / 2}};

    std::vector<float> input_data = {
        -1.0408f, 0.9166f,  -1.3042f, -1.1097f, -1.2188f, 1.1676f,  1.0076f,  -0.7529f, -0.2250f,
        -0.4327f, -1.5071f, -0.4586f, -0.8663f, -0.2656f, 0.1665f,  0.7911f,  -0.9320f, -0.8579f,
        -1.0574f, -0.1188f, -0.9078f, 0.3452f,  -0.5713f, -0.2351f, -0.8480f, 0.5266f,  -1.2944f,
        -0.0243f, -0.2354f, -0.7087f, -0.9647f, -0.0991f, -0.2994f, -0.0650f, -1.5720f, -1.3211f};

    std::vector<int> pos_ids_data = {0, 1};

    std::vector<float> cos_cache_data = {1.0000f,
                                         1.0000f,
                                         1.0000f,
                                         0.5403f,
                                         0.9989f,
                                         1.0000f,
                                         -0.4161f,
                                         0.9957f,
                                         1.0000f,
                                         -0.9900f,
                                         0.9903f,
                                         1.0000f};

    std::vector<float> sin_cache_data = {0.0000f,
                                         0.0000f,
                                         0.0000f,
                                         0.8415f,
                                         0.0464f,
                                         0.0022f,
                                         0.9093f,
                                         0.0927f,
                                         0.0043f,
                                         0.1411f,
                                         0.1388f,
                                         0.0065f};

    migraphx::parameter_map param_map;
    param_map["input"]     = migraphx::argument(input_s, input_data.data());
    param_map["pos_ids"]   = migraphx::argument(pos_ids_s, pos_ids_data.data());
    param_map["cos_cache"] = migraphx::argument(cache_s, cos_cache_data.data());
    param_map["sin_cache"] = migraphx::argument(cache_s, sin_cache_data.data());

    auto result = p.eval(param_map).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        -1.0408f, 0.9166f,  -1.3042f, -1.1097f, -1.2188f, 1.1676f,  0.9085f,  -0.6821f, -0.2240f,
        0.6141f,  -1.5404f, -0.4591f, -0.8663f, -0.2656f, 0.1665f,  0.7911f,  -0.9320f, -0.8579f,
        -0.8618f, -0.0922f, -0.9073f, -0.7033f, -0.5762f, -0.2371f, -0.8480f, 0.5266f,  -1.2944f,
        -0.0243f, -0.2354f, -0.7087f, -0.4665f, -0.0261f, -0.2965f, -0.8469f, -1.5749f, -1.3218f};

    EXPECT(migraphx::verify::verify_range_with_tolerance(
        result_vector, migraphx::verify::expected{gold}, migraphx::verify::tolerance{1e-3}));
}

TEST_CASE(rotary_embedding_1s_verify_test)
{
    std::size_t batch_size          = 1;
    std::size_t sequence_length     = 2;
    std::size_t num_heads           = 3;
    std::size_t head_size           = 6;
    std::size_t max_sequence_length = 4;

    migraphx::program p = read_onnx("rotary_embedding_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_s{migraphx::shape::half_type,
                            {batch_size, sequence_length, num_heads * head_size}};
    migraphx::shape pos_ids_s{migraphx::shape::int32_type, {batch_size, sequence_length}};
    migraphx::shape cache_s{migraphx::shape::half_type, {max_sequence_length, head_size / 2}};

    std::vector<float> input(input_s.elements(), 1.0);
    std::vector<migraphx::half> input_data{input.cbegin(), input.cend()};

    std::vector<int> pos_ids_data = {0, 1};

    std::vector<float> cos_cache(cache_s.elements(), 1.0);
    std::vector<migraphx::half> cos_cache_data{cos_cache.cbegin(), cos_cache.cend()};

    std::vector<float> sin_cache(cache_s.elements(), 1.0);
    std::vector<migraphx::half> sin_cache_data{sin_cache.cbegin(), sin_cache.cend()};

    migraphx::parameter_map param_map;
    param_map["input"]     = migraphx::argument(input_s, input_data.data());
    param_map["pos_ids"]   = migraphx::argument(pos_ids_s, pos_ids_data.data());
    param_map["cos_cache"] = migraphx::argument(cache_s, cos_cache_data.data());
    param_map["sin_cache"] = migraphx::argument(cache_s, sin_cache_data.data());

    auto result = p.eval(param_map).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold(input_s.elements(), 0.0);
    for(auto i = 0; i < gold.size(); ++i)
    {
        if(i % 6 > 2)
            gold[i] = 2.0;
    }

    EXPECT(migraphx::verify::verify_range_with_tolerance(
        result_vector, migraphx::verify::expected{gold}, migraphx::verify::tolerance{1e-3}));
}

TEST_CASE(rotary_embedding_interleaved_verify_test)
{
    std::size_t batch_size          = 1;
    std::size_t sequence_length     = 3;
    std::size_t num_heads           = 2;
    std::size_t head_size           = 4;
    std::size_t max_sequence_length = 8;

    migraphx::program p = read_onnx("rotary_embedding_interleaved_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_s{migraphx::shape::half_type,
                            {batch_size, sequence_length, num_heads * head_size}};
    migraphx::shape pos_ids_s{migraphx::shape::int32_type, {1}};
    migraphx::shape cache_s{migraphx::shape::half_type, {max_sequence_length, head_size / 2}};

    std::vector<float> input = {-1.0408f, 0.9166f,  -1.3042f, -1.1097f, -0.1320f, -0.2751f,
                                -0.2350f, 0.0937f,  -1.2188f, 1.1676f,  -1.0574f, -0.1188f,
                                -0.7396f, -1.2425f, -0.1752f, 0.6990f,  -0.8110f, 0.6737f,
                                -1.1233f, -0.0919f, -0.6861f, 0.7202f,  0.1963f,  0.6142f};
    std::vector<migraphx::half> input_data{input.cbegin(), input.cend()};

    std::vector<int> pos_ids_data = {0};

    std::vector<float> cos_cache = {1.0000f,
                                    1.0000f,
                                    0.5403f,
                                    0.9999f,
                                    -0.4161f,
                                    0.9998f,
                                    -0.9900f,
                                    0.9996f,
                                    -0.6536f,
                                    0.9992f,
                                    0.2837f,
                                    0.9988f,
                                    0.9602f,
                                    0.9982f,
                                    0.7539f,
                                    0.9976f};
    std::vector<migraphx::half> cos_cache_data{cos_cache.cbegin(), cos_cache.cend()};

    std::vector<float> sin_cache = {0.0000f,
                                    0.0000f,
                                    0.8415f,
                                    0.0100f,
                                    0.9093f,
                                    0.0200f,
                                    0.1411f,
                                    0.0300f,
                                    -0.7568f,
                                    0.0400f,
                                    -0.9589f,
                                    0.0500f,
                                    -0.2794f,
                                    0.0600f,
                                    0.6570f,
                                    0.0699f};
    std::vector<migraphx::half> sin_cache_data{sin_cache.cbegin(), sin_cache.cend()};

    migraphx::parameter_map param_map;
    param_map["input"]     = migraphx::argument(input_s, input_data.data());
    param_map["pos_ids"]   = migraphx::argument(pos_ids_s, pos_ids_data.data());
    param_map["cos_cache"] = migraphx::argument(cache_s, cos_cache_data.data());
    param_map["sin_cache"] = migraphx::argument(cache_s, sin_cache_data.data());

    auto result = p.eval(param_map).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {-1.0410f, 0.9165f,  -1.3047f, -1.1094f, 0.1602f,  -0.2595f,
                               -0.2358f, 0.0913f,  -0.5547f, -1.5928f, -1.0547f, -0.1399f,
                               -0.7397f, -1.2422f, -0.1752f, 0.6992f,  -1.0049f, -0.3181f,
                               -1.1221f, -0.1031f, -0.3694f, -0.9229f, 0.1840f,  0.6182f};

    EXPECT(migraphx::verify::verify_range_with_tolerance(
        result_vector, migraphx::verify::expected{gold}, migraphx::verify::tolerance{1e-3}));
}

TEST_CASE(rotary_embedding_interleaved_large_verify_test)
{
    std::size_t batch_size          = 2;
    std::size_t sequence_length     = 8;
    std::size_t num_heads           = 4;
    std::size_t head_size           = 6;
    std::size_t max_sequence_length = 16;

    migraphx::program p = read_onnx("rotary_embedding_interleaved_large_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_s{migraphx::shape::half_type,
                            {batch_size, sequence_length, num_heads * head_size}};
    migraphx::shape pos_ids_s{migraphx::shape::int32_type, {1}};
    migraphx::shape cache_s{migraphx::shape::half_type, {max_sequence_length, head_size / 2}};

    std::vector<float> input = {
        -1.0408f, 0.9166f,  -1.3042f, -1.1097f, -1.2188f, 1.1676f,  -1.0190f, 0.3157f,  -1.6036f,
        1.8493f,  0.0447f,  1.5853f,  0.1036f,  -0.3514f, 0.2421f,  0.6463f,  0.8730f,  -0.9276f,
        1.0311f,  -1.9557f, -0.1482f, 1.7376f,  2.2039f,  -0.6589f, -1.0574f, -0.1188f, -0.9078f,
        0.3452f,  -0.5713f, -0.2351f, -0.5912f, 1.1312f,  0.7562f,  -1.2023f, -0.5833f, -0.4407f,
        0.1766f,  1.0224f,  -0.4826f, -0.5421f, -0.5342f, -0.6413f, 1.3314f,  -0.4498f, 0.5493f,
        0.0539f,  0.2601f,  0.8570f,  1.0076f,  -0.7529f, -0.2250f, -0.4327f, -1.5071f, -0.4586f,
        -1.9791f, 0.7787f,  -0.7749f, -0.1398f, 1.1414f,  -0.6354f, 0.0352f,  -0.4765f, -0.0409f,
        1.1993f,  0.5374f,  -0.1930f, 2.5211f,  -0.0452f, -0.3105f, -0.9407f, -0.0034f, 1.5199f,
        -0.8480f, 0.5266f,  0.0299f,  -0.0498f, 1.0651f,  0.8860f,  -1.4702f, -0.2134f, -0.8707f,
        1.6159f,  -0.2356f, 0.9444f,  0.5937f,  0.7203f,  0.5061f,  1.5192f,  -0.4897f, 0.9231f,
        0.2654f,  -0.1441f, 0.5407f,  -1.5476f, 0.6455f,  -1.1382f, 0.4640f,  -0.4986f, 0.1289f,
        2.7631f,  0.1405f,  1.1191f,  2.1134f,  -0.9754f, 0.1757f,  -0.1319f, -0.2735f, 0.3355f,
        -0.6008f, -1.1164f, 0.2577f,  -0.7226f, -0.9244f, 1.8737f,  0.6052f,  1.1904f,  1.2195f,
        -0.0470f, -1.0914f, 1.0223f,  0.3152f,  1.7528f,  -0.7650f, 1.8299f,  -0.2784f, -0.2719f,
        0.1885f,  2.1432f,  0.8527f,  0.0965f,  -0.0625f, 0.8269f,  1.0122f,  -1.4482f, -0.0644f,
        0.3215f,  0.5908f,  -1.4197f, 0.2113f,  0.0306f,  0.3604f,  0.3166f,  -0.8975f, -0.6393f,
        -1.2944f, -0.0243f, -0.2354f, -0.7087f, 1.1566f,  0.4296f,  0.5599f,  -0.7776f, 0.3339f,
        0.1759f,  2.1108f,  1.0702f,  0.8279f,  -0.2969f, 0.7120f,  -0.2068f, -0.1548f, 0.1553f,
        0.6207f,  -0.1690f, -0.5816f, 1.2632f,  0.0695f,  1.1862f,  -1.1874f, -0.7468f, -0.9320f,
        -0.8579f, -0.9647f, -0.0991f, 0.0195f,  1.1213f,  -1.4873f, -0.2043f, -1.0466f, -1.5772f,
        -0.0489f, 0.3430f,  0.1264f,  0.1519f,  -1.3639f, -1.6593f, 1.8127f,  -1.4459f, -0.2158f,
        -0.9792f, -1.4392f, 0.6508f,  0.8964f,  0.5717f,  -0.2390f, 0.6983f,  -1.3416f, 0.2715f,
        -0.2852f, 0.6051f,  0.2167f,  -0.2181f, -1.6306f, 1.4788f,  0.2754f,  -0.0261f, -0.4618f,
        -0.5646f, -1.0389f, 0.5819f,  1.3697f,  0.0002f,  1.5333f,  -1.0556f, -0.1254f, 0.1527f,
        -0.5996f, -1.0962f, 1.6327f,  1.3951f,  0.8784f,  0.3389f,  1.2907f,  0.3124f,  0.7299f,
        1.4220f,  0.3375f,  0.0438f,  1.8698f,  -0.2635f, -2.0799f, -0.6313f, 0.4090f,  -1.1458f,
        0.0784f,  -1.8848f, -1.6165f, 0.6179f,  0.9905f,  -0.0729f, 0.5054f,  -0.6681f, -1.4382f,
        1.7547f,  -0.9605f, -0.4558f, -1.6105f, 0.2979f,  1.1537f,  -1.5604f, 1.2779f,  -1.2514f,
        0.6056f,  0.5763f,  -3.3558f, 0.2836f,  0.6909f,  -0.7631f, 2.4451f,  -0.3500f, 1.3289f,
        -0.6494f, 0.3478f,  1.0038f,  -0.2937f, 0.9238f,  -1.2185f, 0.4138f,  0.5033f,  0.9174f,
        1.8131f,  1.4436f,  -0.4207f, 0.0220f,  -0.6807f, -1.3306f, 1.5646f,  0.3338f,  0.7105f,
        0.4683f,  -0.6179f, 0.0818f,  -0.0488f, -0.9810f, -1.3632f, 0.0929f,  -1.7926f, -0.2921f,
        -0.4792f, 0.6756f,  -0.3413f, -0.2242f, -0.2111f, 0.6282f,  0.1667f,  -1.4055f, 1.5895f,
        1.0838f,  -0.9077f, -0.8060f, 0.7967f,  -2.9351f, 2.4179f,  -0.4026f, 0.6451f,  1.6845f,
        -0.0901f, 0.6106f,  2.3603f,  1.3908f,  -0.7917f, -0.6734f, -0.1213f, -1.1116f, -0.7401f,
        -0.7879f, 0.0606f,  -2.3337f, -1.2603f, -1.7245f, -0.3533f, -0.9421f, -0.1776f, 0.3992f,
        -1.7142f, -0.5319f, -0.8848f, 0.6513f,  1.0002f,  -1.4699f, -1.4254f, 0.7013f,  0.2414f,
        0.2551f,  -0.7457f, 0.3133f,  -1.0941f, -0.3682f, -0.0163f, -0.0645f, -0.8101f, 0.1415f,
        0.0551f,  0.5873f,  -0.5887f, -1.4733f, -0.8565f, 0.7400f,  -0.5033f, 0.0553f,  0.9265f,
        -0.8652f, -0.0288f, -0.2209f, 0.0610f,  0.6776f,  0.4361f,  -0.8052f, 0.3955f,  0.8988f,
        0.8238f,  0.2262f,  1.2912f,  0.6488f,  1.2114f,  1.3569f,  0.2983f,  0.4718f,  -1.1936f,
        0.7928f,  -0.8665f, 0.9468f,  1.1629f,  0.0616f,  -1.3136f, -0.2764f, 0.0277f,  -0.1126f,
        0.2342f,  -0.5866f, -1.8219f, 1.1079f,  0.5795f,  -1.4249f};
    std::vector<migraphx::half> input_data{input.cbegin(), input.cend()};

    std::vector<int> pos_ids_data = {0};

    std::vector<float> cos_cache = {
        1.0000f,  1.0000f,  1.0000f, 0.5403f,  0.9989f,  1.0000f,  -0.4161f, 0.9957f,
        1.0000f,  -0.9900f, 0.9903f, 1.0000f,  -0.6536f, 0.9828f,  1.0000f,  0.2837f,
        0.9732f,  0.9999f,  0.9602f, 0.9615f,  0.9999f,  0.7539f,  0.9477f,  0.9999f,
        -0.1455f, 0.9318f,  0.9999f, -0.9111f, 0.9140f,  0.9998f,  -0.8391f, 0.8942f,
        0.9998f,  0.0044f,  0.8725f, 0.9997f,  0.8439f,  0.8488f,  0.9997f,  0.9074f,
        0.8234f,  0.9996f,  0.1367f, 0.7962f,  0.9995f,  -0.7597f, 0.7673f,  0.9995f};
    std::vector<migraphx::half> cos_cache_data{cos_cache.cbegin(), cos_cache.cend()};

    std::vector<float> sin_cache = {
        0.0000f, 0.0000f,  0.0000f,  0.8415f, 0.0464f,  0.0022f, 0.9093f,  0.0927f,
        0.0043f, 0.1411f,  0.1388f,  0.0065f, -0.7568f, 0.1846f, 0.0086f,  -0.9589f,
        0.2300f, 0.0108f,  -0.2794f, 0.2749f, 0.0129f,  0.6570f, 0.3192f,  0.0151f,
        0.9894f, 0.3629f,  0.0172f,  0.4121f, 0.4057f,  0.0194f, -0.5440f, 0.4477f,
        0.0215f, -1.0000f, 0.4887f,  0.0237f, -0.5366f, 0.5286f, 0.0259f,  0.4202f,
        0.5675f, 0.0280f,  0.9906f,  0.6050f, 0.0302f,  0.6503f, 0.6413f,  0.0323f};
    std::vector<migraphx::half> sin_cache_data{sin_cache.cbegin(), sin_cache.cend()};

    migraphx::parameter_map param_map;
    param_map["input"]     = migraphx::argument(input_s, input_data.data());
    param_map["pos_ids"]   = migraphx::argument(pos_ids_s, pos_ids_data.data());
    param_map["cos_cache"] = migraphx::argument(cache_s, cos_cache_data.data());
    param_map["sin_cache"] = migraphx::argument(cache_s, sin_cache_data.data());

    auto result = p.eval(param_map).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        -1.0410f, 0.9165f,  -1.3047f, -1.1094f, -1.2188f, 1.1680f,  -0.8154f, -0.6855f, -1.6865f,
        1.7725f,  0.0412f,  1.5850f,  0.2761f,  0.2402f,  0.1810f,  0.6655f,  0.8770f,  -0.9238f,
        -0.7446f, 2.0820f,  -0.3877f, 1.6982f,  2.2070f,  -0.6440f, 0.6016f,  0.8779f,  -0.9556f,
        0.1716f,  -0.5688f, -0.2400f, 0.9160f,  0.8877f,  1.0117f,  -0.9951f, -0.5786f, -0.4468f,
        0.4548f,  0.9321f,  -0.3149f, -0.6533f, -0.5259f, -0.6479f, 1.2979f,  0.5356f,  0.5029f,
        0.2262f,  0.2471f,  0.8608f,  1.0078f,  -0.7529f, -0.2250f, -0.4326f, -1.5068f, -0.4585f,
        -1.7236f, -1.2441f, -0.7671f, -0.1754f, 1.1426f,  -0.6323f, 0.4185f,  0.2302f,  -0.1519f,
        1.1895f,  0.5381f,  -0.1907f, -2.4883f, 0.4004f,  -0.1769f, -0.9746f, -0.0133f, 1.5186f,
        0.9521f,  0.2976f,  0.0386f,  -0.0434f, 1.0576f,  0.8950f,  -0.6211f, 1.3486f,  -1.2188f,
        1.3721f,  -0.2457f, 0.9414f,  0.7705f,  0.5249f,  0.0688f,  1.5986f,  -0.5015f, 0.9170f,
        0.2944f,  0.0657f,  1.0059f,  -1.2939f, 0.6626f,  -1.1289f, 0.4641f,  -0.4985f, 0.1289f,
        2.7637f,  0.1405f,  1.1191f,  1.9619f,  1.2500f,  0.1815f,  -0.1235f, -0.2742f, 0.3347f,
        1.2637f,  -0.0815f, 0.3235f,  -0.6953f, -0.9321f, 1.8691f,  -0.7666f, -1.0928f, 1.2129f,
        0.1227f,  -1.0977f, 1.0146f,  1.1201f,  -1.3838f, -1.0889f, 1.6562f,  -0.2759f, -0.2742f,
        2.1074f,  0.4268f,  0.8071f,  0.2898f,  -0.0714f, 0.8257f,  0.5669f,  -1.6719f, -0.1503f,
        0.2913f,  0.6089f,  -1.4121f, 0.1392f,  0.1617f,  0.2402f,  0.4148f,  -0.8877f, -0.6523f,
        -1.2939f, -0.0243f, -0.2354f, -0.7085f, 1.1562f,  0.4297f,  0.9565f,  0.0505f,  0.3252f,
        0.1910f,  2.1074f,  1.0742f,  -0.0747f, 0.8755f,  0.7275f,  -0.1398f, -0.1554f, 0.1545f,
        -0.5903f, 0.2546f,  -0.7510f, 1.1699f,  0.0618f,  1.1865f,  0.2114f,  1.3857f,  -0.7573f,
        -1.0146f, -0.9639f, -0.1074f, 1.0791f,  0.2991f,  -1.3994f, -0.5405f, -1.0293f, -1.5879f,
        0.0489f,  0.3425f,  0.0797f,  0.1807f,  -1.3428f, -1.6768f, 2.3164f,  0.1006f,  0.1078f,
        -0.9966f, -1.4492f, 0.6289f,  0.8965f,  0.5718f,  -0.2390f, 0.6982f,  -1.3418f, 0.2715f,
        -0.6626f, 0.0870f,  0.2264f,  -0.2078f, -1.6338f, 1.4746f,  -0.0908f, 0.2610f,  -0.4072f,
        -0.6040f, -1.0410f, 0.5771f,  -1.3564f, 0.1930f,  1.6641f,  -0.8320f, -0.1263f, 0.1519f,
        -0.4377f, 1.1699f,  1.3467f,  1.6719f,  0.8755f,  0.3462f,  0.6655f,  -1.1484f, 0.3831f,
        1.5498f,  0.3369f,  0.0474f,  1.7207f,  -0.7744f, -1.8252f, -1.1787f, 0.4236f,  -1.1396f,
        1.2969f,  -1.3691f, -1.7275f, 0.0693f,  0.9917f,  -0.0579f, 0.5054f,  -0.6680f, -1.4385f,
        1.7549f,  -0.9604f, -0.4558f, -1.1201f, -1.1934f, 1.2236f,  -1.5049f, 1.2803f,  -1.2480f,
        -0.7749f, 0.3105f,  -3.3652f, -0.0288f, 0.6938f,  -0.7598f, -2.3691f, 0.6914f,  1.4053f,
        -0.4585f, 0.3413f,  1.0059f,  0.8906f,  -0.3818f, -1.2734f, 0.1819f,  0.4954f,  0.9214f,
        1.8975f,  -1.3281f, -0.4141f, -0.0753f, -0.6660f, -1.3379f, 1.5938f,  -0.1165f, 0.5537f,
        0.6450f,  -0.6187f, 0.0738f,  0.6074f,  -0.7710f, -1.3213f, -0.3467f, -1.7881f, -0.3188f,
        -0.4792f, 0.6758f,  -0.3413f, -0.2242f, -0.2111f, 0.6284f,  1.2715f,  -0.6187f, 1.5371f,
        1.1553f,  -0.9058f, -0.8081f, 2.3359f,  1.9443f,  2.4434f,  -0.1765f, 0.6377f,  1.6865f,
        0.0030f,  -0.6172f, 2.1426f,  1.7041f,  -0.7871f, -0.6782f, -0.7612f, 0.8179f,  -0.5820f,
        -0.9106f, 0.0806f,  -2.3320f, -2.0098f, 0.7197f,  -0.1272f, -0.9976f, -0.1819f, 0.3972f,
        -1.7930f, -0.0317f, -1.0293f, 0.3828f,  1.0186f,  -1.4561f, -1.5342f, -0.4087f, 0.1473f,
        0.3186f,  -0.7500f, 0.3018f,  -1.0938f, -0.3682f, -0.0163f, -0.0645f, -0.8101f, 0.1415f,
        -0.4644f, 0.3635f,  -0.5195f, -1.4980f, -0.8579f, 0.7383f,  0.1591f,  -0.4805f, 1.0020f,
        -0.7754f, -0.0278f, -0.2211f, -0.1559f, -0.6621f, 0.5430f,  -0.7363f, 0.3896f,  0.9014f,
        -0.3669f, -0.7705f, 1.1484f,  0.8760f,  1.1992f,  1.3662f,  0.5366f,  -0.1521f, -1.3428f,
        0.4971f,  -0.8765f, 0.9370f,  1.1328f,  -0.2654f, -1.1865f, -0.6265f, 0.0291f,  -0.1122f,
        0.5615f,  -0.2878f, -2.0781f, 0.4678f,  0.6011f,  -1.4160f};

    EXPECT(migraphx::verify::verify_range_with_tolerance(
        result_vector, migraphx::verify::expected{gold}, migraphx::verify::tolerance{1e-3}));
}

TEST_CASE(rotary_embedding_dim_verify_test)
{
    std::size_t batch_size           = 1;
    std::size_t sequence_length      = 2;
    std::size_t num_heads            = 1;
    std::size_t head_size            = 6;
    std::size_t rotary_embedding_dim = 4;
    std::size_t max_sequence_length  = 2;

    migraphx::program p = read_onnx("rotary_embedding_dim_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_s{migraphx::shape::half_type,
                            {batch_size, sequence_length, num_heads * head_size}};
    migraphx::shape pos_ids_s{migraphx::shape::int32_type, {1}};
    migraphx::shape cache_s{migraphx::shape::half_type,
                            {max_sequence_length, rotary_embedding_dim / 2}};

    std::vector<float> input = {-1.0408f,
                                0.9166f,
                                -1.3042f,
                                -1.1097f,
                                -1.2188f,
                                1.1676f,
                                1.0076f,
                                -0.7529f,
                                -0.2250f,
                                -0.4327f,
                                -1.5071f,
                                -0.4586f};
    std::vector<migraphx::half> input_data{input.cbegin(), input.cend()};

    std::vector<int> pos_ids_data = {0, 1};

    std::vector<float> cos_cache = {1.0000f, 1.0000f, 1.0000f, 0.5403f};
    std::vector<migraphx::half> cos_cache_data{cos_cache.cbegin(), cos_cache.cend()};

    std::vector<float> sin_cache = {0.0000f, 0.0000f, 0.0000f, 0.8415f};
    std::vector<migraphx::half> sin_cache_data{sin_cache.cbegin(), sin_cache.cend()};

    migraphx::parameter_map param_map;
    param_map["input"]     = migraphx::argument(input_s, input_data.data());
    param_map["pos_ids"]   = migraphx::argument(pos_ids_s, pos_ids_data.data());
    param_map["cos_cache"] = migraphx::argument(cache_s, cos_cache_data.data());
    param_map["sin_cache"] = migraphx::argument(cache_s, sin_cache_data.data());

    auto result = p.eval(param_map).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {-1.0408f,
                               0.9166f,
                               -1.3042f,
                               -1.1097f,
                               -1.2188f,
                               1.1676f,
                               1.0076f,
                               -0.0427f,
                               -0.2250f,
                               -0.8673f,
                               -1.5071f,
                               -0.4586f};

    EXPECT(migraphx::verify::verify_range_with_tolerance(
        result_vector, migraphx::verify::expected{gold}, migraphx::verify::tolerance{1e-3}));
}
