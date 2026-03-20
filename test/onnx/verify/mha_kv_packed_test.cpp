/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

TEST_CASE(multi_head_attention_kv_packed_test)
{
    migraphx::program p = read_onnx("mha_kv_packed_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape q_s{migraphx::shape::float_type, {1, 2, 4}};
    migraphx::shape kv_s{migraphx::shape::float_type, {1, 2, 2, 2, 2}};
    std::vector<float> query     = {1, 3, 5, 7, 2, 4, 6, 8};
    std::vector<float> key_value = {1, 3, 1, 3, 5, 7, 5, 7, 2, 4, 2, 4, 6, 8, 6, 8};

    migraphx::parameter_map pp;
    pp["q"]  = migraphx::argument(q_s, query.data());
    pp["kv"] = migraphx::argument(kv_s, key_value.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {
        1.9441926, 3.9441926, 5.9997935, 7.999794, 1.9858339, 3.9858341, 5.99995, 7.9999495};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(multi_head_attention_kv_packed_bias_key_padding_mask_test)
{
    migraphx::program p = read_onnx("mha_kv_packed_bias_key_padding_mask_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape q_s{migraphx::shape::float_type, {1, 2, 4}};
    migraphx::shape kv_s{migraphx::shape::float_type, {1, 2, 2, 2, 2}};
    migraphx::shape bias_s{migraphx::shape::float_type, {12}};
    migraphx::shape mask_s{migraphx::shape::int32_type, {1, 2}};

    std::vector<float> query = {1.63389, 1.35173, 1.3877, 1.52187, 0.901106, 0.730935, 0.639857, 0.926961};
    std::vector<float> key_value = {1.61202, 1.33192, 1.61202, 1.33192, 1.03611, 1.09385, 1.03611, 1.09385, 
                                    0.776542, 0.692002, 0.776542, 0.692002, 0.544437, 0.687849, 0.544437, 0.687849};
    std::vector<float> bias_data(12, 0.0f);
    std::vector<int32_t> mask_data = {1, 1};

    migraphx::parameter_map pp;
    pp["q"]                = migraphx::argument(q_s, query.data());
    pp["kv"]               = migraphx::argument(kv_s, key_value.data());
    pp["bias"]             = migraphx::argument(bias_s, bias_data.data());
    pp["key_padding_mask"] = migraphx::argument(mask_s, mask_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    
    // Gold data computed from reference implementation with KV packed format
    std::vector<float> gold = {
        1.380390f, 1.002805f, 0.6146411f, 0.791859f, 1.28671f, 0.919257f, 0.5846517f, 0.756680f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
