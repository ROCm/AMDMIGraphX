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

TEST_CASE(mha_bias_key_padding_mask_attention_bias_test)
{
    migraphx::program p = read_onnx("mha_attention_bias_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape q_s{migraphx::shape::float_type, {1, 2, 4}};
    migraphx::shape k_s{migraphx::shape::float_type, {1, 2, 4}};
    migraphx::shape v_s{migraphx::shape::float_type, {1, 2, 4}};
    migraphx::shape bias_s{migraphx::shape::float_type, {12}};
    migraphx::shape mask_s{migraphx::shape::int32_type, {1, 2}};
    migraphx::shape attn_bias_s{migraphx::shape::float_type, {1, 2, 2, 2}};

    std::vector<float> query = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> key   = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> value = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> bias_data(12, 0);
    std::vector<int32_t> mask_data = {1, 1};
    std::vector<float> attn_bias_data(8, 0);

    migraphx::parameter_map pp;
    pp["q"]                = migraphx::argument(q_s, query.data());
    pp["k"]                = migraphx::argument(k_s, key.data());
    pp["v"]                = migraphx::argument(v_s, value.data());
    pp["bias"]             = migraphx::argument(bias_s, bias_data.data());
    pp["key_padding_mask"] = migraphx::argument(mask_s, mask_data.data());
    pp["attention_bias"]   = migraphx::argument(attn_bias_s, attn_bias_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {4.99917, 5.99917, 7, 8, 5, 6, 7, 8};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
