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

#include <algorithm>

namespace {

std::vector<std::vector<float>> run_gqa(const std::string& model,
                                        const migraphx::shape& qkv_shape,
                                        std::vector<float> qkv_data,
                                        bool symbolic)
{
    migraphx::program p;
    if(symbolic)
    {
        auto sequence = migraphx::sym::var("sequence", {1, 8}, {1, 8});
        migraphx::onnx_options options;
        options.use_symbolic_shapes = true;
        options.map_dyn_input_dims["qkv"] =
            sym_dims({migraphx::sym::lit(1), sequence, migraphx::sym::lit(96)});
        p = read_onnx(model, options);
    }
    else
    {
        p = read_onnx(model);
    }
    p.compile(migraphx::make_target("ref"));

    migraphx::shape scalar_shape{migraphx::shape::float_type, {1}};
    std::vector<float> scalar_data = {0};
    migraphx::shape cache_shape{migraphx::shape::half_type, {1, 2, 10, 16}};
    std::vector<float> key_cache_data(cache_shape.elements());
    std::generate(key_cache_data.begin(), key_cache_data.end(), [i = 0]() mutable {
        return static_cast<float>((i++ % 17) - 8) / 16.0f;
    });
    auto value_cache_data = key_cache_data;
    migraphx::shape seqlens_shape{migraphx::shape::int32_type, {1, 1}};
    std::vector<int> seqlens_data = {8};

    migraphx::parameter_map parameters;
    parameters["qkv"]                   = migraphx::argument(qkv_shape, qkv_data.data());
    parameters["key"]                   = migraphx::argument(scalar_shape, scalar_data.data());
    parameters["value"]                 = migraphx::argument(scalar_shape, scalar_data.data());
    parameters["past_key_values_key"]   = migraphx::argument(cache_shape, key_cache_data.data());
    parameters["past_key_values_value"] = migraphx::argument(cache_shape, value_cache_data.data());
    parameters["seqlens_k"]             = migraphx::argument(seqlens_shape, seqlens_data.data());

    std::vector<std::vector<float>> result;
    for(const auto& output : p.eval(parameters))
    {
        std::vector<float> values;
        output.visit([&](auto data) { values.assign(data.begin(), data.end()); });
        result.push_back(std::move(values));
    }
    return result;
}

void verify_symbolic_gqa(const std::string& model, const migraphx::shape& qkv_shape)
{
    std::vector<float> qkv_data(qkv_shape.elements());
    std::generate(qkv_data.begin(), qkv_data.end(), [i = 0]() mutable {
        return static_cast<float>((i++ % 23) - 11) / 8.0f;
    });
    auto static_results   = run_gqa(model, qkv_shape, qkv_data, false);
    auto symbolic_results = run_gqa(model, qkv_shape, qkv_data, true);
    EXPECT(static_results.size() == symbolic_results.size());
    if(static_results.size() != symbolic_results.size())
        return;
    EXPECT(std::equal(static_results.begin(),
                      static_results.end(),
                      symbolic_results.begin(),
                      [](const auto& static_result, const auto& symbolic_result) {
                          return migraphx::verify::verify_rms_range(static_result, symbolic_result);
                      }));
}

} // namespace

TEST_CASE(group_query_attention_symbolic_rotary_local_window_verify)
{
    verify_symbolic_gqa("group_query_attention_decode_local_test.onnx",
                        migraphx::shape{migraphx::shape::half_type, {1, 1, 96}});
    verify_symbolic_gqa("group_query_attention_prefill_local_test.onnx",
                        migraphx::shape{migraphx::shape::half_type, {1, 8, 96}});
}
