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

#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/split_sym_dim.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

static std::vector<std::vector<float>>
eval_group_query_attention(migraphx::program& p, std::size_t sequence, int past_length)
{
    migraphx::shape qkv_shape{migraphx::shape::half_type, {1, sequence, 96}};
    migraphx::shape past_shape{migraphx::shape::half_type, {1, 2, 10, 16}};
    migraphx::shape key_value_shape{migraphx::shape::float_type, {1}};
    migraphx::shape slk_shape{migraphx::shape::int32_type, {1, 1}};
    std::vector<float> qkv_data(qkv_shape.elements());
    std::vector<float> past_data(past_shape.elements());
    std::size_t n = 0;
    std::generate(qkv_data.begin(), qkv_data.end(), [&] { return ((n++ * 37) % 19) / 4.0f - 2; });
    std::generate(past_data.begin(), past_data.end(), [&] { return ((n++ * 23) % 17) / 4.0f - 2; });

    migraphx::parameter_map pp;
    pp["qkv"]                   = migraphx::literal{qkv_shape, qkv_data}.get_argument();
    pp["key"]                   = migraphx::literal{key_value_shape, {0.0f}}.get_argument();
    pp["value"]                 = migraphx::literal{key_value_shape, {0.0f}}.get_argument();
    pp["past_key_values_key"]   = migraphx::literal{past_shape, past_data}.get_argument();
    pp["past_key_values_value"] = migraphx::literal{past_shape, past_data}.get_argument();
    pp["seqlens_k"]             = migraphx::literal{slk_shape, {past_length}}.get_argument();

    auto outputs = p.eval(pp);
    std::vector<std::vector<float>> result(outputs.size());
    std::transform(outputs.begin(), outputs.end(), result.begin(), [](const auto& output) {
        return output.template to_vector<float>();
    });
    return result;
}

// A symbolic sequence length picks the prompt or single-token positions and local window at
// runtime, which has to match parsing with the length fixed. The ref target cannot evaluate a
// reshape to symbolic dims, so split_sym_dim specializes the graph first, as on the GPU.
TEST_CASE(group_query_attention_symbolic_prefill_local_test)
{
    migraphx::onnx_options symbolic_options;
    symbolic_options.use_symbolic_shapes       = true;
    symbolic_options.map_dyn_input_dims["qkv"] = {{1, 1}, {1, 8}, {96, 96}};
    auto symbolic = read_onnx("group_query_attention_prefill_local_test.onnx", symbolic_options);
    migraphx::run_passes(symbolic, {migraphx::split_sym_dim{}, migraphx::dead_code_elimination{}});
    symbolic.compile(migraphx::make_target("ref"));

    migraphx::onnx_options static_options;
    static_options.map_input_dims["qkv"] = {1, 8, 96};
    auto fixed = read_onnx("group_query_attention_prefill_local_test.onnx", static_options);
    fixed.compile(migraphx::make_target("ref"));

    auto result = eval_group_query_attention(symbolic, 8, 7);
    auto gold   = eval_group_query_attention(fixed, 8, 7);
    EXPECT(result.size() == 3);
    EXPECT(result.size() == gold.size());
    EXPECT(migraphx::verify::verify_rms_range(result[0], gold[0]));
    EXPECT(migraphx::verify::verify_rms_range(result[1], gold[1]));
    EXPECT(migraphx::verify::verify_rms_range(result[2], gold[2]));
}

TEST_CASE(group_query_attention_symbolic_decode_local_test)
{
    migraphx::onnx_options symbolic_options;
    symbolic_options.use_symbolic_shapes       = true;
    symbolic_options.map_dyn_input_dims["qkv"] = {{1, 1}, {1, 8}, {96, 96}};
    auto symbolic = read_onnx("group_query_attention_prefill_local_test.onnx", symbolic_options);
    migraphx::run_passes(symbolic, {migraphx::split_sym_dim{}, migraphx::dead_code_elimination{}});
    symbolic.compile(migraphx::make_target("ref"));

    migraphx::onnx_options static_options;
    static_options.map_input_dims["qkv"] = {1, 1, 96};
    auto fixed = read_onnx("group_query_attention_prefill_local_test.onnx", static_options);
    fixed.compile(migraphx::make_target("ref"));

    auto result = eval_group_query_attention(symbolic, 1, 5);
    auto gold   = eval_group_query_attention(fixed, 1, 5);
    EXPECT(result.size() == 3);
    EXPECT(result.size() == gold.size());
    EXPECT(migraphx::verify::verify_rms_range(result[0], gold[0]));
    EXPECT(migraphx::verify::verify_rms_range(result[1], gold[1]));
    EXPECT(migraphx::verify::verify_rms_range(result[2], gold[2]));
}
