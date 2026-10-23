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
#include <algorithm>

TEST_CASE(group_query_attention_symbolic_parse_test)
{
    auto sequence = migraphx::sym::var("sequence", {1, 8});

    migraphx::onnx_options options;
    options.use_symbolic_shapes = true;
    options.map_dyn_input_dims["qkv"] =
        sym_dims({migraphx::sym::lit(1), sequence, migraphx::sym::lit(12288)});
    options.map_dyn_input_dims["past_key_values_key"] = sym_dims(
        {migraphx::sym::lit(1), migraphx::sym::lit(32), sequence, migraphx::sym::lit(128)});
    options.map_dyn_input_dims["past_key_values_value"] =
        options.map_dyn_input_dims.at("past_key_values_key");

    auto p = read_onnx("group_query_attention_defaults_test.onnx", options);
    auto expected_output =
        migraphx::shape{migraphx::shape::half_type,
                        sym_dims({migraphx::sym::lit(1), sequence, migraphx::sym::lit(4096)})};
    auto expected_cache = migraphx::shape{
        migraphx::shape::half_type,
        sym_dims(
            {migraphx::sym::lit(1), migraphx::sym::lit(32), sequence, migraphx::sym::lit(128)})};
    EXPECT(p.get_output_shapes() ==
           std::vector<migraphx::shape>{expected_output, expected_cache, expected_cache});
    EXPECT(std::count_if(p.get_main_module()->begin(), p.get_main_module()->end(), [](auto ins) {
               return ins.name() == "dynamic_range";
           }) == 2);
    EXPECT(std::count_if(p.get_main_module()->begin(), p.get_main_module()->end(), [](auto ins) {
               return ins.name() == "concat_past_present";
           }) == 2);
    EXPECT(std::all_of(p.get_main_module()->begin(), p.get_main_module()->end(), [](auto ins) {
        return ins.name() != "concat_past_present" or ins.inputs().size() == 4;
    }));
    EXPECT(std::none_of(p.get_main_module()->begin(), p.get_main_module()->end(), [](auto ins) {
        return ins.name() == "reshape" and ins.get_shape().symbolic() and ins.inputs().size() == 1;
    }));
}

TEST_CASE(group_query_attention_symbolic_query_static_cache_parse_test)
{
    auto sequence = migraphx::sym::var("sequence", {1, 8});

    migraphx::onnx_options options;
    options.use_symbolic_shapes = true;
    options.map_dyn_input_dims["qkv"] =
        sym_dims({migraphx::sym::lit(1), sequence, migraphx::sym::lit(12288)});

    auto p = read_onnx("group_query_attention_defaults_test.onnx", options);
    auto expected_output =
        migraphx::shape{migraphx::shape::half_type,
                        sym_dims({migraphx::sym::lit(1), sequence, migraphx::sym::lit(4096)})};
    auto expected_cache = migraphx::shape{migraphx::shape::half_type, {1, 32, 4096, 128}};
    EXPECT(p.get_output_shapes() ==
           std::vector<migraphx::shape>{expected_output, expected_cache, expected_cache});
    EXPECT(std::count_if(p.get_main_module()->begin(), p.get_main_module()->end(), [](auto ins) {
               return ins.name() == "dynamic_range";
           }) == 2);
    EXPECT(std::count_if(p.get_main_module()->begin(), p.get_main_module()->end(), [](auto ins) {
               return ins.name() == "concat_past_present";
           }) == 2);
}

TEST_CASE(group_query_attention_batched_seqlens_symbolic_parse_test)
{
    auto sequence = migraphx::sym::var("sequence", {1, 2});

    migraphx::onnx_options options;
    options.use_symbolic_shapes = true;
    options.map_dyn_input_dims["qkv"] =
        sym_dims({migraphx::sym::lit(2), sequence, migraphx::sym::lit(16)});

    auto p = read_onnx("group_query_attention_batched_seqlens_test.onnx", options);
    auto expected_output =
        migraphx::shape{migraphx::shape::half_type,
                        sym_dims({migraphx::sym::lit(2), sequence, migraphx::sym::lit(8)})};
    auto expected_cache = migraphx::shape{migraphx::shape::half_type, {2, 2, 4, 2}};
    EXPECT(p.get_output_shapes() ==
           std::vector<migraphx::shape>{expected_output, expected_cache, expected_cache});
    EXPECT(std::count_if(p.get_main_module()->begin(), p.get_main_module()->end(), [](auto ins) {
               return ins.name() == "unsqueeze";
           }) == 3);
}

TEST_CASE(group_query_attention_head_sink_symbolic_parse_test)
{
    auto sequence = migraphx::sym::var("sequence", {1, 4});

    migraphx::onnx_options options;
    options.use_symbolic_shapes = true;
    options.map_dyn_input_dims["qkv"] =
        sym_dims({migraphx::sym::lit(1), sequence, migraphx::sym::lit(96)});

    auto p = read_onnx("group_query_attention_head_sink_decode_test.onnx", options);
    auto expected_output =
        migraphx::shape{migraphx::shape::half_type,
                        sym_dims({migraphx::sym::lit(1), sequence, migraphx::sym::lit(32)})};
    auto expected_cache = migraphx::shape{migraphx::shape::half_type, {1, 2, 10, 16}};
    EXPECT(p.get_output_shapes() ==
           std::vector<migraphx::shape>{expected_output, expected_cache, expected_cache});
    EXPECT(std::count_if(p.get_main_module()->begin(), p.get_main_module()->end(), [](auto ins) {
               return ins.name() == "dyn_slice";
           }) == 1);
}
