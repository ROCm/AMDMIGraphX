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
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <algorithm>

namespace {

std::vector<migraphx::module_ref> find_specializations(migraphx::const_module_ref mod)
{
    auto instructions = migraphx::iterator_for(*mod);
    auto select       = std::find_if(instructions.begin(), instructions.end(), [](auto ins) {
        return ins->name() == "select_module";
    });
    if(select == instructions.end())
        return {};
    return (*select)->module_inputs();
}

std::size_t count_literals(migraphx::const_module_ref mod)
{
    auto instructions = migraphx::iterator_for(*mod);
    return std::count_if(instructions.begin(), instructions.end(), [](auto ins) {
        return ins->name() == "@literal";
    });
}

migraphx::onnx_options split_options(std::size_t max_sequence_length)
{
    migraphx::onnx_options options;
    options.split_prefill_decode          = true;
    options.use_symbolic_shapes           = true;
    options.dim_params["sequence_length"] = {1, max_sequence_length};
    return options;
}

} // namespace

TEST_CASE(split_prefill_decode_test)
{
    auto p = read_onnx("split_prefill_decode_test.onnx", split_options(4));

    auto specializations = find_specializations(p.get_main_module());
    EXPECT(specializations.size() == 2);
    EXPECT(specializations.at(0)->name() == "main:split:decode");
    EXPECT(specializations.at(1)->name() == "main:split:prefill");
    EXPECT(specializations.at(0)->get_parameter_shape("x") ==
           migraphx::shape{migraphx::shape::float_type, {1, 1, 2}});
    EXPECT(specializations.at(1)->get_parameter_shape("x") ==
           migraphx::shape{migraphx::shape::float_type, {1, 4, 2}});

    // The initializer stays in the main module and reaches both specializations as a parameter.
    EXPECT(count_literals(p.get_main_module()) == 1);
    EXPECT(count_literals(specializations.at(0)) == 0);
    EXPECT(count_literals(specializations.at(1)) == 0);
    EXPECT(specializations.at(0)->get_parameter_shape("one") ==
           migraphx::shape{migraphx::shape::float_type, {1}});
    EXPECT(specializations.at(1)->get_parameter_shape("one") ==
           migraphx::shape{migraphx::shape::float_type, {1}});
}

// The split has to happen while parsing because a kv-cache attention operator cannot be parsed
// with a symbolic sequence length at all.
TEST_CASE(split_prefill_decode_group_query_attention_test)
{
    auto p = read_onnx("group_query_attention_symbolic_test.onnx", split_options(8));

    auto specializations = find_specializations(p.get_main_module());
    EXPECT(specializations.size() == 2);
    EXPECT(specializations.at(0)->get_parameter_shape("qkv").lens() ==
           std::vector<std::size_t>{1, 1, 96});
    EXPECT(specializations.at(1)->get_parameter_shape("qkv").lens() ==
           std::vector<std::size_t>{1, 8, 96});

    // The caches are initializers, so they are literals of the main module and reach the
    // specializations as parameters. Whatever constants the operator parser itself adds are not.
    EXPECT(count_literals(p.get_main_module()) == 3);
    EXPECT(specializations.at(0)->get_parameter_shape("cos_cache").lens() ==
           std::vector<std::size_t>{10, 8});
    EXPECT(specializations.at(1)->get_parameter_shape("cos_cache").lens() ==
           std::vector<std::size_t>{10, 8});
}

// The split describes how the model is used, not how its shapes are spelled, so it applies to
// plain dynamic dimensions too.
TEST_CASE(split_prefill_decode_without_symbolic_shapes_test)
{
    auto options                = split_options(4);
    options.use_symbolic_shapes = false;
    auto p                      = read_onnx("split_prefill_decode_test.onnx", options);

    auto specializations = find_specializations(p.get_main_module());
    EXPECT(specializations.size() == 2);
    EXPECT(specializations.at(0)->get_parameter_shape("x") ==
           migraphx::shape{migraphx::shape::float_type, {1, 1, 2}});
    EXPECT(specializations.at(1)->get_parameter_shape("x") ==
           migraphx::shape{migraphx::shape::float_type, {1, 4, 2}});
}

// Without the opt-in a sequence_length range is an ordinary dynamic dimension.
TEST_CASE(split_prefill_decode_not_enabled_test)
{
    auto options                 = split_options(4);
    options.split_prefill_decode = false;
    auto p                       = read_onnx("split_prefill_decode_test.onnx", options);

    EXPECT(find_specializations(p.get_main_module()).empty());
}

// Asking for the split without giving it something to split on is worth reporting: the
// alternative is a program that silently handles only one of the two phases.
TEST_CASE(split_prefill_decode_missing_dim_param_test)
{
    migraphx::onnx_options options;
    options.split_prefill_decode = true;
    options.use_symbolic_shapes  = true;
    EXPECT(test::throws([&] { read_onnx("split_prefill_decode_test.onnx", options); }));
}

// Only a range that bottoms out at a single token describes decoding.
TEST_CASE(split_prefill_decode_range_does_not_start_at_one_test)
{
    auto options                          = split_options(4);
    options.dim_params["sequence_length"] = {2, 4};
    EXPECT(test::throws([&] { read_onnx("split_prefill_decode_test.onnx", options); }));
}

// Explicit dims replace the dim-param, so there is no sequence length left to specialize on.
TEST_CASE(split_prefill_decode_input_dim_override_test)
{
    auto options                = split_options(4);
    options.map_input_dims["x"] = {1, 4, 2};
    EXPECT(test::throws([&] { read_onnx("split_prefill_decode_test.onnx", options); }));
}
