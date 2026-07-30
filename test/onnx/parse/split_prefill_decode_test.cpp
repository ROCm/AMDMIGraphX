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
#include <migraphx/register_target.hpp>
#include <algorithm>
#include <cassert>

namespace {

migraphx::instruction_ref find_select_module(migraphx::module_ref mod)
{
    auto result = std::find_if(
        mod->begin(), mod->end(), [](const auto& ins) { return ins.name() == "select_module"; });
    assert(result != mod->end());
    return result;
}

std::size_t count_literals(migraphx::const_module_ref mod)
{
    auto instructions = migraphx::iterator_for(*mod);
    return std::count_if(instructions.begin(), instructions.end(), [](auto ins) {
        return ins->name() == "@literal";
    });
}

std::vector<float>
run_program(migraphx::program& p, const std::vector<std::size_t>& lens, std::vector<float> data)
{
    migraphx::parameter_map params;
    params["x"] =
        migraphx::argument{migraphx::shape{migraphx::shape::float_type, lens}, data.data()};
    auto result = p.eval(params).back();
    std::vector<float> output;
    result.visit([&](auto view) { output.assign(view.begin(), view.end()); });
    return output;
}

} // namespace

TEST_CASE(split_prefill_decode_test)
{
    migraphx::onnx_options options;
    options.use_symbolic_shapes           = true;
    options.dim_params["sequence_length"] = {1, 4};
    auto p                                = read_onnx("split_prefill_decode_test.onnx", options);

    auto* root      = p.get_main_module();
    auto select     = find_select_module(root);
    auto submodules = select->module_inputs();
    EXPECT(submodules.size() == 2);
    EXPECT(submodules.at(0)->name() == "main:split_prefill_decode:decode");
    EXPECT(submodules.at(1)->name() == "main:split_prefill_decode:prefill");
    EXPECT(submodules.at(0)->get_parameter_shape("x").lens() == std::vector<std::size_t>{1, 1, 2});
    EXPECT(submodules.at(1)->get_parameter_shape("x").lens() == std::vector<std::size_t>{1, 4, 2});

    EXPECT(count_literals(root) == 1);
    EXPECT(count_literals(submodules.at(0)) == 0);
    EXPECT(count_literals(submodules.at(1)) == 0);

    p.compile(migraphx::make_target("ref"));
    EXPECT(run_program(p, {1, 1, 2}, {1.0f, 2.0f}) == std::vector<float>{2.0f, 3.0f});
    EXPECT(run_program(p, {1, 4, 2}, std::vector<float>(8, 2.0f)) == std::vector<float>(8, 3.0f));
}

TEST_CASE(group_query_attention_symbolic_test)
{
    migraphx::onnx_options options;
    options.use_symbolic_shapes           = true;
    options.dim_params["sequence_length"] = {1, 8};
    auto p = read_onnx("group_query_attention_symbolic_test.onnx", options);

    auto* root      = p.get_main_module();
    auto select     = find_select_module(root);
    auto submodules = select->module_inputs();
    EXPECT(submodules.size() == 2);
    EXPECT(submodules.at(0)->get_parameter_shape("qkv").lens() ==
           std::vector<std::size_t>{1, 1, 96});
    EXPECT(submodules.at(1)->get_parameter_shape("qkv").lens() ==
           std::vector<std::size_t>{1, 8, 96});
    EXPECT(count_literals(root) == 3);
    EXPECT(submodules.at(0)->get_parameter_shape("cos_cache").lens() ==
           std::vector<std::size_t>{10, 8});
    EXPECT(submodules.at(1)->get_parameter_shape("cos_cache").lens() ==
           std::vector<std::size_t>{10, 8});
}
