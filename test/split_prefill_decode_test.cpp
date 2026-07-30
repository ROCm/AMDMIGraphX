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
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/split_prefill_decode.hpp>
#include <migraphx/sym.hpp>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <string>
#include <vector>
#include <test.hpp>

namespace {

using dd = migraphx::shape::dynamic_dimension;

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::split_prefill_decode{}, migraphx::dead_code_elimination{}});
}

migraphx::instruction_ref find_select_module(migraphx::module_ref mod)
{
    auto result = std::find_if(
        mod->begin(), mod->end(), [](const auto& ins) { return ins.name() == "select_module"; });
    assert(result != mod->end());
    return result;
}

migraphx::program make_relu_program(const migraphx::sym::expr& sequence_length,
                                    const migraphx::sym::expr& batch = migraphx::sym::lit(2))
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto data = mm->add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{batch}, dd{sequence_length}}});
    auto result = mm->add_instruction(migraphx::make_op("relu"), data);
    mm->add_return({result});
    return p;
}

std::vector<float>
run_relu(migraphx::program& p, const std::vector<std::size_t>& lens, std::vector<float> data)
{
    migraphx::parameter_map params;
    params["data"] =
        migraphx::argument{migraphx::shape{migraphx::shape::float_type, lens}, data.data()};
    auto result = p.eval(params).back();
    std::vector<float> output;
    result.visit([&](auto view) { output.assign(view.begin(), view.end()); });
    return output;
}

std::size_t count_literals(migraphx::const_module_ref mod)
{
    auto instructions = migraphx::iterator_for(*mod);
    return std::count_if(instructions.begin(), instructions.end(), [](auto ins) {
        return ins->name() == "@literal";
    });
}

} // namespace

TEST_CASE(split_prefill_decode_modules)
{
    auto sequence_length = migraphx::sym::var("sequence_length", {1, 8});
    auto p               = make_relu_program(sequence_length);
    run_pass(p);

    auto* mm        = p.get_main_module();
    auto select     = find_select_module(mm);
    auto submodules = select->module_inputs();
    EXPECT(submodules.size() == 2);
    EXPECT(submodules.at(0)->name() == "main:split_prefill_decode:decode");
    EXPECT(submodules.at(1)->name() == "main:split_prefill_decode:prefill");
    EXPECT(submodules.at(0)->get_parameter_shape("data") ==
           migraphx::shape{migraphx::shape::float_type, {2, 1}});
    EXPECT(submodules.at(1)->get_parameter_shape("data") ==
           migraphx::shape{migraphx::shape::float_type, {2, 8}});
}

TEST_CASE(split_prefill_decode_multiple_inputs_outputs)
{
    auto sequence_length = migraphx::sym::var("sequence_length", {1, 4});
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto ids = mm->add_parameter("input_ids",
                                 migraphx::shape{migraphx::shape::float_type,
                                                 {dd{migraphx::sym::lit(2)}, dd{sequence_length}}});
    auto embeds =
        mm->add_parameter("inputs_embeds",
                          migraphx::shape{migraphx::shape::float_type,
                                          {dd{sequence_length}, dd{migraphx::sym::lit(3)}}});
    auto ids_out    = mm->add_instruction(migraphx::make_op("relu"), ids);
    auto embeds_out = mm->add_instruction(migraphx::make_op("relu"), embeds);
    mm->add_return({ids_out, embeds_out});

    run_pass(p);

    auto select     = find_select_module(mm);
    auto submodules = select->module_inputs();
    EXPECT(submodules.at(0)->get_parameter_shape("input_ids").lens() ==
           std::vector<std::size_t>{2, 1});
    EXPECT(submodules.at(0)->get_parameter_shape("inputs_embeds").lens() ==
           std::vector<std::size_t>{1, 3});
    EXPECT(submodules.at(1)->get_parameter_shape("input_ids").lens() ==
           std::vector<std::size_t>{2, 4});
    EXPECT(submodules.at(1)->get_parameter_shape("inputs_embeds").lens() ==
           std::vector<std::size_t>{4, 3});
    EXPECT(mm->get_output_shapes().size() == 2);
}

TEST_CASE(split_prefill_decode_dispatch)
{
    auto sequence_length = migraphx::sym::var("sequence_length", {1, 4});
    auto p               = make_relu_program(sequence_length);
    run_pass(p);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> decode_data{-1.0f, 2.0f};
    EXPECT(run_relu(p, {2, 1}, decode_data) == std::vector<float>{0.0f, 2.0f});

    std::vector<float> prefill_data{-1.0f, 2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, 8.0f};
    EXPECT(run_relu(p, {2, 4}, prefill_data) ==
           std::vector<float>{0.0f, 2.0f, 3.0f, 0.0f, 5.0f, 0.0f, 7.0f, 8.0f});
}

TEST_CASE(split_prefill_decode_other_dynamic_dimension)
{
    auto sequence_length = migraphx::sym::var("sequence_length", {1, 4});
    auto batch           = migraphx::sym::var("batch", {1, 2});
    auto p               = make_relu_program(sequence_length, batch);
    run_pass(p);
    p.compile(migraphx::make_target("ref"));

    EXPECT(run_relu(p, {2, 1}, {-1.0f, 2.0f}) == std::vector<float>{0.0f, 2.0f});
    EXPECT(run_relu(p, {1, 4}, {-1.0f, 2.0f, 3.0f, -4.0f}) ==
           std::vector<float>{0.0f, 2.0f, 3.0f, 0.0f});

    migraphx::parameter_map params;
    std::vector<float> unsupported_data(4, 1.0f);
    params["data"] = migraphx::argument{migraphx::shape{migraphx::shape::float_type, {2, 2}},
                                        unsupported_data.data()};
    EXPECT(test::throws([&] { std::ignore = p.eval(params); }));
}

TEST_CASE(split_prefill_decode_preserves_layout)
{
    auto sequence_length = migraphx::sym::var("sequence_length", {1, 4});
    auto batch           = migraphx::sym::var("batch", {1, 2});
    auto input_shape     = migraphx::shape::from_permutation(
        migraphx::shape::float_type, {dd{batch}, dd{sequence_length}}, {1, 0});
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto data   = mm->add_parameter("data", input_shape);
    auto result = mm->add_instruction(migraphx::make_op("relu"), data);
    mm->add_return({result});
    run_pass(p);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> values(8, 1.0f);
    auto transposed_shape =
        migraphx::shape::from_permutation(migraphx::shape::float_type, {2, 4}, {1, 0});
    migraphx::parameter_map params;
    params["data"] = migraphx::argument{transposed_shape, values.data()};
    EXPECT(p.eval(params).back().get_shape() == transposed_shape);

    params["data"] =
        migraphx::argument{migraphx::shape{migraphx::shape::float_type, {2, 4}}, values.data()};
    EXPECT(test::throws([&] { std::ignore = p.eval(params); }));
}

TEST_CASE(split_prefill_decode_shared_literal_roundtrip)
{
    auto sequence_length = migraphx::sym::var("sequence_length", {1, 4});
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto data =
        mm->add_parameter("data",
                          migraphx::shape{migraphx::shape::float_type,
                                          {dd{migraphx::sym::lit(2)}, dd{sequence_length}}});
    auto one       = mm->add_literal(1.0f);
    auto broadcast = mm->add_instruction(migraphx::make_op("multibroadcast"), one, data);
    auto result    = mm->add_instruction(migraphx::make_op("add"), data, broadcast);
    mm->add_return({result});

    run_pass(p);

    auto select = find_select_module(mm);
    EXPECT(count_literals(mm) == 1);
    EXPECT(std::all_of(select->module_inputs().begin(),
                       select->module_inputs().end(),
                       [](auto mod) { return count_literals(mod) == 0; }));

    auto buffer = migraphx::save_buffer(p);
    auto loaded = migraphx::load_buffer(buffer);
    EXPECT(p.sort() == loaded.sort());
    EXPECT(count_literals(loaded.get_main_module()) == 1);
    auto loaded_select = find_select_module(loaded.get_main_module());
    EXPECT(std::all_of(loaded_select->module_inputs().begin(),
                       loaded_select->module_inputs().end(),
                       [](auto mod) { return count_literals(mod) == 0; }));
}

TEST_CASE(split_prefill_decode_noop)
{
    auto wrong_bounds = make_relu_program(migraphx::sym::var("sequence_length", {2, 4}));
    auto expected     = wrong_bounds;
    run_pass(wrong_bounds);
    EXPECT(wrong_bounds == expected);

    auto unrelated = make_relu_program(migraphx::sym::var("tokens", {1, 4}));
    expected       = unrelated;
    run_pass(unrelated);
    EXPECT(unrelated == expected);
}

TEST_CASE(split_prefill_decode_idempotent)
{
    auto p = make_relu_program(migraphx::sym::var("sequence_length", {1, 4}));
    run_pass(p);
    auto expected = p;
    run_pass(p);
    EXPECT(p == expected);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
