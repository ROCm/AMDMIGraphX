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
#include <migraphx/argument.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/value.hpp>

#include <test.hpp>

#include <algorithm>
#include <iterator>
#include <vector>

TEST_CASE(fixed_pad_int64_dynamic_input)
{
    auto n = migraphx::sym::var("n", {1, 4});
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter(
        "input",
        migraphx::shape{migraphx::shape::int64_type, {migraphx::shape::dynamic_dimension{n}}});
    auto output = mm->add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), input);
    mm->add_return({output});

    auto target = migraphx::make_target("gpu");
    p.compile(target);
    EXPECT(migraphx::none_of(migraphx::iterator_for(*p.get_main_module()), [](auto ins) {
        return ins->name() == "gpu::dynamic_code_object_op";
    }));

    std::vector<int64_t> input_data = {3, 4};
    migraphx::parameter_map params;
    auto parameter_shapes = p.get_parameter_shapes();
    std::transform(
        parameter_shapes.begin(),
        parameter_shapes.end(),
        std::inserter(params, params.end()),
        [&](const auto& entry) {
            if(entry.first == "input")
                return std::make_pair(
                    entry.first,
                    target.copy_to(migraphx::argument{
                        migraphx::shape{migraphx::shape::int64_type, {2}}, input_data.data()}));
            return std::make_pair(entry.first, target.allocate(entry.second));
        });
    auto result = target.copy_from(p.eval(params).back());
    EXPECT(result.to_vector<int64_t>() == std::vector<int64_t>{3, 4, 0, 0});
}

TEST_CASE(fixed_pad_int64_nonstandard_dynamic_input)
{
    auto n = migraphx::sym::var("n", {1, 4});
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter(
        "input",
        migraphx::shape{migraphx::shape::int64_type,
                        {migraphx::shape::dynamic_dimension{n},
                         migraphx::shape::dynamic_dimension{migraphx::sym::lit(2)}},
                        {migraphx::sym::lit(1), n}});
    auto output = mm->add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), input);
    mm->add_return({output});

    auto target = migraphx::make_target("gpu");
    p.compile(target);
    EXPECT(migraphx::none_of(migraphx::iterator_for(*p.get_main_module()), [](auto ins) {
        return ins->name() == "gpu::dynamic_code_object_op";
    }));

    std::vector<int64_t> input_data = {1, 2, 3, 4};
    migraphx::parameter_map params;
    auto parameter_shapes = p.get_parameter_shapes();
    std::transform(parameter_shapes.begin(),
                   parameter_shapes.end(),
                   std::inserter(params, params.end()),
                   [&](const auto& entry) {
                       if(entry.first == "input")
                           return std::make_pair(
                               entry.first,
                               target.copy_to(migraphx::argument{
                                   migraphx::shape{migraphx::shape::int64_type, {2, 2}, {1, 2}},
                                   input_data.data()}));
                       return std::make_pair(entry.first, target.allocate(entry.second));
                   });
    auto result = target.copy_from(p.eval(params).back());
    EXPECT(result.to_vector<int64_t>() == std::vector<int64_t>{1, 3, 2, 4, 0, 0, 0, 0});
}

TEST_CASE(split_sym_dim_topk_gather_has_no_dynamic_code_object)
{
    auto n              = migraphx::sym::var("n", {1, 4});
    auto runtime_k_expr = migraphx::sym::resolve_min(migraphx::sym::lit(2), n);
    auto runtime_k      = migraphx::sym::var("runtime_k", {0, 2});
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto scores = mm->add_parameter(
        "scores",
        migraphx::shape{migraphx::shape::float_type, {migraphx::shape::dynamic_dimension{n}}});
    auto labels = mm->add_parameter(
        "labels",
        migraphx::shape{migraphx::shape::int64_type, {migraphx::shape::dynamic_dimension{n}}});
    auto topk = mm->add_instruction(
        migraphx::make_op("topk", {{"k", 2}, {"axis", 0}, {"largest", true}}), scores);
    auto indices = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), topk);
    auto starts  = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto runtime_k_value = mm->add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions",
                            migraphx::to_value(std::vector<migraphx::sym::expr>{runtime_k_expr})}}),
        scores);
    indices = mm->add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {0}},
                           {"starts", {0}},
                           {"ends", migraphx::value::array{migraphx::to_value(runtime_k)}},
                           {"always_leq", true}}),
        indices,
        starts,
        runtime_k_value);
    auto output = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), labels, indices);
    mm->add_return({output});

    auto target = migraphx::make_target("gpu");
    p.compile(target);
    EXPECT(std::all_of(p.get_modules().begin(), p.get_modules().end(), [](auto* mod) {
        return migraphx::none_of(migraphx::iterator_for(*mod), [](auto ins) {
            return ins->name() == "gpu::dynamic_code_object_op";
        });
    }));

    std::vector<float> score_data   = {0.1f, 0.9f, 0.5f};
    std::vector<int64_t> label_data = {10, 11, 12};
    migraphx::parameter_map params;
    auto parameter_shapes = p.get_parameter_shapes();
    std::transform(
        parameter_shapes.begin(),
        parameter_shapes.end(),
        std::inserter(params, params.end()),
        [&](const auto& entry) {
            if(entry.first == "scores")
                return std::make_pair(
                    entry.first,
                    target.copy_to(migraphx::argument{
                        migraphx::shape{migraphx::shape::float_type, {3}}, score_data.data()}));
            if(entry.first == "labels")
                return std::make_pair(
                    entry.first,
                    target.copy_to(migraphx::argument{
                        migraphx::shape{migraphx::shape::int64_type, {3}}, label_data.data()}));
            return std::make_pair(entry.first, target.allocate(entry.second));
        });
    auto result = target.copy_from(p.eval(params).back());
    EXPECT(result.to_vector<int64_t>() == std::vector<int64_t>{11, 12});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
