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
#include <migraphx/dim_like.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/split_sym_dim.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/value.hpp>
#include <test.hpp>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace {

using dd = migraphx::shape::dynamic_dimension;
using se = migraphx::sym::expr;
using migraphx::sym::lit;
using migraphx::sym::var;

struct clone_spec
{
    std::size_t min;
    std::size_t max;
};

void run_pass(migraphx::program& p, std::size_t max_clones = migraphx::split_sym_dim{}.max_clones)
{
    migraphx::run_passes(p,
                         {migraphx::split_sym_dim{max_clones}, migraphx::dead_code_elimination{}});
}

migraphx::instruction_ref add_expected_clone_extent(migraphx::module& m,
                                                    const se& expression,
                                                    const se& root,
                                                    const clone_spec& clone,
                                                    migraphx::instruction_ref input)
{
    if(clone.min != clone.max)
        return m.add_instruction(
            migraphx::make_op("eval_expr_from_shape",
                              {{"expressions", migraphx::to_value(std::vector<se>{expression})}}),
            input);
    auto value = expression.eval_uint({{migraphx::sym::as_symbol(root), clone.min}});
    return m.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {int64_t(value)}});
}

migraphx::instruction_ref add_mask(migraphx::module& m,
                                   migraphx::instruction_ref input,
                                   migraphx::instruction_ref indices,
                                   migraphx::instruction_ref extent,
                                   migraphx::instruction_ref fill,
                                   int64_t axis)
{
    const auto& lens = input->get_shape().lens();
    auto index       = m.add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", lens}}), indices);
    auto limit =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), extent);
    auto less      = m.add_instruction(migraphx::make_op("less"), index, limit);
    auto condition = m.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), less);
    auto fill_value =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), fill);
    return m.add_instruction(migraphx::make_op("where"), condition, input, fill_value);
}

migraphx::instruction_ref add_expected_scatternd(migraphx::module& m,
                                                 migraphx::instruction_ref data,
                                                 migraphx::instruction_ref indices,
                                                 migraphx::instruction_ref updates,
                                                 const se& extent_expression,
                                                 const se& root,
                                                 const clone_spec& clone)
{
    const auto& index_shape = indices->get_shape();
    auto index_lens         = index_shape.dynamic() ? index_shape.max_lens() : index_shape.lens();
    auto index_depth        = index_lens.back();
    index_lens.pop_back();

    std::vector<int64_t> positions_data(index_lens.front());
    std::iota(positions_data.begin(), positions_data.end(), 0);
    auto positions = m.add_literal(migraphx::literal{
        migraphx::shape{migraphx::shape::int64_type, {index_lens.front()}}, positions_data});
    positions      = m.add_instruction(
        migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", index_lens}}), positions);

    migraphx::instruction_ref extent;
    if(clone.min == clone.max)
    {
        auto value = extent_expression.eval_uint({{migraphx::sym::as_symbol(root), clone.min}});
        extent     = m.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {int64_t(value)}});
    }
    else
    {
        extent = m.add_instruction(
            migraphx::make_op(
                "eval_expr_from_shape",
                {{"expressions", migraphx::to_value(std::vector<se>{extent_expression})}}),
            m.get_parameters());
    }
    extent =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", index_lens}}), extent);
    auto valid = m.add_instruction(migraphx::make_op("less"), positions, extent);

    auto effective_index_depth = index_depth;
    if(index_depth == 0)
    {
        if(data->get_shape().scalar())
            data = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1}}}), data);
        else
            data = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), data);
        effective_index_depth = 1;
    }

    auto data_lens = data->get_shape().lens();
    std::vector<int64_t> pads(data_lens.size() * 2, 0);
    std::fill(pads.begin() + data_lens.size(),
              pads.begin() + data_lens.size() + effective_index_depth,
              int64_t{1});
    auto padded_data = m.add_instruction(migraphx::make_op("pad", {{"pads", pads}}), data);

    auto rewritten_index_lens = index_lens;
    rewritten_index_lens.push_back(effective_index_depth);
    auto condition =
        m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {index_lens.size()}}}), valid);
    condition = m.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", rewritten_index_lens}}), condition);

    std::vector<int64_t> sink_values(effective_index_depth);
    std::transform(data_lens.begin(),
                   data_lens.begin() + effective_index_depth,
                   sink_values.begin(),
                   [](auto x) { return static_cast<int64_t>(x); });
    auto sink = m.add_literal(migraphx::literal{
        migraphx::shape{migraphx::shape::int64_type, {effective_index_depth}}, sink_values});
    sink      = m.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", rewritten_index_lens}}), sink);

    if(index_depth == 0)
    {
        std::vector<int64_t> zeros(effective_index_depth, 0);
        indices = m.add_literal(migraphx::literal{
            migraphx::shape{migraphx::shape::int64_type, {effective_index_depth}}, zeros});
        indices = m.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", rewritten_index_lens}}), indices);
    }
    indices = m.add_instruction(migraphx::make_op("where"), condition, indices, sink);

    auto scattered =
        m.add_instruction(migraphx::make_op("scatternd_none"), padded_data, indices, updates);
    std::vector<int64_t> axes(effective_index_depth);
    std::iota(axes.begin(), axes.end(), int64_t{0});
    std::vector<int64_t> starts(effective_index_depth, 0);
    std::vector<int64_t> ends(effective_index_depth);
    std::transform(data_lens.begin(),
                   data_lens.begin() + effective_index_depth,
                   ends.begin(),
                   [](auto x) { return static_cast<int64_t>(x); });
    auto result = m.add_instruction(
        migraphx::make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}),
        scattered);
    if(index_depth == 0)
        result = m.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), result);
    return m.add_instruction(migraphx::make_op("contiguous"), result);
}

template <class Spec, class F>
std::vector<migraphx::module_ref>
add_clones(migraphx::program& p, std::size_t block, const std::vector<Spec>& specs, F f)
{
    std::vector<migraphx::module_ref> modules;
    for(std::size_t i = 0; i < specs.size(); ++i)
    {
        auto* sm = p.create_module("main:split_sym_dim_" + std::to_string(block) + "_" +
                                   std::to_string(i));
        f(*sm, specs.at(i));
        // Fixed parameters stay symbolic if shape evaluations need them or output layouts change.
        if(migraphx::none_of(*sm,
                             [](const auto& ins) { return ins.name() == "eval_expr_from_shape"; }))
        {
            auto static_clone = *sm;
            for(auto parameter : static_clone.get_parameters())
            {
                const auto& s = parameter->get_shape();
                if(s.symbolic() and s.is_fixed() and
                   migraphx::all_of(s.dyn_strides(), [](const auto& stride) {
                       return migraphx::sym::fixed_value(stride).has_value();
                   }))
                    migraphx::instruction::replace(
                        parameter, parameter->get_operator(), s.to_static(), {});
            }
            if(static_clone.get_output_shapes() == sm->get_output_shapes())
                *sm = std::move(static_clone);
        }
        modules.push_back(sm);
    }
    return modules;
}

migraphx::instruction_ref
add_residual_diamonds(migraphx::module& m, migraphx::instruction_ref input, std::size_t count)
{
    auto diamonds = migraphx::range(count);
    return std::accumulate(diamonds.begin(), diamonds.end(), input, [&](auto current, auto) {
        auto branch = m.add_instruction(migraphx::make_op("relu"), current);
        return m.add_instruction(migraphx::make_op("add"), current, branch);
    });
}

} // namespace

TEST_CASE(split_sym_dim_skips_unbounded_root)
{
    auto n = var("n", {0, std::numeric_limits<int64_t>::max()});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x =
        m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto y = m.add_instruction(migraphx::make_op("relu"), x);
    m.add_return({y});
    auto expected = p;

    run_pass(p);

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_supports_zero_length_root)
{
    auto n = var("n", {0, 4});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto input =
        m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{n}}});
    auto output = m.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), input);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{0, 0}, {1, 4}};
    auto modules        = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_n     = var("n", {clone.min, clone.max});
        auto clone_input = sm.add_parameter(
            "data", migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{clone_n}}});
        auto padded =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_input);
        auto masked = padded;
        if(clone.max != 0)
        {
            auto negative_infinity =
                sm.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}},
                                                 {-std::numeric_limits<float>::infinity()}});
            std::vector<int64_t> indices_data(clone.max);
            std::iota(indices_data.begin(), indices_data.end(), 0);
            auto indices = sm.add_literal(migraphx::literal{
                migraphx::shape{migraphx::shape::int64_type, {clone.max}}, indices_data});
            auto extent  = add_expected_clone_extent(sm, n, n, clone, clone_input);
            masked       = add_mask(sm, padded, indices, extent, negative_infinity, 1);
        }
        auto clone_output = sm.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), masked);
        sm.add_return({clone_output});
    });
    auto& expected_main = *expected.get_main_module();
    auto expected_input = expected_main.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{n}}});
    auto target_n = var("split_sym_dim_n_target", {0, 4}, {0, 4});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{target_n}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_input},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{1}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_covers_full_interval)
{
    auto n = var("n", {1, 8}, {2, 4});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x =
        m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto y = m.add_instruction(migraphx::make_op("relu"), x);
    m.add_return({y});
    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}, {5, 8}};
    auto modules        = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto input =
            sm.add_parameter("data",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{var("n", {clone.min, clone.max})}, dd{lit(4)}}});
        auto pad    = sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), input);
        auto output = sm.add_instruction(migraphx::make_op("relu"), pad);
        sm.add_return({output});
    });
    auto& expected_main = *expected.get_main_module();
    auto input          = expected_main.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto target_n = var("split_sym_dim_n_target", {1, 8}, {1, 2, 4, 8});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type, {dd{target_n}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{input},
        modules);
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{input});
    auto output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{input});
    output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        output,
        output_start,
        output_end);
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_supports_one_to_one_axis_transforms)
{
    auto n = var("n", {1, 8}, {2, 4});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x  = m.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{n}, dd{lit(4)}}});
    x = m.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), x);
    x = m.add_instruction(migraphx::make_op("reshape", {{"dims", {0, 0}}}), x);
    x = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), x);
    m.add_return({x});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}, {5, 8}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto input = sm.add_parameter(
            "data",
            migraphx::shape{migraphx::shape::float_type,
                            {dd{lit(1)}, dd{var("n", {clone.min, clone.max})}, dd{lit(4)}}});
        auto output = sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), input);
        output      = sm.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), output);
        output      = sm.add_instruction(migraphx::make_op("reshape", {{"dims", {0, 0}}}), output);
        output =
            sm.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), output);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto input          = expected_main.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{n}, dd{lit(4)}}});
    auto target_n = var("split_sym_dim_n_target", {1, 8}, {1, 2, 4, 8});
    migraphx::shape output_shape{
        migraphx::shape::float_type, {dd{lit(4)}, dd{target_n}}, {lit(1), lit(4)}};
    auto select = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{output_shape}})}}),
        std::vector<migraphx::instruction_ref>{input},
        modules);
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{input});
    auto output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{input});
    output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{1}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        output,
        output_start,
        output_end);
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_stepped_unsqueeze_is_noop)
{
    auto n = var("n", {4, 8}, {8});
    migraphx::program p;
    auto& m    = *p.get_main_module();
    auto input = m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto output =
        m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}, {"steps", {2}}}), input);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    auto& expected_main = *expected.get_main_module();
    auto expected_input =
        expected_main.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto expected_output = expected_main.add_instruction(
        migraphx::make_op("unsqueeze", {{"axes", {0}}, {"steps", {2}}}), expected_input);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_freezes_one_input_reshape_symbolic_dims)
{
    auto n = var("n", {1, 8}, {2, 4});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto input =
        m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    std::vector<migraphx::dim_like> reshape_dims = {dd{n}, 4, 1};
    auto output                                  = m.add_instruction(
        migraphx::make_op("reshape", {{"dims", migraphx::to_value(reshape_dims)}}), input);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}, {5, 8}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_input =
            sm.add_parameter("data",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{var("n", {clone.min, clone.max})}, dd{lit(4)}}});
        auto padded_input =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_input);
        auto clone_output = sm.add_instruction(
            migraphx::make_op("reshape", {{"dims", {clone.max, 4, 1}}}), padded_input);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_input = expected_main.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto target_n = var("split_sym_dim_n_target", {1, 8}, {1, 2, 4, 8});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{migraphx::shape{
                  migraphx::shape::float_type, {dd{target_n}, dd{lit(4)}, dd{lit(1)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_input},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    auto output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        output_start,
        output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_freezes_non_unit_one_input_reshape_symbolic_dims)
{
    auto n = var("n", {1, 8}, {2, 4});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto input =
        m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    std::vector<migraphx::dim_like> reshape_dims = {dd{n}, 2, 2};
    auto output                                  = m.add_instruction(
        migraphx::make_op("reshape", {{"dims", migraphx::to_value(reshape_dims)}}), input);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}, {5, 8}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_input =
            sm.add_parameter("data",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{var("n", {clone.min, clone.max})}, dd{lit(4)}}});
        auto padded_input =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_input);
        auto clone_output = sm.add_instruction(
            migraphx::make_op("reshape", {{"dims", {clone.max, 2, 2}}}), padded_input);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_input = expected_main.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto target_n = var("split_sym_dim_n_target", {1, 8}, {1, 2, 4, 8});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{migraphx::shape{
                  migraphx::shape::float_type, {dd{target_n}, dd{lit(2)}, dd{lit(2)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_input},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    auto output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        output_start,
        output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_splits_at_nonparallel_symbolic_reshape)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto data =
        m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto target = m.add_parameter(
        "target", migraphx::shape{migraphx::shape::float_type, {dd{lit(4)}, dd{n}}});
    auto reshape = m.add_instruction(migraphx::make_op("reshape"), data, target);
    auto output  = m.add_instruction(migraphx::make_op("relu"), reshape);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_n = var("n", {clone.min, clone.max});
        auto input   = sm.add_parameter(
            "#split_sym_dim_input_0_0",
            migraphx::shape{migraphx::shape::float_type, {dd{lit(4)}, dd{clone_n}}});
        sm.add_parameter("data",
                         migraphx::shape{migraphx::shape::float_type, {dd{clone_n}, dd{lit(4)}}});
        auto padded = sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), input);
        auto result = sm.add_instruction(migraphx::make_op("relu"), padded);
        sm.add_return({result});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto expected_target = expected_main.add_parameter(
        "target", migraphx::shape{migraphx::shape::float_type, {dd{lit(4)}, dd{n}}});
    auto expected_reshape =
        expected_main.add_instruction(migraphx::make_op("reshape"), expected_data, expected_target);
    auto target_n = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type, {dd{lit(4)}, dd{target_n}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_reshape, expected_data},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_target, expected_data});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_target, expected_data});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{1}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_materializes_symbolic_multibroadcast)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto data =
        m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(3)}}});
    auto bias  = m.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {3}});
    auto bcast = m.add_instruction(
        migraphx::make_op(
            "multibroadcast",
            {{"out_dyn_dims", migraphx::to_value(std::vector<dd>{dd{n}, dd{lit(3)}})}}),
        std::vector<migraphx::instruction_ref>{bias, data});
    auto output = m.add_instruction(migraphx::make_op("add"), data, bcast);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_bias =
            sm.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {3}});
        auto clone_data =
            sm.add_parameter("data",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{var("n", {clone.min, clone.max})}, dd{lit(3)}}});
        auto clone_bcast = sm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {clone.max, 3}}}), clone_bias);
        auto padded_data =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_data);
        auto clone_output = sm.add_instruction(migraphx::make_op("add"), padded_data, clone_bcast);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(3)}}});
    auto expected_bias =
        expected_main.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {3}});
    auto target_n = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type, {dd{target_n}, dd{lit(3)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_bias, expected_data},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_bias, expected_data});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_bias, expected_data});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_materializes_symbolic_multibroadcast_with_multiple_shape_inputs)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto data =
        m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(3)}}});
    auto shape_input =
        m.add_parameter("shape_input", migraphx::shape{migraphx::shape::float_type, {1, 3}});
    auto bias  = m.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {3}});
    auto bcast = m.add_instruction(
        migraphx::make_op(
            "multibroadcast",
            {{"out_dyn_dims", migraphx::to_value(std::vector<dd>{dd{n}, dd{lit(3)}})}}),
        std::vector<migraphx::instruction_ref>{bias, data, shape_input});
    auto output = m.add_instruction(migraphx::make_op("add"), data, bcast);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_bias =
            sm.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {3}});
        auto clone_data =
            sm.add_parameter("data",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{var("n", {clone.min, clone.max})}, dd{lit(3)}}});
        auto clone_bcast = sm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {clone.max, 3}}}), clone_bias);
        auto padded_data =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_data);
        auto clone_output = sm.add_instruction(migraphx::make_op("add"), padded_data, clone_bcast);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(3)}}});
    auto expected_shape_input = expected_main.add_parameter(
        "shape_input", migraphx::shape{migraphx::shape::float_type, {1, 3}});
    auto expected_bias =
        expected_main.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {3}});
    auto target_n = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type, {dd{target_n}, dd{lit(3)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_bias, expected_data},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_bias, expected_shape_input, expected_data});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_bias, expected_shape_input, expected_data});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_absorbs_fixed_symbolic_multibroadcast)
{
    auto fixed_batch = var("fixed_batch", {1, 1});
    auto sequence    = var("sequence", {1, 4}, {2});
    migraphx::program p;
    auto& m   = *p.get_main_module();
    auto data = m.add_parameter(
        "data",
        migraphx::shape{migraphx::shape::float_type, {dd{fixed_batch}, dd{sequence}, dd{lit(4)}}});
    auto weights = m.add_parameter("weights", migraphx::shape{migraphx::shape::float_type, {4, 4}});
    auto target  = m.add_parameter(
        "target",
        migraphx::shape{migraphx::shape::float_type, {dd{fixed_batch}, dd{lit(4)}, dd{lit(4)}}});
    auto broadcast = m.add_instruction(
        migraphx::make_op(
            "multibroadcast",
            {{"out_dyn_dims",
              migraphx::to_value(std::vector<dd>{dd{fixed_batch}, dd{lit(4)}, dd{lit(4)}})}}),
        std::vector<migraphx::instruction_ref>{weights, target});
    auto output = m.add_instruction(migraphx::make_op("dot"), data, broadcast);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_batch    = var("fixed_batch", {1, 1});
        auto clone_sequence = var("sequence", {clone.min, clone.max});
        auto clone_data =
            sm.add_parameter("data",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{clone_batch}, dd{clone_sequence}, dd{lit(4)}}});
        auto clone_weights =
            sm.add_parameter("weights", migraphx::shape{migraphx::shape::float_type, {4, 4}});
        auto clone_broadcast = sm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 4, 4}}}), clone_weights);
        auto padded_data =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_data);
        auto padded_broadcast =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_broadcast);
        auto clone_output =
            sm.add_instruction(migraphx::make_op("dot"), padded_data, padded_broadcast);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter(
        "data",
        migraphx::shape{migraphx::shape::float_type, {dd{fixed_batch}, dd{sequence}, dd{lit(4)}}});
    auto expected_weights = expected_main.add_parameter(
        "weights", migraphx::shape{migraphx::shape::float_type, {4, 4}});
    auto expected_target = expected_main.add_parameter(
        "target",
        migraphx::shape{migraphx::shape::float_type, {dd{fixed_batch}, dd{lit(4)}, dd{lit(4)}}});
    auto target_sequence = var("split_sym_dim_sequence_target", {1, 4}, {1, 2, 4});
    auto select          = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{migraphx::shape{
                  migraphx::shape::float_type, {dd{lit(1)}, dd{target_sequence}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_data, expected_weights},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_target, expected_weights, expected_data});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{sequence})}}),
        std::vector<migraphx::instruction_ref>{expected_target, expected_weights, expected_data});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{1}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{sequence})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_materializes_symbolic_broadcast)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m   = *p.get_main_module();
    auto data = m.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(3)}, dd{lit(4)}}});
    auto bias  = m.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {3}});
    auto bcast = m.add_instruction(
        migraphx::make_op(
            "broadcast",
            {{"axis", 1},
             {"out_dyn_dims", migraphx::to_value(std::vector<dd>{dd{n}, dd{lit(3)}, dd{lit(4)}})}}),
        bias,
        data);
    auto output = m.add_instruction(migraphx::make_op("add"), data, bcast);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_bias =
            sm.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {3}});
        auto clone_data = sm.add_parameter(
            "data",
            migraphx::shape{migraphx::shape::float_type,
                            {dd{var("n", {clone.min, clone.max})}, dd{lit(3)}, dd{lit(4)}}});
        auto clone_bcast = sm.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {clone.max, 3, 4}}}),
            clone_bias);
        auto padded_data =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_data);
        auto clone_output = sm.add_instruction(migraphx::make_op("add"), padded_data, clone_bcast);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(3)}, dd{lit(4)}}});
    auto expected_bias =
        expected_main.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {3}});
    auto target_n = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{migraphx::shape{
                  migraphx::shape::float_type, {dd{target_n}, dd{lit(3)}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_bias, expected_data},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_bias, expected_data});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_bias, expected_data});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_materializes_symbolic_broadcast_with_dims)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m   = *p.get_main_module();
    auto data = m.add_parameter(
        "data",
        migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{lit(1)}, dd{n}, dd{n}}});
    auto dims =
        m.add_parameter("dims", migraphx::shape{migraphx::shape::int64_type, {std::size_t{4}}});
    auto output = m.add_instruction(
        migraphx::make_op(
            "broadcast_with_dims",
            {{"out_dyn_dims",
              migraphx::to_value(std::vector<dd>{dd{lit(1)}, dd{lit(1)}, dd{n}, dd{n}})}}),
        data,
        dims);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_data = sm.add_parameter("data",
                                           migraphx::shape{migraphx::shape::float_type,
                                                           {dd{lit(1)},
                                                            dd{lit(1)},
                                                            dd{var("n", {clone.min, clone.max})},
                                                            dd{var("n", {clone.min, clone.max})}}});
        auto padded_data =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_data);
        auto clone_output = sm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, clone.max, clone.max}}}),
            padded_data);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter(
        "data",
        migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{lit(1)}, dd{n}, dd{n}}});
    auto expected_dims = expected_main.add_parameter(
        "dims", migraphx::shape{migraphx::shape::int64_type, {std::size_t{4}}});
    auto target_n = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type,
                                  {dd{lit(1)}, dd{lit(1)}, dd{target_n}, dd{target_n}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_data},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0), lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_dims, expected_data});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n, n})}}),
        std::vector<migraphx::instruction_ref>{expected_dims, expected_data});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{2, 3}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0), lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n, n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_coalesces_into_symbolic_broadcast_with_dims)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto data =
        m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{n}}});
    auto dims =
        m.add_parameter("dims", migraphx::shape{migraphx::shape::int64_type, {std::size_t{3}}});
    auto relu   = m.add_instruction(migraphx::make_op("relu"), data);
    auto output = m.add_instruction(
        migraphx::make_op(
            "broadcast_with_dims",
            {{"out_dyn_dims", migraphx::to_value(std::vector<dd>{dd{lit(1)}, dd{n}, dd{n}})}}),
        relu,
        dims);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_data =
            sm.add_parameter("data",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{lit(1)}, dd{var("n", {clone.min, clone.max})}}});
        auto padded_data =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_data);
        auto clone_relu   = sm.add_instruction(migraphx::make_op("relu"), padded_data);
        auto clone_output = sm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, clone.max, clone.max}}}),
            clone_relu);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{n}}});
    auto expected_dims = expected_main.add_parameter(
        "dims", migraphx::shape{migraphx::shape::int64_type, {std::size_t{3}}});
    auto target_n = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    migraphx::shape output_shape{migraphx::shape::float_type,
                                 {dd{lit(1)}, dd{target_n}, dd{target_n}},
                                 {lit(0), lit(0), lit(1)}};
    auto select = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{output_shape}})}}),
        std::vector<migraphx::instruction_ref>{expected_data},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0), lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_dims, expected_data});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n, n})}}),
        std::vector<migraphx::instruction_ref>{expected_dims, expected_data});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{1, 2}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0), lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n, n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_materializes_symbolic_allocate)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    m.add_parameter("source", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto dims =
        m.add_parameter("dims", migraphx::shape{migraphx::shape::int64_type, {std::size_t{2}}});
    auto output_shape = migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}};
    auto output       = m.add_instruction(
        migraphx::make_op("allocate", {{"shape", migraphx::to_value(output_shape)}}), dims);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        sm.add_parameter(
            "source",
            migraphx::shape{migraphx::shape::float_type, {dd{var("n", {clone.min, clone.max})}}});
        migraphx::shape clone_output_shape{migraphx::shape::float_type, {clone.max, 4}};
        auto clone_output = sm.add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(clone_output_shape)}}));
        sm.add_return({clone_output});
    });

    auto& expected_main  = *expected.get_main_module();
    auto expected_source = expected_main.add_parameter(
        "source", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto expected_dims = expected_main.add_parameter(
        "dims", migraphx::shape{migraphx::shape::int64_type, {std::size_t{2}}});
    auto target_n = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type, {dd{target_n}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_source},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_dims, expected_source});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_dims, expected_source});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_materializes_two_input_symbolic_reshape)
{
    auto batch = var("batch", {1, 1});
    auto n     = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m   = *p.get_main_module();
    auto data = m.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{batch}, dd{n}, dd{lit(16)}}});
    auto target = m.add_parameter(
        "target",
        migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{n}, dd{lit(4)}, dd{lit(4)}}});
    auto reshape = m.add_instruction(migraphx::make_op("reshape"), data, target);
    m.add_return({reshape});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_data = sm.add_parameter(
            "data",
            migraphx::shape{
                migraphx::shape::float_type,
                {dd{var("batch", {1, 1})}, dd{var("n", {clone.min, clone.max})}, dd{lit(16)}}});
        auto padded_data =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_data);
        auto clone_output = sm.add_instruction(
            migraphx::make_op("reshape", {{"dims", {1, clone.max, 4, 4}}}), padded_data);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{batch}, dd{n}, dd{lit(16)}}});
    auto expected_target = expected_main.add_parameter(
        "target",
        migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{n}, dd{lit(4)}, dd{lit(4)}}});
    auto target_n = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type,
                                  {dd{lit(1)}, dd{target_n}, dd{lit(4)}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_data},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_target, expected_data});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_target, expected_data});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{1}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_specializes_two_input_symbolic_reshape_with_unit_axis)
{
    auto batch = var("batch", {1, 1});
    auto n     = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m   = *p.get_main_module();
    auto data = m.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{batch}, dd{n}, dd{lit(4)}}});
    auto target = m.add_parameter(
        "target",
        migraphx::shape{migraphx::shape::float_type, {dd{batch}, dd{lit(1)}, dd{n}, dd{lit(4)}}});
    auto reshape = m.add_instruction(migraphx::make_op("reshape"), data, target);
    m.add_return({reshape});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_data = sm.add_parameter(
            "data",
            migraphx::shape{
                migraphx::shape::float_type,
                {dd{var("batch", {1, 1})}, dd{var("n", {clone.min, clone.max})}, dd{lit(4)}}});
        auto padded_data =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_data);
        auto clone_output = sm.add_instruction(
            migraphx::make_op("reshape", {{"dims", {1, 1, clone.max, 4}}}), padded_data);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{batch}, dd{n}, dd{lit(4)}}});
    auto expected_target = expected_main.add_parameter(
        "target",
        migraphx::shape{migraphx::shape::float_type, {dd{batch}, dd{lit(1)}, dd{n}, dd{lit(4)}}});
    auto target_n = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type,
                                  {dd{lit(1)}, dd{lit(1)}, dd{target_n}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_data},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_target, expected_data});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_target, expected_data});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{2}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_absorbs_fixed_symbolic_reshape)
{
    auto fixed_batch = var("fixed_batch", {1, 1});
    auto sequence    = var("sequence", {1, 4}, {2});
    migraphx::program p;
    auto& m   = *p.get_main_module();
    auto data = m.add_parameter(
        "data",
        migraphx::shape{migraphx::shape::float_type, {dd{fixed_batch}, dd{sequence}, dd{lit(4)}}});
    auto weights = m.add_parameter(
        "weights", migraphx::shape{migraphx::shape::float_type, {dd{fixed_batch}, dd{lit(16)}}});
    auto target = m.add_parameter(
        "target",
        migraphx::shape{migraphx::shape::float_type, {dd{fixed_batch}, dd{lit(4)}, dd{lit(4)}}});
    auto reshape = m.add_instruction(migraphx::make_op("reshape"), weights, target);
    auto output  = m.add_instruction(migraphx::make_op("dot"), data, reshape);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_batch    = var("fixed_batch", {1, 1});
        auto clone_sequence = var("sequence", {clone.min, clone.max});
        auto clone_data =
            sm.add_parameter("data",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{clone_batch}, dd{clone_sequence}, dd{lit(4)}}});
        auto clone_weights =
            sm.add_parameter("weights", migraphx::shape{migraphx::shape::float_type, {1, 16}});
        auto clone_reshape =
            sm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 4}}}), clone_weights);
        auto padded_data =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_data);
        auto padded_reshape =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_reshape);
        auto clone_output =
            sm.add_instruction(migraphx::make_op("dot"), padded_data, padded_reshape);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter(
        "data",
        migraphx::shape{migraphx::shape::float_type, {dd{fixed_batch}, dd{sequence}, dd{lit(4)}}});
    auto expected_weights = expected_main.add_parameter(
        "weights", migraphx::shape{migraphx::shape::float_type, {dd{fixed_batch}, dd{lit(16)}}});
    auto expected_target = expected_main.add_parameter(
        "target",
        migraphx::shape{migraphx::shape::float_type, {dd{fixed_batch}, dd{lit(4)}, dd{lit(4)}}});
    auto target_sequence = var("split_sym_dim_sequence_target", {1, 4}, {1, 2, 4});
    auto select          = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{migraphx::shape{
                  migraphx::shape::float_type, {dd{lit(1)}, dd{target_sequence}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_data, expected_weights},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_target, expected_weights, expected_data});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{sequence})}}),
        std::vector<migraphx::instruction_ref>{expected_target, expected_weights, expected_data});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{1}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{sequence})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_drops_absorbed_reshape_shape_dependencies)
{
    auto fixed_batch = var("fixed_batch", {1, 1});
    auto sequence    = var("sequence", {1, 4}, {2});
    migraphx::program p;
    auto& m   = *p.get_main_module();
    auto data = m.add_parameter(
        "data",
        migraphx::shape{migraphx::shape::float_type, {dd{fixed_batch}, dd{sequence}, dd{lit(4)}}});
    auto weights = m.add_parameter(
        "weights", migraphx::shape{migraphx::shape::float_type, {dd{fixed_batch}, dd{lit(16)}}});
    auto target_dims = m.add_instruction(
        migraphx::make_op(
            "eval_expr_from_shape",
            {{"expressions", migraphx::to_value(std::vector<se>{fixed_batch, lit(4), lit(4)})}}),
        weights);
    auto target = m.add_instruction(
        migraphx::make_op(
            "allocate",
            {{"shape",
              migraphx::to_value(migraphx::shape{migraphx::shape::float_type,
                                                 {dd{fixed_batch}, dd{lit(4)}, dd{lit(4)}}})}}),
        target_dims);
    auto reshape = m.add_instruction(migraphx::make_op("reshape"), weights, target);
    auto output  = m.add_instruction(migraphx::make_op("dot"), data, reshape);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_batch    = var("fixed_batch", {1, 1});
        auto clone_sequence = var("sequence", {clone.min, clone.max});
        auto clone_data =
            sm.add_parameter("data",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{clone_batch}, dd{clone_sequence}, dd{lit(4)}}});
        auto clone_weights =
            sm.add_parameter("weights", migraphx::shape{migraphx::shape::float_type, {1, 16}});
        auto clone_reshape =
            sm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 4}}}), clone_weights);
        auto padded_data =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_data);
        auto padded_reshape =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_reshape);
        auto clone_output =
            sm.add_instruction(migraphx::make_op("dot"), padded_data, padded_reshape);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter(
        "data",
        migraphx::shape{migraphx::shape::float_type, {dd{fixed_batch}, dd{sequence}, dd{lit(4)}}});
    auto expected_weights = expected_main.add_parameter(
        "weights", migraphx::shape{migraphx::shape::float_type, {dd{fixed_batch}, dd{lit(16)}}});
    auto target_sequence = var("split_sym_dim_sequence_target", {1, 4}, {1, 2, 4});
    auto select          = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{migraphx::shape{
                  migraphx::shape::float_type, {dd{lit(1)}, dd{target_sequence}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_data, expected_weights},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_weights, expected_data});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{sequence})}}),
        std::vector<migraphx::instruction_ref>{expected_weights, expected_data});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{1}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{sequence})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_keeps_padded_reshape_dependency_at_boundary)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto data =
        m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{lit(4)}, dd{n}}});
    auto target = m.add_parameter(
        "target", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto before  = m.add_instruction(migraphx::make_op("relu"), data);
    auto reshape = m.add_instruction(migraphx::make_op("reshape"), before, target);
    auto output  = m.add_instruction(migraphx::make_op("relu"), reshape);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto before_modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_n    = var("n", {clone.min, clone.max});
        auto clone_data = sm.add_parameter(
            "data", migraphx::shape{migraphx::shape::float_type, {dd{lit(4)}, dd{clone_n}}});
        auto padded_data =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_data);
        auto clone_output = sm.add_instruction(migraphx::make_op("relu"), padded_data);
        sm.add_return({clone_output});
    });
    auto after_modules  = add_clones(expected, 1, clones, [&](auto& sm, const auto& clone) {
        auto clone_n = var("n", {clone.min, clone.max});
        sm.add_parameter("data",
                         migraphx::shape{migraphx::shape::float_type, {dd{lit(4)}, dd{clone_n}}});
        auto clone_input = sm.add_parameter(
            "#split_sym_dim_input_1_0",
            migraphx::shape{migraphx::shape::float_type, {dd{clone_n}, dd{lit(4)}}});
        auto padded_input =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_input);
        auto clone_output = sm.add_instruction(migraphx::make_op("relu"), padded_input);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{lit(4)}, dd{n}}});
    auto expected_target = expected_main.add_parameter(
        "target", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto target_n      = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select_before = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type, {dd{lit(4)}, dd{target_n}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_data},
        before_modules);
    auto before_output = expected_main.add_instruction(
        migraphx::make_op("get_tuple_elem", {{"index", 0}}), select_before);
    auto before_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_target, expected_data});
    auto before_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_target, expected_data});
    before_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{1}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        before_output,
        before_output_start,
        before_output_end);
    auto expected_reshape =
        expected_main.add_instruction(migraphx::make_op("reshape"), before_output, expected_target);
    auto select_after = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type, {dd{target_n}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_reshape, expected_data},
        after_modules);
    auto expected_output = expected_main.add_instruction(
        migraphx::make_op("get_tuple_elem", {{"index", 0}}), select_after);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_target, expected_data});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_target, expected_data});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_retries_affected_connected_edge)
{
    auto n = var("n", {1, 2}, {1, 2});
    auto m = var("m", {1, 2}, {1, 2});
    auto q = var("q", {1, 2}, {1, 2});
    migraphx::program p;
    auto& main = *p.get_main_module();
    auto input = main.add_parameter(
        "input", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto other = main.add_parameter(
        "other", migraphx::shape{migraphx::shape::float_type, {dd{m}, dd{lit(4)}}});
    auto q_source = main.add_parameter(
        "q_source", migraphx::shape{migraphx::shape::float_type, {dd{lit(4)}, dd{q}}});
    auto before   = main.add_instruction(migraphx::make_op("relu"), input);
    other         = main.add_instruction(migraphx::make_op("relu"), other);
    auto reduced  = main.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), input);
    auto expanded = main.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 1}}}), reduced);
    expanded      = main.add_instruction(
        migraphx::make_op(
            "multibroadcast",
            {{"out_dyn_dims", migraphx::to_value(std::vector<dd>{dd{lit(4)}, dd{q}})}}),
        std::vector<migraphx::instruction_ref>{expanded, q_source});
    auto packed = main.add_instruction(migraphx::make_op("contiguous"), expanded);
    auto output = main.add_instruction(migraphx::make_op("dot"), before, packed);
    main.add_return({output, other});

    run_pass(p, 4);

    migraphx::program expected;
    struct connected_clone_spec
    {
        std::size_t n;
        std::size_t q;
    };
    std::vector<connected_clone_spec> connected_clones = {{1, 1}, {2, 1}, {1, 2}, {2, 2}};
    auto connected_modules =
        add_clones(expected, 0, connected_clones, [&](auto& sm, const auto& clone) {
            sm.add_parameter("q_source",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{lit(4)}, dd{var("q", {clone.q, clone.q})}}});
            auto clone_input =
                sm.add_parameter("input",
                                 migraphx::shape{migraphx::shape::float_type,
                                                 {dd{var("n", {clone.n, clone.n})}, dd{lit(4)}}});
            auto padded_input =
                sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_input);
            auto clone_before = sm.add_instruction(migraphx::make_op("relu"), padded_input);
            auto reduced_input =
                sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_input);
            auto clone_reduced =
                sm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), reduced_input);
            auto clone_expanded =
                sm.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 1}}}), clone_reduced);
            clone_expanded = sm.add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", {4, clone.q}}}), clone_expanded);
            auto clone_packed = sm.add_instruction(migraphx::make_op("contiguous"), clone_expanded);
            auto clone_output =
                sm.add_instruction(migraphx::make_op("dot"), clone_before, clone_packed);
            sm.add_return({clone_output});
        });

    std::vector<clone_spec> other_clones = {{1, 1}, {2, 2}};
    auto other_modules = add_clones(expected, 1, other_clones, [&](auto& sm, const auto& clone) {
        auto clone_other =
            sm.add_parameter("other",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{var("m", {clone.min, clone.max})}, dd{lit(4)}}});
        auto padded_other =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_other);
        auto clone_output = sm.add_instruction(migraphx::make_op("relu"), padded_other);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_input = expected_main.add_parameter(
        "input", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto expected_other = expected_main.add_parameter(
        "other", migraphx::shape{migraphx::shape::float_type, {dd{m}, dd{lit(4)}}});
    auto expected_q_source = expected_main.add_parameter(
        "q_source", migraphx::shape{migraphx::shape::float_type, {dd{lit(4)}, dd{q}}});
    std::vector<migraphx::instruction_ref> runtime_sources = {
        expected_q_source, expected_other, expected_input};

    auto target_n  = var("split_sym_dim_n_target", {1, 2}, {1, 2});
    auto target_q  = var("split_sym_dim_q_target", {1, 2}, {1, 2});
    auto select_nq = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type, {dd{target_n}, dd{target_q}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_input, expected_q_source},
        connected_modules);
    auto output_nq = expected_main.add_instruction(
        migraphx::make_op("get_tuple_elem", {{"index", 0}}), select_nq);
    auto output_nq_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0), lit(0)})}}),
        runtime_sources);
    auto output_nq_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n, q})}}),
        runtime_sources);
    output_nq = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0, 1}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0), lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n, q})}}),
        output_nq,
        output_nq_start,
        output_nq_end);

    auto target_m = var("split_sym_dim_m_target", {1, 2}, {1, 2});
    auto select_m = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type, {dd{target_m}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_other},
        other_modules);
    auto output_m = expected_main.add_instruction(
        migraphx::make_op("get_tuple_elem", {{"index", 0}}), select_m);
    auto output_m_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        runtime_sources);
    auto output_m_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{m})}}),
        runtime_sources);
    output_m = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{m})}}),
        output_m,
        output_m_start,
        output_m_end);
    expected_main.add_return({output_nq, output_m});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_preserves_two_input_reshape_target_layout)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto data =
        m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    migraphx::shape target_shape{
        migraphx::shape::float_type, {dd{lit(4)}, dd{n}}, {lit(1), lit(4)}};
    auto target  = m.add_parameter("target", target_shape);
    auto reshape = m.add_instruction(migraphx::make_op("reshape"), data, target);
    m.add_return({reshape});

    auto expected = p;
    run_pass(p);
    EXPECT(p == expected);
}

TEST_CASE(split_sym_dim_preserves_unroutable_symbolic_stride)
{
    auto n      = var("n", {1, 4}, {2});
    auto stride = var("stride", {4, 8}, {4});
    migraphx::program p;
    auto& m = *p.get_main_module();
    migraphx::shape input_shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}, {stride, lit(1)}};
    auto input  = m.add_parameter("input", input_shape);
    auto output = m.add_instruction(migraphx::make_op("relu"), input);
    m.add_return({output});

    auto expected = p;
    run_pass(p);
    EXPECT(p == expected);
}

TEST_CASE(split_sym_dim_preserves_routed_symbolic_parameter_strides)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    migraphx::shape data_shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}, {lit(1), n}};
    migraphx::shape weights_shape{
        migraphx::shape::float_type, {dd{lit(4)}, dd{lit(4)}}, {n + 4, lit(1)}};
    auto data    = m.add_parameter("data", data_shape);
    auto weights = m.add_parameter("weights", weights_shape);
    auto output  = m.add_instruction(migraphx::make_op("dot"), data, weights);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        migraphx::shape clone_data_shape;
        migraphx::shape clone_weights_shape;
        if(clone.min == clone.max)
        {
            clone_data_shape    = {migraphx::shape::float_type, {clone.min, 4}, {1, clone.min}};
            clone_weights_shape = {migraphx::shape::float_type, {4, 4}, {clone.min + 4, 1}};
        }
        else
        {
            auto clone_n     = var("n", {clone.min, clone.max});
            clone_data_shape = {
                migraphx::shape::float_type, {dd{clone_n}, dd{lit(4)}}, {lit(1), clone_n}};
            clone_weights_shape = {
                migraphx::shape::float_type, {dd{lit(4)}, dd{lit(4)}}, {clone_n + 4, lit(1)}};
        }

        auto clone_weights = sm.add_parameter("weights", clone_weights_shape);
        auto clone_data    = sm.add_parameter("data", clone_data_shape);
        auto padded_data =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_data);
        auto padded_weights =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_weights);
        auto clone_output =
            sm.add_instruction(migraphx::make_op("dot"), padded_data, padded_weights);
        sm.add_return({clone_output});
    });

    auto& expected_main   = *expected.get_main_module();
    auto expected_data    = expected_main.add_parameter("data", data_shape);
    auto expected_weights = expected_main.add_parameter("weights", weights_shape);
    auto target_n         = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select           = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type, {dd{target_n}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_data, expected_weights},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_weights, expected_data});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_weights, expected_data});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_keeps_variable_stride_dependency_at_boundary)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto data =
        m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    migraphx::shape weights_shape{
        migraphx::shape::float_type, {dd{lit(4)}, dd{lit(4)}}, {n + 4, lit(1)}};
    auto weights  = m.add_parameter("weights", weights_shape);
    auto identity = m.add_instruction(migraphx::make_op("identity"), weights);
    auto output   = m.add_instruction(migraphx::make_op("dot"), data, identity);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_n    = var("n", {clone.min, clone.max});
        auto clone_data = sm.add_parameter(
            "data", migraphx::shape{migraphx::shape::float_type, {dd{clone_n}, dd{lit(4)}}});
        migraphx::shape clone_dependency_shape;
        if(clone.min == clone.max)
        {
            clone_dependency_shape = {migraphx::shape::float_type, {4, 4}, {clone.min + 4, 1}};
        }
        else
        {
            clone_dependency_shape = {
                migraphx::shape::float_type, {dd{lit(4)}, dd{lit(4)}}, {clone_n + 4, lit(1)}};
        }
        auto clone_dependency =
            sm.add_parameter("#split_sym_dim_input_0_0", clone_dependency_shape);
        auto padded_data =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_data);
        auto padded_dependency =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_dependency);
        auto clone_output =
            sm.add_instruction(migraphx::make_op("dot"), padded_data, padded_dependency);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto expected_weights = expected_main.add_parameter("weights", weights_shape);
    auto expected_identity =
        expected_main.add_instruction(migraphx::make_op("identity"), expected_weights);
    auto target_n = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type, {dd{target_n}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_identity, expected_data},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_weights, expected_data});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_weights, expected_data});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_uses_clone_output_layout_for_dispatch)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    migraphx::shape input_shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}, {lit(1), n}};
    auto input  = m.add_parameter("input", input_shape);
    auto output = m.add_instruction(migraphx::make_op("relu"), input);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_n = var("n", {clone.min, clone.max});
        migraphx::shape clone_input_shape{
            migraphx::shape::float_type, {dd{clone_n}, dd{lit(4)}}, {lit(1), clone_n}};
        auto clone_input = sm.add_parameter("input", clone_input_shape);
        auto padded_input =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_input);
        auto clone_output = sm.add_instruction(migraphx::make_op("relu"), padded_input);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_input = expected_main.add_parameter("input", input_shape);
    auto target_n       = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select         = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type, {dd{target_n}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_input},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_uses_non_degenerate_clone_layout_for_dispatch)
{
    auto n = var("n", {1, 3}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    migraphx::shape input_shape{migraphx::shape::float_type, {dd{n}, dd{n}}, {lit(1), n}};
    auto input = m.add_parameter("input", input_shape);
    auto output =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), input);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 3}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_n = var("n", {clone.min, clone.max});
        migraphx::shape clone_input_shape{
            migraphx::shape::float_type, {dd{clone_n}, dd{clone_n}}, {lit(1), clone_n}};
        auto clone_input = sm.add_parameter("input", clone_input_shape);
        auto padded_input =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_input);
        auto clone_output = sm.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {1, 0}}}), padded_input);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_input = expected_main.add_parameter("input", input_shape);
    auto target_n       = var("split_sym_dim_n_target", {1, 3}, {1, 2, 3});
    migraphx::shape output_shape{
        migraphx::shape::float_type, {dd{target_n}, dd{target_n}}, {lit(1), target_n}};
    auto select = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{output_shape}})}}),
        std::vector<migraphx::instruction_ref>{expected_input},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0), lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n, n})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0, 1}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0), lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n, n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_materializes_reshape_with_shape_chain_target)
{
    auto batch = var("batch", {1, 1});
    auto n     = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m   = *p.get_main_module();
    auto data = m.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{batch}, dd{n}, dd{lit(16)}}});
    auto shape =
        m.add_instruction(migraphx::make_op("dimensions_of", {{"start", 0}, {"end", 3}}), data);
    auto index0 = m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}},
                                                  std::vector<int64_t>{0}});
    auto index1 = m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}},
                                                  std::vector<int64_t>{1}});
    auto batch_dim = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), shape, index0);
    auto sequence_dim =
        m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), shape, index1);
    auto four = m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}},
                                                std::vector<int64_t>{4}});
    auto target_dims = m.add_instruction(
        migraphx::make_op("concat", {{"axis", 0}}), batch_dim, sequence_dim, four, four);
    auto target = m.add_instruction(
        migraphx::make_op(
            "allocate",
            {{"shape",
              migraphx::to_value(migraphx::shape{migraphx::shape::float_type,
                                                 {dd{lit(1)}, dd{n}, dd{lit(4)}, dd{lit(4)}}})}}),
        target_dims);
    auto reshape = m.add_instruction(migraphx::make_op("reshape"), data, target);
    m.add_return({reshape});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_data = sm.add_parameter(
            "data",
            migraphx::shape{
                migraphx::shape::float_type,
                {dd{var("batch", {1, 1})}, dd{var("n", {clone.min, clone.max})}, dd{lit(16)}}});
        auto padded_data =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_data);
        auto clone_output = sm.add_instruction(
            migraphx::make_op("reshape", {{"dims", {1, clone.max, 4, 4}}}), padded_data);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{batch}, dd{n}, dd{lit(16)}}});
    auto target_n = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type,
                                  {dd{lit(1)}, dd{target_n}, dd{lit(4)}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_data},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_data});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_data});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{1}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_multi_input_multibroadcast_is_noop)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto shape_input =
        m.add_parameter("shape", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(3)}}});
    auto data   = m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {3}});
    auto output = m.add_instruction(migraphx::make_op("multibroadcast"), data, shape_input);
    m.add_return({output});

    auto expected = p;
    run_pass(p);
    EXPECT(p == expected);
}

TEST_CASE(split_sym_dim_two_input_broadcast_is_noop)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m          = *p.get_main_module();
    auto shape_input = m.add_parameter(
        "shape", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(3)}, dd{lit(4)}}});
    auto data = m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {3}});
    auto output =
        m.add_instruction(migraphx::make_op("broadcast", {{"axis", 1}}), data, shape_input);
    m.add_return({output});

    auto expected = p;
    run_pass(p);
    EXPECT(p == expected);
}

TEST_CASE(split_sym_dim_uses_interval_bounds_without_optimals)
{
    auto n = var("n", {1, 8});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x =
        m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto y = m.add_instruction(migraphx::make_op("relu"), x);
    m.add_return({y});
    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 8}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto input =
            sm.add_parameter("data",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{var("n", {clone.min, clone.max})}, dd{lit(4)}}});
        auto pad    = sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), input);
        auto output = sm.add_instruction(migraphx::make_op("relu"), pad);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto input          = expected_main.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto target_n = var("split_sym_dim_n_target", {1, 8}, {1, 8});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type, {dd{target_n}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{input},
        modules);
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{input});
    auto output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{input});
    output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        output,
        output_start,
        output_end);
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_clone_cap_is_noop)
{
    auto n = var("n", {1, 8}, {2, 4});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x =
        m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto y = m.add_instruction(migraphx::make_op("relu"), x);
    m.add_return({y});
    auto expected = p;
    run_pass(p, 2);
    EXPECT(p == expected);
}

TEST_CASE(split_sym_dim_default_clone_cap_is_noop)
{
    const migraphx::split_sym_dim pass{};
    const auto max_clones = pass.max_clones;
    std::set<migraphx::sym::scalar> optimals;
    auto optimal_values = migraphx::range(std::size_t{2}, max_clones + 1);
    std::transform(optimal_values.begin(),
                   optimal_values.end(),
                   std::inserter(optimals, optimals.end()),
                   [](auto value) { return static_cast<int64_t>(value); });
    auto n = var("n", {1, static_cast<int64_t>(max_clones + 1)}, optimals);

    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x =
        m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto y = m.add_instruction(migraphx::make_op("relu"), x);
    m.add_return({y});

    auto expected = p;
    run_pass(p);
    EXPECT(p == expected);
}

TEST_CASE(split_sym_dim_specializes_static_output_reduce)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m     = *p.get_main_module();
    auto data   = m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto output = m.add_instruction(migraphx::make_op("reduce_max", {{"axes", {0}}}), data);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto input = sm.add_parameter(
            "data",
            migraphx::shape{migraphx::shape::float_type, {dd{var("n", {clone.min, clone.max})}}});
        input = sm.add_instruction(
            migraphx::make_op("fixed_pad", {{"value", -std::numeric_limits<float>::max()}}), input);
        auto clone_output =
            sm.add_instruction(migraphx::make_op("reduce_max", {{"axes", {0}}}), input);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto input =
        expected_main.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto select = expected_main.add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes",
                            migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                                migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}}}}})}}),
        std::vector<migraphx::instruction_ref>{input},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_specializes_static_output_gemm)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto a =
        m.add_parameter("a", migraphx::shape{migraphx::shape::float_type, {dd{lit(4)}, dd{n}}});
    auto b =
        m.add_parameter("b", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto output = m.add_instruction(migraphx::make_op("dot"), a, b);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_n = var("n", {clone.min, clone.max});
        auto clone_a = sm.add_parameter(
            "a", migraphx::shape{migraphx::shape::float_type, {dd{lit(4)}, dd{clone_n}}});
        auto clone_b = sm.add_parameter(
            "b", migraphx::shape{migraphx::shape::float_type, {dd{clone_n}, dd{lit(4)}}});
        auto padded_a =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_a);
        auto padded_b =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_b);
        std::vector<int64_t> indices_data(clone.max);
        std::iota(indices_data.begin(), indices_data.end(), 0);
        auto indices = sm.add_literal(migraphx::literal{
            migraphx::shape{migraphx::shape::int64_type, {clone.max}}, indices_data});
        auto extent  = add_expected_clone_extent(sm, n, n, clone, clone_a);
        auto zero    = sm.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {0.0f}});
        auto masked_a     = add_mask(sm, padded_a, indices, extent, zero, 1);
        auto masked_b     = add_mask(sm, padded_b, indices, extent, zero, 0);
        auto clone_output = sm.add_instruction(migraphx::make_op("dot"), masked_a, masked_b);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_a     = expected_main.add_parameter(
        "a", migraphx::shape{migraphx::shape::float_type, {dd{lit(4)}, dd{n}}});
    auto expected_b = expected_main.add_parameter(
        "b", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto select = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type, {dd{lit(4)}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_a, expected_b},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_reduce_mean_is_noop)
{
    auto b = var("b", {1, 4}, {2, 4});
    auto s = var("s", {2, 8}, {4, 8});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x  = m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{b}, dd{s}}});
    auto y  = m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {1}}}), x);
    m.add_return({y});

    auto expected = p;
    run_pass(p);
    EXPECT(p == expected);
}

TEST_CASE(split_sym_dim_argmax_is_noop)
{
    auto b = var("b", {1, 4}, {2, 4});
    auto s = var("s", {2, 8}, {4, 8});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x  = m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{b}, dd{s}}});
    auto y  = m.add_instruction(migraphx::make_op("argmax", {{"axis", 1}}), x);
    m.add_return({y});

    auto expected = p;
    run_pass(p);
    EXPECT(p == expected);
}

TEST_CASE(split_sym_dim_reduce_all_uses_one)
{
    auto b = var("b", {1, 2}, {2});
    auto s = var("s", {2, 4}, {4});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x  = m.add_parameter("data", migraphx::shape{migraphx::shape::bool_type, {dd{b}, dd{s}}});
    auto y  = m.add_instruction(migraphx::make_op("reduce_all", {{"axes", {1}}}), x);
    m.add_return({y});
    run_pass(p);

    migraphx::program expected;
    struct reduce_clone_spec
    {
        std::size_t b_min;
        std::size_t b_max;
        std::size_t s_min;
        std::size_t s_max;
    };
    std::vector<reduce_clone_spec> clones = {
        {1, 1, 2, 2}, {2, 2, 2, 2}, {1, 1, 3, 4}, {2, 2, 3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto input  = sm.add_parameter("data",
                                       migraphx::shape{migraphx::shape::bool_type,
                                                       {dd{var("b", {clone.b_min, clone.b_max})},
                                                        dd{var("s", {clone.s_min, clone.s_max})}}});
        auto pad    = sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 1.0f}}), input);
        auto output = sm.add_instruction(migraphx::make_op("reduce_all", {{"axes", {1}}}), pad);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto input          = expected_main.add_parameter(
        "data", migraphx::shape{migraphx::shape::bool_type, {dd{b}, dd{s}}});
    auto target_b = var("split_sym_dim_b_target", {1, 2}, {1, 2});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::bool_type, {dd{target_b}, dd{lit(1)}}}}})}}),
        std::vector<migraphx::instruction_ref>{input},
        modules);
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{input});
    auto output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{b})}}),
        std::vector<migraphx::instruction_ref>{input});
    output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{b})}}),
        output,
        output_start,
        output_end);
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_pads_absorbed_reduction_input)
{
    auto batch = var("batch", {1, 1});
    auto n     = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto mask =
        m.add_parameter("mask", migraphx::shape{migraphx::shape::int64_type, {dd{batch}, dd{n}}});
    auto values =
        m.add_parameter("values", migraphx::shape{migraphx::shape::int32_type, {dd{batch}, dd{n}}});
    auto reduced = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), mask);
    reduced      = m.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), reduced);
    auto expanded = m.add_instruction(
        migraphx::make_op(
            "multibroadcast",
            {{"out_dyn_dims", migraphx::to_value(std::vector<dd>{dd{batch}, dd{n}})}}),
        std::vector<migraphx::instruction_ref>{reduced, values});
    auto output = m.add_instruction(migraphx::make_op("add"), values, expanded);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_batch = var("batch", {1, 1});
        auto clone_n     = var("n", {clone.min, clone.max});
        auto clone_mask  = sm.add_parameter(
            "mask", migraphx::shape{migraphx::shape::int64_type, {dd{clone_batch}, dd{clone_n}}});
        auto clone_values = sm.add_parameter(
            "values", migraphx::shape{migraphx::shape::int32_type, {dd{clone_batch}, dd{clone_n}}});
        auto padded_mask =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_mask);
        auto clone_reduced =
            sm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), padded_mask);
        clone_reduced = sm.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}),
            clone_reduced);
        clone_reduced =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_reduced);
        auto clone_expanded = sm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, clone.max}}}), clone_reduced);
        auto padded_values =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_values);
        auto clone_output =
            sm.add_instruction(migraphx::make_op("add"), padded_values, clone_expanded);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_mask  = expected_main.add_parameter(
        "mask", migraphx::shape{migraphx::shape::int64_type, {dd{batch}, dd{n}}});
    auto expected_values = expected_main.add_parameter(
        "values", migraphx::shape{migraphx::shape::int32_type, {dd{batch}, dd{n}}});
    auto target_n = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::int32_type, {dd{lit(1)}, dd{target_n}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_mask, expected_values},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_values, expected_mask});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_values, expected_mask});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{1}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_auto_pad_spatial_is_noop)
{
    auto s = var("s", {4, 8}, {6, 8});
    migraphx::program p;
    auto& m    = *p.get_main_module();
    auto input = m.add_parameter(
        "data",
        migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{lit(1)}, dd{s}, dd{s}}});
    auto weights =
        m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 1, 3, 3}}));
    auto conv = m.add_instruction(
        migraphx::make_op("convolution",
                          {{"padding_mode", migraphx::op::padding_mode_t::same_upper},
                           {"stride", {2, 2}},
                           {"dilation", {1, 1}}}),
        input,
        weights);
    m.add_return({conv});

    auto expected = p;
    run_pass(p);
    EXPECT(p == expected);
}

TEST_CASE(split_sym_dim_preserves_ceil_average_pooling)
{
    auto n = var("n", {2, 4});
    migraphx::program p;
    auto& m    = *p.get_main_module();
    auto input = m.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{lit(1)}, dd{n}}});
    auto output =
        m.add_instruction(migraphx::make_op("pooling",
                                            {{"mode", migraphx::op::pooling_mode::average},
                                             {"padding", {0}},
                                             {"stride", {2}},
                                             {"lengths", {2}},
                                             {"dilations", {1}},
                                             {"ceil_mode", true},
                                             {"count_include_pad", true}}),
                          input);
    m.add_return({output});

    auto expected = p;
    run_pass(p);
    EXPECT(p == expected);
}

TEST_CASE(split_sym_dim_global_max_pooling_preserves_runtime_spatial_extent)
{
    auto n = var("n", {1, 8}, {8});
    migraphx::program p;
    auto& m    = *p.get_main_module();
    auto input = m.add_parameter(
        "data",
        migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{lit(1)}, dd{n}, dd{lit(1)}}});
    auto neg    = m.add_instruction(migraphx::make_op("neg"), input);
    auto output = m.add_instruction(
        migraphx::make_op("pooling",
                          {{"mode", migraphx::op::pooling_mode::max}, {"dyn_global", true}}),
        neg);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 8}};
    auto neg_modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_n = var("n", {clone.min, clone.max});
        auto clone_input =
            sm.add_parameter("data",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{lit(1)}, dd{lit(1)}, dd{clone_n}, dd{lit(1)}}});
        clone_input =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_input);
        auto clone_output = sm.add_instruction(migraphx::make_op("neg"), clone_input);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_input = expected_main.add_parameter(
        "data",
        migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{lit(1)}, dd{n}, dd{lit(1)}}});
    auto target_n   = var("split_sym_dim_n_target", {1, 8}, {1, 8});
    auto neg_select = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type,
                                  {dd{lit(1)}, dd{lit(1)}, dd{target_n}, dd{lit(1)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_input},
        neg_modules);
    auto neg_output = expected_main.add_instruction(
        migraphx::make_op("get_tuple_elem", {{"index", 0}}), neg_select);
    auto output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    auto output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    neg_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{2}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        neg_output,
        output_start,
        output_end);

    auto pooling_modules = add_clones(expected, 1, clones, [&](auto& sm, const auto& clone) {
        auto clone_n = var("n", {clone.min, clone.max});
        sm.add_parameter("data",
                         migraphx::shape{migraphx::shape::float_type,
                                         {dd{lit(1)}, dd{lit(1)}, dd{clone_n}, dd{lit(1)}}});
        auto clone_extent = migraphx::sym::min(clone_n, lit(clone.max));
        auto boundary =
            sm.add_parameter("#split_sym_dim_input_1_0",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{lit(1)}, dd{lit(1)}, dd{clone_extent}, dd{lit(1)}},
                                             {target_n, target_n, lit(1), lit(1)}});
        boundary = sm.add_instruction(
            migraphx::make_op("fixed_pad", {{"value", std::numeric_limits<float>::lowest()}}),
            boundary);
        auto clone_output = sm.add_instruction(
            migraphx::make_op("pooling",
                              {{"mode", migraphx::op::pooling_mode::max}, {"dyn_global", true}}),
            boundary);
        sm.add_return({clone_output});
    });

    auto pool_select = expected_main.add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes",
                            migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                                migraphx::shape{migraphx::shape::float_type, {1, 1, 1, 1}}}})}}),
        std::vector<migraphx::instruction_ref>{neg_output, expected_input},
        pooling_modules);
    auto expected_output = expected_main.add_instruction(
        migraphx::make_op("get_tuple_elem", {{"index", 0}}), pool_select);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_splits_windowed_boundary)
{
    auto s = var("s", {8, 16}, {12, 16});
    migraphx::program p;
    auto& m    = *p.get_main_module();
    auto input = m.add_parameter(
        "data",
        migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{lit(1)}, dd{s}, dd{s}}});
    auto weights =
        m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 1, 3, 3}}));
    auto conv = m.add_instruction(migraphx::make_op("convolution"), input, weights);
    auto pool = m.add_instruction(migraphx::make_op("pooling",
                                                    {{"mode", migraphx::op::pooling_mode::max},
                                                     {"padding", {1, 1}},
                                                     {"stride", {2, 2}},
                                                     {"lengths", {3, 3}}}),
                                  conv);
    m.add_return({pool});

    run_pass(p);

    migraphx::program expected;
    migraphx::shape weights_shape{migraphx::shape::float_type, {1, 1, 3, 3}};
    std::vector<clone_spec> clones = {{8, 8}, {9, 12}, {13, 16}};
    auto convolution_modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_s = var("s", {clone.min, clone.max});
        auto clone_input =
            sm.add_parameter("data",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{lit(1)}, dd{lit(1)}, dd{clone_s}, dd{clone_s}}});
        auto clone_weights = sm.add_literal(migraphx::generate_literal(weights_shape));
        auto pad =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_input);
        auto output = sm.add_instruction(migraphx::make_op("convolution"), pad, clone_weights);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_input = expected_main.add_parameter(
        "data",
        migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{lit(1)}, dd{s}, dd{s}}});
    auto target_s    = var("split_sym_dim_s_target", {8, 16}, {8, 12, 16});
    auto conv_extent = target_s - 2;
    auto conv_select = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type,
                                  {dd{lit(1)}, dd{lit(1)}, dd{conv_extent}, dd{conv_extent}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_input},
        convolution_modules);
    auto conv_output = expected_main.add_instruction(
        migraphx::make_op("get_tuple_elem", {{"index", 0}}), conv_select);
    auto conv_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0), lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    auto conv_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{s - 2, s - 2})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    conv_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{2, 3}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0), lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{s - 2, s - 2})}}),
        conv_output,
        conv_output_start,
        conv_output_end);

    auto pooling_modules = add_clones(expected, 1, clones, [&](auto& sm, const auto& clone) {
        auto clone_s = var("s", {clone.min, clone.max});
        sm.add_parameter("data",
                         migraphx::shape{migraphx::shape::float_type,
                                         {dd{lit(1)}, dd{lit(1)}, dd{clone_s}, dd{clone_s}}});
        auto clone_extent         = migraphx::sym::min(clone_s - 2, lit(clone.max - 2));
        auto clone_boundary_shape = migraphx::shape{
            migraphx::shape::float_type,
            {dd{lit(1)}, dd{lit(1)}, dd{clone_extent}, dd{clone_extent}},
            {conv_extent * conv_extent, conv_extent * conv_extent, conv_extent, lit(1)}};
        auto boundary = sm.add_parameter("#split_sym_dim_input_1_0", clone_boundary_shape);
        auto pad      = sm.add_instruction(
            migraphx::make_op("fixed_pad", {{"value", std::numeric_limits<float>::lowest()}}),
            boundary);
        auto output =
            sm.add_instruction(migraphx::make_op("pooling",
                                                 {{"mode", migraphx::op::pooling_mode::max},
                                                  {"padding", {1, 1}},
                                                  {"stride", {2, 2}},
                                                  {"lengths", {3, 3}}}),
                               pad);
        sm.add_return({output});
    });

    auto pooled_extent = (target_s - 3) / 2 + 1;
    auto pool_select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{migraphx::shape{
                  migraphx::shape::float_type,
                  {dd{lit(1)}, dd{lit(1)}, dd{pooled_extent}, dd{pooled_extent}}}}})}}),
        std::vector<migraphx::instruction_ref>{conv_output, expected_input},
        pooling_modules);
    auto output = expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}),
                                                pool_select);
    auto output_extent = (s - 3) / 2 + 1;
    auto output_start  = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0), lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    auto output_end = expected_main.add_instruction(
        migraphx::make_op(
            "eval_expr_from_shape",
            {{"expressions", migraphx::to_value(std::vector<se>{output_extent, output_extent})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    output = expected_main.add_instruction(
        migraphx::make_op(
            "dyn_slice",
            {{"axes", std::vector<int64_t>{2, 3}},
             {"starts", migraphx::to_value(std::vector<se>{lit(0), lit(0)})},
             {"ends", migraphx::to_value(std::vector<se>{output_extent, output_extent})}}),
        output,
        output_start,
        output_end);
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_retains_only_nonparallel_boundary_axes)
{
    auto batch   = var("batch", {1, 2}, {2});
    auto spatial = var("spatial", {8, 16}, {12, 16});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto input =
        m.add_parameter("data",
                        migraphx::shape{migraphx::shape::float_type,
                                        {dd{batch}, dd{lit(1)}, dd{spatial}, dd{spatial}}});
    auto weights =
        m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 1, 3, 3}}));
    auto conv = m.add_instruction(migraphx::make_op("convolution"), input, weights);
    auto pool = m.add_instruction(migraphx::make_op("pooling",
                                                    {{"mode", migraphx::op::pooling_mode::max},
                                                     {"padding", {1, 1}},
                                                     {"stride", {2, 2}},
                                                     {"lengths", {3, 3}}}),
                                  conv);
    m.add_return({pool});

    run_pass(p);

    migraphx::program expected;
    migraphx::shape weights_shape{migraphx::shape::float_type, {1, 1, 3, 3}};
    struct boundary_clone_spec
    {
        std::size_t batch;
        std::size_t spatial_min;
        std::size_t spatial_max;
    };
    std::vector<boundary_clone_spec> clones = {
        {1, 8, 8}, {2, 8, 8}, {1, 9, 12}, {2, 9, 12}, {1, 13, 16}, {2, 13, 16}};
    auto convolution_modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_batch   = var("batch", {clone.batch, clone.batch});
        auto clone_spatial = var("spatial", {clone.spatial_min, clone.spatial_max});
        auto clone_input   = sm.add_parameter(
            "data",
            migraphx::shape{migraphx::shape::float_type,
                            {dd{clone_batch}, dd{lit(1)}, dd{clone_spatial}, dd{clone_spatial}}});
        auto clone_weights = sm.add_literal(migraphx::generate_literal(weights_shape));
        auto pad =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_input);
        auto output = sm.add_instruction(migraphx::make_op("convolution"), pad, clone_weights);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_input = expected_main.add_parameter(
        "data",
        migraphx::shape{migraphx::shape::float_type,
                        {dd{batch}, dd{lit(1)}, dd{spatial}, dd{spatial}}});
    auto target_batch   = var("split_sym_dim_batch_target", {1, 2}, {1, 2});
    auto target_spatial = var("split_sym_dim_spatial_target", {8, 16}, {8, 12, 16});
    auto conv_extent    = target_spatial - 2;
    auto conv_select    = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{migraphx::shape{
                  migraphx::shape::float_type,
                  {dd{target_batch}, dd{lit(1)}, dd{conv_extent}, dd{conv_extent}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_input},
        convolution_modules);
    auto conv_output = expected_main.add_instruction(
        migraphx::make_op("get_tuple_elem", {{"index", 0}}), conv_select);
    auto conv_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0), lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    auto conv_output_end = expected_main.add_instruction(
        migraphx::make_op(
            "eval_expr_from_shape",
            {{"expressions", migraphx::to_value(std::vector<se>{spatial - 2, spatial - 2})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    conv_output = expected_main.add_instruction(
        migraphx::make_op(
            "dyn_slice",
            {{"axes", std::vector<int64_t>{2, 3}},
             {"starts", migraphx::to_value(std::vector<se>{lit(0), lit(0)})},
             {"ends", migraphx::to_value(std::vector<se>{spatial - 2, spatial - 2})}}),
        conv_output,
        conv_output_start,
        conv_output_end);

    auto pooling_modules = add_clones(expected, 1, clones, [&](auto& sm, const auto& clone) {
        auto clone_batch   = var("batch", {clone.batch, clone.batch});
        auto clone_spatial = var("spatial", {clone.spatial_min, clone.spatial_max});
        sm.add_parameter(
            "data",
            migraphx::shape{migraphx::shape::float_type,
                            {dd{clone_batch}, dd{lit(1)}, dd{clone_spatial}, dd{clone_spatial}}});
        auto clone_extent = migraphx::sym::min(clone_spatial - 2, lit(clone.spatial_max - 2));
        auto clone_boundary_shape = migraphx::shape{
            migraphx::shape::float_type,
            {dd{lit(clone.batch)}, dd{lit(1)}, dd{clone_extent}, dd{clone_extent}},
            {conv_extent * conv_extent, conv_extent * conv_extent, conv_extent, lit(1)}};
        auto boundary = sm.add_parameter("#split_sym_dim_input_1_0", clone_boundary_shape);
        auto pad      = sm.add_instruction(
            migraphx::make_op("fixed_pad", {{"value", std::numeric_limits<float>::lowest()}}),
            boundary);
        auto output =
            sm.add_instruction(migraphx::make_op("pooling",
                                                 {{"mode", migraphx::op::pooling_mode::max},
                                                  {"padding", {1, 1}},
                                                  {"stride", {2, 2}},
                                                  {"lengths", {3, 3}}}),
                               pad);
        sm.add_return({output});
    });

    auto pooled_extent = (target_spatial - 3) / 2 + 1;
    auto pool_select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{migraphx::shape{
                  migraphx::shape::float_type,
                  {dd{target_batch}, dd{lit(1)}, dd{pooled_extent}, dd{pooled_extent}}}}})}}),
        std::vector<migraphx::instruction_ref>{conv_output, expected_input},
        pooling_modules);
    auto output = expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}),
                                                pool_select);
    auto output_extent = (spatial - 3) / 2 + 1;
    auto output_start  = expected_main.add_instruction(
        migraphx::make_op(
            "eval_expr_from_shape",
            {{"expressions", migraphx::to_value(std::vector<se>{lit(0), lit(0), lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    auto output_end = expected_main.add_instruction(
        migraphx::make_op(
            "eval_expr_from_shape",
            {{"expressions",
              migraphx::to_value(std::vector<se>{batch, output_extent, output_extent})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    output = expected_main.add_instruction(
        migraphx::make_op(
            "dyn_slice",
            {{"axes", std::vector<int64_t>{0, 2, 3}},
             {"starts", migraphx::to_value(std::vector<se>{lit(0), lit(0), lit(0)})},
             {"ends", migraphx::to_value(std::vector<se>{batch, output_extent, output_extent})}}),
        output,
        output_start,
        output_end);
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_merges_independent_supported_branches)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x0 =
        m.add_parameter("x0", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto x1 =
        m.add_parameter("x1", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto y0 = m.add_instruction(migraphx::make_op("relu"), x0);
    auto y1 = m.add_instruction(migraphx::make_op("relu"), x1);
    m.add_return({y0, y1});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_n = var("n", {clone.min, clone.max});
        auto x1      = sm.add_parameter(
            "x1", migraphx::shape{migraphx::shape::float_type, {dd{clone_n}, dd{lit(4)}}});
        auto x0 = sm.add_parameter(
            "x0", migraphx::shape{migraphx::shape::float_type, {dd{clone_n}, dd{lit(4)}}});
        auto pad0 = sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), x0);
        auto y0   = sm.add_instruction(migraphx::make_op("relu"), pad0);
        auto pad1 = sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), x1);
        auto y1   = sm.add_instruction(migraphx::make_op("relu"), pad1);
        sm.add_return({y0, y1});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_x0    = expected_main.add_parameter(
        "x0", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto expected_x1 = expected_main.add_parameter(
        "x1", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto target_n     = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto output_shape = migraphx::shape{migraphx::shape::float_type, {dd{target_n}, dd{lit(4)}}};
    auto select       = expected_main.add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes",
                            migraphx::to_value(migraphx::shape{
                                std::vector<migraphx::shape>{output_shape, output_shape}})}}),
        std::vector<migraphx::instruction_ref>{expected_x0, expected_x1},
        modules);
    auto output0 =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto output0_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_x1, expected_x0});
    auto output0_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_x1, expected_x0});
    output0 = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        output0,
        output0_start,
        output0_end);
    auto output1 =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), select);
    auto output1_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_x1, expected_x0});
    auto output1_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_x1, expected_x0});
    output1 = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        output1,
        output1_start,
        output1_end);
    expected_main.add_return({output0, output1});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_orders_merged_blocks_by_dependencies)
{
    auto n = var("n", {1, 4}, {2});
    auto k = var("k", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x  = m.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto y  = m.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {dd{k}}});

    auto independent = m.add_instruction(migraphx::make_op("relu"), x);
    auto producer    = m.add_instruction(migraphx::make_op("relu"), y);
    auto reduced  = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), producer);
    auto consumer = m.add_instruction(
        migraphx::make_op("multibroadcast",
                          {{"out_dyn_dims", migraphx::to_value(std::vector<dd>{dd{n}})}}),
        std::vector<migraphx::instruction_ref>{reduced, x});
    m.add_return({independent, producer, consumer});

    run_pass(p, 3);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto producer_modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_k = var("k", {clone.min, clone.max});
        auto input =
            sm.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {dd{clone_k}}});
        input       = sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), input);
        auto output = sm.add_instruction(migraphx::make_op("relu"), input);
        sm.add_return({output});
    });
    auto target_k         = var("split_sym_dim_k_target", {1, 4}, {1, 2, 4});
    auto consumer_modules = add_clones(expected, 1, clones, [&](auto& sm, const auto& clone) {
        auto clone_n = var("n", {clone.min, clone.max});
        auto input =
            sm.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {dd{clone_n}}});
        auto dependency = sm.add_parameter(
            "#split_sym_dim_input_1_0",
            migraphx::shape{migraphx::shape::float_type, {dd{migraphx::sym::min(k, target_k)}}});
        input      = sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), input);
        auto first = sm.add_instruction(migraphx::make_op("relu"), input);
        dependency =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), dependency);
        dependency =
            sm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), dependency);
        auto second = sm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {clone.max}}}), dependency);
        sm.add_return({first, second});
    });
    auto& expected_main   = *expected.get_main_module();
    auto expected_x =
        expected_main.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto expected_y =
        expected_main.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {dd{k}}});
    std::vector<migraphx::instruction_ref> runtime_sources = {expected_y, expected_x};
    auto producer_select                                   = expected_main.add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes",
                            migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                                migraphx::shape{migraphx::shape::float_type, {dd{target_k}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_y},
        producer_modules);
    auto expected_producer = expected_main.add_instruction(
        migraphx::make_op("get_tuple_elem", {{"index", 0}}), producer_select);
    auto expected_producer_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        runtime_sources);
    auto expected_producer_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{k})}}),
        runtime_sources);
    expected_producer = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{k})}}),
        expected_producer,
        expected_producer_start,
        expected_producer_end);

    auto target_n = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    migraphx::shape broadcast_shape{migraphx::shape::float_type, {dd{target_n}}, {lit(0)}};
    auto consumer_select = expected_main.add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes",
                            migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                                migraphx::shape{migraphx::shape::float_type, {dd{target_n}}},
                                broadcast_shape}})}}),
        std::vector<migraphx::instruction_ref>{expected_producer, expected_x},
        consumer_modules);
    auto first = expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}),
                                               consumer_select);
    auto first_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        runtime_sources);
    auto first_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        runtime_sources);
    first = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        first,
        first_start,
        first_end);
    auto second = expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}),
                                                consumer_select);
    auto second_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        runtime_sources);
    auto second_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        runtime_sources);
    second = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        second,
        second_start,
        second_end);
    expected_main.add_return({first, expected_producer, second});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_rejects_cyclic_residual_merge)
{
    migraphx::program p;
    {
        auto spatial = var("spatial", {4, 8}, {4});
        auto& m      = *p.get_main_module();
        auto x =
            m.add_parameter("x",
                            migraphx::shape{migraphx::shape::float_type,
                                            {dd{lit(1)}, dd{lit(4)}, dd{spatial}, dd{spatial}}});
        auto w =
            m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4, 3, 3}}));
        auto activation  = m.add_instruction(migraphx::make_op("relu"), x);
        auto convolution = m.add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1}}}), activation, w);
        auto output = m.add_instruction(migraphx::make_op("add"), activation, convolution);
        m.add_return({output});
    }
    run_pass(p);

    migraphx::program expected;
    {
        auto spatial                   = var("spatial", {4, 8}, {4});
        auto target_spatial            = var("split_sym_dim_spatial_target", {4, 8}, {4, 8});
        std::vector<clone_spec> clones = {{4, 4}, {5, 8}};
        auto activation_modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
            auto clone_spatial = var("spatial", {clone.min, clone.max});
            auto x             = sm.add_parameter(
                "x",
                migraphx::shape{migraphx::shape::float_type,
                                {dd{lit(1)}, dd{lit(4)}, dd{clone_spatial}, dd{clone_spatial}}});
            x           = sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), x);
            auto output = sm.add_instruction(migraphx::make_op("relu"), x);
            sm.add_return({output});
        });
        auto convolution_modules =
            add_clones(expected, 1, clones, [&](auto& sm, const auto& clone) {
                auto clone_spatial = var("spatial", {clone.min, clone.max});
                auto w             = sm.add_literal(
                    migraphx::generate_literal({migraphx::shape::float_type, {4, 4, 3, 3}}));
                sm.add_parameter(
                    "x",
                    migraphx::shape{
                        migraphx::shape::float_type,
                        {dd{lit(1)}, dd{lit(4)}, dd{clone_spatial}, dd{clone_spatial}}});
                auto clone_extent = migraphx::sym::min(clone_spatial, lit(clone.max));
                migraphx::shape boundary_shape{
                    migraphx::shape::float_type,
                    {dd{lit(1)}, dd{lit(4)}, dd{clone_extent}, dd{clone_extent}},
                    {lit(4) * target_spatial * target_spatial,
                     target_spatial * target_spatial,
                     target_spatial,
                     lit(1)}};
                auto activation = sm.add_parameter("#split_sym_dim_input_1_0", boundary_shape);
                activation = sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}),
                                                activation);
                auto convolution = sm.add_instruction(
                    migraphx::make_op("convolution", {{"padding", {1, 1}}}), activation, w);
                auto output = sm.add_instruction(migraphx::make_op("add"), activation, convolution);
                sm.add_return({output});
            });
        auto& m = *expected.get_main_module();
        auto x =
            m.add_parameter("x",
                            migraphx::shape{migraphx::shape::float_type,
                                            {dd{lit(1)}, dd{lit(4)}, dd{spatial}, dd{spatial}}});
        auto output_shape =
            migraphx::shape{migraphx::shape::float_type,
                            {dd{lit(1)}, dd{lit(4)}, dd{target_spatial}, dd{target_spatial}}};
        auto select_activation = m.add_instruction(
            migraphx::make_op("select_module",
                              {{"output_dyn_shapes",
                                migraphx::to_value(
                                    migraphx::shape{std::vector<migraphx::shape>{output_shape}})}}),
            std::vector<migraphx::instruction_ref>{x},
            activation_modules);
        auto activation = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}),
                                            select_activation);
        auto activation_start = m.add_instruction(
            migraphx::make_op(
                "eval_expr_from_shape",
                {{"expressions", migraphx::to_value(std::vector<se>{lit(0), lit(0)})}}),
            std::vector<migraphx::instruction_ref>{x});
        auto activation_end = m.add_instruction(
            migraphx::make_op(
                "eval_expr_from_shape",
                {{"expressions", migraphx::to_value(std::vector<se>{spatial, spatial})}}),
            std::vector<migraphx::instruction_ref>{x});
        activation = m.add_instruction(
            migraphx::make_op("dyn_slice",
                              {{"axes", std::vector<int64_t>{2, 3}},
                               {"starts", migraphx::to_value(std::vector<se>{lit(0), lit(0)})},
                               {"ends", migraphx::to_value(std::vector<se>{spatial, spatial})}}),
            activation,
            activation_start,
            activation_end);
        auto select_convolution = m.add_instruction(
            migraphx::make_op("select_module",
                              {{"output_dyn_shapes",
                                migraphx::to_value(
                                    migraphx::shape{std::vector<migraphx::shape>{output_shape}})}}),
            std::vector<migraphx::instruction_ref>{activation, x},
            convolution_modules);
        auto output       = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}),
                                              select_convolution);
        auto output_start = m.add_instruction(
            migraphx::make_op(
                "eval_expr_from_shape",
                {{"expressions", migraphx::to_value(std::vector<se>{lit(0), lit(0)})}}),
            std::vector<migraphx::instruction_ref>{x});
        auto output_end = m.add_instruction(
            migraphx::make_op(
                "eval_expr_from_shape",
                {{"expressions", migraphx::to_value(std::vector<se>{spatial, spatial})}}),
            std::vector<migraphx::instruction_ref>{x});
        output = m.add_instruction(
            migraphx::make_op("dyn_slice",
                              {{"axes", std::vector<int64_t>{2, 3}},
                               {"starts", migraphx::to_value(std::vector<se>{lit(0), lit(0)})},
                               {"ends", migraphx::to_value(std::vector<se>{spatial, spatial})}}),
            output,
            output_start,
            output_end);
        m.add_return({output});
    }
    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_rejects_cyclic_residual_merge_reversed_inputs)
{
    migraphx::program p;
    {
        auto spatial = var("spatial", {4, 8}, {4});
        auto& m      = *p.get_main_module();
        auto x =
            m.add_parameter("x",
                            migraphx::shape{migraphx::shape::float_type,
                                            {dd{lit(1)}, dd{lit(4)}, dd{spatial}, dd{spatial}}});
        auto w =
            m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4, 3, 3}}));
        auto activation  = m.add_instruction(migraphx::make_op("relu"), x);
        auto convolution = m.add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1}}}), activation, w);
        auto output = m.add_instruction(migraphx::make_op("add"), convolution, activation);
        m.add_return({output});
    }
    run_pass(p);

    migraphx::program expected;
    {
        auto spatial                   = var("spatial", {4, 8}, {4});
        auto target_spatial            = var("split_sym_dim_spatial_target", {4, 8}, {4, 8});
        std::vector<clone_spec> clones = {{4, 4}, {5, 8}};
        auto activation_modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
            auto clone_spatial = var("spatial", {clone.min, clone.max});
            auto x             = sm.add_parameter(
                "x",
                migraphx::shape{migraphx::shape::float_type,
                                {dd{lit(1)}, dd{lit(4)}, dd{clone_spatial}, dd{clone_spatial}}});
            x           = sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), x);
            auto output = sm.add_instruction(migraphx::make_op("relu"), x);
            sm.add_return({output});
        });
        auto convolution_modules =
            add_clones(expected, 1, clones, [&](auto& sm, const auto& clone) {
                auto clone_spatial = var("spatial", {clone.min, clone.max});
                auto w             = sm.add_literal(
                    migraphx::generate_literal({migraphx::shape::float_type, {4, 4, 3, 3}}));
                sm.add_parameter(
                    "x",
                    migraphx::shape{
                        migraphx::shape::float_type,
                        {dd{lit(1)}, dd{lit(4)}, dd{clone_spatial}, dd{clone_spatial}}});
                auto clone_extent = migraphx::sym::min(clone_spatial, lit(clone.max));
                migraphx::shape boundary_shape{
                    migraphx::shape::float_type,
                    {dd{lit(1)}, dd{lit(4)}, dd{clone_extent}, dd{clone_extent}},
                    {lit(4) * target_spatial * target_spatial,
                     target_spatial * target_spatial,
                     target_spatial,
                     lit(1)}};
                auto activation = sm.add_parameter("#split_sym_dim_input_1_0", boundary_shape);
                activation = sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}),
                                                activation);
                auto convolution = sm.add_instruction(
                    migraphx::make_op("convolution", {{"padding", {1, 1}}}), activation, w);
                auto output = sm.add_instruction(migraphx::make_op("add"), convolution, activation);
                sm.add_return({output});
            });
        auto& m = *expected.get_main_module();
        auto x =
            m.add_parameter("x",
                            migraphx::shape{migraphx::shape::float_type,
                                            {dd{lit(1)}, dd{lit(4)}, dd{spatial}, dd{spatial}}});
        auto output_shape =
            migraphx::shape{migraphx::shape::float_type,
                            {dd{lit(1)}, dd{lit(4)}, dd{target_spatial}, dd{target_spatial}}};
        auto select_activation = m.add_instruction(
            migraphx::make_op("select_module",
                              {{"output_dyn_shapes",
                                migraphx::to_value(
                                    migraphx::shape{std::vector<migraphx::shape>{output_shape}})}}),
            std::vector<migraphx::instruction_ref>{x},
            activation_modules);
        auto activation = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}),
                                            select_activation);
        auto activation_start = m.add_instruction(
            migraphx::make_op(
                "eval_expr_from_shape",
                {{"expressions", migraphx::to_value(std::vector<se>{lit(0), lit(0)})}}),
            std::vector<migraphx::instruction_ref>{x});
        auto activation_end = m.add_instruction(
            migraphx::make_op(
                "eval_expr_from_shape",
                {{"expressions", migraphx::to_value(std::vector<se>{spatial, spatial})}}),
            std::vector<migraphx::instruction_ref>{x});
        activation = m.add_instruction(
            migraphx::make_op("dyn_slice",
                              {{"axes", std::vector<int64_t>{2, 3}},
                               {"starts", migraphx::to_value(std::vector<se>{lit(0), lit(0)})},
                               {"ends", migraphx::to_value(std::vector<se>{spatial, spatial})}}),
            activation,
            activation_start,
            activation_end);
        auto select_convolution = m.add_instruction(
            migraphx::make_op("select_module",
                              {{"output_dyn_shapes",
                                migraphx::to_value(
                                    migraphx::shape{std::vector<migraphx::shape>{output_shape}})}}),
            std::vector<migraphx::instruction_ref>{activation, x},
            convolution_modules);
        auto output       = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}),
                                              select_convolution);
        auto output_start = m.add_instruction(
            migraphx::make_op(
                "eval_expr_from_shape",
                {{"expressions", migraphx::to_value(std::vector<se>{lit(0), lit(0)})}}),
            std::vector<migraphx::instruction_ref>{x});
        auto output_end = m.add_instruction(
            migraphx::make_op(
                "eval_expr_from_shape",
                {{"expressions", migraphx::to_value(std::vector<se>{spatial, spatial})}}),
            std::vector<migraphx::instruction_ref>{x});
        output = m.add_instruction(
            migraphx::make_op("dyn_slice",
                              {{"axes", std::vector<int64_t>{2, 3}},
                               {"starts", migraphx::to_value(std::vector<se>{lit(0), lit(0)})},
                               {"ends", migraphx::to_value(std::vector<se>{spatial, spatial})}}),
            output,
            output_start,
            output_end);
        m.add_return({output});
    }
    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_materializes_fixed_axis_slice)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x =
        m.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto y = m.add_instruction(migraphx::make_op("relu"), x);
    y      = m.add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {2}}}), y);
    y = m.add_instruction(migraphx::make_op("relu"), y);
    m.add_return({y});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto input =
            sm.add_parameter("x",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{var("n", {clone.min, clone.max})}, dd{lit(4)}}});
        auto pad    = sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), input);
        auto output = sm.add_instruction(migraphx::make_op("relu"), pad);
        output      = sm.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {2}}}), output);
        output = sm.add_instruction(migraphx::make_op("relu"), output);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto input          = expected_main.add_parameter(
        "x", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto target_n = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type, {dd{target_n}, dd{lit(2)}}}}})}}),
        std::vector<migraphx::instruction_ref>{input},
        modules);
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{input});
    auto output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{input});
    output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        output,
        output_start,
        output_end);
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_preserves_nonprefix_dyn_slice)
{
    auto rows    = var("rows", {1, 4}, {2});
    auto columns = var("columns", {2, 4});
    migraphx::program p;
    auto& m   = *p.get_main_module();
    auto data = m.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{rows}, dd{columns}}});
    auto start_expr = columns - 2;
    auto start      = m.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{start_expr})}}),
        data);
    auto end = m.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{columns})}}),
        data);
    auto output = m.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {1}},
                           {"starts", migraphx::to_value(std::vector<se>{start_expr})},
                           {"ends", migraphx::to_value(std::vector<se>{columns})}}),
        data,
        start,
        end);
    m.add_return({output});

    auto expected = p;
    run_pass(p);
    EXPECT(p == expected);
}

TEST_CASE(split_sym_dim_preserves_negative_bound_dyn_slice)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto data =
        m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto start =
        m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {0}});
    auto end =
        m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {-1}});
    auto output = m.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {1}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{lit(-1)})}}),
        data,
        start,
        end);
    m.add_return({output});

    auto expected = p;
    run_pass(p);

    EXPECT(p == expected);
}

TEST_CASE(split_sym_dim_materializes_gather_concat_and_dyn_slice)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m    = *p.get_main_module();
    auto ids   = m.add_parameter("ids", migraphx::shape{migraphx::shape::int64_type, {dd{n}}});
    auto table = m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}));
    auto gathered = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), table, ids);
    auto start =
        m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {0}});
    auto end =
        m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {2}});
    auto sliced = m.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {1}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{lit(2)})}}),
        gathered,
        start,
        end);
    auto joined = m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), sliced, sliced);
    auto output = m.add_instruction(migraphx::make_op("relu"), joined);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_table =
            sm.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}));
        auto clone_ids = sm.add_parameter(
            "ids",
            migraphx::shape{migraphx::shape::int64_type, {dd{var("n", {clone.min, clone.max})}}});
        auto padded_ids =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_ids);
        auto clone_gather =
            sm.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), clone_table, padded_ids);
        auto clone_slice = sm.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {2}}}),
            clone_gather);
        auto clone_concat = sm.add_instruction(
            migraphx::make_op("concat", {{"axis", 1}}), clone_slice, clone_slice);
        auto clone_output = sm.add_instruction(migraphx::make_op("relu"), clone_concat);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_ids =
        expected_main.add_parameter("ids", migraphx::shape{migraphx::shape::int64_type, {dd{n}}});
    auto target_n = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type, {dd{target_n}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_ids},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_ids});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_ids});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_materializes_gathernd)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m      = *p.get_main_module();
    auto indices = m.add_parameter(
        "indices", migraphx::shape{migraphx::shape::int64_type, {dd{lit(1)}, dd{n}, dd{lit(1)}}});
    auto table  = m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}));
    auto output = m.add_instruction(migraphx::make_op("gathernd"), table, indices);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_table =
            sm.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}));
        auto clone_indices = sm.add_parameter(
            "indices",
            migraphx::shape{migraphx::shape::int64_type,
                            {dd{lit(1)}, dd{var("n", {clone.min, clone.max})}, dd{lit(1)}}});
        auto padded_indices =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_indices);
        auto clone_output =
            sm.add_instruction(migraphx::make_op("gathernd"), clone_table, padded_indices);
        sm.add_return({clone_output});
    });

    auto& expected_main   = *expected.get_main_module();
    auto expected_indices = expected_main.add_parameter(
        "indices", migraphx::shape{migraphx::shape::int64_type, {dd{lit(1)}, dd{n}, dd{lit(1)}}});
    auto target_n = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{migraphx::shape{
                  migraphx::shape::float_type, {dd{lit(1)}, dd{target_n}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_indices},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_indices});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_indices});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{1}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_mixed_static_symbolic_concat_is_noop)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto static_data =
        m.add_parameter("static_data", migraphx::shape{migraphx::shape::float_type, {2, 4}});
    auto symbolic_data = m.add_parameter(
        "symbolic_data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto output =
        m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), static_data, symbolic_data);
    m.add_return({output});

    auto expected = p;
    run_pass(p);

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_concat_past_present_is_noop)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m      = *p.get_main_module();
    auto present = m.add_parameter(
        "present",
        migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{lit(2)}, dd{n}, dd{lit(4)}}});
    auto seqlens = m.add_parameter("seqlens", {migraphx::shape::int32_type, {1}});
    auto past    = m.add_parameter(
        "past",
        migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{lit(2)}, dd{n}, dd{lit(4)}}});
    auto output = m.add_instruction(
        migraphx::make_op("concat_past_present", {{"kv_num_heads", 2}}), present, seqlens, past);
    m.add_return({output});

    auto expected = p;
    run_pass(p);

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_concat_past_present_static_cache_is_noop)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m      = *p.get_main_module();
    auto present = m.add_parameter(
        "present",
        migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{lit(2)}, dd{n}, dd{lit(4)}}});
    auto seqlens = m.add_parameter("seqlens", {migraphx::shape::int32_type, {1}});
    auto past    = m.add_parameter("past", {migraphx::shape::float_type, {1, 2, 4, 4}});
    auto output  = m.add_instruction(
        migraphx::make_op("concat_past_present", {{"kv_num_heads", 2}}), present, seqlens, past);
    m.add_return({output});

    auto expected = p;
    run_pass(p);

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_materializes_fill_range_and_scatter)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto data_buffer =
        m.add_parameter("data_buffer", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto update_buffer =
        m.add_parameter("update_buffer", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto zero =
        m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {0.0f}});
    auto one =
        m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {1.0f}});
    auto start =
        m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {0}});
    auto delta =
        m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {1}});
    auto limit   = m.add_instruction(migraphx::make_op("dimensions_of", {{"start", 0}, {"end", 1}}),
                                     data_buffer);
    auto indices = m.add_instruction(
        migraphx::make_op("dynamic_range", {{"output_dim", migraphx::to_value(dd{n})}}),
        start,
        limit,
        delta);
    indices      = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), indices);
    auto data    = m.add_instruction(migraphx::make_op("fill"), zero, data_buffer);
    auto updates = m.add_instruction(migraphx::make_op("fill"), one, update_buffer);
    auto output  = m.add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto zero = sm.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {0.0f}});
        auto one = sm.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {1.0f}});
        auto clone_start = sm.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {0}});
        auto clone_delta = sm.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {1}});
        auto clone_n = var("n", {clone.min, clone.max});
        sm.add_parameter("#split_sym_dim_input_0_1",
                         migraphx::shape{migraphx::shape::int64_type, {1}});
        auto clone_data = sm.add_parameter(
            "data_buffer", migraphx::shape{migraphx::shape::float_type, {dd{clone_n}}});
        auto clone_updates = sm.add_parameter(
            "update_buffer", migraphx::shape{migraphx::shape::float_type, {dd{clone_n}}});

        std::vector<int64_t> range_indices_data(clone.max);
        std::iota(range_indices_data.begin(), range_indices_data.end(), 0);
        auto range_indices = sm.add_literal(migraphx::literal{
            migraphx::shape{migraphx::shape::int64_type, {clone.max}}, range_indices_data});
        auto starts        = sm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {clone.max}}}), clone_start);
        auto deltas = sm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {clone.max}}}), clone_delta);
        range_indices = sm.add_instruction(migraphx::make_op("mul"), range_indices, deltas);
        range_indices = sm.add_instruction(migraphx::make_op("add"), starts, range_indices);
        range_indices =
            sm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), range_indices);

        auto padded_data =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_data);
        auto filled_data = sm.add_instruction(migraphx::make_op("fill"), zero, padded_data);
        auto padded_updates =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_updates);
        auto filled_updates = sm.add_instruction(migraphx::make_op("fill"), one, padded_updates);
        auto clone_output =
            add_expected_scatternd(sm, filled_data, range_indices, filled_updates, n, n, clone);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter(
        "data_buffer", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto expected_updates = expected_main.add_parameter(
        "update_buffer", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto expected_extent = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        expected_data);
    auto target_n = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select   = expected_main.add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes",
                            migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                                migraphx::shape{migraphx::shape::float_type, {dd{target_n}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_extent, expected_data, expected_updates},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_updates, expected_data});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_updates, expected_data});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_materializes_scatternd_with_static_indices)
{
    auto n = var("n", {1, 4});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto data =
        m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto indices = m.add_parameter("indices", {migraphx::shape::int64_type, {1, 1}});
    auto updates = m.add_parameter("updates", {migraphx::shape::float_type, {1, 4}});
    auto output  = m.add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_n    = var("n", {clone.min, clone.max});
        auto clone_data = sm.add_parameter(
            "data", migraphx::shape{migraphx::shape::float_type, {dd{clone_n}, dd{lit(4)}}});
        auto clone_indices = sm.add_parameter("indices", {migraphx::shape::int64_type, {1, 1}});
        auto clone_updates = sm.add_parameter("updates", {migraphx::shape::float_type, {1, 4}});
        auto padded_data =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_data);
        auto clone_output = sm.add_instruction(
            migraphx::make_op("scatternd_none"), padded_data, clone_indices, clone_updates);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto expected_indices =
        expected_main.add_parameter("indices", {migraphx::shape::int64_type, {1, 1}});
    auto expected_updates =
        expected_main.add_parameter("updates", {migraphx::shape::float_type, {1, 4}});
    auto target_n = var("split_sym_dim_n_target", {1, 4}, {1, 4});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type, {dd{target_n}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_data, expected_indices, expected_updates},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_updates, expected_indices, expected_data});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_updates, expected_indices, expected_data});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_rewrites_padded_scatter_indices)
{
    auto n = var("n", {1, 4});
    migraphx::program p;
    auto& m      = *p.get_main_module();
    auto data    = m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto indices = m.add_parameter(
        "indices", migraphx::shape{migraphx::shape::int64_type, {dd{n}, dd{lit(1)}}});
    auto updates =
        m.add_parameter("updates", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto output = m.add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_n = var("n", {clone.min, clone.max});
        auto clone_data =
            sm.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{clone_n}}});
        auto clone_indices = sm.add_parameter(
            "indices", migraphx::shape{migraphx::shape::int64_type, {dd{clone_n}, dd{lit(1)}}});
        auto clone_updates = sm.add_parameter(
            "updates", migraphx::shape{migraphx::shape::float_type, {dd{clone_n}}});
        auto padded_data =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_data);
        auto padded_indices =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_indices);
        auto padded_updates =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_updates);
        auto clone_output =
            add_expected_scatternd(sm, padded_data, padded_indices, padded_updates, n, n, clone);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data =
        expected_main.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto expected_indices = expected_main.add_parameter(
        "indices", migraphx::shape{migraphx::shape::int64_type, {dd{n}, dd{lit(1)}}});
    auto expected_updates = expected_main.add_parameter(
        "updates", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto target_n = var("split_sym_dim_n_target", {1, 4}, {1, 4});
    auto select   = expected_main.add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes",
                            migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                                migraphx::shape{migraphx::shape::float_type, {dd{target_n}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_data, expected_indices, expected_updates},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_updates, expected_indices, expected_data});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_updates, expected_indices, expected_data});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_rewrites_zero_depth_scatter_indices)
{
    auto m_dim = var("m", {1, 2});
    auto n     = var("n", {1, 4});
    migraphx::program p;
    auto& m   = *p.get_main_module();
    auto data = m.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{m_dim}, dd{lit(2)}}});
    auto indices = m.add_parameter(
        "indices", migraphx::shape{migraphx::shape::int64_type, {dd{n}, dd{lit(0)}}});
    auto updates = m.add_parameter(
        "updates", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{m_dim}, dd{lit(2)}}});
    auto output = m.add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
    m.add_return({output});

    run_pass(p);

    struct scatter_clone_spec
    {
        clone_spec n;
        clone_spec m;
    };

    migraphx::program expected;
    std::vector<scatter_clone_spec> clones = {
        {{1, 1}, {1, 1}}, {{1, 1}, {2, 2}}, {{2, 4}, {1, 1}}, {{2, 4}, {2, 2}}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_n    = var("n", {clone.n.min, clone.n.max});
        auto clone_m    = var("m", {clone.m.min, clone.m.max});
        auto clone_data = sm.add_parameter(
            "data", migraphx::shape{migraphx::shape::float_type, {dd{clone_m}, dd{lit(2)}}});
        auto clone_indices = sm.add_parameter(
            "indices", migraphx::shape{migraphx::shape::int64_type, {dd{clone_n}, dd{lit(0)}}});
        auto clone_updates = sm.add_parameter(
            "updates",
            migraphx::shape{migraphx::shape::float_type, {dd{clone_n}, dd{clone_m}, dd{lit(2)}}});
        auto padded_data =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_data);
        auto padded_updates =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_updates);
        auto clone_output =
            add_expected_scatternd(sm, padded_data, clone_indices, padded_updates, n, n, clone.n);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {dd{m_dim}, dd{lit(2)}}});
    auto expected_indices = expected_main.add_parameter(
        "indices", migraphx::shape{migraphx::shape::int64_type, {dd{n}, dd{lit(0)}}});
    auto expected_updates = expected_main.add_parameter(
        "updates", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{m_dim}, dd{lit(2)}}});
    auto target_m = var("split_sym_dim_m_target", {1, 2}, {1, 2});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type, {dd{target_m}, dd{lit(2)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_data, expected_indices, expected_updates},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_updates, expected_indices, expected_data});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{m_dim})}}),
        std::vector<migraphx::instruction_ref>{expected_updates, expected_indices, expected_data});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{m_dim})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_rewrites_scalar_zero_depth_scatter_indices)
{
    auto n = var("n", {1, 4});
    migraphx::program p;
    auto& m      = *p.get_main_module();
    auto data    = m.add_parameter("data", migraphx::shape{migraphx::shape::float_type});
    auto indices = m.add_parameter(
        "indices", migraphx::shape{migraphx::shape::int64_type, {dd{n}, dd{lit(0)}}});
    auto updates = m.add_parameter(
        "updates", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(1)}}});
    auto output = m.add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_n       = var("n", {clone.min, clone.max});
        auto clone_data    = sm.add_parameter("data", migraphx::shape{migraphx::shape::float_type});
        auto clone_indices = sm.add_parameter(
            "indices", migraphx::shape{migraphx::shape::int64_type, {dd{clone_n}, dd{lit(0)}}});
        auto clone_updates = sm.add_parameter(
            "updates", migraphx::shape{migraphx::shape::float_type, {dd{clone_n}, dd{lit(1)}}});
        auto padded_updates =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_updates);
        auto clone_output =
            add_expected_scatternd(sm, clone_data, clone_indices, padded_updates, n, n, clone);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data =
        expected_main.add_parameter("data", migraphx::shape{migraphx::shape::float_type});
    auto expected_indices = expected_main.add_parameter(
        "indices", migraphx::shape{migraphx::shape::int64_type, {dd{n}, dd{lit(0)}}});
    auto expected_updates = expected_main.add_parameter(
        "updates", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(1)}}});
    auto select = expected_main.add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes",
                            migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                                migraphx::shape{migraphx::shape::float_type, {1}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_data, expected_indices, expected_updates},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_materializes_attention_chain)
{
    auto sequence = var("sequence_length", {1, 4}, {2});
    migraphx::program p;
    auto& m  = *p.get_main_module();
    auto ids = m.add_parameter(
        "input_ids", migraphx::shape{migraphx::shape::int64_type, {dd{lit(1)}, dd{sequence}}});
    auto mask_buffer = m.add_parameter(
        "mask_buffer", migraphx::shape{migraphx::shape::float_type, {dd{sequence}}});
    auto update_buffer = m.add_parameter(
        "update_buffer", migraphx::shape{migraphx::shape::float_type, {dd{sequence}}});
    auto table  = m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}));
    auto hidden = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), table, ids);
    auto key =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), hidden);
    auto scores = m.add_instruction(migraphx::make_op("dot"), hidden, key);

    auto start =
        m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {0}});
    auto delta =
        m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {1}});
    auto limit =
        m.add_instruction(migraphx::make_op("dimensions_of", {{"start", 1}, {"end", 2}}), ids);
    auto indices = m.add_instruction(
        migraphx::make_op("dynamic_range", {{"output_dim", migraphx::to_value(dd{sequence})}}),
        start,
        limit,
        delta);
    indices = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), indices);
    auto zero =
        m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {0.0f}});
    auto one =
        m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {1.0f}});
    auto mask_data = m.add_instruction(migraphx::make_op("fill"), zero, mask_buffer);
    auto updates   = m.add_instruction(migraphx::make_op("fill"), one, update_buffer);
    auto mask = m.add_instruction(migraphx::make_op("scatternd_none"), mask_data, indices, updates);
    mask      = m.add_instruction(
        migraphx::make_op(
            "multibroadcast",
            {{"out_dyn_dims",
              migraphx::to_value(std::vector<dd>{dd{lit(1)}, dd{sequence}, dd{sequence}})}}),
        std::vector<migraphx::instruction_ref>{mask, scores});
    scores       = m.add_instruction(migraphx::make_op("add"), scores, mask);
    auto probs   = m.add_instruction(migraphx::make_op("softmax", {{"axis", 2}}), scores);
    auto context = m.add_instruction(migraphx::make_op("dot"), probs, hidden);
    auto output  = m.add_instruction(migraphx::make_op("relu"), context);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto negative_infinity =
            sm.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}},
                                             {-std::numeric_limits<float>::infinity()}});
        auto zero = sm.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {0.0f}});
        auto contracted_zero = sm.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {0.0f}});
        auto one = sm.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {1.0f}});
        auto clone_start = sm.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {0}});
        auto clone_delta = sm.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {1}});
        auto clone_table =
            sm.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}));
        auto clone_sequence = var("sequence_length", {clone.min, clone.max});
        sm.add_parameter("#split_sym_dim_input_0_2",
                         migraphx::shape{migraphx::shape::int64_type, {1}});
        auto clone_ids = sm.add_parameter(
            "input_ids",
            migraphx::shape{migraphx::shape::int64_type, {dd{lit(1)}, dd{clone_sequence}}});
        auto clone_mask_buffer = sm.add_parameter(
            "mask_buffer", migraphx::shape{migraphx::shape::float_type, {dd{clone_sequence}}});
        auto clone_update_buffer = sm.add_parameter(
            "update_buffer", migraphx::shape{migraphx::shape::float_type, {dd{clone_sequence}}});

        auto padded_ids =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_ids);
        auto clone_hidden =
            sm.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), clone_table, padded_ids);
        auto clone_key = sm.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), clone_hidden);
        auto clone_scores = sm.add_instruction(migraphx::make_op("dot"), clone_hidden, clone_key);

        std::vector<int64_t> range_indices_data(clone.max);
        std::iota(range_indices_data.begin(), range_indices_data.end(), 0);
        auto range_indices = sm.add_literal(migraphx::literal{
            migraphx::shape{migraphx::shape::int64_type, {clone.max}}, range_indices_data});
        auto starts        = sm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {clone.max}}}), clone_start);
        auto deltas = sm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {clone.max}}}), clone_delta);
        range_indices = sm.add_instruction(migraphx::make_op("mul"), range_indices, deltas);
        range_indices = sm.add_instruction(migraphx::make_op("add"), starts, range_indices);
        range_indices =
            sm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), range_indices);

        auto padded_mask_buffer = sm.add_instruction(
            migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_mask_buffer);
        auto mask_data = sm.add_instruction(migraphx::make_op("fill"), zero, padded_mask_buffer);
        auto padded_update_buffer = sm.add_instruction(
            migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_update_buffer);
        auto mask_updates =
            sm.add_instruction(migraphx::make_op("fill"), one, padded_update_buffer);
        auto clone_mask = add_expected_scatternd(
            sm, mask_data, range_indices, mask_updates, sequence, sequence, clone);
        clone_mask = sm.add_instruction(
            migraphx::make_op("multibroadcast",
                              {{"out_lens", {std::size_t{1}, clone.max, clone.max}}}),
            clone_mask);
        clone_scores = sm.add_instruction(migraphx::make_op("add"), clone_scores, clone_mask);

        std::vector<int64_t> softmax_indices_data(clone.max);
        std::iota(softmax_indices_data.begin(), softmax_indices_data.end(), 0);
        auto softmax_indices = sm.add_literal(migraphx::literal{
            migraphx::shape{migraphx::shape::int64_type, {clone.max}}, softmax_indices_data});
        auto softmax_extent  = add_expected_clone_extent(sm, sequence, sequence, clone, clone_ids);
        clone_scores =
            add_mask(sm, clone_scores, softmax_indices, softmax_extent, negative_infinity, 2);
        auto probabilities =
            sm.add_instruction(migraphx::make_op("softmax", {{"axis", 2}}), clone_scores);
        auto masked_hidden =
            add_mask(sm, clone_hidden, softmax_indices, softmax_extent, contracted_zero, 1);
        auto context = sm.add_instruction(migraphx::make_op("dot"), probabilities, masked_hidden);
        auto clone_output = sm.add_instruction(migraphx::make_op("relu"), context);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_ids   = expected_main.add_parameter(
        "input_ids", migraphx::shape{migraphx::shape::int64_type, {dd{lit(1)}, dd{sequence}}});
    auto expected_mask_buffer = expected_main.add_parameter(
        "mask_buffer", migraphx::shape{migraphx::shape::float_type, {dd{sequence}}});
    auto expected_update_buffer = expected_main.add_parameter(
        "update_buffer", migraphx::shape{migraphx::shape::float_type, {dd{sequence}}});
    auto expected_extent = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{sequence})}}),
        expected_ids);
    auto target_sequence = var("split_sym_dim_sequence_length_target", {1, 4}, {1, 2, 4});
    auto select          = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{migraphx::shape{
                  migraphx::shape::float_type, {dd{lit(1)}, dd{target_sequence}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{
            expected_extent, expected_ids, expected_mask_buffer, expected_update_buffer},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{
            expected_update_buffer, expected_mask_buffer, expected_ids});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{sequence})}}),
        std::vector<migraphx::instruction_ref>{
            expected_update_buffer, expected_mask_buffer, expected_ids});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{1}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{sequence})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_clones_fixed_shape_dependencies_into_cases)
{
    auto sequence = var("sequence_length", {1, 4}, {2});
    migraphx::program p;
    auto& m    = *p.get_main_module();
    auto input = m.add_parameter(
        "input", migraphx::shape{migraphx::shape::float_type, {dd{sequence}, dd{lit(4)}}});
    auto dims =
        m.add_instruction(migraphx::make_op("dimensions_of", {{"start", 1}, {"end", 2}}), input);
    auto extent = m.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), dims);
    auto scale = m.add_instruction(migraphx::make_op("sqrt"), extent);
    scale      = m.add_instruction(
        migraphx::make_op(
            "multibroadcast",
            {{"out_dyn_dims", migraphx::to_value(std::vector<dd>{dd{sequence}, dd{lit(4)}})}}),
        std::vector<migraphx::instruction_ref>{scale, input});
    auto output = m.add_instruction(migraphx::make_op("mul"), input, scale);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_sequence = var("sequence_length", {clone.min, clone.max});
        auto clone_input    = sm.add_parameter(
            "input",
            migraphx::shape{migraphx::shape::float_type, {dd{clone_sequence}, dd{lit(4)}}});
        auto clone_extent = sm.add_parameter("#split_sym_dim_input_0_0",
                                             migraphx::shape{migraphx::shape::int64_type, {1}});
        auto clone_scale  = sm.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}),
            clone_extent);
        clone_scale = sm.add_instruction(migraphx::make_op("sqrt"), clone_scale);
        clone_scale = sm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {clone.max, std::size_t{4}}}}),
            clone_scale);
        auto padded_input =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_input);
        auto clone_output = sm.add_instruction(migraphx::make_op("mul"), padded_input, clone_scale);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_input = expected_main.add_parameter(
        "input", migraphx::shape{migraphx::shape::float_type, {dd{sequence}, dd{lit(4)}}});
    auto expected_extent = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(4)})}}),
        expected_input);
    auto target_sequence = var("split_sym_dim_sequence_length_target", {1, 4}, {1, 2, 4});
    auto select          = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{migraphx::shape{
                  migraphx::shape::float_type, {dd{target_sequence}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_extent, expected_input},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{sequence})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{sequence})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_resolves_symbolic_shape_queries_without_model_compute)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m    = *p.get_main_module();
    auto ids   = m.add_parameter("ids", migraphx::shape{migraphx::shape::int64_type, {dd{n}}});
    auto table = m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}));
    auto gathered = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), table, ids);
    auto dims =
        m.add_instruction(migraphx::make_op("dimensions_of", {{"start", 0}, {"end", 2}}), gathered);
    m.add_return({dims});

    run_pass(p);

    migraphx::program expected;
    auto& expected_main = *expected.get_main_module();
    auto expected_ids =
        expected_main.add_parameter("ids", migraphx::shape{migraphx::shape::int64_type, {dd{n}}});
    auto expected_dims = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n, lit(4)})}}),
        expected_ids);
    expected_main.add_return({expected_dims});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_resolves_eval_expr_from_intermediate_instruction)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m    = *p.get_main_module();
    auto ids   = m.add_parameter("ids", migraphx::shape{migraphx::shape::int64_type, {dd{n}}});
    auto table = m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}));
    auto gathered = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), table, ids);
    auto dims     = m.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n, lit(4)})}}),
        gathered);
    m.add_return({dims});

    run_pass(p);

    migraphx::program expected;
    auto& expected_main = *expected.get_main_module();
    auto expected_ids =
        expected_main.add_parameter("ids", migraphx::shape{migraphx::shape::int64_type, {dd{n}}});
    auto expected_dims = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n, lit(4)})}}),
        expected_ids);
    expected_main.add_return({expected_dims});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_preserves_shape_query_without_direct_root)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto input =
        m.add_parameter("input", migraphx::shape{migraphx::shape::float_type, {dd{n + 1}}});
    auto dims =
        m.add_instruction(migraphx::make_op("dimensions_of", {{"start", 0}, {"end", 1}}), input);
    m.add_return({dims});

    auto expected = p;
    run_pass(p);
    EXPECT(p == expected);
}

TEST_CASE(split_sym_dim_applies_clone_cap_per_block)
{
    auto n     = var("n", {1, 4}, {2});
    auto m_dim = var("m", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x0 =
        m.add_parameter("x0", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto x1 = m.add_parameter(
        "x1", migraphx::shape{migraphx::shape::float_type, {dd{m_dim}, dd{lit(4)}}});
    auto y0 = m.add_instruction(migraphx::make_op("relu"), x0);
    auto y1 = m.add_instruction(migraphx::make_op("relu"), x1);
    m.add_return({y0, y1});

    run_pass(p, 4);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto n_modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto input =
            sm.add_parameter("x0",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{var("n", {clone.min, clone.max})}, dd{lit(4)}}});
        auto pad    = sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), input);
        auto output = sm.add_instruction(migraphx::make_op("relu"), pad);
        sm.add_return({output});
    });
    auto m_modules = add_clones(expected, 1, clones, [&](auto& sm, const auto& clone) {
        auto input =
            sm.add_parameter("x1",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{var("m", {clone.min, clone.max})}, dd{lit(4)}}});
        auto pad    = sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), input);
        auto output = sm.add_instruction(migraphx::make_op("relu"), pad);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_x0    = expected_main.add_parameter(
        "x0", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto expected_x1 = expected_main.add_parameter(
        "x1", migraphx::shape{migraphx::shape::float_type, {dd{m_dim}, dd{lit(4)}}});
    auto runtime_sources = std::vector<migraphx::instruction_ref>{expected_x1, expected_x0};

    auto target_n = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select_n = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type, {dd{target_n}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_x0},
        n_modules);
    auto output_n = expected_main.add_instruction(
        migraphx::make_op("get_tuple_elem", {{"index", 0}}), select_n);
    auto output_n_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        runtime_sources);
    auto output_n_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        runtime_sources);
    output_n = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        output_n,
        output_n_start,
        output_n_end);

    auto target_m = var("split_sym_dim_m_target", {1, 4}, {1, 2, 4});
    auto select_m = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type, {dd{target_m}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_x1},
        m_modules);
    auto output_m = expected_main.add_instruction(
        migraphx::make_op("get_tuple_elem", {{"index", 0}}), select_m);
    auto output_m_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        runtime_sources);
    auto output_m_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{m_dim})}}),
        runtime_sources);
    output_m = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{m_dim})}}),
        output_m,
        output_m_start,
        output_m_end);
    expected_main.add_return({output_n, output_m});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_mixed_range_parameter_is_noop)
{
    auto n = var("n", {1, 4}, {2, 4});
    migraphx::program p;
    auto& m       = *p.get_main_module();
    auto symbolic = m.add_parameter(
        "symbolic", migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto ranged =
        m.add_parameter("ranged", migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
    auto y0 = m.add_instruction(migraphx::make_op("relu"), symbolic);
    auto y1 = m.add_instruction(migraphx::make_op("relu"), ranged);
    m.add_return({y0, y1});

    auto expected = p;
    run_pass(p);
    EXPECT(p == expected);
}

TEST_CASE(split_sym_dim_keeps_literals_in_clones)
{
    auto n = var("n", {1, 4}, {2, 4});
    migraphx::program p;
    auto& m    = *p.get_main_module();
    auto input = m.add_parameter(
        "data",
        migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(1)}, dd{lit(5)}, dd{lit(5)}}});
    auto weights =
        m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 1, 3, 3}}));
    auto conv = m.add_instruction(migraphx::make_op("convolution"), input, weights);
    m.add_return({conv});
    run_pass(p);

    migraphx::program expected;
    migraphx::shape weights_shape{migraphx::shape::float_type, {1, 1, 3, 3}};
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_input = sm.add_parameter(
            "data",
            migraphx::shape{
                migraphx::shape::float_type,
                {dd{var("n", {clone.min, clone.max})}, dd{lit(1)}, dd{lit(5)}, dd{lit(5)}}});
        auto clone_weights = sm.add_literal(migraphx::generate_literal(weights_shape));
        auto pad =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_input);
        auto output = sm.add_instruction(migraphx::make_op("convolution"), pad, clone_weights);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_input = expected_main.add_parameter(
        "data",
        migraphx::shape{migraphx::shape::float_type, {dd{n}, dd{lit(1)}, dd{lit(5)}, dd{lit(5)}}});
    auto target_n = var("split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select   = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type,
                                  {dd{target_n}, dd{lit(1)}, dd{lit(3)}, dd{lit(3)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_input},
        modules);
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    auto output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        output,
        output_start,
        output_end);
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_coalesces_spatial_cnn)
{
    auto spatial = var("spatial", {8, 16}, {12});

    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x  = m.add_parameter("image",
                              migraphx::shape{migraphx::shape::float_type,
                                              {dd{lit(1)}, dd{lit(3)}, dd{spatial}, dd{spatial}}});
    auto w0 =
        m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 3, 3, 3}}));
    auto w1 =
        m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {6, 4, 3, 3}}));

    auto conv0 = m.add_instruction(migraphx::make_op("convolution"), x, w0);
    auto relu0 = m.add_instruction(migraphx::make_op("relu"), conv0);
    auto conv1 = m.add_instruction(migraphx::make_op("convolution"), relu0, w1);
    auto relu1 = m.add_instruction(migraphx::make_op("relu"), conv1);
    auto pool  = m.add_instruction(migraphx::make_op("pooling",
                                                     {{"mode", migraphx::op::pooling_mode::max},
                                                      {"padding", {0, 0}},
                                                      {"stride", {2, 2}},
                                                      {"lengths", {2, 2}}}),
                                   relu1);
    m.add_return({pool});

    run_pass(p);

    migraphx::program expected;
    migraphx::shape w0_shape{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape w1_shape{migraphx::shape::float_type, {6, 4, 3, 3}};
    std::vector<clone_spec> clones = {{8, 8}, {9, 12}, {13, 16}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_spatial = var("spatial", {clone.min, clone.max});
        auto input         = sm.add_parameter(
            "image",
            migraphx::shape{migraphx::shape::float_type,
                            {dd{lit(1)}, dd{lit(3)}, dd{clone_spatial}, dd{clone_spatial}}});
        auto clone_w0 = sm.add_literal(migraphx::generate_literal(w0_shape));
        auto clone_w1 = sm.add_literal(migraphx::generate_literal(w1_shape));
        auto pad   = sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), input);
        auto conv0 = sm.add_instruction(migraphx::make_op("convolution"), pad, clone_w0);
        auto relu0 = sm.add_instruction(migraphx::make_op("relu"), conv0);
        auto conv1 = sm.add_instruction(migraphx::make_op("convolution"), relu0, clone_w1);
        auto relu1 = sm.add_instruction(migraphx::make_op("relu"), conv1);
        auto output =
            sm.add_instruction(migraphx::make_op("pooling",
                                                 {{"mode", migraphx::op::pooling_mode::max},
                                                  {"padding", {0, 0}},
                                                  {"stride", {2, 2}},
                                                  {"lengths", {2, 2}}}),
                               relu1);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto input          = expected_main.add_parameter(
        "image",
        migraphx::shape{migraphx::shape::float_type,
                        {dd{lit(1)}, dd{lit(3)}, dd{spatial}, dd{spatial}}});
    auto target_spatial = var("split_sym_dim_spatial_target", {8, 16}, {8, 12, 16});
    auto output_extent  = (target_spatial - 6) / 2 + 1;
    auto select         = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{migraphx::shape{
                  migraphx::shape::float_type,
                  {dd{lit(1)}, dd{lit(6)}, dd{output_extent}, dd{output_extent}}}}})}}),
        std::vector<migraphx::instruction_ref>{input},
        modules);
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto real_extent  = (spatial - 6) / 2 + 1;
    auto output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0), lit(0)})}}),
        std::vector<migraphx::instruction_ref>{input});
    auto output_end = expected_main.add_instruction(
        migraphx::make_op(
            "eval_expr_from_shape",
            {{"expressions", migraphx::to_value(std::vector<se>{real_extent, real_extent})}}),
        std::vector<migraphx::instruction_ref>{input});
    output = expected_main.add_instruction(
        migraphx::make_op(
            "dyn_slice",
            {{"axes", std::vector<int64_t>{2, 3}},
             {"starts", migraphx::to_value(std::vector<se>{lit(0), lit(0)})},
             {"ends", migraphx::to_value(std::vector<se>{real_extent, real_extent})}}),
        output,
        output_start,
        output_end);
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_preserves_compound_mask_extent)
{
    auto sequence = var("sequence", {4, 16}, {8});

    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x  = m.add_parameter(
        "x", migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{lit(1)}, dd{sequence}}});
    auto weights =
        m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 1, 3}}));
    auto conv = m.add_instruction(
        migraphx::make_op("convolution", {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
        x,
        weights);
    auto output = m.add_instruction(migraphx::make_op("softmax", {{"axis", 2}}), conv);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    migraphx::shape weights_shape{migraphx::shape::float_type, {1, 1, 3}};
    std::vector<clone_spec> clones = {{4, 4}, {5, 8}, {9, 16}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto negative_infinity =
            sm.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}},
                                             {-std::numeric_limits<float>::infinity()}});
        std::vector<int64_t> indices_data(clone.max - 2);
        std::iota(indices_data.begin(), indices_data.end(), 0);
        auto indices        = sm.add_literal(migraphx::literal{
            migraphx::shape{migraphx::shape::int64_type, {clone.max - 2}}, indices_data});
        auto clone_sequence = var("sequence", {clone.min, clone.max});
        auto input =
            sm.add_parameter("x",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{lit(1)}, dd{lit(1)}, dd{clone_sequence}}});
        auto clone_weights = sm.add_literal(migraphx::generate_literal(weights_shape));
        auto pad = sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), input);
        auto convolution = sm.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
            pad,
            clone_weights);
        auto extent       = add_expected_clone_extent(sm, sequence - 2, sequence, clone, input);
        auto masked       = add_mask(sm, convolution, indices, extent, negative_infinity, 2);
        auto clone_output = sm.add_instruction(migraphx::make_op("softmax", {{"axis", 2}}), masked);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_input = expected_main.add_parameter(
        "x", migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{lit(1)}, dd{sequence}}});
    auto target_sequence = var("split_sym_dim_sequence_target", {4, 16}, {4, 8, 16});
    auto target_extent   = target_sequence - 2;
    auto select          = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{migraphx::shape{
                  migraphx::shape::float_type, {dd{lit(1)}, dd{lit(1)}, dd{target_extent}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_input},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{sequence - 2})}}),
        std::vector<migraphx::instruction_ref>{expected_input});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{2}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{sequence - 2})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_freezes_fixed_roots_in_mask_extents)
{
    auto sequence = var("sequence", {4, 8}, {6});
    auto kernel   = var("kernel", {3, 3});
    migraphx::program p;
    auto& m    = *p.get_main_module();
    auto input = m.add_parameter(
        "input",
        migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{lit(1)}, dd{sequence}}});
    auto weights = m.add_parameter(
        "weights",
        migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{lit(1)}, dd{kernel}}});
    auto conv = m.add_instruction(
        migraphx::make_op("convolution", {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
        input,
        weights);
    auto output = m.add_instruction(migraphx::make_op("softmax", {{"axis", 2}}), conv);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    migraphx::shape weights_shape{migraphx::shape::float_type, {1, 1, 3}};
    std::vector<clone_spec> clones = {{4, 4}, {5, 6}, {7, 8}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto negative_infinity =
            sm.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}},
                                             {-std::numeric_limits<float>::infinity()}});
        std::vector<int64_t> indices_data(clone.max - 2);
        std::iota(indices_data.begin(), indices_data.end(), 0);
        auto indices        = sm.add_literal(migraphx::literal{
            migraphx::shape{migraphx::shape::int64_type, {clone.max - 2}}, indices_data});
        auto clone_sequence = var("sequence", {clone.min, clone.max});
        auto clone_weights  = sm.add_parameter("weights", weights_shape);
        auto clone_input =
            sm.add_parameter("input",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{lit(1)}, dd{lit(1)}, dd{clone_sequence}}});
        auto padded_input =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_input);
        auto padded_weights =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_weights);
        auto convolution = sm.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
            padded_input,
            padded_weights);

        migraphx::instruction_ref extent;
        if(clone.min == clone.max)
        {
            extent = sm.add_literal(migraphx::literal{
                migraphx::shape{migraphx::shape::int64_type, {1}}, {int64_t(clone.min - 2)}});
        }
        else
        {
            extent = sm.add_instruction(
                migraphx::make_op(
                    "eval_expr_from_shape",
                    {{"expressions", migraphx::to_value(std::vector<se>{sequence - 2})}}),
                clone_weights,
                clone_input);
        }
        auto masked       = add_mask(sm, convolution, indices, extent, negative_infinity, 2);
        auto clone_output = sm.add_instruction(migraphx::make_op("softmax", {{"axis", 2}}), masked);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_input = expected_main.add_parameter(
        "input",
        migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{lit(1)}, dd{sequence}}});
    auto expected_weights = expected_main.add_parameter(
        "weights",
        migraphx::shape{migraphx::shape::float_type, {dd{lit(1)}, dd{lit(1)}, dd{kernel}}});
    auto target_sequence = var("split_sym_dim_sequence_target", {4, 8}, {4, 6, 8});
    auto target_extent   = target_sequence - 2;
    auto select          = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{migraphx::shape{
                  migraphx::shape::float_type, {dd{lit(1)}, dd{lit(1)}, dd{target_extent}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_input, expected_weights},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_weights, expected_input});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op(
            "eval_expr_from_shape",
            {{"expressions", migraphx::to_value(std::vector<se>{-kernel + sequence + 1})}}),
        std::vector<migraphx::instruction_ref>{expected_weights, expected_input});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{2}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{-kernel + sequence + 1})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_specializes_transformer_core)
{
    auto sequence = var("sequence", {4, 16}, {8});
    auto c2       = lit(2);
    auto c8       = lit(8);

    migraphx::program p;
    auto& m = *p.get_main_module();
    auto q  = m.add_parameter(
        "query", migraphx::shape{migraphx::shape::float_type, {dd{c2}, dd{sequence}, dd{c8}}});
    auto kt = m.add_parameter(
        "key_transposed",
        migraphx::shape{migraphx::shape::float_type, {dd{c2}, dd{c8}, dd{sequence}}});
    auto v = m.add_parameter(
        "value", migraphx::shape{migraphx::shape::float_type, {dd{c2}, dd{sequence}, dd{c8}}});
    auto scores   = m.add_instruction(migraphx::make_op("dot"), q, kt);
    auto context  = m.add_instruction(migraphx::make_op("dot"), scores, v);
    auto residual = m.add_instruction(migraphx::make_op("add"), context, q);
    auto output   = m.add_instruction(migraphx::make_op("relu"), residual);
    m.add_return({output});
    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{4, 4}, {5, 8}, {9, 16}};
    auto modules        = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto zero = sm.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {0.0f}});
        std::vector<int64_t> indices_data(clone.max);
        std::iota(indices_data.begin(), indices_data.end(), 0);
        auto indices        = sm.add_literal(migraphx::literal{
            migraphx::shape{migraphx::shape::int64_type, {clone.max}}, indices_data});
        auto clone_sequence = var("sequence", {clone.min, clone.max});
        auto value =
            sm.add_parameter("value",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{lit(2)}, dd{clone_sequence}, dd{lit(8)}}});
        auto query =
            sm.add_parameter("query",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{lit(2)}, dd{clone_sequence}, dd{lit(8)}}});
        auto key = sm.add_parameter("key_transposed",
                                    migraphx::shape{migraphx::shape::float_type,
                                                    {dd{lit(2)}, dd{lit(8)}, dd{clone_sequence}}});

        auto padded_query =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), query);
        auto padded_key =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), key);
        auto scores = sm.add_instruction(migraphx::make_op("dot"), padded_query, padded_key);
        auto padded_value =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), value);
        auto extent        = add_expected_clone_extent(sm, sequence, sequence, clone, query);
        auto masked_scores = add_mask(sm, scores, indices, extent, zero, 2);
        auto masked_value  = add_mask(sm, padded_value, indices, extent, zero, 1);
        auto context  = sm.add_instruction(migraphx::make_op("dot"), masked_scores, masked_value);
        auto residual = sm.add_instruction(migraphx::make_op("add"), context, padded_query);
        auto output   = sm.add_instruction(migraphx::make_op("relu"), residual);
        sm.add_return({output});
    });
    auto& expected_main = *expected.get_main_module();
    auto query          = expected_main.add_parameter(
        "query",
        migraphx::shape{migraphx::shape::float_type, {dd{lit(2)}, dd{sequence}, dd{lit(8)}}});
    auto key = expected_main.add_parameter(
        "key_transposed",
        migraphx::shape{migraphx::shape::float_type, {dd{lit(2)}, dd{lit(8)}, dd{sequence}}});
    auto value = expected_main.add_parameter(
        "value",
        migraphx::shape{migraphx::shape::float_type, {dd{lit(2)}, dd{sequence}, dd{lit(8)}}});
    auto target_sequence = var("split_sym_dim_sequence_target", {4, 16}, {4, 8, 16});
    auto select          = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{migraphx::shape{
                  migraphx::shape::float_type, {dd{lit(2)}, dd{target_sequence}, dd{lit(8)}}}}})}}),
        std::vector<migraphx::instruction_ref>{key, query, value},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{value, key, query});
    auto output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{sequence})}}),
        std::vector<migraphx::instruction_ref>{value, key, query});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{1}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{sequence})}}),
        expected_output,
        output_start,
        output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_specializes_transformer)
{
    auto sequence = var("sequence", {4, 16}, {8});
    auto c2       = lit(2);
    auto c8       = lit(8);

    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x  = m.add_parameter(
        "x", migraphx::shape{migraphx::shape::float_type, {dd{c2}, dd{sequence}, dd{c8}}});
    migraphx::shape input_weight_shape{migraphx::shape::float_type, {2, 8, 8}};
    auto wq = m.add_literal(migraphx::generate_literal(input_weight_shape, 0));
    auto wk = m.add_literal(migraphx::generate_literal(input_weight_shape, 1));
    auto wv = m.add_literal(migraphx::generate_literal(input_weight_shape, 2));
    auto wo = m.add_literal(migraphx::generate_literal(input_weight_shape, 3));

    auto q  = m.add_instruction(migraphx::make_op("dot"), x, wq);
    auto k  = m.add_instruction(migraphx::make_op("dot"), x, wk);
    auto v  = m.add_instruction(migraphx::make_op("dot"), x, wv);
    auto kt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), k);
    auto scores  = m.add_instruction(migraphx::make_op("dot"), q, kt);
    auto probs   = m.add_instruction(migraphx::make_op("softmax", {{"axis", 2}}), scores);
    auto context = m.add_instruction(migraphx::make_op("dot"), probs, v);
    auto proj    = m.add_instruction(migraphx::make_op("dot"), context, wo);
    auto output  = m.add_instruction(migraphx::make_op("add"), proj, x);
    m.add_return({output});
    run_pass(p);

    migraphx::program expected;
    migraphx::shape weight_shape{migraphx::shape::float_type, {2, 8, 8}};
    std::vector<clone_spec> clones = {{4, 4}, {5, 8}, {9, 16}};
    auto modules        = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto zero = sm.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {0.0f}});
        auto negative_infinity =
            sm.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}},
                                             {-std::numeric_limits<float>::infinity()}});
        std::vector<int64_t> indices_data(clone.max);
        std::iota(indices_data.begin(), indices_data.end(), 0);
        auto indices        = sm.add_literal(migraphx::literal{
            migraphx::shape{migraphx::shape::int64_type, {clone.max}}, indices_data});
        auto clone_sequence = var("sequence", {clone.min, clone.max});
        auto input =
            sm.add_parameter("x",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{lit(2)}, dd{clone_sequence}, dd{lit(8)}}});
        auto wq = sm.add_literal(migraphx::generate_literal(weight_shape, 0));
        auto wk = sm.add_literal(migraphx::generate_literal(weight_shape, 1));
        auto wv = sm.add_literal(migraphx::generate_literal(weight_shape, 2));
        auto wo = sm.add_literal(migraphx::generate_literal(weight_shape, 3));

        auto padded = sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), input);
        auto query  = sm.add_instruction(migraphx::make_op("dot"), padded, wq);
        auto key    = sm.add_instruction(migraphx::make_op("dot"), padded, wk);
        auto value  = sm.add_instruction(migraphx::make_op("dot"), padded, wv);
        auto key_transposed =
            sm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), key);
        auto scores        = sm.add_instruction(migraphx::make_op("dot"), query, key_transposed);
        auto extent        = add_expected_clone_extent(sm, sequence, sequence, clone, input);
        auto masked_scores = add_mask(sm, scores, indices, extent, negative_infinity, 2);
        auto probabilities =
            sm.add_instruction(migraphx::make_op("softmax", {{"axis", 2}}), masked_scores);
        auto masked_value = add_mask(sm, value, indices, extent, zero, 1);
        auto context    = sm.add_instruction(migraphx::make_op("dot"), probabilities, masked_value);
        auto projection = sm.add_instruction(migraphx::make_op("dot"), context, wo);
        auto output     = sm.add_instruction(migraphx::make_op("add"), projection, padded);
        sm.add_return({output});
    });
    auto& expected_main = *expected.get_main_module();
    auto input          = expected_main.add_parameter(
        "x", migraphx::shape{migraphx::shape::float_type, {dd{lit(2)}, dd{sequence}, dd{lit(8)}}});
    auto target_sequence = var("split_sym_dim_sequence_target", {4, 16}, {4, 8, 16});
    auto select          = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{migraphx::shape{
                  migraphx::shape::float_type, {dd{lit(2)}, dd{target_sequence}, dd{lit(8)}}}}})}}),
        std::vector<migraphx::instruction_ref>{input},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{input});
    auto output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{sequence})}}),
        std::vector<migraphx::instruction_ref>{input});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{1}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{sequence})}}),
        expected_output,
        output_start,
        output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_masks_contraction_operands_independently)
{
    auto sequence = var("sequence", {1, 8}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto scores =
        m.add_parameter("scores",
                        migraphx::shape{migraphx::shape::float_type,
                                        {dd{lit(1)}, dd{lit(1)}, dd{sequence}, dd{sequence}}});
    auto a = m.add_parameter("a",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{lit(1)}, dd{lit(1)}, dd{sequence}, dd{lit(4)}}});
    auto b = m.add_parameter("b",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{lit(1)}, dd{lit(1)}, dd{sequence}, dd{lit(4)}}});
    auto probabilities = m.add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), scores);
    auto values        = m.add_instruction(migraphx::make_op("div"), a, b);
    auto output        = m.add_instruction(migraphx::make_op("dot"), probabilities, values);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 8}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto zero = sm.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {0.0f}});
        auto negative_infinity =
            sm.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}},
                                             {-std::numeric_limits<float>::infinity()}});
        std::vector<int64_t> indices_data(clone.max);
        std::iota(indices_data.begin(), indices_data.end(), 0);
        auto indices        = sm.add_literal(migraphx::literal{
            migraphx::shape{migraphx::shape::int64_type, {clone.max}}, indices_data});
        auto clone_sequence = var("sequence", {clone.min, clone.max});
        auto clone_a        = sm.add_parameter(
            "a",
            migraphx::shape{migraphx::shape::float_type,
                            {dd{lit(1)}, dd{lit(1)}, dd{clone_sequence}, dd{lit(4)}}});
        auto clone_b = sm.add_parameter(
            "b",
            migraphx::shape{migraphx::shape::float_type,
                            {dd{lit(1)}, dd{lit(1)}, dd{clone_sequence}, dd{lit(4)}}});
        auto clone_scores = sm.add_parameter(
            "scores",
            migraphx::shape{migraphx::shape::float_type,
                            {dd{lit(1)}, dd{lit(1)}, dd{clone_sequence}, dd{clone_sequence}}});

        auto padded_scores =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_scores);
        auto extent        = add_expected_clone_extent(sm, sequence, sequence, clone, clone_scores);
        auto masked_scores = add_mask(sm, padded_scores, indices, extent, negative_infinity, 3);
        auto clone_probabilities =
            sm.add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), masked_scores);
        auto padded_a =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_a);
        auto padded_b =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_b);
        auto clone_values  = sm.add_instruction(migraphx::make_op("div"), padded_a, padded_b);
        auto masked_values = add_mask(sm, clone_values, indices, extent, zero, 2);
        auto clone_output =
            sm.add_instruction(migraphx::make_op("dot"), clone_probabilities, masked_values);
        sm.add_return({clone_output});
    });

    auto& expected_main  = *expected.get_main_module();
    auto expected_scores = expected_main.add_parameter(
        "scores",
        migraphx::shape{migraphx::shape::float_type,
                        {dd{lit(1)}, dd{lit(1)}, dd{sequence}, dd{sequence}}});
    auto expected_a = expected_main.add_parameter(
        "a",
        migraphx::shape{migraphx::shape::float_type,
                        {dd{lit(1)}, dd{lit(1)}, dd{sequence}, dd{lit(4)}}});
    auto expected_b = expected_main.add_parameter(
        "b",
        migraphx::shape{migraphx::shape::float_type,
                        {dd{lit(1)}, dd{lit(1)}, dd{sequence}, dd{lit(4)}}});
    auto target_sequence = var("split_sym_dim_sequence_target", {1, 8}, {1, 2, 8});
    auto select          = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                  migraphx::shape{migraphx::shape::float_type,
                                  {dd{lit(1)}, dd{lit(1)}, dd{target_sequence}, dd{lit(4)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_a, expected_b, expected_scores},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_b, expected_a, expected_scores});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{sequence})}}),
        std::vector<migraphx::instruction_ref>{expected_b, expected_a, expected_scores});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{2}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{sequence})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_keeps_softmax_mask_off_contract_axis)
{
    auto sequence = var("sequence", {4, 16}, {8});
    auto c2       = lit(2);
    auto c8       = lit(8);

    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x  = m.add_parameter(
        "x", migraphx::shape{migraphx::shape::float_type, {dd{c2}, dd{sequence}, dd{sequence}}});
    auto value = m.add_parameter(
        "value", migraphx::shape{migraphx::shape::float_type, {dd{c2}, dd{sequence}, dd{c8}}});
    auto probs  = m.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), x);
    auto output = m.add_instruction(migraphx::make_op("dot"), probs, value);
    m.add_return({output});
    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{4, 4}, {5, 8}, {9, 16}};
    auto modules        = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto zero = sm.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {0.0f}});
        auto negative_infinity =
            sm.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}},
                                             {-std::numeric_limits<float>::infinity()}});
        std::vector<int64_t> indices_data(clone.max);
        std::iota(indices_data.begin(), indices_data.end(), 0);
        auto indices        = sm.add_literal(migraphx::literal{
            migraphx::shape{migraphx::shape::int64_type, {clone.max}}, indices_data});
        auto clone_sequence = var("sequence", {clone.min, clone.max});
        auto input =
            sm.add_parameter("x",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{lit(2)}, dd{clone_sequence}, dd{clone_sequence}}});
        auto value =
            sm.add_parameter("value",
                             migraphx::shape{migraphx::shape::float_type,
                                             {dd{lit(2)}, dd{clone_sequence}, dd{lit(8)}}});

        auto padded_input =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), input);
        auto extent       = add_expected_clone_extent(sm, sequence, sequence, clone, input);
        auto masked_input = add_mask(sm, padded_input, indices, extent, negative_infinity, 1);
        auto probabilities =
            sm.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), masked_input);
        auto padded_value =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), value);
        auto masked_probabilities = add_mask(sm, probabilities, indices, extent, zero, 2);
        auto masked_value         = add_mask(sm, padded_value, indices, extent, zero, 1);
        auto output =
            sm.add_instruction(migraphx::make_op("dot"), masked_probabilities, masked_value);
        sm.add_return({output});
    });
    auto& expected_main = *expected.get_main_module();
    auto input          = expected_main.add_parameter(
        "x",
        migraphx::shape{migraphx::shape::float_type, {dd{lit(2)}, dd{sequence}, dd{sequence}}});
    auto expected_value = expected_main.add_parameter(
        "value",
        migraphx::shape{migraphx::shape::float_type, {dd{lit(2)}, dd{sequence}, dd{lit(8)}}});
    auto target_sequence = var("split_sym_dim_sequence_target", {4, 16}, {4, 8, 16});
    auto select          = expected_main.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{migraphx::shape{
                  migraphx::shape::float_type, {dd{lit(2)}, dd{target_sequence}, dd{lit(8)}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_value, input},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_value, input});
    auto output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{sequence})}}),
        std::vector<migraphx::instruction_ref>{expected_value, input});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{1}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{sequence})}}),
        expected_output,
        output_start,
        output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_keeps_reduction_of_padded_body_outside_clone)
{
    auto n = var("n", {1, 8}, {2});
    migraphx::program p;
    auto& m       = *p.get_main_module();
    auto x        = m.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto exp      = m.add_instruction(migraphx::make_op("exp"), x);
    auto sum      = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), exp);
    auto expanded = m.add_instruction(
        migraphx::make_op("multibroadcast",
                          {{"out_dyn_dims", migraphx::to_value(std::vector<dd>{dd{n}})}}),
        std::vector<migraphx::instruction_ref>{sum, x});
    auto output = m.add_instruction(migraphx::make_op("sub"), exp, expanded);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 8}};
    auto exp_modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_n = var("n", {clone.min, clone.max});
        auto clone_x =
            sm.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {dd{clone_n}}});
        auto padded_x =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_x);
        auto clone_output = sm.add_instruction(migraphx::make_op("exp"), padded_x);
        sm.add_return({clone_output});
    });
    auto sub_modules = add_clones(expected, 1, clones, [&](auto& sm, const auto& clone) {
        auto clone_n      = var("n", {clone.min, clone.max});
        auto clone_extent = migraphx::sym::min(clone_n, lit(clone.max));
        sm.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {dd{clone_n}}});
        auto clone_exp =
            sm.add_parameter("#split_sym_dim_input_1_1",
                             migraphx::shape{migraphx::shape::float_type, {dd{clone_extent}}});
        clone_exp =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_exp);
        auto clone_reduce =
            sm.add_parameter("#split_sym_dim_input_1_0",
                             migraphx::shape{migraphx::shape::float_type, {dd{clone_extent}}});
        clone_reduce =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_reduce);
        auto clone_sum =
            sm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), clone_reduce);
        auto clone_expanded = sm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {clone.max}}}), clone_sum);
        auto clone_output = sm.add_instruction(migraphx::make_op("sub"), clone_exp, clone_expanded);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_x =
        expected_main.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto target_n   = var("split_sym_dim_n_target", {1, 8}, {1, 2, 8});
    auto select_exp = expected_main.add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes",
                            migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                                migraphx::shape{migraphx::shape::float_type, {dd{target_n}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_x},
        exp_modules);
    auto expected_exp = expected_main.add_instruction(
        migraphx::make_op("get_tuple_elem", {{"index", 0}}), select_exp);
    auto expected_exp_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_x});
    auto expected_exp_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_x});
    expected_exp = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_exp,
        expected_exp_start,
        expected_exp_end);
    auto select_sub = expected_main.add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes",
                            migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                                migraphx::shape{migraphx::shape::float_type, {dd{target_n}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_exp, expected_exp, expected_x},
        sub_modules);
    auto expected_output = expected_main.add_instruction(
        migraphx::make_op("get_tuple_elem", {{"index", 0}}), select_sub);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_x});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_x});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_keeps_unsupported_dependency_outside_clone)
{
    auto n = var("n", {1, 8}, {2});
    migraphx::program p;
    auto& m       = *p.get_main_module();
    auto x        = m.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto concat   = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, x);
    auto sum      = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), concat);
    auto expanded = m.add_instruction(
        migraphx::make_op("multibroadcast",
                          {{"out_dyn_dims", migraphx::to_value(std::vector<dd>{dd{n}})}}),
        std::vector<migraphx::instruction_ref>{sum, x});
    auto output = m.add_instruction(migraphx::make_op("sub"), x, expanded);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 8}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_n = var("n", {clone.min, clone.max});
        auto clone_x =
            sm.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {dd{clone_n}}});
        clone_x = sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_x);
        auto clone_concat =
            sm.add_parameter("#split_sym_dim_input_0_0",
                             migraphx::shape{migraphx::shape::float_type, {dd{lit(2) * clone_n}}});
        clone_concat =
            sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), clone_concat);
        auto clone_sum =
            sm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), clone_concat);
        auto clone_expanded = sm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {clone.max}}}), clone_sum);
        auto clone_output = sm.add_instruction(migraphx::make_op("sub"), clone_x, clone_expanded);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_x =
        expected_main.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto expected_concat = expected_main.add_instruction(
        migraphx::make_op("concat", {{"axis", 0}}), expected_x, expected_x);
    auto target_n = var("split_sym_dim_n_target", {1, 8}, {1, 2, 8});
    auto select   = expected_main.add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes",
                            migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                                migraphx::shape{migraphx::shape::float_type, {dd{target_n}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_concat, expected_x},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_x});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_x});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_memoizes_unchanged_shared_dependencies)
{
    auto n = var("n", {1, 8}, {2});
    migraphx::program p;
    auto& m          = *p.get_main_module();
    auto x           = m.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto output      = m.add_instruction(migraphx::make_op("relu"), x);
    auto static_path = m.add_parameter(
        "static_path", migraphx::shape{migraphx::shape::float_type, {std::size_t{4}}});
    static_path = add_residual_diamonds(m, static_path, 50);
    m.add_return({output, static_path});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 8}};
    auto modules        = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_n = var("n", {clone.min, clone.max});
        auto input =
            sm.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {dd{clone_n}}});
        input       = sm.add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), input);
        auto result = sm.add_instruction(migraphx::make_op("relu"), input);
        sm.add_return({result});
    });
    auto& expected_main = *expected.get_main_module();
    auto expected_x =
        expected_main.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
    auto expected_static_input = expected_main.add_parameter(
        "static_path", migraphx::shape{migraphx::shape::float_type, {std::size_t{4}}});
    auto expected_static_path = add_residual_diamonds(expected_main, expected_static_input, 50);
    auto target_n             = var("split_sym_dim_n_target", {1, 8}, {1, 2, 8});
    auto select               = expected_main.add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes",
                            migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{
                                migraphx::shape{migraphx::shape::float_type, {dd{target_n}}}}})}}),
        std::vector<migraphx::instruction_ref>{expected_x},
        modules);
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto expected_output_start = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{lit(0)})}}),
        std::vector<migraphx::instruction_ref>{expected_static_input, expected_x});
    auto expected_output_end = expected_main.add_instruction(
        migraphx::make_op("eval_expr_from_shape",
                          {{"expressions", migraphx::to_value(std::vector<se>{n})}}),
        std::vector<migraphx::instruction_ref>{expected_static_input, expected_x});
    expected_output = expected_main.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", std::vector<int64_t>{0}},
                           {"starts", migraphx::to_value(std::vector<se>{lit(0)})},
                           {"ends", migraphx::to_value(std::vector<se>{n})}}),
        expected_output,
        expected_output_start,
        expected_output_end);
    expected_main.add_return({expected_output, expected_static_path});

    EXPECT(p.sort() == expected.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
