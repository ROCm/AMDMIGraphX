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
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
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
#include <string>
#include <unordered_set>
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

migraphx::shape symbolic_shape(std::initializer_list<se> dims,
                               migraphx::shape::type_t type = migraphx::shape::float_type)
{
    return {type, std::vector<dd>(dims.begin(), dims.end())};
}

void run_pass(migraphx::program& p, std::size_t max_clones = 64)
{
    migraphx::run_passes(p,
                         {migraphx::split_sym_dim{max_clones}, migraphx::dead_code_elimination{}});
}

migraphx::operation fixed_pad(float value = 0.0f)
{
    return migraphx::make_op("fixed_pad", {{"value", value}});
}

migraphx::instruction_ref add_symbolic_multibroadcast(
    migraphx::module& m,
    std::initializer_list<se> dims,
    migraphx::instruction_ref input,
    migraphx::instruction_ref shape_input,
    std::initializer_list<migraphx::instruction_ref> additional_shape_inputs = {})
{
    std::vector<dd> output_dims(dims.begin(), dims.end());
    std::vector<migraphx::instruction_ref> inputs = {input, shape_input};
    inputs.insert(inputs.end(), additional_shape_inputs);
    return m.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_dyn_dims", migraphx::to_value(output_dims)}}),
        inputs);
}

migraphx::operation symbolic_broadcast(std::size_t axis, std::initializer_list<se> dims)
{
    std::vector<dd> output_dims(dims.begin(), dims.end());
    return migraphx::make_op("broadcast",
                             {{"axis", axis}, {"out_dyn_dims", migraphx::to_value(output_dims)}});
}

migraphx::operation symbolic_broadcast_with_dims(std::initializer_list<se> dims)
{
    std::vector<dd> output_dims(dims.begin(), dims.end());
    return migraphx::make_op("broadcast_with_dims",
                             {{"out_dyn_dims", migraphx::to_value(output_dims)}});
}

migraphx::instruction_ref add_select_module(migraphx::module& m,
                                            const std::vector<migraphx::instruction_ref>& inputs,
                                            const std::vector<migraphx::module_ref>& modules,
                                            const std::vector<migraphx::shape>& output_shapes)
{
    return m.add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes", migraphx::to_value(migraphx::shape{output_shapes})}}),
        inputs,
        modules);
}

migraphx::instruction_ref add_back_slice(migraphx::module& m,
                                         migraphx::instruction_ref input,
                                         const std::vector<migraphx::instruction_ref>& sources,
                                         const std::vector<int64_t>& axes,
                                         const std::vector<se>& ends)
{
    std::vector<se> starts(ends.size(), lit(0));
    std::vector<int64_t> start_values(starts.size(), 0);
    auto start = m.add_literal(
        migraphx::literal{{migraphx::shape::int64_type, {start_values.size()}}, start_values});
    std::unordered_set<se> required;
    for(const auto& expression : ends)
    {
        auto variables = migraphx::sym::find_variables(expression);
        required.insert(variables.begin(), variables.end());
    }
    std::vector<se> ordered_variables(required.begin(), required.end());
    std::sort(ordered_variables.begin(), ordered_variables.end(), [](const auto& x, const auto& y) {
        return x.to_string() < y.to_string();
    });
    std::vector<migraphx::instruction_ref> expression_sources;
    for(const auto& variable : ordered_variables)
    {
        for(const auto& name : m.get_parameter_names())
        {
            auto source = m.get_parameter(name);
            if(migraphx::contains(expression_sources, source) or
               not migraphx::contains(sources, source) or not source->get_shape().symbolic())
                continue;
            if(migraphx::any_of(source->get_shape().dyn_dims(), [&](const auto& d) {
                   return d.is_symbolic() and d.sym_expr.name() == "variable" and
                          migraphx::sym::as_symbol(d.sym_expr) == variable;
               }))
            {
                expression_sources.push_back(source);
                break;
            }
        }
    }
    auto end = m.add_instruction(
        migraphx::make_op("eval_expr_from_shape", {{"expressions", migraphx::to_value(ends)}}),
        expression_sources);
    auto result = m.add_instruction(migraphx::make_op("dyn_slice",
                                                      {{"axes", axes},
                                                       {"starts", migraphx::to_value(starts)},
                                                       {"ends", migraphx::to_value(ends)},
                                                       {"always_leq", true}}),
                                    input,
                                    start,
                                    end);
    result->set_normalized();
    return result;
}

migraphx::instruction_ref add_iota(migraphx::module& m, std::size_t elements)
{
    std::vector<int64_t> data(elements);
    std::iota(data.begin(), data.end(), 0);
    return m.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {elements}}, data});
}

migraphx::instruction_ref add_fill(migraphx::module& m, float value)
{
    return m.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {value}});
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

template <class F>
void for_each_clone_instruction(migraphx::program& p, F f)
{
    for(auto* mod : p.get_modules())
    {
        if(mod == p.get_main_module())
            continue;
        for(auto&& ins : *mod)
            f(ins);
    }
}

migraphx::program make_relu_program(const se& n)
{
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x  = m.add_parameter("data", symbolic_shape({n, lit(4)}));
    auto y  = m.add_instruction(migraphx::make_op("relu"), x);
    m.add_return({y});
    return p;
}

migraphx::program make_transformer_program()
{
    auto sequence = var("sequence", {4, 16}, {8});
    auto c2       = lit(2);
    auto c8       = lit(8);

    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x  = m.add_parameter("x", symbolic_shape({c2, sequence, c8}));
    migraphx::shape weight_shape{migraphx::shape::float_type, {2, 8, 8}};
    auto wq = m.add_literal(migraphx::generate_literal(weight_shape, 0));
    auto wk = m.add_literal(migraphx::generate_literal(weight_shape, 1));
    auto wv = m.add_literal(migraphx::generate_literal(weight_shape, 2));
    auto wo = m.add_literal(migraphx::generate_literal(weight_shape, 3));

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
    return p;
}

migraphx::program make_transformer_core_program(const se& sequence)
{
    auto c2 = lit(2);
    auto c8 = lit(8);

    migraphx::program p;
    auto& m       = *p.get_main_module();
    auto q        = m.add_parameter("query", symbolic_shape({c2, sequence, c8}));
    auto kt       = m.add_parameter("key_transposed", symbolic_shape({c2, c8, sequence}));
    auto v        = m.add_parameter("value", symbolic_shape({c2, sequence, c8}));
    auto scores   = m.add_instruction(migraphx::make_op("dot"), q, kt);
    auto context  = m.add_instruction(migraphx::make_op("dot"), scores, v);
    auto residual = m.add_instruction(migraphx::make_op("add"), context, q);
    auto output   = m.add_instruction(migraphx::make_op("relu"), residual);
    m.add_return({output});
    return p;
}

migraphx::program make_softmax_off_contract_axis_program()
{
    auto sequence = var("sequence", {4, 16}, {8});
    auto c2       = lit(2);
    auto c8       = lit(8);

    migraphx::program p;
    auto& m     = *p.get_main_module();
    auto x      = m.add_parameter("x", symbolic_shape({c2, sequence, sequence}));
    auto value  = m.add_parameter("value", symbolic_shape({c2, sequence, c8}));
    auto probs  = m.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), x);
    auto output = m.add_instruction(migraphx::make_op("dot"), probs, value);
    m.add_return({output});
    return p;
}

} // namespace

TEST_CASE(split_sym_dim_covers_full_interval)
{
    auto n = var("n", {1, 8}, {2, 4});
    auto p = make_relu_program(n);
    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}, {5, 8}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto input =
            sm.add_parameter("data", symbolic_shape({var("n", {clone.min, clone.max}), lit(4)}));
        auto pad    = sm.add_instruction(fixed_pad(), input);
        auto output = sm.add_instruction(migraphx::make_op("relu"), pad);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto input          = expected_main.add_parameter("data", symbolic_shape({n, lit(4)}));
    auto target_n       = var("_split_sym_dim_n_target", {1, 8}, {1, 2, 4, 8});
    auto select =
        add_select_module(expected_main, {input}, modules, {symbolic_shape({target_n, lit(4)})});
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    output = add_back_slice(expected_main, output, {input}, {0}, {n});
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_supports_one_to_one_axis_transforms)
{
    auto n = var("n", {1, 8}, {2, 4});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x  = m.add_parameter("data", symbolic_shape({lit(1), n, lit(4)}));
    x       = m.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), x);
    x       = m.add_instruction(migraphx::make_op("reshape", {{"dims", {0, 0}}}), x);
    x       = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), x);
    m.add_return({x});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}, {5, 8}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto input = sm.add_parameter(
            "data", symbolic_shape({lit(1), var("n", {clone.min, clone.max}), lit(4)}));
        auto output = sm.add_instruction(fixed_pad(), input);
        output      = sm.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), output);
        output      = sm.add_instruction(migraphx::make_op("reshape", {{"dims", {0, 0}}}), output);
        output =
            sm.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), output);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto input          = expected_main.add_parameter("data", symbolic_shape({lit(1), n, lit(4)}));
    auto target_n       = var("_split_sym_dim_n_target", {1, 8}, {1, 2, 4, 8});
    migraphx::shape output_shape{
        migraphx::shape::float_type, {dd{lit(4)}, dd{target_n}}, {lit(1), lit(4)}};
    auto select = add_select_module(expected_main, {input}, modules, {output_shape});
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    output = add_back_slice(expected_main, output, {input}, {1}, {n});
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_splits_at_nonparallel_symbolic_reshape)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m      = *p.get_main_module();
    auto data    = m.add_parameter("data", symbolic_shape({n, lit(4)}));
    auto target  = m.add_parameter("target", symbolic_shape({lit(4), n}));
    auto reshape = m.add_instruction(migraphx::make_op("reshape"), data, target);
    auto output  = m.add_instruction(migraphx::make_op("relu"), reshape);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_n = var("n", {clone.min, clone.max});
        auto input =
            sm.add_parameter("#split_sym_dim_input_0_0", symbolic_shape({lit(4), clone_n}));
        sm.add_parameter("data", symbolic_shape({clone_n, lit(4)}));
        auto padded = sm.add_instruction(fixed_pad(), input);
        auto result = sm.add_instruction(migraphx::make_op("relu"), padded);
        sm.add_return({result});
    });

    auto& expected_main  = *expected.get_main_module();
    auto expected_data   = expected_main.add_parameter("data", symbolic_shape({n, lit(4)}));
    auto expected_target = expected_main.add_parameter("target", symbolic_shape({lit(4), n}));
    auto expected_reshape =
        expected_main.add_instruction(migraphx::make_op("reshape"), expected_data, expected_target);
    auto target_n = var("_split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select   = add_select_module(expected_main,
                                      {expected_reshape, expected_data},
                                    modules,
                                      {symbolic_shape({lit(4), target_n})});
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    expected_output =
        add_back_slice(expected_main, expected_output, {expected_target, expected_data}, {1}, {n});
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_materializes_symbolic_multibroadcast)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m     = *p.get_main_module();
    auto data   = m.add_parameter("data", symbolic_shape({n, lit(3)}));
    auto bias   = m.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {3}});
    auto bcast  = add_symbolic_multibroadcast(m, {n, lit(3)}, bias, data);
    auto output = m.add_instruction(migraphx::make_op("add"), data, bcast);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_bias =
            sm.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {3}});
        auto clone_data =
            sm.add_parameter("data", symbolic_shape({var("n", {clone.min, clone.max}), lit(3)}));
        auto clone_bcast = sm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {clone.max, 3}}}), clone_bias);
        auto padded_data  = sm.add_instruction(fixed_pad(), clone_data);
        auto clone_output = sm.add_instruction(migraphx::make_op("add"), padded_data, clone_bcast);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter("data", symbolic_shape({n, lit(3)}));
    auto expected_bias =
        expected_main.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {3}});
    auto target_n = var("_split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select   = add_select_module(expected_main,
                                      {expected_bias, expected_data},
                                    modules,
                                      {symbolic_shape({target_n, lit(3)})});
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    expected_output =
        add_back_slice(expected_main, expected_output, {expected_bias, expected_data}, {0}, {n});
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_materializes_symbolic_multibroadcast_with_multiple_shape_inputs)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m   = *p.get_main_module();
    auto data = m.add_parameter("data", symbolic_shape({n, lit(3)}));
    auto shape_input =
        m.add_parameter("shape_input", migraphx::shape{migraphx::shape::float_type, {1, 3}});
    auto bias   = m.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {3}});
    auto bcast  = add_symbolic_multibroadcast(m, {n, lit(3)}, bias, data, {shape_input});
    auto output = m.add_instruction(migraphx::make_op("add"), data, bcast);
    m.add_return({output});

    run_pass(p);

    std::size_t static_broadcasts = 0;
    for_each_clone_instruction(p, [&](auto& ins) {
        if(ins.name() != "multibroadcast")
            return;
        EXPECT(ins.inputs().size() == 1);
        EXPECT(not ins.get_shape().dynamic());
        ++static_broadcasts;
    });
    EXPECT(static_broadcasts == 3);
}

TEST_CASE(split_sym_dim_absorbs_fixed_symbolic_multibroadcast)
{
    auto fixed_batch = var("fixed_batch", {1, 1});
    auto sequence    = var("sequence", {1, 4}, {2});
    migraphx::program p;
    auto& m      = *p.get_main_module();
    auto data    = m.add_parameter("data", symbolic_shape({fixed_batch, sequence, lit(4)}));
    auto weights = m.add_parameter("weights", migraphx::shape{migraphx::shape::float_type, {4, 4}});
    auto target  = m.add_parameter("target", symbolic_shape({fixed_batch, lit(4), lit(4)}));
    auto broadcast = add_symbolic_multibroadcast(m, {fixed_batch, lit(4), lit(4)}, weights, target);
    auto output    = m.add_instruction(migraphx::make_op("dot"), data, broadcast);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_batch    = var("fixed_batch", {1, 1});
        auto clone_sequence = var("sequence", {clone.min, clone.max});
        auto clone_data =
            sm.add_parameter("data", symbolic_shape({clone_batch, clone_sequence, lit(4)}));
        sm.add_parameter("target", migraphx::shape{migraphx::shape::float_type, {1, 4, 4}});
        auto clone_weights =
            sm.add_parameter("weights", migraphx::shape{migraphx::shape::float_type, {4, 4}});
        auto clone_broadcast = sm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 4, 4}}}), clone_weights);
        auto padded_data      = sm.add_instruction(fixed_pad(), clone_data);
        auto padded_broadcast = sm.add_instruction(fixed_pad(), clone_broadcast);
        auto clone_output =
            sm.add_instruction(migraphx::make_op("dot"), padded_data, padded_broadcast);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data =
        expected_main.add_parameter("data", symbolic_shape({fixed_batch, sequence, lit(4)}));
    auto expected_weights = expected_main.add_parameter(
        "weights", migraphx::shape{migraphx::shape::float_type, {4, 4}});
    auto expected_target =
        expected_main.add_parameter("target", symbolic_shape({fixed_batch, lit(4), lit(4)}));
    auto target_sequence = var("_split_sym_dim_sequence_target", {1, 4}, {1, 2, 4});
    auto select          = add_select_module(expected_main,
                                             {expected_data, expected_target, expected_weights},
                                    modules,
                                             {symbolic_shape({lit(1), target_sequence, lit(4)})});
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    expected_output = add_back_slice(expected_main,
                                     expected_output,
                                     {expected_target, expected_weights, expected_data},
                                     {1},
                                     {sequence});
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_materializes_symbolic_broadcast)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m     = *p.get_main_module();
    auto data   = m.add_parameter("data", symbolic_shape({n, lit(3), lit(4)}));
    auto bias   = m.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {3}});
    auto bcast  = m.add_instruction(symbolic_broadcast(1, {n, lit(3), lit(4)}), bias, data);
    auto output = m.add_instruction(migraphx::make_op("add"), data, bcast);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_bias =
            sm.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {3}});
        auto clone_data = sm.add_parameter(
            "data", symbolic_shape({var("n", {clone.min, clone.max}), lit(3), lit(4)}));
        auto clone_bcast = sm.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {clone.max, 3, 4}}}),
            clone_bias);
        auto padded_data  = sm.add_instruction(fixed_pad(), clone_data);
        auto clone_output = sm.add_instruction(migraphx::make_op("add"), padded_data, clone_bcast);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter("data", symbolic_shape({n, lit(3), lit(4)}));
    auto expected_bias =
        expected_main.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {3}});
    auto target_n = var("_split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select   = add_select_module(expected_main,
                                      {expected_bias, expected_data},
                                    modules,
                                      {symbolic_shape({target_n, lit(3), lit(4)})});
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    expected_output =
        add_back_slice(expected_main, expected_output, {expected_bias, expected_data}, {0}, {n});
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_materializes_symbolic_broadcast_with_dims)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m   = *p.get_main_module();
    auto data = m.add_parameter("data", symbolic_shape({lit(1), lit(1), n, n}));
    auto dims =
        m.add_parameter("dims", migraphx::shape{migraphx::shape::int64_type, {std::size_t{4}}});
    auto output =
        m.add_instruction(symbolic_broadcast_with_dims({lit(1), lit(1), n, n}), data, dims);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_data   = sm.add_parameter("data",
                                           symbolic_shape({lit(1),
                                                             lit(1),
                                                             var("n", {clone.min, clone.max}),
                                                             var("n", {clone.min, clone.max})}));
        auto padded_data  = sm.add_instruction(fixed_pad(), clone_data);
        auto clone_output = sm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, clone.max, clone.max}}}),
            padded_data);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data =
        expected_main.add_parameter("data", symbolic_shape({lit(1), lit(1), n, n}));
    auto expected_dims = expected_main.add_parameter(
        "dims", migraphx::shape{migraphx::shape::int64_type, {std::size_t{4}}});
    auto target_n = var("_split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    migraphx::shape output_shape{migraphx::shape::float_type,
                                 {dd{lit(1)}, dd{lit(1)}, dd{target_n}, dd{target_n}},
                                 {target_n * target_n, target_n * target_n, target_n, lit(1)}};
    auto select = add_select_module(expected_main, {expected_data}, modules, {output_shape});
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    expected_output = add_back_slice(
        expected_main, expected_output, {expected_dims, expected_data}, {2, 3}, {n, n});
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_coalesces_into_symbolic_broadcast_with_dims)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m   = *p.get_main_module();
    auto data = m.add_parameter("data", symbolic_shape({lit(1), n}));
    auto dims =
        m.add_parameter("dims", migraphx::shape{migraphx::shape::int64_type, {std::size_t{3}}});
    auto relu   = m.add_instruction(migraphx::make_op("relu"), data);
    auto output = m.add_instruction(symbolic_broadcast_with_dims({lit(1), n, n}), relu, dims);
    m.add_return({output});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_data =
            sm.add_parameter("data", symbolic_shape({lit(1), var("n", {clone.min, clone.max})}));
        auto padded_data  = sm.add_instruction(fixed_pad(), clone_data);
        auto clone_relu   = sm.add_instruction(migraphx::make_op("relu"), padded_data);
        auto clone_output = sm.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, clone.max, clone.max}}}),
            clone_relu);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter("data", symbolic_shape({lit(1), n}));
    auto expected_dims  = expected_main.add_parameter(
        "dims", migraphx::shape{migraphx::shape::int64_type, {std::size_t{3}}});
    auto target_n = var("_split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    migraphx::shape output_shape{migraphx::shape::float_type,
                                 {dd{lit(1)}, dd{target_n}, dd{target_n}},
                                 {lit(0), lit(0), lit(1)}};
    auto select = add_select_module(expected_main, {expected_data}, modules, {output_shape});
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    expected_output = add_back_slice(
        expected_main, expected_output, {expected_dims, expected_data}, {1, 2}, {n, n});
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_materializes_symbolic_allocate)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    m.add_parameter("source", symbolic_shape({n}));
    auto dims =
        m.add_parameter("dims", migraphx::shape{migraphx::shape::int64_type, {std::size_t{2}}});
    auto output_shape = symbolic_shape({n, lit(4)});
    auto output       = m.add_instruction(
        migraphx::make_op("allocate", {{"shape", migraphx::to_value(output_shape)}}), dims);
    m.add_return({output});

    run_pass(p);

    std::size_t static_allocations = 0;
    for_each_clone_instruction(p, [&](auto& ins) {
        if(ins.name() != "allocate")
            return;
        EXPECT(ins.inputs().empty());
        EXPECT(not ins.get_shape().dynamic());
        ++static_allocations;
    });
    EXPECT(static_allocations == 3);
    EXPECT(p.get_output_shapes().front().symbolic());
}

TEST_CASE(split_sym_dim_materializes_two_input_symbolic_reshape)
{
    auto batch = var("batch", {1, 1});
    auto n     = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m      = *p.get_main_module();
    auto data    = m.add_parameter("data", symbolic_shape({batch, n, lit(16)}));
    auto target  = m.add_parameter("target", symbolic_shape({lit(1), n, lit(4), lit(4)}));
    auto reshape = m.add_instruction(migraphx::make_op("reshape"), data, target);
    m.add_return({reshape});

    run_pass(p);

    std::size_t static_reshapes = 0;
    for_each_clone_instruction(p, [&](auto& ins) {
        if(ins.name() != "reshape")
            return;
        EXPECT(ins.inputs().size() == 1);
        EXPECT(not ins.get_shape().dynamic());
        EXPECT(not ins.inputs().front()->get_shape().dynamic());
        ++static_reshapes;
    });
    EXPECT(static_reshapes == 3);
    EXPECT(p.get_output_shapes().front().symbolic());
}

TEST_CASE(split_sym_dim_absorbs_fixed_symbolic_reshape)
{
    auto fixed_batch = var("fixed_batch", {1, 1});
    auto sequence    = var("sequence", {1, 4}, {2});
    migraphx::program p;
    auto& m      = *p.get_main_module();
    auto data    = m.add_parameter("data", symbolic_shape({fixed_batch, sequence, lit(4)}));
    auto weights = m.add_parameter("weights", symbolic_shape({fixed_batch, lit(16)}));
    auto target  = m.add_parameter("target", symbolic_shape({fixed_batch, lit(4), lit(4)}));
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
            sm.add_parameter("data", symbolic_shape({clone_batch, clone_sequence, lit(4)}));
        sm.add_parameter("target", migraphx::shape{migraphx::shape::float_type, {1, 4, 4}});
        auto clone_weights =
            sm.add_parameter("weights", migraphx::shape{migraphx::shape::float_type, {1, 16}});
        auto clone_reshape =
            sm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 4}}}), clone_weights);
        auto padded_data    = sm.add_instruction(fixed_pad(), clone_data);
        auto padded_reshape = sm.add_instruction(fixed_pad(), clone_reshape);
        auto clone_output =
            sm.add_instruction(migraphx::make_op("dot"), padded_data, padded_reshape);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data =
        expected_main.add_parameter("data", symbolic_shape({fixed_batch, sequence, lit(4)}));
    auto expected_weights =
        expected_main.add_parameter("weights", symbolic_shape({fixed_batch, lit(16)}));
    auto expected_target =
        expected_main.add_parameter("target", symbolic_shape({fixed_batch, lit(4), lit(4)}));
    auto target_sequence = var("_split_sym_dim_sequence_target", {1, 4}, {1, 2, 4});
    auto select          = add_select_module(expected_main,
                                             {expected_data, expected_target, expected_weights},
                                    modules,
                                             {symbolic_shape({lit(1), target_sequence, lit(4)})});
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    expected_output = add_back_slice(expected_main,
                                     expected_output,
                                     {expected_target, expected_weights, expected_data},
                                     {1},
                                     {sequence});
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_coalesces_across_absorbable_symbolic_reshape)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m      = *p.get_main_module();
    auto data    = m.add_parameter("data", symbolic_shape({lit(4), n}));
    auto target  = m.add_parameter("target", symbolic_shape({n, lit(4)}));
    auto before  = m.add_instruction(migraphx::make_op("relu"), data);
    auto reshape = m.add_instruction(migraphx::make_op("reshape"), before, target);
    auto output  = m.add_instruction(migraphx::make_op("relu"), reshape);
    m.add_return({output});

    run_pass(p);

    std::size_t selects         = 0;
    std::size_t static_reshapes = 0;
    std::size_t static_relus    = 0;
    for(auto* mod : p.get_modules())
    {
        for(auto&& ins : *mod)
        {
            if(mod == p.get_main_module() and ins.name() == "select_module")
                ++selects;
            if(mod != p.get_main_module() and ins.name() == "reshape")
            {
                EXPECT(not ins.get_shape().dynamic());
                ++static_reshapes;
            }
            if(mod != p.get_main_module() and ins.name() == "relu")
            {
                EXPECT(not ins.get_shape().dynamic());
                ++static_relus;
            }
        }
    }
    EXPECT(selects == 1);
    EXPECT(static_reshapes == 3);
    EXPECT(static_relus == 6);
}

TEST_CASE(split_sym_dim_retries_affected_connected_edge)
{
    auto n = var("n", {1, 2}, {1, 2});
    auto m = var("m", {1, 2}, {1, 2});
    auto q = var("q", {1, 2}, {1, 2});
    migraphx::program p;
    auto& main    = *p.get_main_module();
    auto input    = main.add_parameter("input", symbolic_shape({n, lit(4)}));
    auto other    = main.add_parameter("other", symbolic_shape({m, lit(4)}));
    auto q_source = main.add_parameter("q_source", symbolic_shape({lit(4), q}));
    auto before   = main.add_instruction(migraphx::make_op("relu"), input);
    other         = main.add_instruction(migraphx::make_op("relu"), other);
    auto reduced  = main.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), before);
    auto expanded =
        main.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), reduced);
    expanded    = add_symbolic_multibroadcast(main, {lit(4), q}, expanded, q_source);
    auto packed = main.add_instruction(migraphx::make_op("contiguous"), expanded);
    auto output = main.add_instruction(migraphx::make_op("dot"), before, packed);
    main.add_return({output, other});

    run_pass(p, 4);

    std::size_t selects      = 0;
    std::size_t static_relus = 0;
    for(auto&& ins : *p.get_main_module())
        if(ins.name() == "select_module")
            ++selects;
    for_each_clone_instruction(p, [&](auto& ins) {
        if(ins.name() == "relu")
            ++static_relus;
    });
    EXPECT(selects == 2);
    EXPECT(static_relus == 6);
}

TEST_CASE(split_sym_dim_preserves_two_input_reshape_target_layout)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m   = *p.get_main_module();
    auto data = m.add_parameter("data", symbolic_shape({n, lit(4)}));
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

        auto clone_weights  = sm.add_parameter("weights", clone_weights_shape);
        auto clone_data     = sm.add_parameter("data", clone_data_shape);
        auto padded_data    = sm.add_instruction(fixed_pad(), clone_data);
        auto padded_weights = sm.add_instruction(fixed_pad(), clone_weights);
        auto clone_output =
            sm.add_instruction(migraphx::make_op("dot"), padded_data, padded_weights);
        sm.add_return({clone_output});
    });

    auto& expected_main   = *expected.get_main_module();
    auto expected_data    = expected_main.add_parameter("data", data_shape);
    auto expected_weights = expected_main.add_parameter("weights", weights_shape);
    auto target_n         = var("_split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select           = add_select_module(expected_main,
                                              {expected_data, expected_weights},
                                    modules,
                                              {symbolic_shape({target_n, lit(4)})});
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    expected_output =
        add_back_slice(expected_main, expected_output, {expected_weights, expected_data}, {0}, {n});
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_keeps_variable_stride_dependency_at_boundary)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m   = *p.get_main_module();
    auto data = m.add_parameter("data", symbolic_shape({n, lit(4)}, migraphx::shape::float_type));
    migraphx::shape weights_shape{
        migraphx::shape::float_type, {dd{lit(4)}, dd{lit(4)}}, {n + 4, lit(1)}};
    auto weights  = m.add_parameter("weights", weights_shape);
    auto identity = m.add_instruction(migraphx::make_op("identity"), weights);
    auto output   = m.add_instruction(migraphx::make_op("dot"), data, identity);
    m.add_return({output});

    run_pass(p);

    std::size_t main_identities  = 0;
    std::size_t clone_identities = 0;
    std::size_t static_dots      = 0;
    for(auto* mod : p.get_modules())
    {
        for(auto&& ins : *mod)
        {
            if(ins.name() == "identity")
            {
                if(mod == p.get_main_module())
                    ++main_identities;
                else
                    ++clone_identities;
            }
            if(mod != p.get_main_module() and ins.name() == "dot")
            {
                EXPECT(not ins.get_shape().dynamic());
                ++static_dots;
            }
        }
    }
    EXPECT(main_identities == 1);
    EXPECT(clone_identities == 0);
    EXPECT(static_dots == 3);
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

    EXPECT(p.get_output_shapes().front().standard());
    std::size_t static_relus = 0;
    for_each_clone_instruction(p, [&](auto& ins) {
        if(ins.name() != "relu")
            return;
        EXPECT(not ins.get_shape().dynamic());
        EXPECT(ins.get_shape().standard());
        ++static_relus;
    });
    EXPECT(static_relus == 3);

    p.compile(migraphx::make_target("ref"));
    std::vector<float> logical_values{
        -1.0f, 2.0f, -3.0f, 4.0f, 5.0f, -6.0f, 7.0f, -8.0f, 9.0f, 10.0f, -11.0f, 12.0f};
    std::vector<float> transposed_values(logical_values.size());
    for(std::size_t i = 0; i < 3; ++i)
        for(std::size_t j = 0; j < 4; ++j)
            transposed_values[i + j * 3] = logical_values[i * 4 + j];
    migraphx::parameter_map params;
    params["input"] = migraphx::argument{
        migraphx::shape{migraphx::shape::float_type, {3, 4}, {1, 3}}, transposed_values.data()};
    auto result = p.eval(params).back();
    EXPECT(result.get_shape().standard());
    std::vector<float> result_values;
    result.visit([&](auto values) { result_values.assign(values.begin(), values.end()); });
    std::transform(logical_values.begin(),
                   logical_values.end(),
                   logical_values.begin(),
                   [](auto x) { return std::max(x, 0.0f); });
    EXPECT(result_values == logical_values);
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

    EXPECT(p.get_output_shapes().front().transposed());
    std::size_t static_transposes         = 0;
    std::size_t non_degenerate_transposes = 0;
    for_each_clone_instruction(p, [&](auto& ins) {
        if(ins.name() != "transpose")
            return;
        EXPECT(not ins.get_shape().dynamic());
        ++static_transposes;
        if(ins.get_shape().lens() == std::vector<std::size_t>{1, 1})
            return;
        EXPECT(ins.get_shape().transposed());
        ++non_degenerate_transposes;
    });
    EXPECT(static_transposes == 3);
    EXPECT(non_degenerate_transposes == 2);
}

TEST_CASE(split_sym_dim_materializes_reshape_with_shape_chain_target)
{
    auto batch = var("batch", {1, 1});
    auto n     = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m   = *p.get_main_module();
    auto data = m.add_parameter("data", symbolic_shape({batch, n, lit(16)}));
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
            {{"shape", migraphx::to_value(symbolic_shape({lit(1), n, lit(4), lit(4)}))}}),
        target_dims);
    auto reshape = m.add_instruction(migraphx::make_op("reshape"), data, target);
    m.add_return({reshape});

    run_pass(p);

    std::size_t static_reshapes = 0;
    for_each_clone_instruction(p, [&](auto& ins) {
        if(ins.name() != "reshape")
            return;
        EXPECT(ins.inputs().size() == 1);
        EXPECT(not ins.get_shape().dynamic());
        ++static_reshapes;
    });
    EXPECT(static_reshapes == 3);
}

TEST_CASE(split_sym_dim_specializes_fixed_symbolic_roots)
{
    auto batch = var("batch", {1, 1});
    auto n     = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m  = *p.get_main_module();
    auto lhs = m.add_parameter("lhs", symbolic_shape({batch, n, lit(4)}));
    auto rhs = m.add_parameter("rhs", symbolic_shape({batch, lit(4), lit(4)}));
    auto dot = m.add_instruction(migraphx::make_op("dot"), lhs, rhs);
    m.add_return({dot});

    run_pass(p);

    std::size_t static_dots = 0;
    for_each_clone_instruction(p, [&](auto& ins) {
        if(ins.name() != "dot")
            return;
        EXPECT(not ins.get_shape().dynamic());
        EXPECT(migraphx::none_of(ins.inputs(),
                                 [](auto input) { return input->get_shape().dynamic(); }));
        ++static_dots;
    });
    EXPECT(static_dots == 3);
}

TEST_CASE(split_sym_dim_multi_input_multibroadcast_is_noop)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m          = *p.get_main_module();
    auto shape_input = m.add_parameter("shape", symbolic_shape({n, lit(3)}));
    auto data        = m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {3}});
    auto output      = m.add_instruction(migraphx::make_op("multibroadcast"), data, shape_input);
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
    auto shape_input = m.add_parameter("shape", symbolic_shape({n, lit(3), lit(4)}));
    auto data        = m.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {3}});
    auto output =
        m.add_instruction(migraphx::make_op("broadcast", {{"axis", 1}}), data, shape_input);
    m.add_return({output});

    auto expected = p;
    run_pass(p);
    EXPECT(p == expected);
}

TEST_CASE(split_sym_dim_preserves_original_output_expr)
{
    auto n = var("n", {1, 8}, {2, 4});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x  = m.add_parameter("data", symbolic_shape({n, lit(4)}));
    for(std::size_t i = 0; i < 5; ++i)
        x = m.add_instruction(migraphx::make_op("relu"), x);
    m.add_return({x});
    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}, {5, 8}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto input =
            sm.add_parameter("data", symbolic_shape({var("n", {clone.min, clone.max}), lit(4)}));
        auto output = sm.add_instruction(fixed_pad(), input);
        for(std::size_t i = 0; i < 5; ++i)
            output = sm.add_instruction(migraphx::make_op("relu"), output);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto input          = expected_main.add_parameter("data", symbolic_shape({n, lit(4)}));
    auto target_n       = var("_split_sym_dim_n_target", {1, 8}, {1, 2, 4, 8});
    auto select =
        add_select_module(expected_main, {input}, modules, {symbolic_shape({target_n, lit(4)})});
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    output = add_back_slice(expected_main, output, {input}, {0}, {n});
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_uses_interval_bounds_without_optimals)
{
    auto n = var("n", {1, 8});
    auto p = make_relu_program(n);
    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 8}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto input =
            sm.add_parameter("data", symbolic_shape({var("n", {clone.min, clone.max}), lit(4)}));
        auto pad    = sm.add_instruction(fixed_pad(), input);
        auto output = sm.add_instruction(migraphx::make_op("relu"), pad);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto input          = expected_main.add_parameter("data", symbolic_shape({n, lit(4)}));
    auto target_n       = var("_split_sym_dim_n_target", {1, 8}, {1, 8});
    auto select =
        add_select_module(expected_main, {input}, modules, {symbolic_shape({target_n, lit(4)})});
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    output = add_back_slice(expected_main, output, {input}, {0}, {n});
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_clone_cap_is_noop)
{
    auto p        = make_relu_program(var("n", {1, 8}, {2, 4}));
    auto expected = p;
    run_pass(p, 2);
    EXPECT(p == expected);
}

TEST_CASE(split_sym_dim_reduce_mean_is_noop)
{
    auto b = var("b", {1, 4}, {2, 4});
    auto s = var("s", {2, 8}, {4, 8});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x  = m.add_parameter("data", symbolic_shape({b, s}));
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
    auto x  = m.add_parameter("data", symbolic_shape({b, s}));
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
    auto x  = m.add_parameter("data", symbolic_shape({b, s}, migraphx::shape::bool_type));
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
                                      symbolic_shape({var("b", {clone.b_min, clone.b_max}),
                                                       var("s", {clone.s_min, clone.s_max})},
                                                     migraphx::shape::bool_type));
        auto pad    = sm.add_instruction(fixed_pad(1.0f), input);
        auto output = sm.add_instruction(migraphx::make_op("reduce_all", {{"axes", {1}}}), pad);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto input =
        expected_main.add_parameter("data", symbolic_shape({b, s}, migraphx::shape::bool_type));
    auto target_b = var("_split_sym_dim_b_target", {1, 2}, {1, 2});
    auto select =
        add_select_module(expected_main,
                          {input},
                          modules,
                          {symbolic_shape({target_b, lit(1)}, migraphx::shape::bool_type)});
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    output = add_back_slice(expected_main, output, {input}, {0}, {b});
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_auto_pad_spatial_is_noop)
{
    auto s = var("s", {4, 8}, {6, 8});
    migraphx::program p;
    auto& m    = *p.get_main_module();
    auto input = m.add_parameter("data", symbolic_shape({lit(1), lit(1), s, s}));
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
    auto input = m.add_parameter("data", symbolic_shape({lit(1), lit(1), n}));
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

TEST_CASE(split_sym_dim_splits_windowed_boundary)
{
    auto s = var("s", {8, 16}, {12, 16});
    migraphx::program p;
    auto& m    = *p.get_main_module();
    auto input = m.add_parameter("data", symbolic_shape({lit(1), lit(1), s, s}));
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
            sm.add_parameter("data", symbolic_shape({lit(1), lit(1), clone_s, clone_s}));
        auto clone_weights = sm.add_literal(migraphx::generate_literal(weights_shape));
        auto pad           = sm.add_instruction(fixed_pad(), clone_input);
        auto output = sm.add_instruction(migraphx::make_op("convolution"), pad, clone_weights);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_input =
        expected_main.add_parameter("data", symbolic_shape({lit(1), lit(1), s, s}));
    auto target_s    = var("_split_sym_dim_s_target", {8, 16}, {8, 12, 16});
    auto conv_extent = target_s - 2;
    auto conv_stride = target_s * target_s - lit(4) * target_s + lit(4);
    auto conv_select = add_select_module(
        expected_main,
        {expected_input},
        convolution_modules,
        {migraphx::shape{migraphx::shape::float_type,
                         {dd{lit(1)}, dd{lit(1)}, dd{conv_extent}, dd{conv_extent}},
                         {conv_stride, conv_stride, conv_extent, lit(1)}}});
    auto conv_output = expected_main.add_instruction(
        migraphx::make_op("get_tuple_elem", {{"index", 0}}), conv_select);
    conv_output =
        add_back_slice(expected_main, conv_output, {expected_input}, {2, 3}, {s - 2, s - 2});

    auto pooling_modules = add_clones(expected, 1, clones, [&](auto& sm, const auto& clone) {
        auto clone_s = var("s", {clone.min, clone.max});
        sm.add_parameter("data", symbolic_shape({lit(1), lit(1), clone_s, clone_s}));
        auto clone_boundary_shape =
            migraphx::shape{migraphx::shape::float_type,
                            {dd{lit(1)}, dd{lit(1)}, dd{clone_s - 2}, dd{clone_s - 2}},
                            {conv_stride, conv_stride, conv_extent, lit(1)}};
        auto boundary = sm.add_parameter("#split_sym_dim_input_1_0", clone_boundary_shape);
        auto pad = sm.add_instruction(fixed_pad(std::numeric_limits<float>::lowest()), boundary);
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
    auto pooled_base   = (target_s - 3) / 2;
    auto pooled_stride = pooled_base * pooled_base + lit(2) * pooled_base + lit(1);
    auto pool_select   = add_select_module(
        expected_main,
        {conv_output, expected_input},
        pooling_modules,
        {migraphx::shape{migraphx::shape::float_type,
                           {dd{lit(1)}, dd{lit(1)}, dd{pooled_extent}, dd{pooled_extent}},
                           {pooled_stride, pooled_stride, pooled_extent, lit(1)}}});
    auto output = expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}),
                                                pool_select);
    auto output_extent = (s - 3) / 2 + 1;
    output             = add_back_slice(
        expected_main, output, {expected_input}, {2, 3}, {output_extent, output_extent});
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_retains_only_nonparallel_boundary_axes)
{
    auto batch   = var("batch", {1, 2}, {2});
    auto spatial = var("spatial", {8, 16}, {12, 16});
    migraphx::program p;
    auto& m    = *p.get_main_module();
    auto input = m.add_parameter("data", symbolic_shape({batch, lit(1), spatial, spatial}));
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

    std::vector<std::vector<int64_t>> slice_axes;
    for(auto&& ins : *p.get_main_module())
        if(ins.name() == "dyn_slice")
            slice_axes.push_back(ins.get_operator().to_value().at("axes").to_vector<int64_t>());
    EXPECT(migraphx::contains(slice_axes, std::vector<int64_t>{2, 3}));
    EXPECT(migraphx::contains(slice_axes, std::vector<int64_t>{0, 2, 3}));
}

TEST_CASE(split_sym_dim_merges_independent_supported_branches)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x0 = m.add_parameter("x0", symbolic_shape({n, lit(4)}));
    auto x1 = m.add_parameter("x1", symbolic_shape({n, lit(4)}));
    auto y0 = m.add_instruction(migraphx::make_op("relu"), x0);
    auto y1 = m.add_instruction(migraphx::make_op("relu"), x1);
    m.add_return({y0, y1});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto clone_n = var("n", {clone.min, clone.max});
        auto x1      = sm.add_parameter("x1", symbolic_shape({clone_n, lit(4)}));
        auto x0      = sm.add_parameter("x0", symbolic_shape({clone_n, lit(4)}));
        auto pad0    = sm.add_instruction(fixed_pad(), x0);
        auto y0      = sm.add_instruction(migraphx::make_op("relu"), pad0);
        auto pad1    = sm.add_instruction(fixed_pad(), x1);
        auto y1      = sm.add_instruction(migraphx::make_op("relu"), pad1);
        sm.add_return({y0, y1});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_x0    = expected_main.add_parameter("x0", symbolic_shape({n, lit(4)}));
    auto expected_x1    = expected_main.add_parameter("x1", symbolic_shape({n, lit(4)}));
    auto target_n       = var("_split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto output_shape   = symbolic_shape({target_n, lit(4)});
    auto select         = add_select_module(
        expected_main, {expected_x0, expected_x1}, modules, {output_shape, output_shape});
    auto output0 =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    output0 = add_back_slice(expected_main, output0, {expected_x1, expected_x0}, {0}, {n});
    auto output1 =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), select);
    output1 = add_back_slice(expected_main, output1, {expected_x1, expected_x0}, {0}, {n});
    expected_main.add_return({output0, output1});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_materializes_fixed_axis_slice)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x  = m.add_parameter("x", symbolic_shape({n, lit(4)}));
    auto y  = m.add_instruction(migraphx::make_op("relu"), x);
    y       = m.add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {2}}}), y);
    y = m.add_instruction(migraphx::make_op("relu"), y);
    m.add_return({y});

    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto input =
            sm.add_parameter("x", symbolic_shape({var("n", {clone.min, clone.max}), lit(4)}));
        auto pad    = sm.add_instruction(fixed_pad(), input);
        auto output = sm.add_instruction(migraphx::make_op("relu"), pad);
        output      = sm.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {2}}}), output);
        output = sm.add_instruction(migraphx::make_op("relu"), output);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto input          = expected_main.add_parameter("x", symbolic_shape({n, lit(4)}));
    auto target_n       = var("_split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select =
        add_select_module(expected_main, {input}, modules, {symbolic_shape({target_n, lit(2)})});
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    output = add_back_slice(expected_main, output, {input}, {0}, {n});
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_preserves_nonprefix_dyn_slice)
{
    auto rows    = var("rows", {1, 4}, {2});
    auto columns = var("columns", {2, 4});
    migraphx::program p;
    auto& m         = *p.get_main_module();
    auto data       = m.add_parameter("data", symbolic_shape({rows, columns}));
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

TEST_CASE(split_sym_dim_materializes_gather_concat_and_dyn_slice)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m    = *p.get_main_module();
    auto ids   = m.add_parameter("ids", symbolic_shape({n}, migraphx::shape::int64_type));
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

    for(auto&& ins : *p.get_main_module())
        EXPECT(not migraphx::contains({"gather", "concat", "relu"}, ins.name()));

    std::size_t static_gathers = 0;
    std::size_t static_slices  = 0;
    std::size_t static_concats = 0;
    for_each_clone_instruction(p, [&](auto& ins) {
        if(ins.name() == "gather")
            ++static_gathers;
        if(ins.name() == "slice")
            ++static_slices;
        if(ins.name() == "concat")
            ++static_concats;
        if(migraphx::contains({"gather", "slice", "concat"}, ins.name()))
            EXPECT(not ins.get_shape().dynamic());
    });
    EXPECT(static_gathers == 3);
    EXPECT(static_slices == 3);
    EXPECT(static_concats == 3);
}

TEST_CASE(split_sym_dim_materializes_fill_range_and_scatter)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m            = *p.get_main_module();
    auto data_buffer   = m.add_parameter("data_buffer", symbolic_shape({n}));
    auto update_buffer = m.add_parameter("update_buffer", symbolic_shape({n}));
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

    for(auto&& ins : *p.get_main_module())
    {
        EXPECT(ins.name() != "dynamic_range");
        EXPECT(ins.name() != "fill");
        EXPECT(ins.name() != "scatternd_none");
    }

    std::size_t static_fills    = 0;
    std::size_t static_scatters = 0;
    for_each_clone_instruction(p, [&](auto& ins) {
        EXPECT(ins.name() != "dynamic_range");
        if(ins.name() == "fill")
        {
            EXPECT(not ins.get_shape().dynamic());
            ++static_fills;
        }
        if(ins.name() == "scatternd_none")
        {
            EXPECT(not ins.get_shape().dynamic());
            EXPECT(migraphx::none_of(ins.inputs(),
                                     [](auto input) { return input->get_shape().dynamic(); }));
            ++static_scatters;
        }
    });
    EXPECT(static_fills == 6);
    EXPECT(static_scatters == 3);
}

TEST_CASE(split_sym_dim_rewrites_padded_scatter_indices)
{
    auto n = var("n", {1, 4});
    migraphx::program p;
    auto& m   = *p.get_main_module();
    auto data = m.add_parameter("data", symbolic_shape({n}));
    auto indices =
        m.add_parameter("indices", symbolic_shape({n, lit(1)}, migraphx::shape::int64_type));
    auto updates = m.add_parameter("updates", symbolic_shape({n}));
    auto output  = m.add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
    m.add_return({output});

    run_pass(p);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data_values{10.0f, 20.0f};
    std::vector<int64_t> index_values{0, 0};
    std::vector<float> update_values{1.0f, 2.0f};
    migraphx::parameter_map params;
    params["data"] =
        migraphx::argument{migraphx::shape{migraphx::shape::float_type, {2}}, data_values.data()};
    params["indices"] = migraphx::argument{migraphx::shape{migraphx::shape::int64_type, {2, 1}},
                                           index_values.data()};
    params["updates"] =
        migraphx::argument{migraphx::shape{migraphx::shape::float_type, {2}}, update_values.data()};
    auto result = p.eval(params).back();
    std::vector<float> result_values;
    result.visit([&](auto values) { result_values.assign(values.begin(), values.end()); });
    EXPECT(result_values == std::vector<float>{2.0f, 20.0f});
}

TEST_CASE(split_sym_dim_rewrites_zero_depth_scatter_indices)
{
    auto m_dim = var("m", {1, 2});
    auto n     = var("n", {1, 4});
    migraphx::program p;
    auto& m   = *p.get_main_module();
    auto data = m.add_parameter("data", symbolic_shape({m_dim, lit(2)}));
    auto indices =
        m.add_parameter("indices", symbolic_shape({n, lit(0)}, migraphx::shape::int64_type));
    auto updates = m.add_parameter("updates", symbolic_shape({n, m_dim, lit(2)}));
    auto output  = m.add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
    m.add_return({output});

    run_pass(p);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data_values{10.0f, 20.0f};
    std::vector<float> update_values{1.0f, 2.0f, 3.0f, 4.0f};
    migraphx::parameter_map params;
    params["data"]    = migraphx::argument{migraphx::shape{migraphx::shape::float_type, {1, 2}},
                                        data_values.data()};
    params["indices"] = migraphx::argument{migraphx::shape{migraphx::shape::int64_type, {2, 0}}};
    params["updates"] = migraphx::argument{migraphx::shape{migraphx::shape::float_type, {2, 1, 2}},
                                           update_values.data()};
    auto result       = p.eval(params).back();
    std::vector<float> result_values;
    result.visit([&](auto values) { result_values.assign(values.begin(), values.end()); });
    EXPECT(result_values == std::vector<float>{3.0f, 4.0f});
}

TEST_CASE(split_sym_dim_staticizes_llama_attention_chain)
{
    auto sequence = var("sequence_length", {1, 4}, {2});
    migraphx::program p;
    auto& m            = *p.get_main_module();
    auto ids           = m.add_parameter("input_ids",
                               symbolic_shape({lit(1), sequence}, migraphx::shape::int64_type));
    auto mask_buffer   = m.add_parameter("mask_buffer", symbolic_shape({sequence}));
    auto update_buffer = m.add_parameter("update_buffer", symbolic_shape({sequence}));
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
    mask      = add_symbolic_multibroadcast(m, {lit(1), sequence, sequence}, mask, scores);
    scores    = m.add_instruction(migraphx::make_op("add"), scores, mask);
    auto probs   = m.add_instruction(migraphx::make_op("softmax", {{"axis", 2}}), scores);
    auto context = m.add_instruction(migraphx::make_op("dot"), probs, hidden);
    auto output  = m.add_instruction(migraphx::make_op("relu"), context);
    m.add_return({output});

    run_pass(p);

    std::size_t selects = 0;
    for(auto&& ins : *p.get_main_module())
    {
        selects += ins.name() == "select_module" ? 1 : 0;
        EXPECT(not migraphx::contains(
            {"gather", "dynamic_range", "fill", "scatternd_none", "dot", "softmax", "relu"},
            ins.name()));
    }
    EXPECT(selects == 1);

    std::size_t static_scatters = 0;
    std::size_t static_dots     = 0;
    for_each_clone_instruction(p, [&](auto& ins) {
        if(ins.name() == "scatternd_none")
            ++static_scatters;
        if(ins.name() == "dot")
            ++static_dots;
        if(migraphx::contains({"scatternd_none", "dot", "softmax"}, ins.name()))
            EXPECT(not ins.get_shape().dynamic());
    });
    EXPECT(static_scatters == 3);
    EXPECT(static_dots == 6);
}

TEST_CASE(split_sym_dim_clones_fixed_shape_dependencies_into_cases)
{
    auto sequence = var("sequence_length", {1, 4}, {2});
    migraphx::program p;
    auto& m = *p.get_main_module();
    auto input =
        m.add_parameter("input", symbolic_shape({sequence, lit(4)}, migraphx::shape::float_type));
    auto dims =
        m.add_instruction(migraphx::make_op("dimensions_of", {{"start", 1}, {"end", 2}}), input);
    auto extent = m.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), dims);
    auto scale  = m.add_instruction(migraphx::make_op("sqrt"), extent);
    scale       = add_symbolic_multibroadcast(m, {sequence, lit(4)}, scale, input);
    auto output = m.add_instruction(migraphx::make_op("mul"), input, scale);
    m.add_return({output});

    run_pass(p);

    std::size_t selects = 0;
    for(auto&& ins : *p.get_main_module())
    {
        EXPECT(migraphx::contains({"@param",
                                   "@literal",
                                   "select_module",
                                   "get_tuple_elem",
                                   "eval_expr_from_shape",
                                   "dyn_slice",
                                   "@return"},
                                  ins.name()));
        if(ins.name() == "select_module")
        {
            ++selects;
            EXPECT(ins.inputs().size() == 1);
        }
    }
    EXPECT(selects == 1);

    std::size_t static_sqrts = 0;
    for_each_clone_instruction(p, [&](auto& ins) {
        if(ins.name() != "sqrt")
            return;
        EXPECT(not ins.get_shape().dynamic());
        ++static_sqrts;
    });
    EXPECT(static_sqrts == 3);
}

TEST_CASE(split_sym_dim_resolves_symbolic_shape_queries_without_model_compute)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m    = *p.get_main_module();
    auto ids   = m.add_parameter("ids", symbolic_shape({n}, migraphx::shape::int64_type));
    auto table = m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}));
    auto gathered = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), table, ids);
    auto dims =
        m.add_instruction(migraphx::make_op("dimensions_of", {{"start", 0}, {"end", 2}}), gathered);
    m.add_return({dims});

    run_pass(p);

    bool found_eval = false;
    for(auto&& ins : *p.get_main_module())
    {
        EXPECT(ins.name() != "gather");
        EXPECT(ins.name() != "dimensions_of");
        found_eval = found_eval or ins.name() == "eval_expr_from_shape";
    }
    EXPECT(found_eval);
}

TEST_CASE(split_sym_dim_preserves_shape_query_without_direct_root)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m    = *p.get_main_module();
    auto input = m.add_parameter("input", symbolic_shape({n + 1}));
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
    auto x0 = m.add_parameter("x0", symbolic_shape({n, lit(4)}));
    auto x1 = m.add_parameter("x1", symbolic_shape({m_dim, lit(4)}));
    auto y0 = m.add_instruction(migraphx::make_op("relu"), x0);
    auto y1 = m.add_instruction(migraphx::make_op("relu"), x1);
    m.add_return({y0, y1});

    run_pass(p, 4);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{1, 1}, {2, 2}, {3, 4}};
    auto n_modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto input =
            sm.add_parameter("x0", symbolic_shape({var("n", {clone.min, clone.max}), lit(4)}));
        auto pad    = sm.add_instruction(fixed_pad(), input);
        auto output = sm.add_instruction(migraphx::make_op("relu"), pad);
        sm.add_return({output});
    });
    auto m_modules = add_clones(expected, 1, clones, [&](auto& sm, const auto& clone) {
        auto input =
            sm.add_parameter("x1", symbolic_shape({var("m", {clone.min, clone.max}), lit(4)}));
        auto pad    = sm.add_instruction(fixed_pad(), input);
        auto output = sm.add_instruction(migraphx::make_op("relu"), pad);
        sm.add_return({output});
    });

    auto& expected_main  = *expected.get_main_module();
    auto expected_x0     = expected_main.add_parameter("x0", symbolic_shape({n, lit(4)}));
    auto expected_x1     = expected_main.add_parameter("x1", symbolic_shape({m_dim, lit(4)}));
    auto runtime_sources = std::vector<migraphx::instruction_ref>{expected_x1, expected_x0};

    auto target_n = var("_split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select_n = add_select_module(
        expected_main, {expected_x0}, n_modules, {symbolic_shape({target_n, lit(4)})});
    auto output_n = expected_main.add_instruction(
        migraphx::make_op("get_tuple_elem", {{"index", 0}}), select_n);
    output_n = add_back_slice(expected_main, output_n, runtime_sources, {0}, {n});

    auto target_m = var("_split_sym_dim_m_target", {1, 4}, {1, 2, 4});
    auto select_m = add_select_module(
        expected_main, {expected_x1}, m_modules, {symbolic_shape({target_m, lit(4)})});
    auto output_m = expected_main.add_instruction(
        migraphx::make_op("get_tuple_elem", {{"index", 0}}), select_m);
    output_m = add_back_slice(expected_main, output_m, runtime_sources, {0}, {m_dim});
    expected_main.add_return({output_n, output_m});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_mixed_range_parameter_is_noop)
{
    auto n = var("n", {1, 4}, {2, 4});
    migraphx::program p;
    auto& m       = *p.get_main_module();
    auto symbolic = m.add_parameter("symbolic", symbolic_shape({n, lit(4)}));
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
    auto input = m.add_parameter("data", symbolic_shape({n, lit(1), lit(5), lit(5)}));
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
            "data", symbolic_shape({var("n", {clone.min, clone.max}), lit(1), lit(5), lit(5)}));
        auto clone_weights = sm.add_literal(migraphx::generate_literal(weights_shape));
        auto pad           = sm.add_instruction(fixed_pad(), clone_input);
        auto output = sm.add_instruction(migraphx::make_op("convolution"), pad, clone_weights);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_input =
        expected_main.add_parameter("data", symbolic_shape({n, lit(1), lit(5), lit(5)}));
    auto target_n = var("_split_sym_dim_n_target", {1, 4}, {1, 2, 4});
    auto select   = add_select_module(expected_main,
                                      {expected_input},
                                    modules,
                                      {symbolic_shape({target_n, lit(1), lit(3), lit(3)})});
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    output = add_back_slice(expected_main, output, {expected_input}, {0}, {n});
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_coalesces_spatial_cnn)
{
    auto spatial = var("spatial", {8, 16}, {12});

    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x  = m.add_parameter("image", symbolic_shape({lit(1), lit(3), spatial, spatial}));
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
            "image", symbolic_shape({lit(1), lit(3), clone_spatial, clone_spatial}));
        auto clone_w0 = sm.add_literal(migraphx::generate_literal(w0_shape));
        auto clone_w1 = sm.add_literal(migraphx::generate_literal(w1_shape));
        auto pad      = sm.add_instruction(fixed_pad(), input);
        auto conv0    = sm.add_instruction(migraphx::make_op("convolution"), pad, clone_w0);
        auto relu0    = sm.add_instruction(migraphx::make_op("relu"), conv0);
        auto conv1    = sm.add_instruction(migraphx::make_op("convolution"), relu0, clone_w1);
        auto relu1    = sm.add_instruction(migraphx::make_op("relu"), conv1);
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
    auto input =
        expected_main.add_parameter("image", symbolic_shape({lit(1), lit(3), spatial, spatial}));
    auto target_spatial = var("_split_sym_dim_spatial_target", {8, 16}, {8, 12, 16});
    auto output_extent  = (target_spatial - 6) / 2 + 1;
    auto output_base    = (target_spatial - 6) / 2;
    auto output_stride  = output_base * output_base + lit(2) * output_base + lit(1);
    migraphx::shape output_shape{
        migraphx::shape::float_type,
        {dd{lit(1)}, dd{lit(6)}, dd{output_extent}, dd{output_extent}},
        {lit(12) * output_base + lit(6) * output_base * output_base + lit(6),
         output_stride,
         output_extent,
         lit(1)}};
    auto select = add_select_module(expected_main, {input}, modules, {output_shape});
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto real_extent = (spatial - 6) / 2 + 1;
    output = add_back_slice(expected_main, output, {input}, {2, 3}, {real_extent, real_extent});
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_preserves_compound_mask_extent)
{
    auto sequence = var("sequence", {4, 16}, {8});

    migraphx::program p;
    auto& m = *p.get_main_module();
    auto x  = m.add_parameter("x", symbolic_shape({lit(1), lit(1), sequence}));
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
        auto indices        = add_iota(sm, clone.max - 2);
        auto clone_sequence = var("sequence", {clone.min, clone.max});
        auto input = sm.add_parameter("x", symbolic_shape({lit(1), lit(1), clone_sequence}));
        auto clone_weights = sm.add_literal(migraphx::generate_literal(weights_shape));
        auto pad           = sm.add_instruction(fixed_pad(), input);
        auto convolution   = sm.add_instruction(
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
    auto expected_input =
        expected_main.add_parameter("x", symbolic_shape({lit(1), lit(1), sequence}));
    auto target_sequence = var("_split_sym_dim_sequence_target", {4, 16}, {4, 8, 16});
    auto target_extent   = target_sequence - 2;
    migraphx::shape output_shape{migraphx::shape::float_type,
                                 {dd{lit(1)}, dd{lit(1)}, dd{target_extent}},
                                 {target_extent, target_extent, lit(1)}};
    auto select = add_select_module(expected_main, {expected_input}, modules, {output_shape});
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    expected_output =
        add_back_slice(expected_main, expected_output, {expected_input}, {2}, {sequence - 2});
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_freezes_fixed_roots_in_mask_extents)
{
    auto sequence = var("sequence", {4, 8}, {6});
    auto kernel   = var("kernel", {3, 3});
    migraphx::program p;
    auto& m      = *p.get_main_module();
    auto input   = m.add_parameter("input", symbolic_shape({lit(1), lit(1), sequence}));
    auto weights = m.add_parameter("weights", symbolic_shape({lit(1), lit(1), kernel}));
    auto conv    = m.add_instruction(
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
        auto negative_infinity = add_fill(sm, -std::numeric_limits<float>::infinity());
        auto indices           = add_iota(sm, clone.max - 2);
        auto clone_sequence    = var("sequence", {clone.min, clone.max});
        auto clone_weights     = sm.add_parameter("weights", weights_shape);
        auto clone_input =
            sm.add_parameter("input", symbolic_shape({lit(1), lit(1), clone_sequence}));
        auto padded_input   = sm.add_instruction(fixed_pad(), clone_input);
        auto padded_weights = sm.add_instruction(fixed_pad(), clone_weights);
        auto convolution    = sm.add_instruction(
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
    auto expected_input =
        expected_main.add_parameter("input", symbolic_shape({lit(1), lit(1), sequence}));
    auto expected_weights =
        expected_main.add_parameter("weights", symbolic_shape({lit(1), lit(1), kernel}));
    auto target_sequence = var("_split_sym_dim_sequence_target", {4, 8}, {4, 6, 8});
    auto target_extent   = (-kernel + sequence + lit(1))
                             .subs({{migraphx::sym::as_symbol(sequence), target_sequence},
                                    {migraphx::sym::as_symbol(kernel), lit(3)}});
    migraphx::shape output_shape{migraphx::shape::float_type,
                                 {dd{lit(1)}, dd{lit(1)}, dd{target_extent}},
                                 {target_extent, target_extent, lit(1)}};
    auto select = add_select_module(
        expected_main, {expected_input, expected_weights}, modules, {output_shape});
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    expected_output = add_back_slice(expected_main,
                                     expected_output,
                                     {expected_weights, expected_input},
                                     {2},
                                     {-kernel + sequence + 1});
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_specializes_transformer_core)
{
    auto sequence = var("sequence", {4, 16}, {8});
    auto p        = make_transformer_core_program(sequence);
    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{4, 4}, {5, 8}, {9, 16}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto zero           = add_fill(sm, 0.0f);
        auto indices        = add_iota(sm, clone.max);
        auto clone_sequence = var("sequence", {clone.min, clone.max});
        auto value = sm.add_parameter("value", symbolic_shape({lit(2), clone_sequence, lit(8)}));
        auto query = sm.add_parameter("query", symbolic_shape({lit(2), clone_sequence, lit(8)}));
        auto key =
            sm.add_parameter("key_transposed", symbolic_shape({lit(2), lit(8), clone_sequence}));

        auto padded_query  = sm.add_instruction(fixed_pad(), query);
        auto padded_key    = sm.add_instruction(fixed_pad(), key);
        auto scores        = sm.add_instruction(migraphx::make_op("dot"), padded_query, padded_key);
        auto padded_value  = sm.add_instruction(fixed_pad(), value);
        auto extent        = add_expected_clone_extent(sm, sequence, sequence, clone, query);
        auto masked_scores = add_mask(sm, scores, indices, extent, zero, 2);
        auto masked_value  = add_mask(sm, padded_value, indices, extent, zero, 1);
        auto context  = sm.add_instruction(migraphx::make_op("dot"), masked_scores, masked_value);
        auto residual = sm.add_instruction(migraphx::make_op("add"), context, padded_query);
        auto output   = sm.add_instruction(migraphx::make_op("relu"), residual);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto query = expected_main.add_parameter("query", symbolic_shape({lit(2), sequence, lit(8)}));
    auto key =
        expected_main.add_parameter("key_transposed", symbolic_shape({lit(2), lit(8), sequence}));
    auto value = expected_main.add_parameter("value", symbolic_shape({lit(2), sequence, lit(8)}));
    auto target_sequence = var("_split_sym_dim_sequence_target", {4, 16}, {4, 8, 16});
    auto select          = add_select_module(expected_main,
                                             {key, query, value},
                                    modules,
                                             {symbolic_shape({lit(2), target_sequence, lit(8)})});
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    output = add_back_slice(expected_main, output, {value, key, query}, {1}, {sequence});
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_specializes_transformer)
{
    auto sequence = var("sequence", {4, 16}, {8});
    auto p        = make_transformer_program();
    run_pass(p);

    migraphx::program expected;
    migraphx::shape weight_shape{migraphx::shape::float_type, {2, 8, 8}};
    std::vector<clone_spec> clones = {{4, 4}, {5, 8}, {9, 16}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto negative_infinity = add_fill(sm, -std::numeric_limits<float>::infinity());
        auto indices           = add_iota(sm, clone.max);
        auto clone_sequence    = var("sequence", {clone.min, clone.max});
        auto input = sm.add_parameter("x", symbolic_shape({lit(2), clone_sequence, lit(8)}));
        auto wq    = sm.add_literal(migraphx::generate_literal(weight_shape, 0));
        auto wk    = sm.add_literal(migraphx::generate_literal(weight_shape, 1));
        auto wv    = sm.add_literal(migraphx::generate_literal(weight_shape, 2));
        auto wo    = sm.add_literal(migraphx::generate_literal(weight_shape, 3));

        auto padded = sm.add_instruction(fixed_pad(), input);
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
        auto context    = sm.add_instruction(migraphx::make_op("dot"), probabilities, value);
        auto projection = sm.add_instruction(migraphx::make_op("dot"), context, wo);
        auto output     = sm.add_instruction(migraphx::make_op("add"), projection, padded);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto input = expected_main.add_parameter("x", symbolic_shape({lit(2), sequence, lit(8)}));
    auto target_sequence = var("_split_sym_dim_sequence_target", {4, 16}, {4, 8, 16});
    auto select          = add_select_module(
        expected_main, {input}, modules, {symbolic_shape({lit(2), target_sequence, lit(8)})});
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    output = add_back_slice(expected_main, output, {input}, {1}, {sequence});
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_keeps_softmax_mask_off_contract_axis)
{
    auto sequence = var("sequence", {4, 16}, {8});
    auto p        = make_softmax_off_contract_axis_program();
    run_pass(p);

    migraphx::program expected;
    std::vector<clone_spec> clones = {{4, 4}, {5, 8}, {9, 16}};
    auto modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto zero              = add_fill(sm, 0.0f);
        auto negative_infinity = add_fill(sm, -std::numeric_limits<float>::infinity());
        auto indices           = add_iota(sm, clone.max);
        auto clone_sequence    = var("sequence", {clone.min, clone.max});
        auto input =
            sm.add_parameter("x", symbolic_shape({lit(2), clone_sequence, clone_sequence}));
        auto value = sm.add_parameter("value", symbolic_shape({lit(2), clone_sequence, lit(8)}));

        auto padded_input = sm.add_instruction(fixed_pad(), input);
        auto extent       = add_expected_clone_extent(sm, sequence, sequence, clone, input);
        auto masked_input = add_mask(sm, padded_input, indices, extent, negative_infinity, 1);
        auto probabilities =
            sm.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), masked_input);
        auto padded_value         = sm.add_instruction(fixed_pad(), value);
        auto masked_probabilities = add_mask(sm, probabilities, indices, extent, zero, 2);
        auto masked_value         = add_mask(sm, padded_value, indices, extent, zero, 1);
        auto output =
            sm.add_instruction(migraphx::make_op("dot"), masked_probabilities, masked_value);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto input = expected_main.add_parameter("x", symbolic_shape({lit(2), sequence, sequence}));
    auto value = expected_main.add_parameter("value", symbolic_shape({lit(2), sequence, lit(8)}));
    auto target_sequence = var("_split_sym_dim_sequence_target", {4, 16}, {4, 8, 16});
    auto select          = add_select_module(expected_main,
                                             {value, input},
                                    modules,
                                             {symbolic_shape({lit(2), target_sequence, lit(8)})});
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    output = add_back_slice(expected_main, output, {value, input}, {1}, {sequence});
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_data_dependent_nonzero_root)
{
    auto count = var("nonzero_count", {0, 4});
    migraphx::program p;
    auto& m      = *p.get_main_module();
    auto data    = m.add_parameter("data", migraphx::shape{migraphx::shape::bool_type, {2, 2}});
    auto nonzero = m.add_instruction(migraphx::make_op("nonzero"), data);
    auto indices = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nonzero);
    auto num_nonzero =
        m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nonzero);
    auto starts   = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto selected = m.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {1}},
                           {"starts", {0}},
                           {"ends", migraphx::value::array{migraphx::to_value(count)}},
                           {"always_leq", true}}),
        indices,
        starts,
        num_nonzero);
    auto converted = m.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), selected);
    auto output = m.add_instruction(migraphx::make_op("relu"), converted);
    m.add_return({output});

    run_pass(p);

    EXPECT(m.has_instruction(selected));
    auto selections = migraphx::find_all(migraphx::iterator_for(m),
                                         [](auto ins) { return ins->name() == "select_module"; });
    EXPECT(selections.size() == 1);
    EXPECT(migraphx::contains(selections.front()->inputs(), selected));
    EXPECT(p.get_modules().size() == 2);
    EXPECT(migraphx::none_of(p.get_modules(),
                             [](auto* mod) { return mod->name() == "main:split_sym_dim_0_0"; }));
    for(auto* mod : p.get_modules())
    {
        if(mod == p.get_main_module())
            continue;
        EXPECT(migraphx::none_of(*mod, [](const auto& ins) {
            return not migraphx::starts_with(ins.name(), "@") and ins.get_shape().dynamic();
        }));
    }

    p.compile(migraphx::make_target("ref"));
    auto check_result = [&](std::vector<char> input_data,
                            const migraphx::shape& expected_shape,
                            const std::vector<float>& expected_values) {
        migraphx::parameter_map params;
        params["data"] = migraphx::argument{migraphx::shape{migraphx::shape::bool_type, {2, 2}},
                                            input_data.data()};
        auto result    = p.eval(params).back();
        std::vector<float> values;
        result.visit(
            [&](auto output_values) { values.assign(output_values.begin(), output_values.end()); });
        EXPECT(result.get_shape() == expected_shape);
        EXPECT(values == expected_values);
    };
    check_result({0, 0, 0, 0}, migraphx::shape{migraphx::shape::float_type, {2, 0}, {0, 1}}, {});
    check_result({0, 1, 1, 0},
                 migraphx::shape{migraphx::shape::float_type, {2, 2}, {4, 1}},
                 {0.0f, 1.0f, 1.0f, 0.0f});
    check_result({1, 1, 1, 1},
                 migraphx::shape{migraphx::shape::float_type, {2, 4}, {4, 1}},
                 {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f});
}

TEST_CASE(split_sym_dim_sequential_nonzero_nms_roots)
{
    auto nonzero_count = var("nonzero_count", {0, 4});
    auto nms_count     = var("nms_count", {0, 4});
    migraphx::program p;
    auto& m        = *p.get_main_module();
    auto condition = m.add_parameter("condition", migraphx::shape{migraphx::shape::bool_type, {4}});
    auto boxes     = m.add_parameter("boxes", migraphx::shape{migraphx::shape::float_type, {4, 4}});
    auto scores    = m.add_parameter("scores", migraphx::shape{migraphx::shape::float_type, {4}});
    auto starts    = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto nonzero   = m.add_instruction(migraphx::make_op("nonzero"), condition);
    auto indices = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nonzero);
    auto num_nonzero =
        m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nonzero);
    auto selected = m.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {1}},
                           {"starts", {0}},
                           {"ends", migraphx::value::array{migraphx::to_value(nonzero_count)}},
                           {"always_leq", true}}),
        indices,
        starts,
        num_nonzero);
    auto selected_1d = m.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), selected);
    auto selected_boxes =
        m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), boxes, selected_1d);
    selected_boxes =
        m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), selected_boxes);
    auto selected_scores =
        m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), scores, selected_1d);
    selected_scores =
        m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 1}}}), selected_scores);
    auto max_output = m.add_literal(int64_t{4});
    auto nms        = m.add_instruction(
        migraphx::make_op("nonmaxsuppression"), selected_boxes, selected_scores, max_output);
    auto nms_indices  = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nms);
    auto num_selected = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nms);
    auto nms_selected = m.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {0}},
                           {"starts", {0}},
                           {"ends", migraphx::value::array{migraphx::to_value(nms_count)}},
                           {"always_leq", true}}),
        nms_indices,
        starts,
        num_selected);
    auto converted = m.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), nms_selected);
    m.add_return({m.add_instruction(migraphx::make_op("relu"), converted)});

    run_pass(p);

    auto selections = migraphx::find_all(migraphx::iterator_for(m),
                                         [](auto ins) { return ins->name() == "select_module"; });
    EXPECT(selections.size() == 2);
    EXPECT(p.get_modules().size() == 3);
    auto main_slices = migraphx::find_all(migraphx::iterator_for(m),
                                          [](auto ins) { return ins->name() == "dyn_slice"; });
    EXPECT(main_slices.size() == 5);
    for(auto* mod : p.get_modules())
    {
        if(mod == p.get_main_module())
            continue;
        EXPECT(migraphx::none_of(*mod, [](const auto& ins) { return ins.name() == "dyn_slice"; }));
    }
}

TEST_CASE(split_sym_dim_ssd_nms_topk_tail)
{
    auto nms_count  = var("ssd_nms_count", {0, 4});
    auto topk_count = var("ssd_topk_count", {0, 4});
    migraphx::program p;
    auto& m    = *p.get_main_module();
    auto boxes = m.add_parameter("boxes", migraphx::shape{migraphx::shape::float_type, {1, 4, 4}});
    auto scores =
        m.add_parameter("scores", migraphx::shape{migraphx::shape::float_type, {1, 1, 4}});
    auto starts     = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto max_output = m.add_literal(int64_t{4});
    auto iou        = m.add_literal(0.5f);
    auto threshold  = m.add_literal(0.0f);
    auto nms        = m.add_instruction(
        migraphx::make_op("nonmaxsuppression"), boxes, scores, max_output, iou, threshold);
    auto nms_indices  = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nms);
    auto num_selected = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nms);
    auto selected     = m.add_instruction(
        migraphx::make_op("dyn_slice",
                              {{"axes", {0}},
                               {"starts", {0}},
                               {"ends", migraphx::value::array{migraphx::to_value(nms_count)}},
                               {"always_leq", true}}),
        nms_indices,
        starts,
        num_selected);
    auto box_column = m.add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {2}}, {"ends", {3}}}), selected);
    auto box_ids = m.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), box_column);
    auto class_column = m.add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), selected);
    auto class_ids = m.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), class_column);
    auto scores_table = m.add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 1}}}), scores);
    auto selected_scores =
        m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), scores_table, box_ids);
    auto selected_count = m.add_instruction(
        migraphx::make_op("dimensions_of", {{"start", 0}, {"end", 1}}), selected_scores);
    auto cap = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {3}});
    auto cap_and_count =
        m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), cap, selected_count);
    auto k = m.add_instruction(migraphx::make_op("reduce_min", {{"axes", {0}}}), cap_and_count);
    auto topk =
        m.add_instruction(migraphx::make_op("topk", {{"k", 4}, {"axis", 0}}), selected_scores);
    auto topk_values = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), topk);
    auto topk_indices =
        m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), topk);
    auto values = m.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {0}},
                           {"starts", {0}},
                           {"ends", migraphx::value::array{migraphx::to_value(topk_count)}},
                           {"always_leq", true}}),
        topk_values,
        starts,
        k);
    auto ranked_indices = m.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {0}},
                           {"starts", {0}},
                           {"ends", migraphx::value::array{migraphx::to_value(topk_count)}},
                           {"always_leq", true}}),
        topk_indices,
        starts,
        k);
    auto ranked_box_ids =
        m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), box_ids, ranked_indices);
    auto boxes_table = m.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), boxes);
    auto output_boxes =
        m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), boxes_table, ranked_box_ids);
    auto output_labels =
        m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), class_ids, ranked_indices);
    m.add_return({output_boxes, output_labels, values});

    run_pass(p);

    auto selections = migraphx::find_all(migraphx::iterator_for(m),
                                         [](auto ins) { return ins->name() == "select_module"; });
    EXPECT(selections.size() == 2);
    for(auto* mod : p.get_modules())
    {
        if(mod == p.get_main_module())
            continue;
        EXPECT(migraphx::none_of(*mod, [](const auto& ins) { return ins.name() == "dyn_slice"; }));
    }

    p.compile(migraphx::make_target("ref"));
    std::vector<float> box_values   = {0.0f,
                                       0.0f,
                                       1.0f,
                                       1.0f,
                                       0.0f,
                                       2.0f,
                                       1.0f,
                                       3.0f,
                                       0.0f,
                                       4.0f,
                                       1.0f,
                                       5.0f,
                                       0.0f,
                                       4.0f,
                                       1.0f,
                                       5.0f};
    std::vector<float> score_values = {0.9f, 0.8f, 0.7f, 0.6f};
    migraphx::parameter_map params;
    params["boxes"]  = migraphx::argument{migraphx::shape{migraphx::shape::float_type, {1, 4, 4}},
                                         box_values.data()};
    params["scores"] = migraphx::argument{migraphx::shape{migraphx::shape::float_type, {1, 1, 4}},
                                          score_values.data()};
    auto results     = p.eval(params);
    EXPECT(
        results.at(0).to_vector<float>() ==
        std::vector<float>{0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 2.0f, 1.0f, 3.0f, 0.0f, 4.0f, 1.0f, 5.0f});
    EXPECT(results.at(1).to_vector<int64_t>() == std::vector<int64_t>{0, 0, 0});
    EXPECT(results.at(2).to_vector<float>() == std::vector<float>{0.9f, 0.8f, 0.7f});
}

TEST_CASE(split_sym_dim_maskrcnn_reduced_nms_fanout)
{
    auto nonzero_count = var("mask_nonzero_count", {0, 4});
    auto nms_count0    = var("mask_nms_count_0", {0, 4});
    auto nms_count1    = var("mask_nms_count_1", {0, 4});
    migraphx::program p;
    auto& m        = *p.get_main_module();
    auto condition = m.add_parameter("condition", migraphx::shape{migraphx::shape::bool_type, {4}});
    auto boxes     = m.add_parameter("boxes", migraphx::shape{migraphx::shape::float_type, {4, 4}});
    auto scores0   = m.add_parameter("scores0", migraphx::shape{migraphx::shape::float_type, {4}});
    auto scores1   = m.add_parameter("scores1", migraphx::shape{migraphx::shape::float_type, {4}});
    auto starts    = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto nonzero   = m.add_instruction(migraphx::make_op("nonzero"), condition);
    auto indices = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nonzero);
    auto num_nonzero =
        m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nonzero);
    auto selected = m.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {1}},
                           {"starts", {0}},
                           {"ends", migraphx::value::array{migraphx::to_value(nonzero_count)}},
                           {"always_leq", true}}),
        indices,
        starts,
        num_nonzero);
    auto selected_1d = m.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), selected);
    auto selected_boxes =
        m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), boxes, selected_1d);
    selected_boxes =
        m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), selected_boxes);
    auto selected_scores0 =
        m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), scores0, selected_1d);
    selected_scores0 =
        m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 1}}}), selected_scores0);
    auto selected_scores1 =
        m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), scores1, selected_1d);
    selected_scores1 =
        m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 1}}}), selected_scores1);
    auto max_output = m.add_literal(int64_t{4});
    auto nms0       = m.add_instruction(
        migraphx::make_op("nonmaxsuppression"), selected_boxes, selected_scores0, max_output);
    auto indices0  = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nms0);
    auto count0    = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nms0);
    auto selected0 = m.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {0}},
                           {"starts", {0}},
                           {"ends", migraphx::value::array{migraphx::to_value(nms_count0)}},
                           {"always_leq", true}}),
        indices0,
        starts,
        count0);
    auto nms1 = m.add_instruction(
        migraphx::make_op("nonmaxsuppression"), selected_boxes, selected_scores1, max_output);
    auto indices1  = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nms1);
    auto count1    = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nms1);
    auto selected1 = m.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {0}},
                           {"starts", {0}},
                           {"ends", migraphx::value::array{migraphx::to_value(nms_count1)}},
                           {"always_leq", true}}),
        indices1,
        starts,
        count1);
    auto joined =
        m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), selected0, selected1);
    auto converted = m.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), joined);
    m.add_return({m.add_instruction(migraphx::make_op("relu"), converted)});

    run_pass(p);

    auto selections = migraphx::find_all(migraphx::iterator_for(m),
                                         [](auto ins) { return ins->name() == "select_module"; });
    EXPECT(selections.size() == 2);
    std::vector<std::size_t> selection_input_counts;
    std::transform(selections.begin(),
                   selections.end(),
                   std::back_inserter(selection_input_counts),
                   [](auto selection) { return selection->inputs().size(); });
    std::sort(selection_input_counts.begin(), selection_input_counts.end());
    EXPECT(selection_input_counts == std::vector<std::size_t>{3, 4});
    EXPECT(p.get_modules().size() == 5);
    for(auto* mod : p.get_modules())
    {
        if(mod == p.get_main_module())
            continue;
        EXPECT(migraphx::none_of(*mod, [](const auto& ins) { return ins.name() == "dyn_slice"; }));
    }

    p.compile(migraphx::make_target("ref"));
    std::vector<float> box_values    = {0.0f,
                                        0.0f,
                                        1.0f,
                                        1.0f,
                                        0.0f,
                                        2.0f,
                                        1.0f,
                                        3.0f,
                                        0.0f,
                                        4.0f,
                                        1.0f,
                                        5.0f,
                                        0.0f,
                                        4.0f,
                                        1.0f,
                                        5.0f};
    std::vector<float> score_values0 = {0.9f, 0.8f, 0.7f, 0.6f};
    std::vector<float> score_values1 = {0.85f, 0.75f, 0.65f, 0.55f};
    auto run                         = [&](std::vector<char> condition_values) {
        migraphx::parameter_map params;
        params["condition"] = migraphx::argument{migraphx::shape{migraphx::shape::bool_type, {4}},
                                                 condition_values.data()};
        params["boxes"]   = migraphx::argument{migraphx::shape{migraphx::shape::float_type, {4, 4}},
                                             box_values.data()};
        params["scores0"] = migraphx::argument{migraphx::shape{migraphx::shape::float_type, {4}},
                                               score_values0.data()};
        params["scores1"] = migraphx::argument{migraphx::shape{migraphx::shape::float_type, {4}},
                                               score_values1.data()};
        return p.eval(params).back();
    };
    auto empty = run({0, 0, 0, 0});
    EXPECT(empty.get_shape().lens() == std::vector<std::size_t>{0, 3});
    EXPECT(empty.to_vector<float>().empty());
    auto nonempty = run({1, 1, 1, 1});
    EXPECT(nonempty.get_shape().lens() == std::vector<std::size_t>{6, 3});
    EXPECT(nonempty.to_vector<float>().size() == 18);
}

TEST_CASE(split_sym_dim_data_dependent_clone_cap_is_noop)
{
    migraphx::program p;
    auto& m     = *p.get_main_module();
    auto starts = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto data0  = m.add_parameter("data0", migraphx::shape{migraphx::shape::float_type, {4}});
    auto count0 = m.add_parameter("count0", migraphx::shape{migraphx::shape::int64_type, {1}});
    auto slice0 = m.add_instruction(
        migraphx::make_op(
            "dyn_slice",
            {{"axes", {0}},
             {"starts", {0}},
             {"ends", migraphx::value::array{migraphx::to_value(var("count_0", {0, 4}))}},
             {"always_leq", true}}),
        data0,
        starts,
        count0);
    auto data1  = m.add_parameter("data1", migraphx::shape{migraphx::shape::float_type, {4}});
    auto count1 = m.add_parameter("count1", migraphx::shape{migraphx::shape::int64_type, {1}});
    auto slice1 = m.add_instruction(
        migraphx::make_op(
            "dyn_slice",
            {{"axes", {0}},
             {"starts", {0}},
             {"ends", migraphx::value::array{migraphx::to_value(var("count_1", {0, 4}))}},
             {"always_leq", true}}),
        data1,
        starts,
        count1);
    auto data2  = m.add_parameter("data2", migraphx::shape{migraphx::shape::float_type, {4}});
    auto count2 = m.add_parameter("count2", migraphx::shape{migraphx::shape::int64_type, {1}});
    auto slice2 = m.add_instruction(
        migraphx::make_op(
            "dyn_slice",
            {{"axes", {0}},
             {"starts", {0}},
             {"ends", migraphx::value::array{migraphx::to_value(var("count_2", {0, 4}))}},
             {"always_leq", true}}),
        data2,
        starts,
        count2);
    auto joined =
        m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), slice0, slice1, slice2);
    m.add_return({m.add_instruction(migraphx::make_op("relu"), joined)});
    auto expected = p;

    run_pass(p, 4);

    EXPECT(p == expected);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
