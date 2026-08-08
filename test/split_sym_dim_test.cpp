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
#include <migraphx/make_op.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/split_sym_dim.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/value.hpp>
#include <test.hpp>
#include <limits>
#include <numeric>
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

migraphx::operation fixed_pad(std::initializer_list<se> dims, float value = 0.0f)
{
    return migraphx::make_op(
        "fixed_pad",
        {{"dims", migraphx::to_value(std::vector<se>(dims.begin(), dims.end()))},
         {"value", value}});
}

migraphx::operation symbolic_multibroadcast(std::initializer_list<se> dims)
{
    std::vector<dd> output_dims(dims.begin(), dims.end());
    return migraphx::make_op("multibroadcast", {{"out_dyn_dims", migraphx::to_value(output_dims)}});
}

migraphx::operation symbolic_broadcast(std::size_t axis, std::initializer_list<se> dims)
{
    std::vector<dd> output_dims(dims.begin(), dims.end());
    return migraphx::make_op("broadcast",
                             {{"axis", axis}, {"out_dyn_dims", migraphx::to_value(output_dims)}});
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
    auto start = m.add_instruction(
        migraphx::make_op("eval_expr_from_shape", {{"expressions", migraphx::to_value(starts)}}),
        sources);
    auto end = m.add_instruction(
        migraphx::make_op("eval_expr_from_shape", {{"expressions", migraphx::to_value(ends)}}),
        sources);
    return m.add_instruction(migraphx::make_op("dyn_slice",
                                               {{"axes", axes},
                                                {"starts", migraphx::to_value(starts)},
                                                {"ends", migraphx::to_value(ends)}}),
                             input,
                             start,
                             end);
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

template <class F>
std::vector<migraphx::module_ref>
add_clones(migraphx::program& p, std::size_t block, const std::vector<clone_spec>& specs, F f)
{
    std::vector<migraphx::module_ref> modules;
    for(std::size_t i = 0; i < specs.size(); ++i)
    {
        auto* sm = p.create_module("main:split_sym_dim_" + std::to_string(block) + "_" +
                                   std::to_string(i));
        f(*sm, specs.at(i));
        modules.push_back(sm);
    }
    return modules;
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

migraphx::program make_transformer_core_program()
{
    auto sequence = var("sequence", {4, 16}, {8});
    auto c2       = lit(2);
    auto c8       = lit(8);

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

// Minimal FCN-style symbolic block.
//
// Before:
//   main: data[n, 4] -> relu -> return
//
// After:
//   main: data
//           -> select_module(clones for [1], [2], [3..4], [5..8])
//           -> get_tuple_elem
//           -> dyn_slice(axis=0, end=n)
//           -> return
//
//   clone for route [lo..hi]:
//     data[n in lo..hi, 4] -> fixed_pad[hi, 4] -> relu -> return
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
        auto pad    = sm.add_instruction(fixed_pad({lit(clone.max), lit(4)}), input);
        auto output = sm.add_instruction(migraphx::make_op("relu"), pad);
        sm.add_return({output});
    });

    auto& m        = *expected.get_main_module();
    auto input     = m.add_parameter("data", symbolic_shape({n, lit(4)}));
    auto optimal_n = var("#split_sym_dim_n_opt", {1, 8}, {1, 2, 4, 8});
    auto select    = add_select_module(m, {input}, modules, {symbolic_shape({optimal_n, lit(4)})});
    auto output    = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    output         = add_back_slice(m, output, {input}, {0}, {n});
    m.add_return({output});

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
        auto output = sm.add_instruction(fixed_pad({lit(1), lit(clone.max), lit(4)}), input);
        output      = sm.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), output);
        output      = sm.add_instruction(migraphx::make_op("reshape", {{"dims", {0, 0}}}), output);
        output =
            sm.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), output);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto input          = expected_main.add_parameter("data", symbolic_shape({lit(1), n, lit(4)}));
    auto optimal_n      = var("#split_sym_dim_n_opt", {1, 8}, {1, 2, 4, 8});
    migraphx::shape output_shape{
        migraphx::shape::float_type, {dd{lit(4)}, dd{optimal_n}}, {lit(1), lit(4)}};
    auto select = add_select_module(expected_main, {input}, modules, {output_shape});
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    output = add_back_slice(expected_main, output, {input}, {1}, {n});
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

TEST_CASE(split_sym_dim_materializes_symbolic_multibroadcast)
{
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m     = *p.get_main_module();
    auto data   = m.add_parameter("data", symbolic_shape({n, lit(3)}));
    auto bias   = m.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {3}});
    auto bcast  = m.add_instruction(symbolic_multibroadcast({n, lit(3)}), bias);
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
        auto padded_data  = sm.add_instruction(fixed_pad({lit(clone.max), lit(3)}), clone_data);
        auto clone_output = sm.add_instruction(migraphx::make_op("add"), padded_data, clone_bcast);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter("data", symbolic_shape({n, lit(3)}));
    auto expected_bias =
        expected_main.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {3}});
    auto optimal_n = var("#split_sym_dim_n_opt", {1, 4}, {1, 2, 4});
    auto select    = add_select_module(expected_main,
                                       {expected_bias, expected_data},
                                       modules,
                                       {symbolic_shape({optimal_n, lit(3)})});
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    expected_output =
        add_back_slice(expected_main, expected_output, {expected_bias, expected_data}, {0}, {n});
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
    auto bcast  = m.add_instruction(symbolic_broadcast(1, {n, lit(3), lit(4)}), bias);
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
        auto padded_data =
            sm.add_instruction(fixed_pad({lit(clone.max), lit(3), lit(4)}), clone_data);
        auto clone_output = sm.add_instruction(migraphx::make_op("add"), padded_data, clone_bcast);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_data  = expected_main.add_parameter("data", symbolic_shape({n, lit(3), lit(4)}));
    auto expected_bias =
        expected_main.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {3}});
    auto optimal_n = var("#split_sym_dim_n_opt", {1, 4}, {1, 2, 4});
    auto select    = add_select_module(expected_main,
                                       {expected_bias, expected_data},
                                       modules,
                                       {symbolic_shape({optimal_n, lit(3), lit(4)})});
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    expected_output =
        add_back_slice(expected_main, expected_output, {expected_bias, expected_data}, {0}, {n});
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
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
        auto output = sm.add_instruction(fixed_pad({lit(clone.max), lit(4)}), input);
        for(std::size_t i = 0; i < 5; ++i)
            output = sm.add_instruction(migraphx::make_op("relu"), output);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto input          = expected_main.add_parameter("data", symbolic_shape({n, lit(4)}));
    auto optimal_n      = var("#split_sym_dim_n_opt", {1, 8}, {1, 2, 4, 8});
    auto select =
        add_select_module(expected_main, {input}, modules, {symbolic_shape({optimal_n, lit(4)})});
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
        auto pad    = sm.add_instruction(fixed_pad({lit(clone.max), lit(4)}), input);
        auto output = sm.add_instruction(migraphx::make_op("relu"), pad);
        sm.add_return({output});
    });

    auto& m        = *expected.get_main_module();
    auto input     = m.add_parameter("data", symbolic_shape({n, lit(4)}));
    auto optimal_n = var("#split_sym_dim_n_opt", {1, 8}, {1, 8});
    auto select    = add_select_module(m, {input}, modules, {symbolic_shape({optimal_n, lit(4)})});
    auto output    = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    output         = add_back_slice(m, output, {input}, {0}, {n});
    m.add_return({output});

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
    std::vector<migraphx::module_ref> modules;
    for(std::size_t i = 0; i < clones.size(); ++i)
    {
        const auto& clone = clones.at(i);
        auto* sm          = expected.create_module("main:split_sym_dim_0_" + std::to_string(i));
        auto input        = sm->add_parameter("data",
                                       symbolic_shape({var("b", {clone.b_min, clone.b_max}),
                                                              var("s", {clone.s_min, clone.s_max})},
                                                      migraphx::shape::bool_type));
        auto pad =
            sm->add_instruction(fixed_pad({lit(clone.b_max), lit(clone.s_max)}, 1.0f), input);
        auto output = sm->add_instruction(migraphx::make_op("reduce_all", {{"axes", {1}}}), pad);
        sm->add_return({output});
        modules.push_back(sm);
    }

    auto& expected_main = *expected.get_main_module();
    auto input =
        expected_main.add_parameter("data", symbolic_shape({b, s}, migraphx::shape::bool_type));
    auto optimal_b = var("#split_sym_dim_b_opt", {1, 2}, {1, 2});
    auto select =
        add_select_module(expected_main,
                          {input},
                          modules,
                          {symbolic_shape({optimal_b, lit(1)}, migraphx::shape::bool_type)});
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
        auto pad = sm.add_instruction(fixed_pad({lit(1), lit(1), lit(clone.max), lit(clone.max)}),
                                      clone_input);
        auto output = sm.add_instruction(migraphx::make_op("convolution"), pad, clone_weights);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_input =
        expected_main.add_parameter("data", symbolic_shape({lit(1), lit(1), s, s}));
    auto optimal_s   = var("#split_sym_dim_s_opt", {8, 16}, {8, 12, 16});
    auto conv_extent = optimal_s - 2;
    auto conv_select =
        add_select_module(expected_main,
                          {expected_input},
                          convolution_modules,
                          {symbolic_shape({lit(1), lit(1), conv_extent, conv_extent})});
    auto conv_output = expected_main.add_instruction(
        migraphx::make_op("get_tuple_elem", {{"index", 0}}), conv_select);
    conv_output =
        add_back_slice(expected_main, conv_output, {expected_input}, {2, 3}, {s - 2, s - 2});

    auto boundary_extent = migraphx::sym::min(s - 2, conv_extent);
    auto boundary_shape  = symbolic_shape({lit(1), lit(1), boundary_extent, boundary_extent});
    auto pooling_modules = add_clones(expected, 1, clones, [&](auto& sm, const auto& clone) {
        auto clone_s = var("s", {clone.min, clone.max});
        sm.add_parameter("data", symbolic_shape({lit(1), lit(1), clone_s, clone_s}));
        auto boundary      = sm.add_parameter("#split_sym_dim_input_1_0", boundary_shape);
        auto target_extent = clone.max - 2;
        auto pad =
            sm.add_instruction(fixed_pad({lit(1), lit(1), lit(target_extent), lit(target_extent)},
                                         std::numeric_limits<float>::lowest()),
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

    auto pooled_extent = (optimal_s - 3) / 2 + 1;
    auto pool_select =
        add_select_module(expected_main,
                          {conv_output, expected_input},
                          pooling_modules,
                          {symbolic_shape({lit(1), lit(1), pooled_extent, pooled_extent})});
    auto output = expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}),
                                                pool_select);
    auto output_extent = (s - 3) / 2 + 1;
    output             = add_back_slice(
        expected_main, output, {expected_input}, {2, 3}, {output_extent, output_extent});
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
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
        auto pad0    = sm.add_instruction(fixed_pad({lit(clone.max), lit(4)}), x0);
        auto y0      = sm.add_instruction(migraphx::make_op("relu"), pad0);
        auto pad1    = sm.add_instruction(fixed_pad({lit(clone.max), lit(4)}), x1);
        auto y1      = sm.add_instruction(migraphx::make_op("relu"), pad1);
        sm.add_return({y0, y1});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_x0    = expected_main.add_parameter("x0", symbolic_shape({n, lit(4)}));
    auto expected_x1    = expected_main.add_parameter("x1", symbolic_shape({n, lit(4)}));
    auto optimal_n      = var("#split_sym_dim_n_opt", {1, 4}, {1, 2, 4});
    auto output_shape   = symbolic_shape({optimal_n, lit(4)});
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

TEST_CASE(split_sym_dim_keeps_unsupported_op_between_blocks)
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
    auto first_modules = add_clones(expected, 0, clones, [&](auto& sm, const auto& clone) {
        auto input =
            sm.add_parameter("x", symbolic_shape({var("n", {clone.min, clone.max}), lit(4)}));
        auto pad    = sm.add_instruction(fixed_pad({lit(clone.max), lit(4)}), input);
        auto output = sm.add_instruction(migraphx::make_op("relu"), pad);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto input          = expected_main.add_parameter("x", symbolic_shape({n, lit(4)}));
    auto optimal_n      = var("#split_sym_dim_n_opt", {1, 4}, {1, 2, 4});
    auto first_select   = add_select_module(
        expected_main, {input}, first_modules, {symbolic_shape({optimal_n, lit(4)})});
    auto first_output = expected_main.add_instruction(
        migraphx::make_op("get_tuple_elem", {{"index", 0}}), first_select);
    first_output = add_back_slice(expected_main, first_output, {input}, {0}, {n});
    auto sliced  = expected_main.add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {2}}}), first_output);
    auto boundary_shape = symbolic_shape({migraphx::sym::min(n, optimal_n), lit(2)});

    auto second_modules = add_clones(expected, 1, clones, [&](auto& sm, const auto& clone) {
        sm.add_parameter("x", symbolic_shape({var("n", {clone.min, clone.max}), lit(4)}));
        auto boundary = sm.add_parameter("#split_sym_dim_input_1_0", boundary_shape);
        auto pad      = sm.add_instruction(fixed_pad({lit(clone.max), lit(2)}), boundary);
        auto output   = sm.add_instruction(migraphx::make_op("relu"), pad);
        sm.add_return({output});
    });
    auto second_select  = add_select_module(
        expected_main, {sliced, input}, second_modules, {symbolic_shape({optimal_n, lit(2)})});
    auto output = expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}),
                                                second_select);
    output      = add_back_slice(expected_main, output, {input}, {0}, {n});
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
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
        auto pad    = sm.add_instruction(fixed_pad({lit(clone.max), lit(4)}), input);
        auto output = sm.add_instruction(migraphx::make_op("relu"), pad);
        sm.add_return({output});
    });
    auto m_modules = add_clones(expected, 1, clones, [&](auto& sm, const auto& clone) {
        auto input =
            sm.add_parameter("x1", symbolic_shape({var("m", {clone.min, clone.max}), lit(4)}));
        auto pad    = sm.add_instruction(fixed_pad({lit(clone.max), lit(4)}), input);
        auto output = sm.add_instruction(migraphx::make_op("relu"), pad);
        sm.add_return({output});
    });

    auto& expected_main  = *expected.get_main_module();
    auto expected_x0     = expected_main.add_parameter("x0", symbolic_shape({n, lit(4)}));
    auto expected_x1     = expected_main.add_parameter("x1", symbolic_shape({m_dim, lit(4)}));
    auto runtime_sources = std::vector<migraphx::instruction_ref>{expected_x1, expected_x0};

    auto optimal_n = var("#split_sym_dim_n_opt", {1, 4}, {1, 2, 4});
    auto select_n  = add_select_module(
        expected_main, {expected_x0}, n_modules, {symbolic_shape({optimal_n, lit(4)})});
    auto output_n = expected_main.add_instruction(
        migraphx::make_op("get_tuple_elem", {{"index", 0}}), select_n);
    output_n = add_back_slice(expected_main, output_n, runtime_sources, {0}, {n});

    auto optimal_m = var("#split_sym_dim_m_opt", {1, 4}, {1, 2, 4});
    auto select_m  = add_select_module(
        expected_main, {expected_x1}, m_modules, {symbolic_shape({optimal_m, lit(4)})});
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
        auto pad =
            sm.add_instruction(fixed_pad({lit(clone.max), lit(1), lit(5), lit(5)}), clone_input);
        auto output = sm.add_instruction(migraphx::make_op("convolution"), pad, clone_weights);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_input =
        expected_main.add_parameter("data", symbolic_shape({n, lit(1), lit(5), lit(5)}));
    auto optimal_n = var("#split_sym_dim_n_opt", {1, 4}, {1, 2, 4});
    auto select    = add_select_module(expected_main,
                                       {expected_input},
                                    modules,
                                       {symbolic_shape({optimal_n, lit(1), lit(3), lit(3)})});
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    output = add_back_slice(expected_main, output, {expected_input}, {0}, {n});
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

// Before:
//   main: image[1, 3, spatial, spatial]
//           -> convolution(w0)
//           -> relu
//           -> convolution(w1)
//           -> relu
//           -> max_pool
//           -> return
//
// After:
//   main: image -> select_module(clones for [8], [9..12], [13..16])
//           -> get_tuple_elem
//           -> dyn_slice(axes={2, 3}, ends={(spatial - 6) / 2 + 1, ...})
//           -> return
//
//   clone for route [lo..hi]:
//     image[spatial in lo..hi]
//       -> fixed_pad[1, 3, hi, hi]
//       -> convolution(w0) -> relu -> convolution(w1) -> relu -> max_pool
//       -> return
//
// The safe interior slices are coalesced, so the complete CNN body stays in one static clone.
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
        auto pad =
            sm.add_instruction(fixed_pad({lit(1), lit(3), lit(clone.max), lit(clone.max)}), input);
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
    auto input =
        expected_main.add_parameter("image", symbolic_shape({lit(1), lit(3), spatial, spatial}));
    auto optimal_spatial = var("#split_sym_dim_spatial_opt", {8, 16}, {8, 12, 16});
    auto output_extent   = (optimal_spatial - 6) / 2 + 1;
    auto select =
        add_select_module(expected_main,
                          {input},
                          modules,
                          {symbolic_shape({lit(1), lit(6), output_extent, output_extent})});
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
        auto pad           = sm.add_instruction(fixed_pad({lit(1), lit(1), lit(clone.max)}), input);
        auto convolution   = sm.add_instruction(
            migraphx::make_op("convolution",
                                {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
            pad,
            clone_weights);
        auto extent = sm.add_instruction(
            migraphx::make_op("eval_expr_from_shape",
                              {{"expressions", migraphx::to_value(std::vector<se>{sequence - 2})}}),
            input);
        auto masked       = add_mask(sm, convolution, indices, extent, negative_infinity, 2);
        auto clone_output = sm.add_instruction(migraphx::make_op("softmax", {{"axis", 2}}), masked);
        sm.add_return({clone_output});
    });

    auto& expected_main = *expected.get_main_module();
    auto expected_input =
        expected_main.add_parameter("x", symbolic_shape({lit(1), lit(1), sequence}));
    auto optimal_sequence = var("#split_sym_dim_sequence_opt", {4, 16}, {4, 8, 16});
    auto optimal_extent   = optimal_sequence - 2;
    auto select           = add_select_module(expected_main,
                                              {expected_input},
                                    modules,
                                              {symbolic_shape({lit(1), lit(1), optimal_extent})});
    auto expected_output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    expected_output =
        add_back_slice(expected_main, expected_output, {expected_input}, {2}, {sequence - 2});
    expected_main.add_return({expected_output});

    EXPECT(p.sort() == expected.sort());
}

// Before:
//   main:
//     scores   = dot(query[2, sequence, 8], key_transposed[2, 8, sequence])
//     context  = dot(scores, value[2, sequence, 8])
//     output   = relu(add(context, query))
//     return output
//
// After:
//   main: key_transposed, query, value
//           -> select_module(clones for [4], [5..8], [9..16])
//           -> get_tuple_elem
//           -> dyn_slice(axis=1, end=sequence)
//           -> return
//
//   clone for route [lo..hi]:
//     padded_query = fixed_pad(query, [2, hi, 8])
//     padded_key   = fixed_pad(key_transposed, [2, 8, hi])
//     padded_value = fixed_pad(value, [2, hi, 8])
//     scores       = mask_zero(dot(padded_query, padded_key), extent=sequence)
//     value        = mask_zero(padded_value, extent=sequence)
//     output       = relu(add(dot(scores, value), padded_query))
//     return output
TEST_CASE(split_sym_dim_specializes_transformer_core)
{
    auto sequence = var("sequence", {4, 16}, {8});
    auto p        = make_transformer_core_program();
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

        auto padded_query = sm.add_instruction(fixed_pad({lit(2), lit(clone.max), lit(8)}), query);
        auto padded_key   = sm.add_instruction(fixed_pad({lit(2), lit(8), lit(clone.max)}), key);
        auto scores       = sm.add_instruction(migraphx::make_op("dot"), padded_query, padded_key);
        auto padded_value = sm.add_instruction(fixed_pad({lit(2), lit(clone.max), lit(8)}), value);
        auto extent       = sm.add_instruction(
            migraphx::make_op("eval_expr_from_shape",
                                    {{"expressions", migraphx::to_value(std::vector<se>{sequence})}}),
            query);
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
    auto optimal_sequence = var("#split_sym_dim_sequence_opt", {4, 16}, {4, 8, 16});
    auto select           = add_select_module(expected_main,
                                              {key, query, value},
                                    modules,
                                              {symbolic_shape({lit(2), optimal_sequence, lit(8)})});
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    output = add_back_slice(expected_main, output, {value, key, query}, {1}, {sequence});
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

// Before:
//   main:
//     query   = dot(x[2, sequence, 8], wq)
//     key     = dot(x, wk)
//     value   = dot(x, wv)
//     scores  = dot(query, transpose(key))
//     context = dot(softmax(scores), value)
//     output  = add(dot(context, wo), x)
//     return output
//
// After:
//   main: x -> select_module(clones for [4], [5..8], [9..16])
//           -> get_tuple_elem
//           -> dyn_slice(axis=1, end=sequence)
//           -> return
//
//   clone for route [lo..hi]:
//     padded_x = fixed_pad(x, [2, hi, 8])
//     query, key, value = dot(padded_x, wq/wk/wv)
//     scores  = mask_neg_inf(dot(query, transpose(key)), extent=sequence)
//     context = dot(softmax(scores), value)
//     output  = add(dot(context, wo), padded_x)
//     return output
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

        auto padded = sm.add_instruction(fixed_pad({lit(2), lit(clone.max), lit(8)}), input);
        auto query  = sm.add_instruction(migraphx::make_op("dot"), padded, wq);
        auto key    = sm.add_instruction(migraphx::make_op("dot"), padded, wk);
        auto value  = sm.add_instruction(migraphx::make_op("dot"), padded, wv);
        auto key_transposed =
            sm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), key);
        auto scores = sm.add_instruction(migraphx::make_op("dot"), query, key_transposed);
        auto extent = sm.add_instruction(
            migraphx::make_op("eval_expr_from_shape",
                              {{"expressions", migraphx::to_value(std::vector<se>{sequence})}}),
            input);
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
    auto optimal_sequence = var("#split_sym_dim_sequence_opt", {4, 16}, {4, 8, 16});
    auto select           = add_select_module(
        expected_main, {input}, modules, {symbolic_shape({lit(2), optimal_sequence, lit(8)})});
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

        auto padded_input =
            sm.add_instruction(fixed_pad({lit(2), lit(clone.max), lit(clone.max)}), input);
        auto extent = sm.add_instruction(
            migraphx::make_op("eval_expr_from_shape",
                              {{"expressions", migraphx::to_value(std::vector<se>{sequence})}}),
            input);
        auto masked_input = add_mask(sm, padded_input, indices, extent, negative_infinity, 1);
        auto probabilities =
            sm.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), masked_input);
        auto padded_value = sm.add_instruction(fixed_pad({lit(2), lit(clone.max), lit(8)}), value);
        auto masked_probabilities = add_mask(sm, probabilities, indices, extent, zero, 2);
        auto masked_value         = add_mask(sm, padded_value, indices, extent, zero, 1);
        auto output =
            sm.add_instruction(migraphx::make_op("dot"), masked_probabilities, masked_value);
        sm.add_return({output});
    });

    auto& expected_main = *expected.get_main_module();
    auto input = expected_main.add_parameter("x", symbolic_shape({lit(2), sequence, sequence}));
    auto value = expected_main.add_parameter("value", symbolic_shape({lit(2), sequence, lit(8)}));
    auto optimal_sequence = var("#split_sym_dim_sequence_opt", {4, 16}, {4, 8, 16});
    auto select           = add_select_module(expected_main,
                                              {value, input},
                                    modules,
                                              {symbolic_shape({lit(2), optimal_sequence, lit(8)})});
    auto output =
        expected_main.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    output = add_back_slice(expected_main, output, {value, input}, {1}, {sequence});
    expected_main.add_return({output});

    EXPECT(p.sort() == expected.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
