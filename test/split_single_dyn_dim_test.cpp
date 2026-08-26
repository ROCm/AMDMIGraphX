/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/split_single_dyn_dim.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/program.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/builtin.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/verify.hpp>
#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <test.hpp>

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::split_single_dyn_dim{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(dynamic_batch)
{
    migraphx::program p0;
    {
        auto* mm0 = p0.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input0 = mm0->add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
        auto literal_ins = mm0->add_literal(migraphx::literal{lit_s, {6}});

        // create batch submodules; each captures the literal instead of copying it
        auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
            auto* submod = p0.create_module(module_name);
            migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
            auto sm_input = submod->add_parameter("data", sm_shape);
            auto broadcast_lit =
                submod->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, sm_input);
            auto add_ins =
                submod->add_instruction(migraphx::make_op("add"), sm_input, broadcast_lit);
            submod->add_return({add_ins});
            return submod;
        };
        auto* dim1 = create_submodule(1, "dim_1");
        auto* dim2 = create_submodule(2, "dim_2");
        auto* dim3 = create_submodule(3, "dim_3");
        auto* dim4 = create_submodule(4, "dim_4");

        std::vector<migraphx::shape> sub_shapes = {};
        sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
        migraphx::shape out_attr = migraphx::shape{sub_shapes};
        auto sm_ins              = mm0->add_instruction(
            migraphx::make_op("select_module",
                              {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
            {input0},
            {dim1, dim2, dim3, dim4});
        auto ret =
            mm0->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
        mm0->add_return({ret});
    }

    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input1 = mm1->add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
        auto literal_ins = mm1->add_literal(migraphx::literal{lit_s, {6}});
        auto broadcast_lit =
            mm1->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, input1);
        auto add_ins = mm1->add_instruction(migraphx::make_op("add"), input1, broadcast_lit);
        mm1->add_return({add_ins});
    }
    run_pass(p1);

    EXPECT(p0 == p1);
}

TEST_CASE(dynamic_batch_multiple_input)
{
    migraphx::program p0;
    {
        auto* mm0 = p0.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input0 = mm0->add_parameter("data0", s);
        auto input1 = mm0->add_parameter("data1", s);
        auto input2 = mm0->add_parameter("data2", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
        auto literal_ins = mm0->add_literal(migraphx::literal{lit_s, {6}});

        // create batch submodules; each captures the literal instead of copying it
        auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
            auto* submod = p0.create_module(module_name);
            migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
            auto sm_input0     = submod->add_parameter("data0", sm_shape);
            auto sm_input1     = submod->add_parameter("data1", sm_shape);
            auto sm_input2     = submod->add_parameter("data2", sm_shape);
            auto broadcast_lit = submod->add_instruction(
                migraphx::make_op("multibroadcast"), literal_ins, sm_input0);
            auto add_ins0 =
                submod->add_instruction(migraphx::make_op("add"), sm_input0, broadcast_lit);
            auto add_ins1 = submod->add_instruction(migraphx::make_op("add"), add_ins0, sm_input1);
            auto add_ins2 = submod->add_instruction(migraphx::make_op("add"), add_ins1, sm_input2);
            submod->add_return({add_ins2});
            return submod;
        };
        auto* dim1 = create_submodule(1, "dim_1");
        auto* dim2 = create_submodule(2, "dim_2");
        auto* dim3 = create_submodule(3, "dim_3");
        auto* dim4 = create_submodule(4, "dim_4");

        std::vector<migraphx::shape> sub_shapes = {};
        sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
        migraphx::shape out_attr = migraphx::shape{sub_shapes};
        auto sm_ins              = mm0->add_instruction(
            migraphx::make_op("select_module",
                              {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
            {input0, input1, input2},
            {dim1, dim2, dim3, dim4});
        auto ret =
            mm0->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
        mm0->add_return({ret});
    }

    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input0 = mm1->add_parameter("data0", s);
        auto input1 = mm1->add_parameter("data1", s);
        auto input2 = mm1->add_parameter("data2", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
        auto literal_ins = mm1->add_literal(migraphx::literal{lit_s, {6}});
        auto broadcast_lit =
            mm1->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, input0);
        auto add_ins0 = mm1->add_instruction(migraphx::make_op("add"), input0, broadcast_lit);
        auto add_ins1 = mm1->add_instruction(migraphx::make_op("add"), add_ins0, input1);
        auto add_ins2 = mm1->add_instruction(migraphx::make_op("add"), add_ins1, input2);
        mm1->add_return({add_ins2});
    }
    run_pass(p1);

    EXPECT(p0.sort() == p1.sort());
}

TEST_CASE(multiple_outputs)
{
    migraphx::program p0;
    {
        auto* mm0 = p0.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input0 = mm0->add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
        auto literal_ins = mm0->add_literal(migraphx::literal{lit_s, {6}});

        // create batch submodules; each captures the literal instead of copying it
        auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
            auto* submod = p0.create_module(module_name);
            migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
            auto sm_input = submod->add_parameter("data", sm_shape);
            auto broadcast_lit =
                submod->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, sm_input);
            auto add0_ins =
                submod->add_instruction(migraphx::make_op("add"), sm_input, broadcast_lit);
            auto add1_ins = submod->add_instruction(migraphx::make_op("add"), sm_input, sm_input);
            submod->add_return({add0_ins, add1_ins});
            return submod;
        };
        auto* dim1 = create_submodule(1, "dim_1");
        auto* dim2 = create_submodule(2, "dim_2");
        auto* dim3 = create_submodule(3, "dim_3");
        auto* dim4 = create_submodule(4, "dim_4");

        std::vector<migraphx::shape> sub_shapes = {};
        migraphx::shape tmp_s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        sub_shapes.push_back(tmp_s);
        sub_shapes.push_back(tmp_s);
        migraphx::shape out_attr = migraphx::shape{sub_shapes};
        auto sm_ins              = mm0->add_instruction(
            migraphx::make_op("select_module",
                              {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
            {input0},
            {dim1, dim2, dim3, dim4});
        auto ret0 =
            mm0->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
        auto ret1 =
            mm0->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), sm_ins);
        mm0->add_return({ret0, ret1});
    }

    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input1 = mm1->add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
        auto literal_ins = mm1->add_literal(migraphx::literal{lit_s, {6}});
        auto broadcast_lit =
            mm1->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, input1);
        auto add0_ins = mm1->add_instruction(migraphx::make_op("add"), input1, broadcast_lit);
        auto add1_ins = mm1->add_instruction(migraphx::make_op("add"), input1, input1);
        mm1->add_return({add0_ins, add1_ins});
    }
    run_pass(p1);

    EXPECT(p0 == p1);
}

// code coverage, does nothing
TEST_CASE(empty_param_shapes)
{
    migraphx::program p0;
    {
        auto* mm0 = p0.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {1, 4}};
        auto input0 = mm0->add_literal(migraphx::literal{s, {0, 1, 2, 3}});
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
        auto literal_ins = mm0->add_literal(migraphx::literal{lit_s, {6}});
        auto broadcast_lit =
            mm0->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, input0);
        auto add0_ins = mm0->add_instruction(migraphx::make_op("add"), input0, broadcast_lit);
        mm0->add_return({add0_ins});
    }
    migraphx::program p1 = p0;
    run_pass(p0);
    EXPECT(p0 == p1);
};

// code coverage, does nothing
TEST_CASE(multiple_non_fixed_dd_in_a_param)
{
    migraphx::program p0;
    {
        auto* mm0 = p0.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 20}}};
        auto input0 = mm0->add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
        auto literal_ins = mm0->add_literal(migraphx::literal{lit_s, {6}});
        auto broadcast_lit =
            mm0->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, input0);
        auto add0_ins = mm0->add_instruction(migraphx::make_op("add"), input0, broadcast_lit);
        mm0->add_return({add0_ins});
    }
    migraphx::program p1 = p0;
    run_pass(p0);
    EXPECT(p0 == p1);
}

// code coverage, does nothing
TEST_CASE(different_non_fixed_dd)
{
    migraphx::program p0;
    {
        auto* mm1 = p0.get_main_module();
        migraphx::shape s0{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        migraphx::shape s1{migraphx::shape::float_type, {{3, 6}, {1, 1}, {4, 4}}};
        auto input0 = mm1->add_parameter("data0", s0);
        auto input1 = mm1->add_parameter("data1", s1);
        auto broadcast_in0 =
            mm1->add_instruction(migraphx::make_op("multibroadcast"), input0, input1);
        auto broadcast_in1 =
            mm1->add_instruction(migraphx::make_op("multibroadcast"), input1, input0);
        auto add0_ins =
            mm1->add_instruction(migraphx::make_op("add"), broadcast_in0, broadcast_in1);
        mm1->add_return({add0_ins});
    }
    migraphx::program p1 = p0;
    run_pass(p0);
    EXPECT(p0 == p1);
}

static std::size_t literal_bytes(const migraphx::module& m)
{
    return migraphx::transform_accumulate(
        m.begin(), m.end(), std::size_t{0}, std::plus<>{}, [](const migraphx::instruction& ins) {
            return ins.name() == "@literal" ? ins.get_shape().bytes() : 0;
        });
}

static std::size_t literal_bytes(const migraphx::program& p)
{
    auto mods = p.get_modules();
    return migraphx::transform_accumulate(
        mods.begin(), mods.end(), std::size_t{0}, std::plus<>{}, [](const migraphx::module* m) {
            return literal_bytes(*m);
        });
}

// Specializing must not duplicate the weights: the literals stay in the main module and every
// submodule captures them.
TEST_CASE(literals_are_captured_not_copied)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
    auto input = mm->add_parameter("data", s);
    migraphx::shape lit_s{migraphx::shape::float_type, {4}};
    auto literal_ins = mm->add_literal(migraphx::literal{lit_s, {1, 2, 3, 4}});
    auto broadcast_lit =
        mm->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, input);
    auto add_ins = mm->add_instruction(migraphx::make_op("add"), input, broadcast_lit);
    mm->add_return({add_ins});

    auto before = literal_bytes(p);
    run_pass(p);

    EXPECT(literal_bytes(p) == before);
    EXPECT(literal_bytes(*mm) == before);
    for(const auto* mod : p.get_modules())
    {
        if(mod == mm)
            continue;
        EXPECT(literal_bytes(*mod) == 0);
    }

    // a capture crosses a module boundary, so it has to survive a serialization round trip
    migraphx::program reloaded;
    reloaded.from_value(p.to_value());
    EXPECT(reloaded == p);

    // the captured literal still feeds the submodules at runtime
    p.compile(migraphx::make_target("ref"));
    for(std::size_t batch : {std::size_t{1}, std::size_t{3}})
    {
        migraphx::shape input_s{migraphx::shape::float_type, {batch, 4}};
        std::vector<float> input_data(input_s.elements(), 10);
        migraphx::parameter_map params;
        params["data"] = migraphx::argument{input_s, input_data.data()};
        auto result    = p.eval(params).back();

        std::vector<float> expected;
        std::generate_n(std::back_inserter(expected), input_s.elements(), [n = 0]() mutable {
            return 11.0f + static_cast<float>(n++ % 4);
        });
        std::vector<float> actual;
        result.visit([&](auto output) { actual.assign(output.begin(), output.end()); });
        EXPECT(actual == expected);
    }
}

// check that the parameter inputs into select_module are lexiographically ordered
TEST_CASE(ordered_inputs_to_select_module)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
    auto input0 = mm->add_parameter("breadfruit", s);
    auto input1 = mm->add_parameter("Apricot", s);
    auto input2 = mm->add_parameter("pineapple", s);
    migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
    auto literal_ins = mm->add_literal(migraphx::literal{lit_s, {6}});
    auto broadcast_lit =
        mm->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, input0);
    auto add_ins0 = mm->add_instruction(migraphx::make_op("add"), input0, broadcast_lit);
    auto add_ins1 = mm->add_instruction(migraphx::make_op("add"), add_ins0, input1);
    auto add_ins2 = mm->add_instruction(migraphx::make_op("add"), add_ins1, input2);
    mm->add_return({add_ins2});
    run_pass(p);

    auto sm_ins = std::find_if(
        mm->begin(), mm->end(), [&](auto&& ins) { return ins.name() == "select_module"; });
    std::vector<std::string> sm_param_names;
    for(auto&& ins : sm_ins->inputs())
    {
        if(ins->name() == "@param")
        {
            auto&& param = migraphx::any_cast<migraphx::builtin::param>(ins->get_operator());
            sm_param_names.push_back(param.parameter);
        }
    }
    EXPECT(std::is_sorted(sm_param_names.begin(), sm_param_names.end()));
}

static void
run_pass(migraphx::program& p, const std::string& symbol, std::vector<std::size_t> sizes)
{
    migraphx::run_passes(p,
                         {migraphx::split_single_dyn_dim{symbol, std::move(sizes)},
                          migraphx::dead_code_elimination{}});
}

static migraphx::sym::expr symbolic_dim(const std::string& name, std::size_t max)
{
    return migraphx::sym::var(name, {1, static_cast<std::int64_t>(max)});
}

// A symbolically parsed shape carries an expression for every dimension, the static ones as
// literals, which is what makes shape::symbolic() true and gives it symbolic strides.
static migraphx::shape symbolic_shape(const std::vector<migraphx::sym::expr>& dims)
{
    std::vector<migraphx::shape::dynamic_dimension> dds;
    std::transform(dims.begin(), dims.end(), std::back_inserter(dds), [](const auto& dim) {
        return migraphx::shape::dynamic_dimension{dim};
    });
    return {migraphx::shape::float_type, dds};
}

static std::vector<migraphx::module_ref> find_specializations(migraphx::const_module_ref mod)
{
    auto select = std::find_if(
        mod->begin(), mod->end(), [](const auto& ins) { return ins.name() == "select_module"; });
    if(select == mod->end())
        return {};
    return select->module_inputs();
}

// A symbolic dimension is specialized by name, and only at the sizes asked for. A model that is
// only ever run at one token or a full prompt does not need the sizes in between.
TEST_CASE(symbolic_split_at_given_sizes)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s   = symbolic_shape(
        {migraphx::sym::lit(1), symbolic_dim("sequence_length", 4), migraphx::sym::lit(2)});
    EXPECT(s.symbolic());
    auto x = mm->add_parameter("x", s);
    auto one =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {1}});
    auto broadcast = mm->add_instruction(migraphx::make_op("multibroadcast"), one, x);
    mm->add_return({mm->add_instruction(migraphx::make_op("add"), x, broadcast)});

    run_pass(p, "sequence_length", {1, 4});

    auto specializations = find_specializations(mm);
    EXPECT(specializations.size() == 2);
    EXPECT(specializations.at(0)->name() == "dim_1");
    EXPECT(specializations.at(1)->name() == "dim_4");
    EXPECT(specializations.at(0)->get_parameter_shape("x") ==
           migraphx::shape{migraphx::shape::float_type, {1, 1, 2}});
    EXPECT(specializations.at(1)->get_parameter_shape("x") ==
           migraphx::shape{migraphx::shape::float_type, {1, 4, 2}});

    // The main module still describes the whole range, so it reports the exact output shape for
    // whatever size turns up.
    EXPECT(mm->get_parameter_shape("x") == s);
    EXPECT(mm->get_output_shapes().at(0) == s);
}

// Specializing one symbol says nothing about the others: a dimension that varies on its own has
// to keep varying, in the specializations and in the output shapes.
TEST_CASE(symbolic_split_keeps_independent_symbol)
{
    auto xs = symbolic_shape(
        {migraphx::sym::lit(1), symbolic_dim("sequence_length", 4), migraphx::sym::lit(2)});
    auto zs = symbolic_shape({symbolic_dim("other_dimension", 3), migraphx::sym::lit(2)});

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", xs);
    auto y   = mm->add_parameter("y", xs);
    auto z   = mm->add_parameter("z", zs);
    auto sum = mm->add_instruction(migraphx::make_op("add"), x, y);
    mm->add_return({sum, mm->add_instruction(migraphx::make_op("identity"), z)});

    run_pass(p, "sequence_length", {1, 4});

    auto specializations = find_specializations(mm);
    EXPECT(specializations.size() == 2);
    EXPECT(specializations.at(0)->get_parameter_shape("z").symbolic());
    EXPECT(specializations.at(0)->get_parameter_shape("z") ==
           specializations.at(1)->get_parameter_shape("z"));
    EXPECT(specializations.at(0)->get_parameter_shape("z") == zs);
    EXPECT(not specializations.at(0)->get_parameter_shape("x").dynamic());

    auto output_shapes = mm->get_output_shapes();
    EXPECT(output_shapes.size() == 2);
    EXPECT(output_shapes.at(0) == xs);
    EXPECT(output_shapes.at(1) == zs);
}

// Static inputs settle most shapes on their own, but an operation holding a symbol in an
// attribute has to be told what the symbol is worth in each specialization.
TEST_CASE(symbolic_split_specializes_op_attributes)
{
    auto xs = symbolic_shape(
        {migraphx::sym::lit(1), symbolic_dim("sequence_length", 8), migraphx::sym::lit(2)});

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", xs);
    auto zero =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {0}});
    auto broadcast = mm->add_instruction(migraphx::make_op("multibroadcast"), zero, x);
    auto sum       = mm->add_instruction(migraphx::make_op("add"), x, broadcast);
    auto alloc =
        mm->add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(xs)}}));
    mm->add_return({sum, alloc});

    run_pass(p, "sequence_length", {8});

    auto specializations = find_specializations(mm);
    EXPECT(specializations.size() == 1);
    auto* submod   = specializations.at(0);
    auto alloc_ins = std::find_if(
        submod->begin(), submod->end(), [](const auto& ins) { return ins.name() == "allocate"; });
    EXPECT(alloc_ins != submod->end());
    EXPECT(alloc_ins->get_shape() == migraphx::shape{migraphx::shape::float_type, {1, 8, 2}});
    // the main module keeps the symbol so it can still report the output shape for any size
    EXPECT(alloc->get_shape() == xs);
}

// A module that does not have the symbol is left alone. The pass sees every module in the
// program, so a specialization it just made must not be split again.
TEST_CASE(symbolic_split_leaves_other_modules_alone)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s   = symbolic_shape(
        {migraphx::sym::lit(1), symbolic_dim("sequence_length", 4), migraphx::sym::lit(2)});
    mm->add_return({mm->add_instruction(migraphx::make_op("identity"), mm->add_parameter("x", s))});
    migraphx::program before = p;

    run_pass(p, "batch_size", {1, 4});

    EXPECT(p == before);
}

TEST_CASE(symbolic_split_size_outside_range_throws)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s   = symbolic_shape(
        {migraphx::sym::lit(1), symbolic_dim("sequence_length", 4), migraphx::sym::lit(2)});
    mm->add_return({mm->add_instruction(migraphx::make_op("identity"), mm->add_parameter("x", s))});

    EXPECT(test::throws([&] { run_pass(p, "sequence_length", {1, 8}); }));
}

TEST_CASE(symbolic_split_is_idempotent)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s   = symbolic_shape(
        {migraphx::sym::lit(1), symbolic_dim("sequence_length", 4), migraphx::sym::lit(2)});
    mm->add_return({mm->add_instruction(migraphx::make_op("identity"), mm->add_parameter("x", s))});

    run_pass(p, "sequence_length", {1, 4});
    migraphx::program once = p;
    run_pass(p, "sequence_length", {1, 4});

    EXPECT(p == once);
}

// The whole point of the split is that the specializations compute what the module computed
// before, at the sizes they were built for.
TEST_CASE(symbolic_split_matches_unsplit_on_ref)
{
    auto xs = symbolic_shape(
        {migraphx::sym::lit(1), symbolic_dim("sequence_length", 4), migraphx::sym::lit(2)});
    auto zs = symbolic_shape({symbolic_dim("other_dimension", 3), migraphx::sym::lit(2)});

    auto make_program = [&] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", xs);
        auto z   = mm->add_parameter("z", zs);
        auto two = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {2}});
        auto broadcast = mm->add_instruction(migraphx::make_op("multibroadcast"), two, x);
        auto scaled    = mm->add_instruction(migraphx::make_op("mul"), x, broadcast);
        mm->add_return({scaled, mm->add_instruction(migraphx::make_op("identity"), z)});
        return p;
    };

    auto split = make_program();
    run_pass(split, "sequence_length", {1, 4});
    EXPECT(find_specializations(split.get_main_module()).size() == 2);

    // the specializations reference literals in the main module, so they have to survive a round
    // trip through a serialized program
    migraphx::program reloaded;
    reloaded.from_value(split.to_value());
    EXPECT(reloaded == split);

    auto unsplit = make_program();
    split.compile(migraphx::make_target("ref"));
    unsplit.compile(migraphx::make_target("ref"));

    for(std::size_t sequence_length : {std::size_t{1}, std::size_t{4}})
    {
        migraphx::shape x_shape{migraphx::shape::float_type, {1, sequence_length, 2}};
        migraphx::shape z_shape{migraphx::shape::float_type, {3, 2}};
        std::vector<float> x_data(x_shape.elements());
        std::iota(x_data.begin(), x_data.end(), 1.0f);
        std::vector<float> z_data(z_shape.elements());
        std::iota(z_data.begin(), z_data.end(), 100.0f);

        migraphx::parameter_map params;
        params["x"] = migraphx::argument{x_shape, x_data.data()};
        params["z"] = migraphx::argument{z_shape, z_data.data()};

        auto split_results   = split.eval(params);
        auto unsplit_results = unsplit.eval(params);
        EXPECT(split_results.size() == unsplit_results.size());
        for(auto i : migraphx::range(split_results.size()))
        {
            EXPECT(split_results.at(i).get_shape() == unsplit_results.at(i).get_shape());
            EXPECT(migraphx::verify::verify_rms_range(split_results.at(i).to_vector<float>(),
                                                      unsplit_results.at(i).to_vector<float>()));
        }
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
