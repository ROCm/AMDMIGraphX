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
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <random>
#include <sstream>

#include <basic_ops.hpp>
#include <pointwise.hpp>
#include <test.hpp>

// Check the module is topologically sorted
// TODO: Use test::make_predicate
static bool is_sorted(migraphx::module& m)
{
    std::unordered_set<migraphx::instruction_ref> visited;
    for(auto ins : migraphx::iterator_for(m))
    {
        visited.insert(ins);
        if(std::any_of(ins->inputs().begin(), ins->inputs().end(), [&](auto i) {
               return not visited.count(i);
           }))
        {
            return false; // Found an input that has not been visited yet
        }
    }
    return true;
}

static void shuffle_module(migraphx::module& m)
{
    if(m.size() < 2)
        return;
    std::vector<std::size_t> permutation(m.size() - 1);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::mt19937 g(permutation.size());
    std::shuffle(permutation.begin(), permutation.end(), g);
    permutation.push_back(permutation.size());
    m.shuffle(permutation);
}

static void reverse_module(migraphx::module& m)
{
    if(m.size() < 2)
        return;
    std::vector<std::size_t> permutation(m.size() - 1);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::reverse(permutation.begin(), permutation.end());
    permutation.push_back(permutation.size());
    m.shuffle(permutation);
}

static migraphx::program create_program()
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x = mm->add_parameter("x", {migraphx::shape::int64_type});
    auto y = mm->add_parameter("y", {migraphx::shape::int64_type});

    auto sum = mm->add_instruction(sum_op{}, x, y);
    auto one = mm->add_literal(1);
    mm->add_instruction(sum_op{}, sum, one);

    return p;
}

TEST_CASE(calc_implict_deps)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape cond_s{migraphx::shape::bool_type};
    migraphx::shape xs{migraphx::shape::float_type, {2, 3}};
    migraphx::shape ys{migraphx::shape::float_type, {3, 3}};
    std::vector<float> datax = {1, 2, 3, 4, 5, 6};
    std::vector<float> datay = {8, 7, 6, 5, 4, 3, 2, 1, 0};

    auto lx   = mm->add_literal(migraphx::literal(xs, datax));
    auto ly   = mm->add_literal(migraphx::literal(ys, datay));
    auto cond = mm->add_parameter("cond", cond_s);
    auto x1   = mm->add_parameter("x1", xs);
    auto x2   = mm->add_parameter("x2", xs);
    auto y2   = mm->add_parameter("y2", ys);

    auto* then_mod = p.create_module("If_5_if");
    auto l1        = then_mod->add_literal(migraphx::literal(ys, datay));
    auto a1        = then_mod->add_instruction(migraphx::make_op("add"), x1, lx);
    then_mod->add_return({a1, l1});

    auto* then_mod1 = p.create_module("If_6_if");
    auto l11        = then_mod1->add_literal(migraphx::literal(ys, datay));
    auto a11        = then_mod1->add_instruction(migraphx::make_op("add"), x2, lx);
    then_mod1->add_return({a11, l11});

    auto* else_mod1 = p.create_module("If_6_else");
    auto l21        = else_mod1->add_literal(migraphx::literal(xs, datax));
    auto a21        = else_mod1->add_instruction(migraphx::make_op("mul"), y2, ly);
    else_mod1->add_return({l21, a21});

    auto* else_mod = p.create_module("If_5_else");
    auto l2        = else_mod->add_literal(migraphx::literal(ys, datay));
    auto a2 = else_mod->add_instruction(migraphx::make_op("if"), {cond}, {then_mod1, else_mod1});
    auto a3 = else_mod->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), a2);
    else_mod->add_return({a3, l2});

    auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
    auto r   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
    mm->add_return({r});

    auto implicit_deps = mm->calc_implicit_deps();
    EXPECT(migraphx::contains(implicit_deps, ret));
    EXPECT(migraphx::contains(implicit_deps.at(ret), x1));
    EXPECT(migraphx::contains(implicit_deps.at(ret), x2));
    EXPECT(migraphx::contains(implicit_deps.at(ret), y2));
    EXPECT(migraphx::contains(implicit_deps.at(ret), lx));
    EXPECT(migraphx::contains(implicit_deps.at(ret), ly));
    // test for sorting
    p.sort();
    auto ret_inputs = ret->inputs();
    ret_inputs.insert(ret_inputs.end(), implicit_deps.at(ret).begin(), implicit_deps.at(ret).end());
    EXPECT(std::all_of(ret_inputs.begin(), ret_inputs.end(), [&](const auto i) {
        return std::distance(mm->begin(), i) < std::distance(mm->begin(), ret);
    }));
}

TEST_CASE(module_annotate)
{
    migraphx::program p1 = create_program();
    migraphx::program p2 = create_program();

    auto* mm1 = p1.get_main_module();
    auto* mm2 = p2.get_main_module();
    EXPECT(*mm1 == *mm2);

    std::stringstream ss1;
    mm1->annotate(ss1, [](auto ins) { std::cout << ins->name() << "_1" << std::endl; });

    std::stringstream ss2;
    mm2->annotate(ss2, [](auto ins) { std::cout << ins->name() << "_1" << std::endl; });

    EXPECT(ss1.str() == ss2.str());
}

TEST_CASE(module_ins_clear)
{
    migraphx::program p1 = create_program();
    migraphx::program p2;

    p2 = p1;

    EXPECT(p1 == p2);
}

TEST_CASE(module_name)
{
    migraphx::module m1("name");
    EXPECT(m1.name() == "name");

    auto m2 = m1; // NOLINT
    EXPECT(m2.name() == "name");
    migraphx::module m3;
    m3 = m1;
    EXPECT(m3.name() == "name");
}

TEST_CASE(module_name_main)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    EXPECT(mm->name() == "main");
}

TEST_CASE(module_print_cpp)
{
    migraphx::program p1 = create_program();
    migraphx::program p2 = create_program();

    auto* mm1 = p1.get_main_module();
    auto* mm2 = p2.get_main_module();

    std::stringstream ss1;
    mm1->print_cpp(ss1);

    std::stringstream ss2;
    mm2->print_cpp(ss2);

    EXPECT(ss1.str() == ss2.str());
}

TEST_CASE(module_print_graph)
{
    migraphx::program p1 = create_program();
    migraphx::program p2 = create_program();

    auto* mm1 = p1.get_main_module();
    auto* mm2 = p2.get_main_module();

    std::stringstream ss1;
    mm1->print_graph(ss1, true);

    std::stringstream ss2;
    mm2->print_graph(ss2, true);

    EXPECT(ss1.str() == ss2.str());
}

TEST_CASE(program_module_assign)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sd{migraphx::shape::float_type, {2, 3}};
    auto x = mm->add_parameter("x", sd);

    std::vector<float> one(sd.elements(), 1);
    std::vector<float> two(sd.elements(), 2);

    auto* then_smod = p.create_module("then_smod");
    auto l1         = then_smod->add_literal(migraphx::literal{sd, one});
    auto r1         = then_smod->add_instruction(migraphx::make_op("add"), x, l1);
    then_smod->add_return({r1});

    auto* else_smod = p.create_module("else_smod");
    auto l2         = else_smod->add_literal(migraphx::literal{sd, two});
    auto r2         = else_smod->add_instruction(migraphx::make_op("mul"), x, l2);
    else_smod->add_return({r2});

    migraphx::shape s_cond{migraphx::shape::bool_type, {1}};
    auto cond = mm->add_parameter("cond", s_cond);
    auto ret  = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_smod, else_smod});
    mm->add_return({ret});

    migraphx::program p1 = p;

    EXPECT(p == p1);
}

TEST_CASE(program_module_replace)
{
    auto create_program = [](bool use_if) {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sd{migraphx::shape::float_type, {2, 3}};
        auto x = mm->add_parameter("x", sd);

        std::vector<float> one(sd.elements(), 1);
        std::vector<float> two(sd.elements(), 2);

        auto* then_smod = p.create_module("then_smod");
        auto l1         = then_smod->add_literal(migraphx::literal{sd, one});
        auto r1         = then_smod->add_instruction(migraphx::make_op("add"), x, l1);
        then_smod->add_return({r1});

        auto* else_smod = p.create_module("else_smod");
        auto l2         = else_smod->add_literal(migraphx::literal{sd, two});
        auto r2         = else_smod->add_instruction(migraphx::make_op("mul"), x, l2);
        else_smod->add_return({r2});

        migraphx::shape s_cond{migraphx::shape::bool_type, {1}};
        auto cond = mm->add_parameter("cond", s_cond);

        migraphx::instruction_ref ret{};

        if(use_if)
        {
            ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_smod, else_smod});
        }
        else
        {
            ret = mm->add_instruction(mod_pass_op{}, {cond}, {then_smod, else_smod});
        }

        mm->add_return({ret});

        return p;
    };

    migraphx::program p1 = create_program(false);
    migraphx::program p2 = create_program(true);
    EXPECT(p1 != p2);

    auto* m1               = p1.get_main_module();
    auto ins_pass          = std::prev(std::prev(m1->end()));
    const auto& inputs     = ins_pass->inputs();
    const auto& mod_inputs = ins_pass->module_inputs();
    m1->replace_instruction(ins_pass, migraphx::make_op("if"), inputs, mod_inputs);

    EXPECT(p1 == p2);
}

TEST_CASE(submodule_copy)
{
    migraphx::module mm("main");
    auto x = mm.add_parameter("x", {migraphx::shape::int64_type});

    migraphx::module sm("sub");
    sm.add_instruction(migraphx::make_op("sin"), x);

    mm.add_instruction(migraphx::make_op("if"), {x}, {&sm, &sm});

    auto mm2 = mm;

    EXPECT(mm == mm2);
    EXPECT(mm.get_sub_modules() == mm2.get_sub_modules());
}

TEST_CASE(parameter_name_order)
{
    migraphx::shape s{migraphx::shape::int32_type, {1}};
    migraphx::module mm("main");
    auto x1 = mm.add_parameter("x1", s);
    auto x2 = mm.add_parameter("x2", s);
    auto x3 = mm.add_parameter("x3", s);
    auto x4 = mm.add_parameter("x4", s);

    std::vector<std::string> param_names = {"x1", "x2", "x3", "x4"};
    auto sum1                            = mm.add_instruction(migraphx::make_op("add"), x1, x2);
    auto sum2                            = mm.add_instruction(migraphx::make_op("add"), x3, x4);
    auto r                               = mm.add_instruction(migraphx::make_op("mul"), sum1, sum2);
    mm.add_return({r});

    auto names = mm.get_parameter_names();
    EXPECT(param_names == names);

    auto m1     = mm;
    auto names1 = m1.get_parameter_names();
    EXPECT(param_names == names1);
}

struct map_ins
{
    using type = std::unordered_map<migraphx::instruction_ref, migraphx::instruction_ref>;
    map_ins(std::initializer_list<type::value_type> x) : m(x) {}

    operator type*() { return &m; }

    type m;
};

TEST_CASE(insert_instructions_module)
{
    migraphx::shape s{migraphx::shape::int32_type, {1}};
    migraphx::module m1("m1");
    auto x1   = m1.add_parameter("x1", s);
    auto sqrt = m1.add_instruction(migraphx::make_op("sqrt"), {x1});
    m1.add_instruction(migraphx::make_op("add"), {sqrt, x1});

    migraphx::module m2("m2");
    auto x2 = m2.add_parameter("x2", s);
    m2.add_instruction(migraphx::make_op("sqrt"), {x2});

    m1.insert_instructions(sqrt, &m2, map_ins{{x2, x1}});

    EXPECT(std::prev(sqrt)->name() == "sqrt");
    EXPECT(std::count_if(m1.begin(), m1.end(), [](auto&& ins) { return ins.name() == "sqrt"; }) ==
           2);
    EXPECT(std::count_if(m1.begin(), m1.end(), [](auto&& ins) { return ins.name() == "@param"; }) ==
           1);
    EXPECT(contains(m1.get_parameter_shapes(), "x1"));
    EXPECT(not contains(m1.get_parameter_shapes(), "x2"));
}

TEST_CASE(add_instructions_module)
{
    migraphx::shape s{migraphx::shape::int32_type, {1}};
    migraphx::module m1("m1");
    auto x1 = m1.add_parameter("x1", s);
    m1.add_instruction(migraphx::make_op("sqrt"), {x1});

    migraphx::module m2("m2");
    auto x2 = m2.add_parameter("x2", s);
    m2.add_instruction(migraphx::make_op("sqrt"), {x2});

    m1.add_instructions(&m2, map_ins{{x2, x1}});

    EXPECT(std::count_if(m1.begin(), m1.end(), [](auto&& ins) { return ins.name() == "sqrt"; }) ==
           2);
    EXPECT(std::count_if(m1.begin(), m1.end(), [](auto&& ins) { return ins.name() == "@param"; }) ==
           1);
    EXPECT(contains(m1.get_parameter_shapes(), "x1"));
    EXPECT(not contains(m1.get_parameter_shapes(), "x2"));
}

TEST_CASE(add_instructions_range)
{
    migraphx::shape s{migraphx::shape::int32_type, {1}};
    migraphx::module m1("m1");
    auto x1 = m1.add_parameter("x1", s);
    m1.add_instruction(migraphx::make_op("sqrt"), {x1});

    migraphx::module m2("m2");
    auto x2    = m2.add_parameter("x2", s);
    auto sqrt2 = m2.add_instruction(migraphx::make_op("sqrt"), {x2});

    m1.add_instructions(sqrt2, m2.end(), map_ins{{x2, x1}});
    EXPECT(std::any_of(
        m1.begin(), m1.end(), [&](auto&& ins) { return migraphx::contains(ins.inputs(), x1); }));

    EXPECT(std::count_if(m1.begin(), m1.end(), [](auto&& ins) { return ins.name() == "sqrt"; }) ==
           2);
    EXPECT(std::count_if(m1.begin(), m1.end(), [](auto&& ins) { return ins.name() == "@param"; }) ==
           1);
    EXPECT(contains(m1.get_parameter_shapes(), "x1"));
    EXPECT(not contains(m1.get_parameter_shapes(), "x2"));
}

TEST_CASE(add_instructions_vector)
{
    migraphx::shape s{migraphx::shape::int32_type, {1}};
    migraphx::module m1("m1");
    auto x1 = m1.add_parameter("x1", s);
    m1.add_instruction(migraphx::make_op("sqrt"), {x1});

    migraphx::module m2("m2");
    auto x2    = m2.add_parameter("x2", s);
    auto sqrt2 = m2.add_instruction(migraphx::make_op("sqrt"), {x2});

    m1.add_instructions({sqrt2}, map_ins{{x2, x1}});
    EXPECT(std::any_of(
        m1.begin(), m1.end(), [&](auto&& ins) { return migraphx::contains(ins.inputs(), x1); }));

    EXPECT(std::count_if(m1.begin(), m1.end(), [](auto&& ins) { return ins.name() == "sqrt"; }) ==
           2);
    EXPECT(std::count_if(m1.begin(), m1.end(), [](auto&& ins) { return ins.name() == "@param"; }) ==
           1);
    EXPECT(contains(m1.get_parameter_shapes(), "x1"));
    EXPECT(not contains(m1.get_parameter_shapes(), "x2"));
}

struct check_for_pass_op
{
    bool* found = nullptr;
    std::string name() const { return "check_for_pass_op"; }
    void apply(migraphx::module& m) const
    {
        *found |= std::any_of(m.begin(), m.end(), [](auto&& ins) { return ins.name() == "pass"; });
    }
};

TEST_CASE(module_bypass)
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto* sub = p.create_module("sub");
    sub->set_bypass();
    sub->add_instruction(pass_op{});
    mm->add_instruction(mod_pass_op{}, {}, {sub});
    bool found = false;
    migraphx::run_passes(p, {check_for_pass_op{&found}});
    EXPECT(not found);
}

TEST_CASE(module_without_bypass)
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto* sub = p.create_module("sub");
    sub->add_instruction(pass_op{});
    mm->add_instruction(mod_pass_op{}, {}, {sub});
    bool found = false;
    migraphx::run_passes(p, {check_for_pass_op{&found}});
    EXPECT(found);
}

TEST_CASE(multiple_module_dependency)
{
    // Test when an instruction from a submodule depends on previous module
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto* sub = p.create_module("sub");
    auto l1   = mm->add_literal(migraphx::literal(3));
    // second same literal to make sure instruction_ref is being compared, rather than the
    // instructions
    sub->add_literal(migraphx::literal(3));
    sub->add_instruction(sum_op{}, l1, l1);
    EXPECT((sub->validate() == sub->end()));
}

TEST_CASE(module_split2)
{
    migraphx::shape s{migraphx::shape::float_type, {1}};
    migraphx::module input_m;
    std::vector<migraphx::instruction_ref> inputs;
    {
        auto x1  = input_m.add_parameter("x1", s);
        auto x2  = input_m.add_parameter("x2", s);
        auto x3  = input_m.add_parameter("x3", s);
        auto sx1 = input_m.add_instruction(migraphx::make_op("sqrt"), x1);
        auto sx2 = input_m.add_instruction(migraphx::make_op("sqrt"), x2);
        auto sx3 = input_m.add_instruction(migraphx::make_op("sqrt"), x3);
        inputs   = {sx1, sx2, sx3};
    }
    migraphx::module m;
    std::vector<migraphx::instruction_ref> splits;
    {
        auto x1  = m.add_parameter("x1", s);
        auto x2  = m.add_parameter("x2", s);
        auto x3  = m.add_parameter("x3", s);
        auto add = m.add_instruction(migraphx::make_op("add"), x1, x2);
        auto mul = m.add_instruction(migraphx::make_op("mul"), add, x3);
        m.add_return({mul});
        splits.push_back(add);
    }
    auto mods = m.split(inputs, splits);

    migraphx::module m1;
    {
        auto x1  = m1.add_parameter("x1", s);
        auto x2  = m1.add_parameter("x2", s);
        auto add = m1.add_instruction(migraphx::make_op("add"), x1, x2);
        m1.add_return({add});
    }
    migraphx::module m2;
    {
        auto x0  = m2.add_parameter("x0", s);
        auto x1  = m2.add_parameter("x1", s);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), x0, x1);
        m2.add_return({mul});
    }
    EXPECT(mods[0].mod.sort() == m1.sort());
    EXPECT(mods[1].mod.sort() == m2.sort());

    EXPECT(mods[0].inputs[0] == inputs[0]);
    EXPECT(mods[0].inputs[1] == inputs[1]);

    EXPECT(mods[1].inputs[0] == splits.front());
    EXPECT(mods[1].inputs[1] == inputs[2]);
}

TEST_CASE(module_split_2_dot_ins)
{
    std::vector<migraphx::instruction_ref> inputs;
    std::vector<migraphx::instruction_ref> mod_0_expected_inputs;
    std::vector<migraphx::instruction_ref> mod_1_expected_inputs;
    migraphx::shape s1 = migraphx::shape{migraphx::shape::float_type, {2, 5}};
    migraphx::shape s2 = migraphx::shape{migraphx::shape::float_type, {5, 5}};
    migraphx::module input_m;
    {
        auto x1               = input_m.add_parameter("x1", s1);
        auto x2               = input_m.add_parameter("x2", s1);
        auto x3               = input_m.add_parameter("x3", s1);
        auto x4               = input_m.add_parameter("x4", s1);
        auto y0               = input_m.add_parameter("y0", s1);
        auto y1               = input_m.add_parameter("y1", s2);
        auto sx1              = input_m.add_instruction(migraphx::make_op("sqrt"), x1);
        auto sx2              = input_m.add_instruction(migraphx::make_op("sqrt"), x2);
        auto sx3              = input_m.add_instruction(migraphx::make_op("sqrt"), x3);
        auto sx4              = input_m.add_instruction(migraphx::make_op("sqrt"), x4);
        auto sy0              = input_m.add_instruction(migraphx::make_op("sqrt"), y0);
        auto sy1              = input_m.add_instruction(migraphx::make_op("sqrt"), y1);
        inputs                = {sx1, sx2, sx3, sx4, sy0, sy1};
        mod_0_expected_inputs = {sy0, sy1};
        mod_1_expected_inputs = {sx4, sx3, sx2, sx1};
    }
    std::vector<migraphx::instruction_ref> splits;
    migraphx::module m1;
    {
        // params --> {x1, x2, x3, x4, y0, y1} binds to input args {sx1, sx2, sx3, sx4, sy0, sy1}
        auto m1_y0 = m1.add_parameter("y0", s1);
        auto m1_y1 = m1.add_parameter("y1", s2);
        auto m1_x1 = m1.add_parameter("x1", s1);
        auto m1_x2 = m1.add_parameter("x2", s1);
        auto m1_x3 = m1.add_parameter("x3", s1);
        auto m1_x4 = m1.add_parameter("x4", s1);
        // m1_dot = dot(y0, y1) --> dot(sy0, sy1)
        auto m1_dot = m1.add_instruction(migraphx::make_op("dot"), m1_y0, m1_y1);
        // m1_add = add(x0, m1_dot)  --> add(sx1, m1_dot)
        auto m1_add_1 = m1.add_instruction(migraphx::make_op("add"), m1_x1, m1_dot);
        // m1_add_2 = add(x2, x3) --> add(sx2, sx3)
        auto m1_add_2 = m1.add_instruction(migraphx::make_op("add"), m1_x2, m1_x3);
        // m1_sub = sub(x4, m1_relu_2) --> sub(sx4, m1_add_2)
        auto m1_sub = m1.add_instruction(migraphx::make_op("sub"), m1_x4, m1_add_2);
        // m1_mul = mul(m1_sub, m1_add_1)
        auto m1_mul = m1.add_instruction(migraphx::make_op("mul"), m1_sub, m1_add_1);
        m1.add_return({m1_mul});
        splits.push_back(m1_dot);
    }

    migraphx::module mod_0;
    {
        auto mod_0_y1 = mod_0.add_parameter("y1", s2);
        auto mod_0_y0 = mod_0.add_parameter("y0", s1);
        // mod_0_dot(y0, y1) --> dot(sy0, sy1)
        auto mod_0_dot = mod_0.add_instruction(migraphx::make_op("dot"), mod_0_y0, mod_0_y1);
        mod_0.add_return({mod_0_dot});
    }
    migraphx::module mod_1;
    {
        // expected input args are {dot_ins, sx4, sx3, sx2, sx1}
        auto mod_1_x0 = mod_1.add_parameter("x0", s1);
        auto mod_1_x1 = mod_1.add_parameter("x1", s1);
        auto mod_1_x2 = mod_1.add_parameter("x2", s1);
        auto mod_1_x3 = mod_1.add_parameter("x3", s1);
        auto mod_1_x4 = mod_1.add_parameter("x4", s1);
        // m1_add = add(x4, m1_dot)  --> add(sx1, m1_dot)
        auto m1_add = mod_1.add_instruction(migraphx::make_op("add"), mod_1_x4, mod_1_x0);
        // m1_add_2 = add(x3, x2) --> add(sx2, sx3)
        auto m1_add_2 = mod_1.add_instruction(migraphx::make_op("add"), mod_1_x3, mod_1_x2);
        // m1_sub = sub(x1, m1_relu_2) --> sub(sx4, m1_add_2)
        auto m1_sub = mod_1.add_instruction(migraphx::make_op("sub"), mod_1_x1, m1_add_2);
        // m1_mul = mul(m1_sub, m1_add)
        auto m1_mul = mod_1.add_instruction(migraphx::make_op("mul"), m1_sub, m1_add);
        mod_1.add_return({m1_mul});
    }
    auto mods = m1.split(inputs, splits);
    EXPECT(mods[0].mod.sort() == mod_0.sort());
    const auto mod_0_inputs = mods[0].inputs;
    EXPECT(mod_0_inputs[0] == mod_0_expected_inputs[0]);
    EXPECT(mod_0_inputs[1] == mod_0_expected_inputs[1]);
    const auto mod_1_inputs = mods[1].inputs;
    // first input arg should be the split instruction
    EXPECT(mods[1].mod.sort() == mod_1.sort());
    EXPECT(mod_1_inputs[0] == splits.front());
    EXPECT(mod_1_inputs[1] == mod_1_expected_inputs[0]);
    EXPECT(mod_1_inputs[2] == mod_1_expected_inputs[1]);
    EXPECT(mod_1_inputs[3] == mod_1_expected_inputs[2]);
    EXPECT(mod_1_inputs[4] == mod_1_expected_inputs[3]);
}

TEST_CASE(module_split3)
{
    migraphx::shape s{migraphx::shape::float_type, {1}};
    migraphx::module input_m;
    std::vector<migraphx::instruction_ref> inputs;
    {
        auto x1  = input_m.add_parameter("x1", s);
        auto x2  = input_m.add_parameter("x2", s);
        auto sx1 = input_m.add_instruction(migraphx::make_op("sqrt"), x1);
        auto sx2 = input_m.add_instruction(migraphx::make_op("sqrt"), x2);
        inputs   = {sx1, sx2};
    }
    migraphx::module m;
    std::vector<migraphx::instruction_ref> splits1;
    std::vector<migraphx::instruction_ref> splits2;
    {
        auto x1   = m.add_parameter("x1", s);
        auto x2   = m.add_parameter("x2", s);
        auto mul  = m.add_instruction(migraphx::make_op("mul"), x1, x2);
        auto sqrt = m.add_instruction(migraphx::make_op("sqrt"), mul);
        auto add  = m.add_instruction(migraphx::make_op("add"), sqrt, mul);
        m.add_return({add});
        splits1.push_back(mul);
        splits2.push_back(sqrt);
    }
    auto mods = m.split(inputs, splits1, splits2);

    migraphx::module m1;
    {
        auto x1  = m1.add_parameter("x1", s);
        auto x2  = m1.add_parameter("x2", s);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), x1, x2);
        m1.add_return({mul});
    }
    migraphx::module m2;
    {
        auto x0   = m2.add_parameter("x0", s);
        auto sqrt = m2.add_instruction(migraphx::make_op("sqrt"), x0);
        m2.add_return({sqrt});
    }
    migraphx::module m3;
    {
        auto x0  = m3.add_parameter("x0", s);
        auto x1  = m3.add_parameter("x1", s);
        auto add = m3.add_instruction(migraphx::make_op("add"), x0, x1);
        m3.add_return({add});
    }
    EXPECT(mods[0].mod.sort() == m1.sort());
    EXPECT(mods[1].mod.sort() == m2.sort());
    EXPECT(mods[2].mod.sort() == m3.sort());

    EXPECT(mods[0].inputs[0] == inputs[0]);
    EXPECT(mods[0].inputs[1] == inputs[1]);

    EXPECT(mods[1].inputs[0] == splits1.front());

    EXPECT(mods[2].inputs[0] == splits2.front());
    EXPECT(mods[2].inputs[1] == splits1.front());
}

TEST_CASE(fuse_module)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m1;
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto z   = mm->add_parameter("z", s);
        auto add = add_pointwise(p, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto mul = add_pointwise(p, "main:pointwise1", {add, z}, single_pointwise("mul"));

        std::unordered_map<migraphx::instruction_ref, migraphx::instruction_ref> map_ins;
        auto rins    = m1.fuse(*add->module_inputs().front(), add->inputs(), &map_ins).front();
        map_ins[add] = rins;
        auto ret     = m1.fuse(*mul->module_inputs().front(), mul->inputs(), &map_ins);
        m1.add_return(ret);
    }
    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x0", s);
        auto y   = m2.add_parameter("x1", s);
        auto z   = m2.add_parameter("x2", s);
        auto add = m2.add_instruction(migraphx::make_op("add"), x, y);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), add, z);
        m2.add_return({mul});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(get_inputs)
{
    // Test a use case for get_inputs
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m1;
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", s);
    auto y   = mm->add_parameter("y", s);
    auto z   = mm->add_parameter("z", s);
    auto add = add_pointwise(p, "main:pointwise0", {x, z}, single_pointwise("add"));
    auto mul = add_pointwise(p, "main:pointwise1", {add, y}, single_pointwise("mul"));

    std::unordered_map<migraphx::instruction_ref, migraphx::instruction_ref> map_ins;
    auto rins    = m1.fuse(*add->module_inputs().front(), add->inputs(), &map_ins).front();
    map_ins[add] = rins;
    auto ret     = m1.fuse(*mul->module_inputs().front(), mul->inputs(), &map_ins);
    m1.add_return(ret);

    // After using the fuse methods above, map_ins contains the following mappings:
    // - instruction in mm -> instruction in m1
    // - instruction in add -> instruction in m1
    // - instruction in mul -> instruction in m1

    // create a map of instructions in m1 to instruction in mm (all the parameters will
    // map to some instruction in mm)
    std::unordered_map<migraphx::instruction_ref, migraphx::instruction_ref> map_m1_to_mm;
    for(auto const& [i1, i2] : map_ins)
    {
        if(contains(*mm, i1))
            map_m1_to_mm[i2] = i1;
    }
    // get_inputs should return the instructions from mm in the correct order that should
    // be the new inputs to m1
    auto inputs = m1.get_inputs(map_m1_to_mm);

    EXPECT(inputs.size() == 3);
    EXPECT(inputs[0] == x);
    EXPECT(inputs[1] == z);
    EXPECT(inputs[2] == y);
}

TEST_CASE(add_params)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m1;
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", s);
    auto y   = mm->add_parameter("y", s);
    auto z   = mm->add_parameter("z", s);
    auto add = mm->add_instruction(migraphx::make_op("add"), x, y);
    auto mul = mm->add_instruction(migraphx::make_op("mul"), add, z);

    // use case: add and mul are to be used as input parameters to a new submodule m1
    std::unordered_map<migraphx::instruction_ref, migraphx::instruction_ref> map_ins;
    m1.add_params({mul, add}, &map_ins);

    migraphx::module m2;
    m2.add_parameter("x0", mul->get_shape());
    m2.add_parameter("x1", add->get_shape());

    // m1 should have parameters x0 and x1 with the shapes of mul and add outputs, respectively
    EXPECT(m1 == m2);
    // map_ins should contain a mapping: mul (in mm) -> x0 (in m1)
    EXPECT(m1.get_parameter("x0") == map_ins[mul]);
    // map_ins should contain a mapping: add (in mm) -> x1 (in m1)
    EXPECT(m1.get_parameter("x1") == map_ins[add]);
}

TEST_CASE(linear_graph_sort)
{
    //
    // Linear chain test - graph structure:
    //
    //  x → abs → neg → tanh → return
    //
    // Tests the most basic case of topological sorting.
    //
    migraphx::module m;
    auto s = migraphx::shape{migraphx::shape::float_type, {1}};
    auto x = m.add_parameter("x", s);
    auto a = m.add_instruction(migraphx::make_op("abs"), x);
    auto n = m.add_instruction(migraphx::make_op("neg"), a);
    auto t = m.add_instruction(migraphx::make_op("tanh"), n);
    m.add_return({t});

    m.sort();
    EXPECT(is_sorted(m));

    reverse_module(m);
    m.sort();
    EXPECT(is_sorted(m));

    shuffle_module(m);
    m.sort();
    EXPECT(is_sorted(m));
}

TEST_CASE(diamond_graph_sort)
{
    //
    // Diamond graph test - graph structure:
    //
    //           ┌─→ abs ─┐
    //           │        ↓
    //  x ───────┼───────→ add → return
    //           │        ↑
    //           └─→ neg ─┘
    //
    // Tests handling of branches and reconvergent paths.
    //
    migraphx::module m;
    auto s   = migraphx::shape{migraphx::shape::float_type, {1}};
    auto x   = m.add_parameter("x", s);
    auto a   = m.add_instruction(migraphx::make_op("abs"), x);
    auto n   = m.add_instruction(migraphx::make_op("neg"), x);
    auto add = m.add_instruction(migraphx::make_op("add"), a, n);
    m.add_return({add});

    m.sort();
    EXPECT(is_sorted(m));

    reverse_module(m);
    m.sort();
    EXPECT(is_sorted(m));

    shuffle_module(m);
    m.sort();
    EXPECT(is_sorted(m));
}

TEST_CASE(multiple_outputs_sort)
{
    //
    // Multiple outputs test - graph structure:
    //
    //           ┌─→ abs → tanh ─┐
    //           │                │
    //  x ───────┤                ├─→ return
    //           │                │
    //           └─→ neg ─────────┘
    //
    // Tests handling of multiple outputs from a single instruction.
    //
    migraphx::module m;
    auto s = migraphx::shape{migraphx::shape::float_type, {1}};
    auto x = m.add_parameter("x", s);
    auto a = m.add_instruction(migraphx::make_op("abs"), x);
    auto n = m.add_instruction(migraphx::make_op("neg"), x);
    auto t = m.add_instruction(migraphx::make_op("tanh"), a);
    m.add_return({t, n});

    m.sort();
    EXPECT(is_sorted(m));

    reverse_module(m);
    m.sort();
    EXPECT(is_sorted(m));

    shuffle_module(m);
    m.sort();
    EXPECT(is_sorted(m));
}

TEST_CASE(dead_code_sort)
{
    //
    // Dead code
    //
    //           ┌─→ abs → tanh ─┐
    //           │               │
    //  x ───────┤               ├─→ return
    //           │
    //           └─→ neg
    //
    // Tests handling of dead code
    //
    migraphx::module m;
    auto s = migraphx::shape{migraphx::shape::float_type, {1}};
    auto x = m.add_parameter("x", s);
    auto a = m.add_instruction(migraphx::make_op("abs"), x);
    m.add_instruction(migraphx::make_op("neg"), x);
    auto t = m.add_instruction(migraphx::make_op("tanh"), a);
    m.add_return({t});

    m.sort();
    EXPECT(is_sorted(m));

    reverse_module(m);
    m.sort();
    EXPECT(is_sorted(m));

    shuffle_module(m);
    m.sort();
    EXPECT(is_sorted(m));
}

TEST_CASE(disconnected_components_sort)
{
    //
    // Disconnected components test - graph structure:
    //
    //  x1 → abs1 ─┐
    //              ├─→ return
    //  x2 → abs2 ─┘
    //
    // Tests sorting of disconnected subgraphs.
    //
    migraphx::module m;
    auto s = migraphx::shape{migraphx::shape::float_type, {1}};

    // First subgraph
    auto x1 = m.add_parameter("x1", s);
    auto a1 = m.add_instruction(migraphx::make_op("abs"), x1);

    // Second subgraph (disconnected)
    auto x2 = m.add_parameter("x2", s);
    auto a2 = m.add_instruction(migraphx::make_op("abs"), x2);

    m.add_return({a1, a2});

    m.sort();
    EXPECT(is_sorted(m));

    reverse_module(m);
    m.sort();
    EXPECT(is_sorted(m));

    shuffle_module(m);
    m.sort();
    EXPECT(is_sorted(m));
}

TEST_CASE(empty_graph_sort)
{
    //
    // Empty graph test - graph structure:
    //
    //  (empty module)
    //
    // Tests sorting an empty module.
    //
    migraphx::module m;
    m.sort();

    // No assertions should fail
    EXPECT(is_sorted(m));
}

TEST_CASE(single_node_sort)
{
    //
    // Single node test - graph structure:
    //
    //  x → return
    //
    // Tests the simplest possible non-empty case.
    //
    migraphx::module m;
    auto s = migraphx::shape{migraphx::shape::float_type, {1}};
    auto x = m.add_parameter("x", s);
    m.add_return({x});

    m.sort();

    EXPECT(is_sorted(m));
}

TEST_CASE(sort_with_non_direct_dependencies)
{
    //
    // Non-direct dependencies test - graph structure:
    //
    //  x → abs ─────────┐
    //        │          │
    //        ↓          ↓
    //       neg → add → return
    //
    // Tests handling of both direct and indirect dependencies.
    // (A is a direct dependency of B and C, and an indirect dependency of C via B)
    //
    migraphx::module m;
    auto s = migraphx::shape{migraphx::shape::float_type, {1}};

    auto x = m.add_parameter("x", s);
    auto a = m.add_instruction(migraphx::make_op("abs"), x);
    auto b = m.add_instruction(migraphx::make_op("neg"), a);
    auto c = m.add_instruction(migraphx::make_op("add"), a, b);
    m.add_return({c});

    m.sort();
    EXPECT(is_sorted(m));

    reverse_module(m);
    m.sort();
    EXPECT(is_sorted(m));

    shuffle_module(m);
    m.sort();
    EXPECT(is_sorted(m));
}

TEST_CASE(dfs_without_visited_set_infinite_loop)
{
    //
    // Highly Connected DAG test - graph structure:
    //
    //           x0
    //          /|\
    //         / | \
    //        /  |  \
    //       v   v   v
    //      a1  b1  c1
    //     /|\ /|\ /|\
    //    / | X | X | \  (crossing connections)
    //   /  |/ \|/ \|  \
    //  v   v   v   v   v
    // a2  a3  b2  b3  c2
    //  \   \   |   /   /
    //   \   \  |  /   /
    //    \   \ | /   /
    //     \   \|/   /
    //      \   v   /
    //       \  d  /
    //        \ | /
    //         \|/
    //          v
    //        return
    //
    // This creates a highly connected directed acyclic graph (DAG) where
    // traversing up from the return node would encounter the same nodes
    // multiple times through different paths.
    //
    // Without a proper visited set, a DFS-based topological sort would
    // potentially re-process the same nodes repeatedly, leading to an
    // exponential runtime or infinite loop in pathological implementations.
    //
    migraphx::module m;
    auto s = migraphx::shape{migraphx::shape::float_type, {1}};

    auto x0 = m.add_parameter("x0", s);

    // First layer of operations
    auto a1 = m.add_instruction(migraphx::make_op("add"), x0, x0);
    auto b1 = m.add_instruction(migraphx::make_op("mul"), x0, x0);
    auto c1 = m.add_instruction(migraphx::make_op("tanh"), x0);

    // Second layer with cross-connections
    auto a2 = m.add_instruction(migraphx::make_op("sqrt"), a1);
    auto a3 = m.add_instruction(migraphx::make_op("mul"), a1, b1);
    auto b2 = m.add_instruction(migraphx::make_op("where"), a1, b1, c1);
    auto b3 = m.add_instruction(migraphx::make_op("mul"), b1, c1);
    auto c2 = m.add_instruction(migraphx::make_op("exp"), c1);

    m.add_return({a2, a3, b2, b3, c2});

    m.sort();
    EXPECT(is_sorted(m));

    reverse_module(m);
    m.sort();
    EXPECT(is_sorted(m));

    shuffle_module(m);
    m.sort();
    EXPECT(is_sorted(m));
}

TEST_CASE(recursive_dag_revisit_test)
{
    //
    // Recursive DAG structure - graph structure:
    //
    //      x
    //     / \
    //    v   v
    //   a1   b1
    //  / \  / \
    // v   vv   v
    // a2  c1   b2
    // |   /\   |
    // |  /  \  |
    // v v    v v
    // a3      b3
    // |        |
    // v        v
    // a4      b4
    // |        |
    // v        v
    // a5      b5
    //  \      /
    //   \    /
    //    v  v
    //     d1
    //     |
    //     v
    //   return
    //
    // This test creates a deeper recursive structure with many opportunities
    // for revisiting nodes. Each node in the middle layers (a2-a5, b2-b5, c1)
    // has multiple paths leading to it when traversing up from the return node.
    //
    // A naive DFS implementation without a visited set would potentially
    // revisit these nodes many times, leading to an exponential number of
    // recursive calls, which could manifest as an infinite loop in practice.
    //
    migraphx::module m;
    auto s = migraphx::shape{migraphx::shape::float_type, {1}};

    auto x = m.add_parameter("x", s);

    // Create two main branches
    auto a1 = m.add_instruction(migraphx::make_op("add"), x, x);
    auto b1 = m.add_instruction(migraphx::make_op("mul"), x, x);

    // Create deeper levels with cross-connections
    auto a2 = m.add_instruction(migraphx::make_op("sqrt"), a1);
    auto c1 = m.add_instruction(migraphx::make_op("add"), a1, b1);

    auto b2 = m.add_instruction(migraphx::make_op("neg"), b1);

    auto a3 = m.add_instruction(migraphx::make_op("add"), a2, c1);

    auto b3 = m.add_instruction(migraphx::make_op("add"), b2, c1);

    // Add more layers to increase the number of paths
    auto a4 = m.add_instruction(migraphx::make_op("log"), a3);
    auto b4 = m.add_instruction(migraphx::make_op("exp"), b3);

    auto a5 = m.add_instruction(migraphx::make_op("cos"), a4);
    auto b5 = m.add_instruction(migraphx::make_op("sin"), b4);

    // Final convergence
    auto d1 = m.add_instruction(migraphx::make_op("add"), a5, b5);

    m.add_return({d1});

    m.sort();
    EXPECT(is_sorted(m));

    reverse_module(m);
    m.sort();
    EXPECT(is_sorted(m));

    shuffle_module(m);
    m.sort();
    EXPECT(is_sorted(m));
}

TEST_CASE(exponential_growth_graph_sort)
{
    //
    // Exponential growth graph structure - graph pattern:
    //
    // Each level i has 2^i nodes, with each node connecting to
    // multiple nodes in the previous level. This creates an exponential
    // number of paths through the graph.
    //
    // This structure is particularly problematic for naive DFS implementations
    // without visited node tracking.
    //

    // Define a large enough graph to potentially cause problems
    // but not so large that it takes too long to create
    const int depth = 10;

    migraphx::module m;
    auto s = migraphx::shape{migraphx::shape::float_type, {1}};

    // Available operations
    std::vector<migraphx::operation> operations = {migraphx::make_op("add"),
                                                   migraphx::make_op("mul"),
                                                   migraphx::make_op("min"),
                                                   migraphx::make_op("max"),
                                                   migraphx::make_op("div")};

    auto x                                             = m.add_parameter("x", s);
    std::vector<migraphx::instruction_ref> first_level = {x};

    // Number of nodes at each level which doubles at each level
    std::array<std::size_t, depth> num_nodes;
    num_nodes.fill(2);
    std::partial_sum(num_nodes.begin(), num_nodes.end(), num_nodes.begin(), std::multiplies<>{});

    // Build all layers, accumulating just the last level
    auto last_level =
        std::accumulate(num_nodes.begin(),
                        num_nodes.end(),
                        first_level,
                        [&](const auto& prev_level, std::size_t nodes_at_level) {
                            // Transform indices into nodes
                            std::vector<migraphx::instruction_ref> current_level;
                            current_level.reserve(nodes_at_level);

                            transform(migraphx::range(nodes_at_level),
                                      std::back_inserter(current_level),
                                      [&](std::size_t node) {
                                          // Select inputs from previous level
                                          int input1_idx = node % prev_level.size();
                                          int input2_idx = (node / 2) % prev_level.size();

                                          auto input1 = prev_level[input1_idx];
                                          auto input2 = prev_level[input2_idx];

                                          // Select operation based on node index
                                          const auto& op = operations[node % operations.size()];

                                          // Create the new node
                                          return m.add_instruction(op, input1, input2);
                                      });

                            // Return the current level (to become prev_level in next iteration)
                            return current_level;
                        });

    // Add return node connected to multiple nodes from the last level
    std::vector<migraphx::instruction_ref> final_inputs;
    std::copy_n(last_level.begin(), last_level.size() / 2, std::back_inserter(final_inputs));
    m.add_return(final_inputs);

    m.sort();
    EXPECT(is_sorted(m));

    reverse_module(m);
    m.sort();
    EXPECT(is_sorted(m));

    shuffle_module(m);
    m.sort();
    EXPECT(is_sorted(m));
}

TEST_CASE(pathological_dfs_graph_sort)
{
    //
    // Pathological DFS Graph - designed to create the maximum number
    // of revisits when traversing from the return node without a visited set
    //
    // This creates a graph where the number of unique paths to each node
    // increases exponentially as you traverse up from the return node,
    // making it a worst-case scenario for DFS without visited tracking.
    //
    migraphx::module m;
    auto s = migraphx::shape{migraphx::shape::float_type, {1}};

    const int num_layers      = 20;
    const int nodes_per_layer = 20;
    const int num_params      = 10;

    // Create parameters
    std::vector<migraphx::instruction_ref> params;
    transform(migraphx::range(num_params), std::back_inserter(params), [&](int i) {
        return m.add_parameter("x" + std::to_string(i), s);
    });

    // Available operations
    std::vector<migraphx::operation> operations = {migraphx::make_op("add"),
                                                   migraphx::make_op("mul"),
                                                   migraphx::make_op("min"),
                                                   migraphx::make_op("max"),
                                                   migraphx::make_op("div")};

    std::vector<migraphx::instruction_ref> first_layer;
    transform(
        migraphx::range(nodes_per_layer), std::back_inserter(first_layer), [&](std::size_t i) {
            // Each node connects to two random parameters
            std::size_t param1 = i % num_params;
            std::size_t param2 = (i + 3) % num_params;

            const auto& op = operations[i % operations.size()];
            return m.add_instruction(op, params[param1], params[param2]);
        });

    // Build all layers, accumulating just the last layer
    auto last_layer = std::accumulate(
        migraphx::iota_iterator{0},
        migraphx::iota_iterator{num_layers},
        first_layer,
        [&](const auto& prev_layer, std::size_t) {
            std::vector<std::size_t> node_indices(nodes_per_layer);
            std::iota(node_indices.begin(), node_indices.end(), 0);

            std::vector<migraphx::instruction_ref> current_layer;

            transform(migraphx::range(nodes_per_layer),
                      std::back_inserter(current_layer),
                      [&](std::size_t i) {
                          // Connect to multiple nodes from previous layer to create path explosion
                          const std::size_t num_inputs = std::min(4, nodes_per_layer);
                          std::vector<migraphx::instruction_ref> inputs;

                          // Transform indices to actual input nodes
                          transform(migraphx::range(num_inputs),
                                    std::back_inserter(inputs),
                                    [&](std::size_t j) {
                                        std::size_t input_idx = (i + j * 3) % prev_layer.size();
                                        return prev_layer[input_idx];
                                    });

                          // Create intermediate nodes with pairs of inputs
                          auto op1  = migraphx::make_op("add");
                          auto op2  = migraphx::make_op("mul");
                          auto ins1 = m.add_instruction(op1, inputs[0], inputs[1]);
                          auto ins2 = m.add_instruction(
                              op2, inputs[2 % inputs.size()], inputs[3 % inputs.size()]);

                          // Combine the results
                          const auto& op3 = operations[(i % 3) + 2]; // Use min, max, or div
                          return m.add_instruction(op3, ins1, ins2);
                      });

            // Return the current layer (to become prev_layer in next iteration)
            return current_layer;
        });

    // Create a sequence of operations that combine all nodes from the last layer
    auto final_result = std::accumulate(
        last_layer.begin() + 1, last_layer.end(), last_layer[0], [&](auto x, auto y) {
            return m.add_instruction(migraphx::make_op("add"), x, y);
        });

    m.add_return({final_result});

    m.sort();
    EXPECT(is_sorted(m));

    reverse_module(m);
    m.sort();
    EXPECT(is_sorted(m));

    shuffle_module(m);
    m.sort();
    EXPECT(is_sorted(m));
}

TEST_CASE(hoist_external_inputs)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::float_type, {2, 4}};
    migraphx::shape s4{migraphx::shape::float_type, {4, 2}};
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto a = mm->add_parameter("a", s1);
    auto b = mm->add_parameter("b", s2);
    auto c = mm->add_parameter("c", s3);
    auto d = mm->add_parameter("d", s4);

    auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b); // {2, 4}
    auto external_relu =
        add_pointwise(p, "main:pointwise1", {d}, single_pointwise("relu")); // {4, 3}
    auto external_mul =
        add_pointwise(p, "main:pointwise2", {external_relu, d}, single_pointwise("mul")); // {4, 3}
    auto add  = add_pointwise(p, "main:pointwise0", {dot1, c}, single_pointwise("add"));  // {2, 4}
    auto dot2 = mm->add_instruction(migraphx::make_op("dot"), add, external_mul);         // {2, 3}
    auto transpose =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), dot2);
    mm->add_return({add, transpose});

    // Hoist external inputs between dot1 and dot2
    mm->hoist_external_inputs(dot1, dot2);

    // Verify the module is still topologically sorted overall
    EXPECT(is_sorted(*mm));

    // Verify external operations moved before the fusion chain
    EXPECT(std::distance(mm->begin(), external_relu) < std::distance(mm->begin(), dot1));
    EXPECT(std::distance(mm->begin(), external_mul) < std::distance(mm->begin(), dot1));

    // Verify the fusion chain ordering is preserved: dot1 < add < dot2
    EXPECT(std::distance(mm->begin(), dot1) < std::distance(mm->begin(), add));
    EXPECT(std::distance(mm->begin(), add) < std::distance(mm->begin(), dot2));

    // Verify external_mul is before dot1 (since dot2 depends on external_mul)
    EXPECT(std::distance(mm->begin(), external_mul) < std::distance(mm->begin(), dot1));

    // Verify transpose remains after dot2
    EXPECT(std::distance(mm->begin(), dot2) < std::distance(mm->begin(), transpose));
}

TEST_CASE(hoist_external_inputs_no_movement_needed)
{
    // Test where external dependencies are already before the fusion chain
    migraphx::shape s1{migraphx::shape::float_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::float_type, {2, 4}};
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto a = mm->add_parameter("a", s1);
    auto b = mm->add_parameter("b", s2);
    auto c = mm->add_parameter("c", s3);

    // External operations already positioned before the fusion chain
    auto external1 = add_pointwise(p, "main:pointwise0", {a}, single_pointwise("relu"));
    auto external2 = add_pointwise(p, "main:pointwise1", {b}, single_pointwise("tanh"));
    auto external3 = add_pointwise(p, "main:pointwise2", {c}, single_pointwise("tanh"));

    // Fusion chain
    auto dot1 = mm->add_instruction(migraphx::make_op("dot"), external1, external2);
    auto add  = add_pointwise(p, "main:pointwise3", {dot1, external3}, single_pointwise("add"));

    mm->add_return({add});

    // Record positions before hoist_external_inputs
    auto dot1_pos_before = std::distance(mm->begin(), dot1);
    auto add_pos_before  = std::distance(mm->begin(), add);

    mm->hoist_external_inputs(dot1, add);

    // Verify positions haven't changed (nothing needed to move)
    EXPECT(std::distance(mm->begin(), dot1) == dot1_pos_before);
    EXPECT(std::distance(mm->begin(), add) == add_pos_before);
    EXPECT(is_sorted(*mm));
}

TEST_CASE(hoist_external_inputs_multiple_external_branches)
{
    // Test with multiple independent external branches that need to be moved
    migraphx::shape s1{migraphx::shape::float_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::float_type, {4, 3}};
    migraphx::shape s4{migraphx::shape::float_type, {2, 4}};
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto a = mm->add_parameter("a", s1);
    auto b = mm->add_parameter("b", s2);
    auto c = mm->add_parameter("c", s4);
    auto d = mm->add_parameter("d", s3);

    // Start of fusion chain
    auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b); // {2, 4}

    // External branch 1
    auto ext1 = add_pointwise(p, "main:pointwise0", {c}, single_pointwise("relu"));   // {2, 4}
    auto ext2 = add_pointwise(p, "main:pointwise1", {ext1}, single_pointwise("neg")); // {2, 4}

    // External branch 2
    auto ext3 = add_pointwise(p, "main:pointwise2", {d}, single_pointwise("tanh"));   // {4, 3}
    auto ext4 = add_pointwise(p, "main:pointwise3", {ext3}, single_pointwise("abs")); // {4, 3}

    // Continue fusion chain using external branches
    auto add1 =
        add_pointwise(p, "main:pointwise4", {dot1, ext2}, single_pointwise("add")); // {2, 4}
    auto dot2 =
        mm->add_instruction(migraphx::make_op("dot"), add1, ext4); // {2, 4} x {4, 3} = {2, 3}

    mm->add_return({dot2});

    mm->hoist_external_inputs(dot1, dot2);

    EXPECT(is_sorted(*mm));

    // Verify all external operations moved before dot1
    EXPECT(std::distance(mm->begin(), ext1) < std::distance(mm->begin(), dot1));
    EXPECT(std::distance(mm->begin(), ext2) < std::distance(mm->begin(), dot1));
    EXPECT(std::distance(mm->begin(), ext3) < std::distance(mm->begin(), dot1));
    EXPECT(std::distance(mm->begin(), ext4) < std::distance(mm->begin(), dot1));

    // Verify fusion chain order preserved
    EXPECT(std::distance(mm->begin(), dot1) < std::distance(mm->begin(), add1));
    EXPECT(std::distance(mm->begin(), add1) < std::distance(mm->begin(), dot2));
}

TEST_CASE(hoist_external_inputs_long_fusion_chain)
{
    // Test with a longer fusion chain
    migraphx::shape s{migraphx::shape::float_type, {4, 4}};
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto a = mm->add_parameter("a", s);
    auto b = mm->add_parameter("b", s);
    auto c = mm->add_parameter("c", s);
    auto d = mm->add_parameter("d", s);

    // Long fusion chain
    auto op1 = mm->add_instruction(migraphx::make_op("dot"), a, b);

    // External operations interspersed
    auto ext1 = add_pointwise(p, "main:pointwise0", {d}, single_pointwise("exp"));

    auto op2 = add_pointwise(p, "main:pointwise1", {op1, c}, single_pointwise("add"));

    auto ext2 = add_pointwise(p, "main:pointwise2", {ext1}, single_pointwise("log"));

    auto op3 = add_pointwise(p, "main:pointwise3", {op2}, single_pointwise("relu"));

    auto ext3 = add_pointwise(p, "main:pointwise4", {ext2}, single_pointwise("abs"));

    auto op4 = add_pointwise(p, "main:pointwise5", {op3}, single_pointwise("tanh"));
    auto op5 = mm->add_instruction(migraphx::make_op("dot"), op4, ext3);

    mm->add_return({op5});

    mm->hoist_external_inputs(op1, op5);

    EXPECT(is_sorted(*mm));

    // All external operations moved before op1
    EXPECT(std::distance(mm->begin(), ext1) < std::distance(mm->begin(), op1));
    EXPECT(std::distance(mm->begin(), ext2) < std::distance(mm->begin(), op1));
    EXPECT(std::distance(mm->begin(), ext3) < std::distance(mm->begin(), op1));

    // Fusion chain order preserved
    EXPECT(std::distance(mm->begin(), op1) < std::distance(mm->begin(), op2));
    EXPECT(std::distance(mm->begin(), op2) < std::distance(mm->begin(), op3));
    EXPECT(std::distance(mm->begin(), op3) < std::distance(mm->begin(), op4));
    EXPECT(std::distance(mm->begin(), op4) < std::distance(mm->begin(), op5));
}

TEST_CASE(hoist_external_inputs_adjacent_instructions)
{
    // Test edge case where start and end are adjacent
    migraphx::shape s1{migraphx::shape::float_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::float_type, {4, 5}};
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto a = mm->add_parameter("a", s1); // {2, 3}
    auto b = mm->add_parameter("b", s2); // {3, 4}
    auto c = mm->add_parameter("c", s3); // {4, 5}

    auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b); // {2, 3} x {3, 4} = {2, 4}

    // External operation between dot1 and dot2
    auto external = add_pointwise(p, "main:pointwise0", {c}, single_pointwise("relu")); // {4, 5}

    auto dot2 =
        mm->add_instruction(migraphx::make_op("dot"), dot1, external); // {2, 4} x {4, 5} = {2, 5}

    mm->add_return({dot2});

    mm->hoist_external_inputs(dot1, dot2);

    EXPECT(is_sorted(*mm));

    // External should be moved before dot1
    EXPECT(std::distance(mm->begin(), external) < std::distance(mm->begin(), dot1));
    EXPECT(std::distance(mm->begin(), dot1) < std::distance(mm->begin(), dot2));
}

TEST_CASE(move_output_instructions_after_single_output)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s);
        auto src  = m1.add_instruction(migraphx::make_op("abs"), x);
        auto out1 = m1.add_instruction(migraphx::make_op("neg"), src);
        auto dst  = m1.add_instruction(migraphx::make_op("relu"), x);
        m1.add_return({out1, dst});
        m1.move_output_instructions_after(src, dst);
    }

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto src  = m2.add_instruction(migraphx::make_op("abs"), x);
        auto dst  = m2.add_instruction(migraphx::make_op("relu"), x);
        auto out1 = m2.add_instruction(migraphx::make_op("neg"), src);
        m2.add_return({out1, dst});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(move_output_instructions_after_transitive_outputs)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};

    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", s);
        auto src = m1.add_instruction(migraphx::make_op("abs"), x);
        auto a   = m1.add_instruction(migraphx::make_op("neg"), src);
        auto b   = m1.add_instruction(migraphx::make_op("relu"), a);
        auto dst = m1.add_instruction(migraphx::make_op("sqrt"), x);
        m1.add_return({b, dst});
        m1.move_output_instructions_after(src, dst);
    }

    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", s);
        auto src = m2.add_instruction(migraphx::make_op("abs"), x);
        auto dst = m2.add_instruction(migraphx::make_op("sqrt"), x);
        auto a   = m2.add_instruction(migraphx::make_op("neg"), src);
        auto b   = m2.add_instruction(migraphx::make_op("relu"), a);
        m2.add_return({b, dst});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(move_output_instructions_after_multiple_direct_outputs)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};

    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", s);
        auto src = m1.add_instruction(migraphx::make_op("abs"), x);
        auto a   = m1.add_instruction(migraphx::make_op("neg"), src);
        auto b   = m1.add_instruction(migraphx::make_op("relu"), src);
        auto dst = m1.add_instruction(migraphx::make_op("sqrt"), x);
        m1.add_return({a, b, dst});
        m1.move_output_instructions_after(src, dst);
    }

    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", s);
        auto src = m2.add_instruction(migraphx::make_op("abs"), x);
        auto dst = m2.add_instruction(migraphx::make_op("sqrt"), x);
        auto a   = m2.add_instruction(migraphx::make_op("neg"), src);
        auto b   = m2.add_instruction(migraphx::make_op("relu"), src);
        m2.add_return({a, b, dst});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(move_output_instructions_after_no_outputs_between)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s);
        auto src  = m1.add_instruction(migraphx::make_op("abs"), x);
        auto dst  = m1.add_instruction(migraphx::make_op("sqrt"), x);
        auto out1 = m1.add_instruction(migraphx::make_op("neg"), src);
        m1.add_return({out1, dst});
        m1.move_output_instructions_after(src, dst);
    }

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto src  = m2.add_instruction(migraphx::make_op("abs"), x);
        auto dst  = m2.add_instruction(migraphx::make_op("sqrt"), x);
        auto out1 = m2.add_instruction(migraphx::make_op("neg"), src);
        m2.add_return({out1, dst});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(move_output_instructions_after_diamond)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};

    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", s);
        auto src = m1.add_instruction(migraphx::make_op("abs"), x);
        auto a   = m1.add_instruction(migraphx::make_op("neg"), src);
        auto b   = m1.add_instruction(migraphx::make_op("relu"), src);
        auto c   = m1.add_instruction(migraphx::make_op("add"), a, b);
        auto dst = m1.add_instruction(migraphx::make_op("sqrt"), x);
        m1.add_return({c, dst});
        m1.move_output_instructions_after(src, dst);
    }

    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", s);
        auto src = m2.add_instruction(migraphx::make_op("abs"), x);
        auto dst = m2.add_instruction(migraphx::make_op("sqrt"), x);
        auto a   = m2.add_instruction(migraphx::make_op("neg"), src);
        auto b   = m2.add_instruction(migraphx::make_op("relu"), src);
        auto c   = m2.add_instruction(migraphx::make_op("add"), a, b);
        m2.add_return({c, dst});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(move_output_instructions_after_mixed)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s);
        auto y    = m1.add_parameter("y", s);
        auto src  = m1.add_instruction(migraphx::make_op("abs"), x);
        auto mid  = m1.add_instruction(migraphx::make_op("neg"), y);
        auto out1 = m1.add_instruction(migraphx::make_op("relu"), src);
        auto dst  = m1.add_instruction(migraphx::make_op("sqrt"), y);
        auto out2 = m1.add_instruction(migraphx::make_op("tanh"), src);
        m1.add_return({out1, out2, dst, mid});
        m1.move_output_instructions_after(src, dst);
    }

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto y    = m2.add_parameter("y", s);
        auto src  = m2.add_instruction(migraphx::make_op("abs"), x);
        auto mid  = m2.add_instruction(migraphx::make_op("neg"), y);
        auto dst  = m2.add_instruction(migraphx::make_op("sqrt"), y);
        auto out1 = m2.add_instruction(migraphx::make_op("relu"), src);
        auto out2 = m2.add_instruction(migraphx::make_op("tanh"), src);
        m2.add_return({out1, out2, dst, mid});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(move_output_instructions_after_dst_depends_on_src)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s);
        auto src  = m1.add_instruction(migraphx::make_op("abs"), x);
        auto out1 = m1.add_instruction(migraphx::make_op("neg"), src);
        auto dst  = m1.add_instruction(migraphx::make_op("relu"), src);
        m1.add_return({out1, dst});
        m1.move_output_instructions_after(src, dst);
    }

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto src  = m2.add_instruction(migraphx::make_op("abs"), x);
        auto dst  = m2.add_instruction(migraphx::make_op("relu"), src);
        auto out1 = m2.add_instruction(migraphx::make_op("neg"), src);
        m2.add_return({out1, dst});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(move_output_instructions_after_module_output)
{
    // When src is referenced by instructions inside a submodule, src->outputs()
    // includes those cross-module instructions. The function resolves them to
    // the instruction in the current module that owns the submodule and moves
    // that instruction instead.
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::shape cond_s{migraphx::shape::bool_type, {1}};

    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto cond = mm->add_parameter("cond", cond_s);
        auto src  = mm->add_instruction(migraphx::make_op("abs"), x);

        auto* then_mod = p1.create_module("then_mod");
        auto sub_neg   = then_mod->add_instruction(migraphx::make_op("neg"), src);
        then_mod->add_return({sub_neg});

        auto* else_mod = p1.create_module("else_mod");
        auto sub_relu  = else_mod->add_instruction(migraphx::make_op("relu"), src);
        else_mod->add_return({sub_relu});

        auto out1   = mm->add_instruction(migraphx::make_op("tanh"), src);
        auto if_ins = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        auto dst    = mm->add_instruction(migraphx::make_op("sqrt"), x);
        mm->add_return({out1, if_ins, dst});
        mm->move_output_instructions_after(src, dst);
    }

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto cond = mm->add_parameter("cond", cond_s);
        auto src  = mm->add_instruction(migraphx::make_op("abs"), x);

        auto* then_mod = p2.create_module("then_mod");
        auto sub_neg   = then_mod->add_instruction(migraphx::make_op("neg"), src);
        then_mod->add_return({sub_neg});

        auto* else_mod = p2.create_module("else_mod");
        auto sub_relu  = else_mod->add_instruction(migraphx::make_op("relu"), src);
        else_mod->add_return({sub_relu});

        auto dst    = mm->add_instruction(migraphx::make_op("sqrt"), x);
        auto out1   = mm->add_instruction(migraphx::make_op("tanh"), src);
        auto if_ins = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        mm->add_return({out1, if_ins, dst});
    }

    EXPECT(p1 == p2);
}

TEST_CASE(move_output_instructions_after_only_cross_module_output)
{
    // src has no direct outputs in the current module between src and dst,
    // only cross-module outputs. The owning instruction (if) is between src
    // and dst, so it gets moved after dst.
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::shape cond_s{migraphx::shape::bool_type, {1}};

    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto cond = mm->add_parameter("cond", cond_s);
        auto src  = mm->add_instruction(migraphx::make_op("abs"), x);

        auto* then_mod = p1.create_module("then_mod");
        auto sub_neg   = then_mod->add_instruction(migraphx::make_op("neg"), src);
        then_mod->add_return({sub_neg});

        auto* else_mod = p1.create_module("else_mod");
        auto sub_relu  = else_mod->add_instruction(migraphx::make_op("relu"), src);
        else_mod->add_return({sub_relu});

        auto if_ins = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        auto dst    = mm->add_instruction(migraphx::make_op("sqrt"), x);
        mm->add_return({if_ins, dst});
        mm->move_output_instructions_after(src, dst);
    }

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto cond = mm->add_parameter("cond", cond_s);
        auto src  = mm->add_instruction(migraphx::make_op("abs"), x);

        auto* then_mod = p2.create_module("then_mod");
        auto sub_neg   = then_mod->add_instruction(migraphx::make_op("neg"), src);
        then_mod->add_return({sub_neg});

        auto* else_mod = p2.create_module("else_mod");
        auto sub_relu  = else_mod->add_instruction(migraphx::make_op("relu"), src);
        else_mod->add_return({sub_relu});

        auto dst    = mm->add_instruction(migraphx::make_op("sqrt"), x);
        auto if_ins = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        mm->add_return({if_ins, dst});
    }

    EXPECT(p1 == p2);
}

TEST_CASE(move_output_instructions_after_cross_module_not_between)
{
    // src has cross-module outputs, but the owning instruction (if) is already
    // after dst. Nothing should move.
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::shape cond_s{migraphx::shape::bool_type, {1}};

    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto cond = mm->add_parameter("cond", cond_s);
        auto src  = mm->add_instruction(migraphx::make_op("abs"), x);

        auto* then_mod = p1.create_module("then_mod");
        auto sub_neg   = then_mod->add_instruction(migraphx::make_op("neg"), src);
        then_mod->add_return({sub_neg});

        auto* else_mod = p1.create_module("else_mod");
        auto sub_relu  = else_mod->add_instruction(migraphx::make_op("relu"), src);
        else_mod->add_return({sub_relu});

        auto dst    = mm->add_instruction(migraphx::make_op("sqrt"), x);
        auto if_ins = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        mm->add_return({if_ins, dst});
        mm->move_output_instructions_after(src, dst);
    }

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto cond = mm->add_parameter("cond", cond_s);
        auto src  = mm->add_instruction(migraphx::make_op("abs"), x);

        auto* then_mod = p2.create_module("then_mod");
        auto sub_neg   = then_mod->add_instruction(migraphx::make_op("neg"), src);
        then_mod->add_return({sub_neg});

        auto* else_mod = p2.create_module("else_mod");
        auto sub_relu  = else_mod->add_instruction(migraphx::make_op("relu"), src);
        else_mod->add_return({sub_relu});

        auto dst    = mm->add_instruction(migraphx::make_op("sqrt"), x);
        auto if_ins = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        mm->add_return({if_ins, dst});
    }

    EXPECT(p1 == p2);
}

TEST_CASE(move_output_instructions_after_cross_module_mixed)
{
    // src has cross-module outputs to submodules of two different instructions:
    // one between src and dst (should move), one after dst (should NOT move).
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::shape cond_s{migraphx::shape::bool_type, {1}};

    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto cond1 = mm->add_parameter("cond1", cond_s);
        auto cond2 = mm->add_parameter("cond2", cond_s);
        auto src   = mm->add_instruction(migraphx::make_op("abs"), x);

        auto* then_mod1 = p1.create_module("then_mod1");
        auto sub1       = then_mod1->add_instruction(migraphx::make_op("neg"), src);
        then_mod1->add_return({sub1});

        auto* else_mod1 = p1.create_module("else_mod1");
        auto sub2       = else_mod1->add_instruction(migraphx::make_op("relu"), src);
        else_mod1->add_return({sub2});

        // if1 is between src and dst — should be moved
        auto if1 =
            mm->add_instruction(migraphx::make_op("if"), {cond1}, {then_mod1, else_mod1});
        auto dst = mm->add_instruction(migraphx::make_op("sqrt"), x);

        auto* then_mod2 = p1.create_module("then_mod2");
        auto sub3       = then_mod2->add_instruction(migraphx::make_op("tanh"), src);
        then_mod2->add_return({sub3});

        auto* else_mod2 = p1.create_module("else_mod2");
        auto sub4       = else_mod2->add_instruction(migraphx::make_op("sin"), src);
        else_mod2->add_return({sub4});

        // if2 is after dst — should NOT be moved
        auto if2 =
            mm->add_instruction(migraphx::make_op("if"), {cond2}, {then_mod2, else_mod2});
        mm->add_return({if1, if2, dst});
        mm->move_output_instructions_after(src, dst);
    }

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto cond1 = mm->add_parameter("cond1", cond_s);
        auto cond2 = mm->add_parameter("cond2", cond_s);
        auto src   = mm->add_instruction(migraphx::make_op("abs"), x);

        auto* then_mod1 = p2.create_module("then_mod1");
        auto sub1       = then_mod1->add_instruction(migraphx::make_op("neg"), src);
        then_mod1->add_return({sub1});

        auto* else_mod1 = p2.create_module("else_mod1");
        auto sub2       = else_mod1->add_instruction(migraphx::make_op("relu"), src);
        else_mod1->add_return({sub2});

        auto* then_mod2 = p2.create_module("then_mod2");
        auto sub3       = then_mod2->add_instruction(migraphx::make_op("tanh"), src);
        then_mod2->add_return({sub3});

        auto* else_mod2 = p2.create_module("else_mod2");
        auto sub4       = else_mod2->add_instruction(migraphx::make_op("sin"), src);
        else_mod2->add_return({sub4});

        // Expected: if1 moved after dst, if2 stays after dst
        auto dst = mm->add_instruction(migraphx::make_op("sqrt"), x);
        auto if1 =
            mm->add_instruction(migraphx::make_op("if"), {cond1}, {then_mod1, else_mod1});
        auto if2 =
            mm->add_instruction(migraphx::make_op("if"), {cond2}, {then_mod2, else_mod2});
        mm->add_return({if1, if2, dst});
    }

    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
