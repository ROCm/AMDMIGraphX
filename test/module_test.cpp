#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ref/target.hpp>
#include <migraphx/ranges.hpp>
#include <sstream>
#include "test.hpp"
#include <migraphx/make_op.hpp>

#include <basic_ops.hpp>

migraphx::program create_program()
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
    auto a3 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), a2);
    else_mod->add_return({a3, l2});

    auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
    auto r   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
    mm->add_return({r});

    auto implicit_deps = mm->calc_implicit_deps();
    EXPECT(migraphx::contains(implicit_deps, ret));
    EXPECT(migraphx::contains(implicit_deps.at(ret), x1));
    EXPECT(migraphx::contains(implicit_deps.at(ret), x2));
    EXPECT(migraphx::contains(implicit_deps.at(ret), y2));
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

int main(int argc, const char* argv[]) { test::run(argc, argv); }
