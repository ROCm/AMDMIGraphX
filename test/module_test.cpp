#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ref/target.hpp>
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

TEST_CASE(module_ins_clear)
{
    migraphx::program p1 = create_program();
    migraphx::program p2;

    p2 = p1;

    EXPECT(p1 == p2);
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
    auto ret  = mm->add_instruction(
        migraphx::make_op("iff",
                          {{"then_sub_graph", "then_smod"}, {"else_sub_graph", "else_smod"}}),
        {cond},
        {then_smod, else_smod});
    mm->add_return({ret});

    migraphx::program p1 = p;

    EXPECT(p == p1);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
