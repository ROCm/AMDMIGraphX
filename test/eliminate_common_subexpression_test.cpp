#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/op/add.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(
        *p.get_main_module(),
        {migraphx::eliminate_common_subexpression{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(cse_test1)
{
    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        auto one  = mm1->add_literal(1);
        auto two  = mm1->add_literal(2);
        auto sum1 = mm1->add_instruction(migraphx::op::add{}, one, two);
        auto sum2 = mm1->add_instruction(migraphx::op::add{}, one, two);
        auto sum3 = mm1->add_instruction(migraphx::op::add{}, sum1, sum2);
        mm1->add_instruction(pass_op{}, sum3);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2 = p2.get_main_module();
        auto one  = mm2->add_literal(1);
        auto two  = mm2->add_literal(2);
        auto sum1 = mm2->add_instruction(migraphx::op::add{}, one, two);
        auto sum3 = mm2->add_instruction(migraphx::op::add{}, sum1, sum1);
        mm2->add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(cse_test2)
{
    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        auto one  = mm1->add_literal(1);
        auto two  = mm1->add_literal(2);
        auto sum1 = mm1->add_instruction(migraphx::op::add{}, one, two);
        auto sum2 = mm1->add_instruction(migraphx::op::add{}, two, one);
        auto sum3 = mm1->add_instruction(migraphx::op::add{}, sum1, sum2);
        mm1->add_instruction(pass_op{}, sum3);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2 = p2.get_main_module();
        auto one  = mm2->add_literal(1);
        auto two  = mm2->add_literal(2);
        auto sum1 = mm2->add_instruction(migraphx::op::add{}, one, two);
        auto sum2 = mm2->add_instruction(migraphx::op::add{}, two, one);
        auto sum3 = mm2->add_instruction(migraphx::op::add{}, sum1, sum2);
        mm2->add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(cse_test3)
{
    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        auto one  = mm1->add_literal(1);
        auto two  = mm1->add_literal(1);
        auto sum1 = mm1->add_instruction(migraphx::op::add{}, one, two);
        auto sum2 = mm1->add_instruction(migraphx::op::add{}, two, one);
        auto sum3 = mm1->add_instruction(migraphx::op::add{}, sum1, sum2);
        mm1->add_instruction(pass_op{}, sum3);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2 = p2.get_main_module();
        auto one  = mm2->add_literal(1);
        auto sum1 = mm2->add_instruction(migraphx::op::add{}, one, one);
        auto sum3 = mm2->add_instruction(migraphx::op::add{}, sum1, sum1);
        mm2->add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(cse_test4)
{
    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        auto one  = mm1->add_literal(1);
        auto two  = mm1->add_literal(1);
        auto sum1 = mm1->add_instruction(migraphx::op::add{}, one, two);
        auto sum2 = mm1->add_instruction(migraphx::op::add{}, two, one);
        auto sum3 = mm1->add_instruction(migraphx::op::add{}, sum1, one);
        auto sum4 = mm1->add_instruction(migraphx::op::add{}, sum2, two);
        auto sum5 = mm1->add_instruction(migraphx::op::add{}, sum4, sum3);
        mm1->add_instruction(pass_op{}, sum5);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2 = p2.get_main_module();
        auto one  = mm2->add_literal(1);
        auto sum1 = mm2->add_instruction(migraphx::op::add{}, one, one);
        auto sum3 = mm2->add_instruction(migraphx::op::add{}, sum1, one);
        auto sum5 = mm2->add_instruction(migraphx::op::add{}, sum3, sum3);
        mm2->add_instruction(pass_op{}, sum5);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(cse_test_literal)
{
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto six1  = mm1->add_literal(6);
        auto zero1 = mm1->add_literal(0);
        auto six2  = mm1->add_literal(6);
        auto zero2 = mm1->add_literal(0);
        auto six3  = mm1->add_literal(6);
        auto zero3 = mm1->add_literal(0);

        auto sum1 = mm1->add_instruction(migraphx::op::add{}, six1, zero1);
        auto sum2 = mm1->add_instruction(migraphx::op::add{}, six2, zero2);
        auto sum3 = mm1->add_instruction(migraphx::op::add{}, six3, zero3);
        auto sum4 = mm1->add_instruction(migraphx::op::add{}, sum1, sum2);
        auto sum5 = mm1->add_instruction(migraphx::op::add{}, sum3, sum4);
        mm1->add_instruction(pass_op{}, sum5);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2 = p2.get_main_module();
        auto six  = mm2->add_literal(6);
        auto zero = mm2->add_literal(0);
        auto sum1 = mm2->add_instruction(migraphx::op::add{}, six, zero);
        auto sum2 = mm2->add_instruction(migraphx::op::add{}, sum1, sum1);
        auto sum3 = mm2->add_instruction(migraphx::op::add{}, sum1, sum2);
        mm2->add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
