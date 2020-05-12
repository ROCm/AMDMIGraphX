#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/op/add.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(
        p, {migraphx::eliminate_common_subexpression{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(cse_test1)
{
    migraphx::program p1;
    {
        auto one  = p1.add_literal(1);
        auto two  = p1.add_literal(2);
        auto sum1 = p1.add_instruction(migraphx::op::add{}, one, two);
        auto sum2 = p1.add_instruction(migraphx::op::add{}, one, two);
        auto sum3 = p1.add_instruction(migraphx::op::add{}, sum1, sum2);
        p1.add_instruction(pass_op{}, sum3);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto one  = p2.add_literal(1);
        auto two  = p2.add_literal(2);
        auto sum1 = p2.add_instruction(migraphx::op::add{}, one, two);
        auto sum3 = p2.add_instruction(migraphx::op::add{}, sum1, sum1);
        p2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(cse_test2)
{
    migraphx::program p1;
    {
        auto one  = p1.add_literal(1);
        auto two  = p1.add_literal(2);
        auto sum1 = p1.add_instruction(migraphx::op::add{}, one, two);
        auto sum2 = p1.add_instruction(migraphx::op::add{}, two, one);
        auto sum3 = p1.add_instruction(migraphx::op::add{}, sum1, sum2);
        p1.add_instruction(pass_op{}, sum3);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto one  = p2.add_literal(1);
        auto two  = p2.add_literal(2);
        auto sum1 = p2.add_instruction(migraphx::op::add{}, one, two);
        auto sum2 = p2.add_instruction(migraphx::op::add{}, two, one);
        auto sum3 = p2.add_instruction(migraphx::op::add{}, sum1, sum2);
        p2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(cse_test3)
{
    migraphx::program p1;
    {
        auto one  = p1.add_literal(1);
        auto two  = p1.add_literal(1);
        auto sum1 = p1.add_instruction(migraphx::op::add{}, one, two);
        auto sum2 = p1.add_instruction(migraphx::op::add{}, two, one);
        auto sum3 = p1.add_instruction(migraphx::op::add{}, sum1, sum2);
        p1.add_instruction(pass_op{}, sum3);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto one  = p2.add_literal(1);
        auto sum1 = p2.add_instruction(migraphx::op::add{}, one, one);
        auto sum3 = p2.add_instruction(migraphx::op::add{}, sum1, sum1);
        p2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(cse_test4)
{
    migraphx::program p1;
    {
        auto one  = p1.add_literal(1);
        auto two  = p1.add_literal(1);
        auto sum1 = p1.add_instruction(migraphx::op::add{}, one, two);
        auto sum2 = p1.add_instruction(migraphx::op::add{}, two, one);
        auto sum3 = p1.add_instruction(migraphx::op::add{}, sum1, one);
        auto sum4 = p1.add_instruction(migraphx::op::add{}, sum2, two);
        auto sum5 = p1.add_instruction(migraphx::op::add{}, sum4, sum3);
        p1.add_instruction(pass_op{}, sum5);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto one  = p2.add_literal(1);
        auto sum1 = p2.add_instruction(migraphx::op::add{}, one, one);
        auto sum3 = p2.add_instruction(migraphx::op::add{}, sum1, one);
        auto sum5 = p2.add_instruction(migraphx::op::add{}, sum3, sum3);
        p2.add_instruction(pass_op{}, sum5);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(cse_test_literal)
{
    migraphx::program p1;
    {
        auto six1  = p1.add_literal(6);
        auto zero1 = p1.add_literal(0);
        auto six2  = p1.add_literal(6);
        auto zero2 = p1.add_literal(0);
        auto six3  = p1.add_literal(6);
        auto zero3 = p1.add_literal(0);

        auto sum1 = p1.add_instruction(migraphx::op::add{}, six1, zero1);
        auto sum2 = p1.add_instruction(migraphx::op::add{}, six2, zero2);
        auto sum3 = p1.add_instruction(migraphx::op::add{}, six3, zero3);
        auto sum4 = p1.add_instruction(migraphx::op::add{}, sum1, sum2);
        auto sum5 = p1.add_instruction(migraphx::op::add{}, sum3, sum4);
        p1.add_instruction(pass_op{}, sum5);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto six  = p2.add_literal(6);
        auto zero = p2.add_literal(0);
        auto sum1 = p2.add_instruction(migraphx::op::add{}, six, zero);
        auto sum2 = p2.add_instruction(migraphx::op::add{}, sum1, sum1);
        auto sum3 = p2.add_instruction(migraphx::op::add{}, sum1, sum2);
        p2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
