#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

void run_pass(migraphx::module& m)
{
    migraphx::run_passes(
        m, {migraphx::eliminate_common_subexpression{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(cse_test1)
{
    migraphx::module m1;
    {
        auto one  = m1.add_literal(1);
        auto two  = m1.add_literal(2);
        auto sum1 = m1.add_instruction(migraphx::make_op("add"), one, two);
        auto sum2 = m1.add_instruction(migraphx::make_op("add"), one, two);
        auto sum3 = m1.add_instruction(migraphx::make_op("add"), sum1, sum2);
        m1.add_instruction(pass_op{}, sum3);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto one  = m2.add_literal(1);
        auto two  = m2.add_literal(2);
        auto sum1 = m2.add_instruction(migraphx::make_op("add"), one, two);
        auto sum3 = m2.add_instruction(migraphx::make_op("add"), sum1, sum1);
        m2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(cse_test2)
{
    migraphx::module m1;
    {
        auto one  = m1.add_literal(1);
        auto two  = m1.add_literal(2);
        auto sum1 = m1.add_instruction(migraphx::make_op("add"), one, two);
        auto sum2 = m1.add_instruction(migraphx::make_op("add"), two, one);
        auto sum3 = m1.add_instruction(migraphx::make_op("add"), sum1, sum2);
        m1.add_instruction(pass_op{}, sum3);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto one  = m2.add_literal(1);
        auto two  = m2.add_literal(2);
        auto sum1 = m2.add_instruction(migraphx::make_op("add"), one, two);
        auto sum2 = m2.add_instruction(migraphx::make_op("add"), two, one);
        auto sum3 = m2.add_instruction(migraphx::make_op("add"), sum1, sum2);
        m2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(cse_test3)
{
    migraphx::module m1;
    {
        auto one  = m1.add_literal(1);
        auto two  = m1.add_literal(1);
        auto sum1 = m1.add_instruction(migraphx::make_op("add"), one, two);
        auto sum2 = m1.add_instruction(migraphx::make_op("add"), two, one);
        auto sum3 = m1.add_instruction(migraphx::make_op("add"), sum1, sum2);
        m1.add_instruction(pass_op{}, sum3);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto one  = m2.add_literal(1);
        auto sum1 = m2.add_instruction(migraphx::make_op("add"), one, one);
        auto sum3 = m2.add_instruction(migraphx::make_op("add"), sum1, sum1);
        m2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(cse_test4)
{
    migraphx::module m1;
    {
        auto one  = m1.add_literal(1);
        auto two  = m1.add_literal(1);
        auto sum1 = m1.add_instruction(migraphx::make_op("add"), one, two);
        auto sum2 = m1.add_instruction(migraphx::make_op("add"), two, one);
        auto sum3 = m1.add_instruction(migraphx::make_op("add"), sum1, one);
        auto sum4 = m1.add_instruction(migraphx::make_op("add"), sum2, two);
        auto sum5 = m1.add_instruction(migraphx::make_op("add"), sum4, sum3);
        m1.add_instruction(pass_op{}, sum5);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto one  = m2.add_literal(1);
        auto sum1 = m2.add_instruction(migraphx::make_op("add"), one, one);
        auto sum3 = m2.add_instruction(migraphx::make_op("add"), sum1, one);
        auto sum5 = m2.add_instruction(migraphx::make_op("add"), sum3, sum3);
        m2.add_instruction(pass_op{}, sum5);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(cse_test_literal)
{
    migraphx::module m1;
    {
        auto six1  = m1.add_literal(6);
        auto zero1 = m1.add_literal(0);
        auto six2  = m1.add_literal(6);
        auto zero2 = m1.add_literal(0);
        auto six3  = m1.add_literal(6);
        auto zero3 = m1.add_literal(0);

        auto sum1 = m1.add_instruction(migraphx::make_op("add"), six1, zero1);
        auto sum2 = m1.add_instruction(migraphx::make_op("add"), six2, zero2);
        auto sum3 = m1.add_instruction(migraphx::make_op("add"), six3, zero3);
        auto sum4 = m1.add_instruction(migraphx::make_op("add"), sum1, sum2);
        auto sum5 = m1.add_instruction(migraphx::make_op("add"), sum3, sum4);
        m1.add_instruction(pass_op{}, sum5);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto six  = m2.add_literal(6);
        auto zero = m2.add_literal(0);
        auto sum1 = m2.add_instruction(migraphx::make_op("add"), six, zero);
        auto sum2 = m2.add_instruction(migraphx::make_op("add"), sum1, sum1);
        auto sum3 = m2.add_instruction(migraphx::make_op("add"), sum1, sum2);
        m2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
