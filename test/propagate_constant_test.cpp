#include <migraphx/propagate_constant.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/op/scalar.hpp>
#include <migraphx/op/mul.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::propagate_constant{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(const_add)
{
    migraphx::program p1;
    auto one = p1.add_literal(1);
    auto two = p1.add_literal(2);
    auto sum = p1.add_instruction(migraphx::op::add{}, one, two);
    p1.add_instruction(pass_op{}, sum);
    run_pass(p1);

    migraphx::program p2;
    auto total = p2.add_literal(3);
    p2.add_instruction(pass_op{}, total);
    EXPECT(p1 == p2);
}

TEST_CASE(const_add_parameter)
{
    migraphx::program p1;
    auto one = p1.add_parameter("one", {migraphx::shape::int32_type, {1}});
    auto two = p1.add_literal(2);
    auto sum = p1.add_instruction(migraphx::op::add{}, one, two);
    p1.add_instruction(pass_op{}, sum);
    run_pass(p1);

    migraphx::program p2;
    auto total = p2.add_literal(3);
    p2.add_instruction(pass_op{}, total);
    EXPECT(p1 != p2);
}

TEST_CASE(const_multiadd)
{
    migraphx::program p1;
    auto one  = p1.add_literal(1);
    auto two  = p1.add_literal(2);
    auto sum1 = p1.add_instruction(migraphx::op::add{}, one, two);
    auto sum2 = p1.add_instruction(migraphx::op::add{}, sum1, two);
    p1.add_instruction(pass_op{}, sum2);
    run_pass(p1);

    migraphx::program p2;
    auto total = p2.add_literal(5);
    p2.add_instruction(pass_op{}, total);
    EXPECT(p1 == p2);
}

TEST_CASE(const_add_mul)
{
    migraphx::program p1;
    auto one  = p1.add_literal(1);
    auto two  = p1.add_literal(2);
    auto mul  = p1.add_instruction(migraphx::op::mul{}, two, two);
    auto sum1 = p1.add_instruction(migraphx::op::add{}, one, mul);
    auto sum2 = p1.add_instruction(migraphx::op::add{}, sum1, two);
    p1.add_instruction(pass_op{}, sum2);
    run_pass(p1);

    migraphx::program p2;
    auto total = p2.add_literal(7);
    p2.add_instruction(pass_op{}, total);
    EXPECT(p1 == p2);
}

TEST_CASE(const_add_scalar)
{
    migraphx::program p1;
    auto one = p1.add_instruction(migraphx::op::scalar{{2, 2}}, p1.add_literal(1));
    auto two = p1.add_instruction(migraphx::op::scalar{{2, 2}}, p1.add_literal(2));
    auto sum = p1.add_instruction(migraphx::op::add{}, one, two);
    p1.add_instruction(pass_op{}, sum);
    run_pass(p1);

    migraphx::program p2;
    auto total =
        p2.add_literal(migraphx::literal{{migraphx::shape::int32_type, {2, 2}}, {3, 3, 3, 3}});
    p2.add_instruction(pass_op{}, total);
    EXPECT(p1 == p2);
}

TEST_CASE(const_scalar)
{
    migraphx::program p1;
    {
        auto one = p1.add_instruction(migraphx::op::scalar{{2, 2}}, p1.add_literal(1));
        p1.add_instruction(pass_op{}, one);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto one = p2.add_instruction(migraphx::op::scalar{{2, 2}}, p2.add_literal(1));
        p2.add_instruction(pass_op{}, one);
    }
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
