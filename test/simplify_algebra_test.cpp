#include <migraphx/simplify_algebra.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/operators.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct simplify_algebra_target
{
    std::string name() const { return "simplify_algebra"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&) const
    {
        return {migraphx::simplify_algebra{}, migraphx::dead_code_elimination{}};
    }
    migraphx::context get_context() const { return {}; }
};

TEST_CASE(simplify_add1)
{
    migraphx::program p1;
    {
        auto x    = p1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = p1.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = p1.add_literal(1);
        auto two  = p1.add_literal(2);
        auto sum1 = p1.add_instruction(migraphx::op::add{}, x, one);
        auto sum2 = p1.add_instruction(migraphx::op::add{}, y, two);
        auto sum3 = p1.add_instruction(migraphx::op::add{}, sum1, sum2);
        p1.add_instruction(pass_op{}, sum3);
    }
    p1.compile(simplify_algebra_target{});

    migraphx::program p2;
    {
        auto x    = p2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = p2.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = p2.add_literal(1);
        auto two  = p2.add_literal(2);
        auto sum1 = p2.add_instruction(migraphx::op::add{}, one, two);
        auto sum2 = p2.add_instruction(migraphx::op::add{}, x, y);
        auto sum3 = p2.add_instruction(migraphx::op::add{}, sum2, sum1);
        p2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_add2)
{
    migraphx::program p1;
    {
        auto x    = p1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = p1.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = p1.add_literal(1);
        auto two  = p1.add_literal(2);
        auto sum1 = p1.add_instruction(migraphx::op::add{}, one, x);
        auto sum2 = p1.add_instruction(migraphx::op::add{}, two, y);
        auto sum3 = p1.add_instruction(migraphx::op::add{}, sum1, sum2);
        p1.add_instruction(pass_op{}, sum3);
    }
    p1.compile(simplify_algebra_target{});

    migraphx::program p2;
    {
        auto x    = p2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = p2.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = p2.add_literal(1);
        auto two  = p2.add_literal(2);
        auto sum1 = p2.add_instruction(migraphx::op::add{}, one, two);
        auto sum2 = p2.add_instruction(migraphx::op::add{}, x, y);
        auto sum3 = p2.add_instruction(migraphx::op::add{}, sum2, sum1);
        p2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_add3)
{
    migraphx::program p1;
    {
        auto x    = p1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto one  = p1.add_literal(1);
        auto two  = p1.add_literal(2);
        auto sum1 = p1.add_instruction(migraphx::op::add{}, one, x);
        auto sum2 = p1.add_instruction(migraphx::op::add{}, one, two);
        auto sum3 = p1.add_instruction(migraphx::op::add{}, sum1, sum2);
        p1.add_instruction(pass_op{}, sum3);
    }
    p1.compile(simplify_algebra_target{});

    migraphx::program p2;
    {
        auto x    = p2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto one  = p2.add_literal(1);
        auto two  = p2.add_literal(2);
        auto sum1 = p2.add_instruction(migraphx::op::add{}, one, x);
        auto sum2 = p2.add_instruction(migraphx::op::add{}, one, two);
        auto sum3 = p2.add_instruction(migraphx::op::add{}, sum1, sum2);
        p2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

// TODO: Add test case
void simplify_add4()
{
    migraphx::program p1;
    {
        auto x    = p1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = p1.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = p1.add_literal(1);
        auto two  = p1.add_literal(2);
        auto sum1 = p1.add_instruction(migraphx::op::add{}, one, x);
        auto sum2 = p1.add_instruction(migraphx::op::add{}, sum1, y);
        auto sum3 = p1.add_instruction(migraphx::op::add{}, sum2, two);
        p1.add_instruction(pass_op{}, sum3);
    }
    p1.compile(simplify_algebra_target{});

    migraphx::program p2;
    {
        auto x    = p2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = p2.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = p2.add_literal(1);
        auto two  = p2.add_literal(2);
        auto sum1 = p2.add_instruction(migraphx::op::add{}, one, two);
        auto sum2 = p2.add_instruction(migraphx::op::add{}, x, y);
        auto sum3 = p2.add_instruction(migraphx::op::add{}, sum2, sum1);
        p2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
