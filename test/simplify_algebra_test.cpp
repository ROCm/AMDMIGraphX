#include <migraph/simplify_algebra.hpp>
#include <migraph/dead_code_elimination.hpp>
#include <migraph/operators.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct simplify_algebra_target
{
    std::string name() const { return "simplify_algebra"; }
    std::vector<migraph::pass> get_passes(migraph::context&) const
    {
        return {migraph::simplify_algebra{}, migraph::dead_code_elimination{}};
    }
    migraph::context get_context() const { return {}; }
};

void simplify_add1()
{
    migraph::program p1;
    {
        auto x    = p1.add_parameter("x", {migraph::shape::int32_type, {1}});
        auto y    = p1.add_parameter("y", {migraph::shape::int32_type, {1}});
        auto one  = p1.add_literal(1);
        auto two  = p1.add_literal(2);
        auto sum1 = p1.add_instruction(migraph::op::add{}, x, one);
        auto sum2 = p1.add_instruction(migraph::op::add{}, y, two);
        auto sum3 = p1.add_instruction(migraph::op::add{}, sum1, sum2);
        p1.add_instruction(pass_op{}, sum3);
    }
    p1.compile(simplify_algebra_target{});

    migraph::program p2;
    {
        auto x    = p2.add_parameter("x", {migraph::shape::int32_type, {1}});
        auto y    = p2.add_parameter("y", {migraph::shape::int32_type, {1}});
        auto one  = p2.add_literal(1);
        auto two  = p2.add_literal(2);
        auto sum1 = p2.add_instruction(migraph::op::add{}, one, two);
        auto sum2 = p2.add_instruction(migraph::op::add{}, x, y);
        auto sum3 = p2.add_instruction(migraph::op::add{}, sum2, sum1);
        p2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

void simplify_add2()
{
    migraph::program p1;
    {
        auto x    = p1.add_parameter("x", {migraph::shape::int32_type, {1}});
        auto y    = p1.add_parameter("y", {migraph::shape::int32_type, {1}});
        auto one  = p1.add_literal(1);
        auto two  = p1.add_literal(2);
        auto sum1 = p1.add_instruction(migraph::op::add{}, one, x);
        auto sum2 = p1.add_instruction(migraph::op::add{}, two, y);
        auto sum3 = p1.add_instruction(migraph::op::add{}, sum1, sum2);
        p1.add_instruction(pass_op{}, sum3);
    }
    p1.compile(simplify_algebra_target{});

    migraph::program p2;
    {
        auto x    = p2.add_parameter("x", {migraph::shape::int32_type, {1}});
        auto y    = p2.add_parameter("y", {migraph::shape::int32_type, {1}});
        auto one  = p2.add_literal(1);
        auto two  = p2.add_literal(2);
        auto sum1 = p2.add_instruction(migraph::op::add{}, one, two);
        auto sum2 = p2.add_instruction(migraph::op::add{}, x, y);
        auto sum3 = p2.add_instruction(migraph::op::add{}, sum2, sum1);
        p2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

void simplify_add3()
{
    migraph::program p1;
    {
        auto x    = p1.add_parameter("x", {migraph::shape::int32_type, {1}});
        auto one  = p1.add_literal(1);
        auto two  = p1.add_literal(2);
        auto sum1 = p1.add_instruction(migraph::op::add{}, one, x);
        auto sum2 = p1.add_instruction(migraph::op::add{}, one, two);
        auto sum3 = p1.add_instruction(migraph::op::add{}, sum1, sum2);
        p1.add_instruction(pass_op{}, sum3);
    }
    p1.compile(simplify_algebra_target{});

    migraph::program p2;
    {
        auto x    = p2.add_parameter("x", {migraph::shape::int32_type, {1}});
        auto one  = p2.add_literal(1);
        auto two  = p2.add_literal(2);
        auto sum1 = p2.add_instruction(migraph::op::add{}, one, x);
        auto sum2 = p2.add_instruction(migraph::op::add{}, one, two);
        auto sum3 = p2.add_instruction(migraph::op::add{}, sum1, sum2);
        p2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

void simplify_add4()
{
    migraph::program p1;
    {
        auto x    = p1.add_parameter("x", {migraph::shape::int32_type, {1}});
        auto y    = p1.add_parameter("y", {migraph::shape::int32_type, {1}});
        auto one  = p1.add_literal(1);
        auto two  = p1.add_literal(2);
        auto sum1 = p1.add_instruction(migraph::op::add{}, one, x);
        auto sum2 = p1.add_instruction(migraph::op::add{}, sum1, y);
        auto sum3 = p1.add_instruction(migraph::op::add{}, sum2, two);
        p1.add_instruction(pass_op{}, sum3);
    }
    p1.compile(simplify_algebra_target{});

    migraph::program p2;
    {
        auto x    = p2.add_parameter("x", {migraph::shape::int32_type, {1}});
        auto y    = p2.add_parameter("y", {migraph::shape::int32_type, {1}});
        auto one  = p2.add_literal(1);
        auto two  = p2.add_literal(2);
        auto sum1 = p2.add_instruction(migraph::op::add{}, one, two);
        auto sum2 = p2.add_instruction(migraph::op::add{}, x, y);
        auto sum3 = p2.add_instruction(migraph::op::add{}, sum2, sum1);
        p2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

int main()
{
    simplify_add1();
    simplify_add2();
    simplify_add3();
    // simplify_add4();
}
