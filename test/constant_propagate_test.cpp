#include <migraph/constant_propagate.hpp>
#include <migraph/dead_code_elimination.hpp>
#include <migraph/operators.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct const_prop_target
{
    std::string name() const { return "const_prop"; }
    std::vector<migraph::pass> get_passes(migraph::context&) const
    {
        return {migraph::constant_propagate{}, migraph::dead_code_elimination{}};
    }
    migraph::context get_context() const { return {}; }
};

void const_add1()
{
    migraph::program p1;
    auto one = p1.add_literal(1);
    auto two = p1.add_literal(2);
    auto sum = p1.add_instruction(migraph::op::add{}, one, two);
    p1.add_instruction(pass_op{}, sum);
    p1.compile(const_prop_target{});

    migraph::program p2;
    auto total = p2.add_literal(3);
    p2.add_instruction(pass_op{}, total);
    EXPECT(p1 == p2);
}

void const_add2()
{
    migraph::program p1;
    auto one = p1.add_parameter("one", {migraph::shape::int32_type, {1}});
    auto two = p1.add_literal(2);
    auto sum = p1.add_instruction(migraph::op::add{}, one, two);
    p1.add_instruction(pass_op{}, sum);
    p1.compile(const_prop_target{});

    migraph::program p2;
    auto total = p2.add_literal(3);
    p2.add_instruction(pass_op{}, total);
    EXPECT(p1 != p2);
}

void const_add3()
{
    migraph::program p1;
    auto one  = p1.add_literal(1);
    auto two  = p1.add_literal(2);
    auto sum1 = p1.add_instruction(migraph::op::add{}, one, two);
    auto sum2 = p1.add_instruction(migraph::op::add{}, sum1, two);
    p1.add_instruction(pass_op{}, sum2);
    p1.compile(const_prop_target{});

    migraph::program p2;
    auto total = p2.add_literal(5);
    p2.add_instruction(pass_op{}, total);
    EXPECT(p1 == p2);
}

int main()
{
    const_add1();
    const_add2();
    const_add3();
}
