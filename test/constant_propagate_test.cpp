#include <migraphx/constant_propagate.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/op/add.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct const_prop_target
{
    std::string name() const { return "const_prop"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&) const
    {
        return {migraphx::constant_propagate{}, migraphx::dead_code_elimination{}};
    }
    migraphx::context get_context() const { return {}; }
};

TEST_CASE(const_add1)
{
    migraphx::program p1;
    auto one = p1.add_literal(1);
    auto two = p1.add_literal(2);
    auto sum = p1.add_instruction(migraphx::op::add{}, one, two);
    p1.add_instruction(pass_op{}, sum);
    p1.compile(const_prop_target{});

    migraphx::program p2;
    auto total = p2.add_literal(3);
    p2.add_instruction(pass_op{}, total);
    EXPECT(p1 == p2);
}

TEST_CASE(const_add2)
{
    migraphx::program p1;
    auto one = p1.add_parameter("one", {migraphx::shape::int32_type, {1}});
    auto two = p1.add_literal(2);
    auto sum = p1.add_instruction(migraphx::op::add{}, one, two);
    p1.add_instruction(pass_op{}, sum);
    p1.compile(const_prop_target{});

    migraphx::program p2;
    auto total = p2.add_literal(3);
    p2.add_instruction(pass_op{}, total);
    EXPECT(p1 != p2);
}

TEST_CASE(const_add3)
{
    migraphx::program p1;
    auto one  = p1.add_literal(1);
    auto two  = p1.add_literal(2);
    auto sum1 = p1.add_instruction(migraphx::op::add{}, one, two);
    auto sum2 = p1.add_instruction(migraphx::op::add{}, sum1, two);
    p1.add_instruction(pass_op{}, sum2);
    p1.compile(const_prop_target{});

    migraphx::program p2;
    auto total = p2.add_literal(5);
    p2.add_instruction(pass_op{}, total);
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
