#include <migraph/common_subexpression_elimination.hpp>
#include <migraph/dead_code_elimination.hpp>
#include <migraph/operators.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct cse_target
{
    std::string name() const { return "dce"; }
    std::vector<migraph::pass> get_passes(migraph::context&) const
    {
        return {migraph::common_subexpression_elimination{}, migraph::dead_code_elimination{}};
    }
    migraph::context get_context() const { return {}; }
};

TEST_CASE(cse_test1)
{
    migraph::program p1;
    {
        auto one  = p1.add_literal(1);
        auto two  = p1.add_literal(2);
        auto sum1 = p1.add_instruction(migraph::op::add{}, one, two);
        auto sum2 = p1.add_instruction(migraph::op::add{}, one, two);
        auto sum3 = p1.add_instruction(migraph::op::add{}, sum1, sum2);
        p1.add_instruction(pass_op{}, sum3);
    }
    p1.compile(cse_target{});

    migraph::program p2;
    {
        auto one  = p2.add_literal(1);
        auto two  = p2.add_literal(2);
        auto sum1 = p2.add_instruction(migraph::op::add{}, one, two);
        auto sum3 = p2.add_instruction(migraph::op::add{}, sum1, sum1);
        p2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(cse_test2)
{
    migraph::program p1;
    {
        auto one  = p1.add_literal(1);
        auto two  = p1.add_literal(2);
        auto sum1 = p1.add_instruction(migraph::op::add{}, one, two);
        auto sum2 = p1.add_instruction(migraph::op::add{}, two, one);
        auto sum3 = p1.add_instruction(migraph::op::add{}, sum1, sum2);
        p1.add_instruction(pass_op{}, sum3);
    }
    p1.compile(cse_target{});

    migraph::program p2;
    {
        auto one  = p2.add_literal(1);
        auto two  = p2.add_literal(2);
        auto sum1 = p2.add_instruction(migraph::op::add{}, one, two);
        auto sum2 = p2.add_instruction(migraph::op::add{}, two, one);
        auto sum3 = p2.add_instruction(migraph::op::add{}, sum1, sum2);
        p2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(cse_test3)
{
    migraph::program p1;
    {
        auto one  = p1.add_literal(1);
        auto two  = p1.add_literal(1);
        auto sum1 = p1.add_instruction(migraph::op::add{}, one, two);
        auto sum2 = p1.add_instruction(migraph::op::add{}, two, one);
        auto sum3 = p1.add_instruction(migraph::op::add{}, sum1, sum2);
        p1.add_instruction(pass_op{}, sum3);
    }
    p1.compile(cse_target{});

    migraph::program p2;
    {
        auto one  = p2.add_literal(1);
        auto sum1 = p2.add_instruction(migraph::op::add{}, one, one);
        auto sum3 = p2.add_instruction(migraph::op::add{}, sum1, sum1);
        p2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(cse_test4)
{
    migraph::program p1;
    {
        auto one  = p1.add_literal(1);
        auto two  = p1.add_literal(1);
        auto sum1 = p1.add_instruction(migraph::op::add{}, one, two);
        auto sum2 = p1.add_instruction(migraph::op::add{}, two, one);
        auto sum3 = p1.add_instruction(migraph::op::add{}, sum1, one);
        auto sum4 = p1.add_instruction(migraph::op::add{}, sum2, two);
        auto sum5 = p1.add_instruction(migraph::op::add{}, sum4, sum3);
        p1.add_instruction(pass_op{}, sum5);
    }
    p1.compile(cse_target{});

    migraph::program p2;
    {
        auto one  = p2.add_literal(1);
        auto sum1 = p2.add_instruction(migraph::op::add{}, one, one);
        auto sum3 = p2.add_instruction(migraph::op::add{}, sum1, one);
        auto sum5 = p2.add_instruction(migraph::op::add{}, sum3, sum3);
        p2.add_instruction(pass_op{}, sum5);
    }
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
