#include <migraphx/dead_code_elimination.hpp>
#include <basic_ops.hpp>
#include <migraphx/op/abnormal_ops.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/op/identity.hpp>
#include <test.hpp>

struct dce_target
{
    std::string name() const { return "dce"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&) const
    {
        return {migraphx::dead_code_elimination{}};
    }
    migraphx::context get_context() const { return {}; }
};

TEST_CASE(simple_test)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, one, two);
    auto count = std::distance(p.begin(), p.end());
    p.compile(dce_target{});
    EXPECT(std::distance(p.begin(), p.end()) == count);
    auto result = p.eval({});
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(simple_test_nop)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(nop{});
    p.add_instruction(sum_op{}, one, two);
    auto count = std::distance(p.begin(), p.end());
    p.compile(dce_target{});
    EXPECT(std::distance(p.begin(), p.end()) == count);
    auto result = p.eval({});
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(simple_test_nop2)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(nop{});
    p.add_instruction(sum_op{}, one, two);
    p.add_instruction(nop{});
    p.compile(dce_target{});
    EXPECT(std::distance(p.begin(), p.end()) == 2);
    auto result = p.eval({});
    EXPECT(result == migraphx::literal{});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(duplicate_test1)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, one, two);
    p.add_instruction(sum_op{}, one, two);
    auto count = std::distance(p.begin(), p.end());
    p.compile(dce_target{});
    EXPECT(std::distance(p.begin(), p.end()) == (count - 1));
    auto result = p.eval({});
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(duplicate_test2)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, one, two);
    p.add_instruction(minus_op{}, one, two);
    p.add_instruction(sum_op{}, one, two);
    auto count = std::distance(p.begin(), p.end());
    p.compile(dce_target{});
    EXPECT(std::distance(p.begin(), p.end()) == (count - 2));
    auto result = p.eval({});
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(depth_test)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto x1  = p.add_instruction(sum_op{}, one, two);
    auto x2  = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(minus_op{}, x1, x2);
    p.add_instruction(minus_op{}, x1, x2);
    p.add_instruction(sum_op{}, one, two);
    auto count = std::distance(p.begin(), p.end());
    p.compile(dce_target{});
    EXPECT(std::distance(p.begin(), p.end()) == (count - 4));
    auto result = p.eval({});
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(undefined_test)
{
    migraphx::program p;

    auto one   = p.add_literal(1);
    auto two   = p.add_literal(2);
    auto undef = p.add_instruction(migraphx::op::undefined{});
    p.add_instruction(sum_op{}, one, two);
    auto count = std::distance(p.begin(), p.end());
    p.compile(dce_target{});
    EXPECT(std::distance(p.begin(), p.end()) == count - 1);
    EXPECT(not p.has_instruction(undef));
    auto result = p.eval({});
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(duplicate_args1)
{
    migraphx::program p;

    auto l0 = p.add_literal(0);
    auto l3 = p.add_literal(3);
    p.add_instruction(migraphx::op::add{}, l3, l3);
    p.add_instruction(migraphx::op::identity{}, l0);
    auto count = std::distance(p.begin(), p.end());
    p.compile(dce_target{});
    EXPECT(std::distance(p.begin(), p.end()) != count);
    EXPECT(std::distance(p.begin(), p.end()) == 2);
    auto result = p.eval({});
    EXPECT(result == migraphx::literal{0});
}

TEST_CASE(duplicate_args2)
{
    migraphx::program p;

    auto l0   = p.add_literal(0);
    auto l3   = p.add_literal(3);
    auto sum1 = p.add_instruction(migraphx::op::add{}, l0, l3);
    p.add_instruction(migraphx::op::add{}, sum1, l3);
    p.add_instruction(migraphx::op::identity{}, l0);
    auto count = std::distance(p.begin(), p.end());
    p.compile(dce_target{});
    EXPECT(std::distance(p.begin(), p.end()) != count);
    EXPECT(std::distance(p.begin(), p.end()) == 2);
    auto result = p.eval({});
    EXPECT(result == migraphx::literal{0});
}

TEST_CASE(duplicate_args3)
{
    migraphx::program p;

    auto l0   = p.add_literal(0);
    auto l3   = p.add_literal(3);
    auto sum1 = p.add_instruction(migraphx::op::add{}, l0, l3);
    auto sum2 = p.add_instruction(migraphx::op::add{}, l0, sum1);
    p.add_instruction(migraphx::op::add{}, sum2, l3);
    p.add_instruction(migraphx::op::identity{}, l0);
    auto count = std::distance(p.begin(), p.end());
    p.compile(dce_target{});
    EXPECT(std::distance(p.begin(), p.end()) != count);
    EXPECT(std::distance(p.begin(), p.end()) == 2);
    auto result = p.eval({});
    EXPECT(result == migraphx::literal{0});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
