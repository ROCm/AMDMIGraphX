#include <migraphx/dead_code_elimination.hpp>
#include <basic_ops.hpp>
#include <migraphx/operators.hpp>
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

int main(int argc, const char* argv[]) { test::run(argc, argv); }
