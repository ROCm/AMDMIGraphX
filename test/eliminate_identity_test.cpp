#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <basic_ops.hpp>
#include <migraphx/op/identity.hpp>
#include <test.hpp>

void run_pass(migraphx::program& p) { migraphx::run_passes(p, {migraphx::eliminate_identity{}}); }

TEST_CASE(simple_test)
{
    migraphx::program p;

    auto one          = p.add_literal(1);
    auto one_identity = p.add_instruction(migraphx::op::identity{}, one);
    auto two          = p.add_literal(2);
    auto two_identity = p.add_instruction(migraphx::op::identity{}, two);
    p.add_instruction(sum_op{}, one_identity, two_identity);
    run_pass(p);
    EXPECT(std::none_of(p.begin(), p.end(), [](const migraphx::instruction& ins) {
        return ins.name() == "identity";
    }));
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3});
}

TEST_CASE(simple_test_end)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto ans = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(migraphx::op::identity{}, ans);
    run_pass(p);
    EXPECT(std::none_of(p.begin(), p.end(), [](const migraphx::instruction& ins) {
        return ins.name() == "identity";
    }));
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3});
}

TEST_CASE(simple_test_end_dependency)
{
    migraphx::program p;

    auto one   = p.add_literal(1.0);
    auto two   = p.add_literal(2.0);
    auto three = p.add_literal(3.0);
    auto ans   = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(sum_op{}, ans, three);
    p.add_instruction(migraphx::op::identity{}, ans);
    run_pass(p);
    EXPECT(std::any_of(p.begin(), p.end(), [](const migraphx::instruction& ins) {
        return ins.name() == "identity";
    }));
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3.0});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
