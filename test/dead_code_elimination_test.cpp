#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(*p.get_main_module(), {migraphx::dead_code_elimination{}});
}

TEST_CASE(simple_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(sum_op{}, one, two);
    auto count = std::distance(mm->begin(), mm->end());
    run_pass(p);
    EXPECT(std::distance(mm->begin(), mm->end()) == count);
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(simple_test_nop)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(nop{});
    mm->add_instruction(sum_op{}, one, two);
    auto count = std::distance(mm->begin(), mm->end());
    run_pass(p);
    EXPECT(std::distance(mm->begin(), mm->end()) == count);
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(simple_test_nop2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(nop{});
    mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(nop{});
    run_pass(p);
    EXPECT(std::distance(mm->begin(), mm->end()) == 2);
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(duplicate_test1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(sum_op{}, one, two);
    auto count = std::distance(mm->begin(), mm->end());
    run_pass(p);
    EXPECT(std::distance(mm->begin(), mm->end()) == (count - 1));
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(duplicate_test2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(minus_op{}, one, two);
    mm->add_instruction(sum_op{}, one, two);
    auto count = std::distance(mm->begin(), mm->end());
    run_pass(p);
    EXPECT(std::distance(mm->begin(), mm->end()) == (count - 2));
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(depth_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto x1  = mm->add_instruction(sum_op{}, one, two);
    auto x2  = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(minus_op{}, x1, x2);
    mm->add_instruction(minus_op{}, x1, x2);
    mm->add_instruction(sum_op{}, one, two);
    auto count = std::distance(mm->begin(), mm->end());
    run_pass(p);
    EXPECT(std::distance(mm->begin(), mm->end()) == (count - 4));
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(undefined_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto one   = mm->add_literal(1);
    auto two   = mm->add_literal(2);
    auto undef = mm->add_instruction(migraphx::make_op("undefined"));
    mm->add_instruction(sum_op{}, one, two);
    auto count = std::distance(mm->begin(), mm->end());
    run_pass(p);
    EXPECT(std::distance(mm->begin(), mm->end()) == count - 1);
    EXPECT(not mm->has_instruction(undef));
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(duplicate_args1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_literal(0);
    auto l3  = mm->add_literal(3);
    mm->add_instruction(migraphx::make_op("add"), l3, l3);
    mm->add_instruction(migraphx::make_op("identity"), l0);
    auto count = std::distance(mm->begin(), mm->end());
    run_pass(p);
    EXPECT(std::distance(mm->begin(), mm->end()) != count);
    EXPECT(std::distance(mm->begin(), mm->end()) == 2);
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{0});
}

TEST_CASE(duplicate_args2)
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto l0   = mm->add_literal(0);
    auto l3   = mm->add_literal(3);
    auto sum1 = mm->add_instruction(migraphx::make_op("add"), l0, l3);
    mm->add_instruction(migraphx::make_op("add"), sum1, l3);
    mm->add_instruction(migraphx::make_op("identity"), l0);
    auto count = std::distance(mm->begin(), mm->end());
    run_pass(p);
    EXPECT(std::distance(mm->begin(), mm->end()) != count);
    EXPECT(std::distance(mm->begin(), mm->end()) == 2);
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{0});
}

TEST_CASE(duplicate_args3)
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto l0   = mm->add_literal(0);
    auto l3   = mm->add_literal(3);
    auto sum1 = mm->add_instruction(migraphx::make_op("add"), l0, l3);
    auto sum2 = mm->add_instruction(migraphx::make_op("add"), l0, sum1);
    mm->add_instruction(migraphx::make_op("add"), sum2, l3);
    mm->add_instruction(migraphx::make_op("identity"), l0);
    auto count = std::distance(mm->begin(), mm->end());
    run_pass(p);
    EXPECT(std::distance(mm->begin(), mm->end()) != count);
    EXPECT(std::distance(mm->begin(), mm->end()) == 2);
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{0});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
