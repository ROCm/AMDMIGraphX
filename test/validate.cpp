#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <basic_ops.hpp>
#include <test.hpp>
#include <rob.hpp>

TEST_CASE(simple_test)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, one, two);
    EXPECT(bool{p.validate() == p.end()});
    auto result = p.eval({});
    EXPECT(result.back() == migraphx::literal{3});
    EXPECT(result.back() != migraphx::literal{4});
}

TEST_CASE(out_of_order)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto ins = p.add_instruction(sum_op{}, one, two);
    p.move_instruction(two, p.end());
    EXPECT(bool{p.validate() == ins});
}

TEST_CASE(incomplete_args)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto ins = p.add_instruction(sum_op{}, one, two);
    ins->clear_arguments();
    EXPECT(bool{p.validate() == ins});
}

MIGRAPHX_ROB(access_ins_arguments,
             std::vector<migraphx::instruction_ref>,
             migraphx::instruction,
             arguments)

TEST_CASE(invalid_args)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto ins = p.add_instruction(sum_op{}, one, two);
    access_ins_arguments(*ins).clear();
    EXPECT(bool{p.validate() == p.begin()});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
