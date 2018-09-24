#include <migraph/program.hpp>
#include <migraph/instruction.hpp>
#include <basic_ops.hpp>
#include <test.hpp>
#include <rob.hpp>

void simple_test()
{
    migraph::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, one, two);
    EXPECT(bool{p.validate() == p.end()});
    auto result = p.eval({});
    EXPECT(result == migraph::literal{3});
    EXPECT(result != migraph::literal{4});
}

void out_of_order()
{
    migraph::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto ins = p.add_instruction(sum_op{}, one, two);
    p.move_instruction(two, p.end());
    EXPECT(bool{p.validate() == ins});
}

void incomplete_args()
{
    migraph::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto ins = p.add_instruction(sum_op{}, one, two);
    ins->clear_arguments();
    EXPECT(bool{p.validate() == ins});
}

MIGRAPH_ROB(access_ins_arguments,
            std::vector<migraph::instruction_ref>,
            migraph::instruction,
            arguments)

void invalid_args()
{
    migraph::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto ins = p.add_instruction(sum_op{}, one, two);
    access_ins_arguments(*ins).clear();
    EXPECT(bool{p.validate() == p.begin()});
}

int main()
{
    simple_test();
    out_of_order();
    incomplete_args();
    invalid_args();
}
