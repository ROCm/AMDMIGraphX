#include <migraph/program.hpp>
#include <migraph/instruction.hpp>
#include <test.hpp>
#include <basic_ops.hpp>

void simple_alias()
{
    migraph::program p;
    auto l  = p.add_literal(1);
    auto p1 = p.add_instruction(pass_op{}, l);
    EXPECT(bool{migraph::instruction::get_output_alias(l) == l});
    EXPECT(bool{migraph::instruction::get_output_alias(p1) == l});
}

void cascade_alias()
{
    migraph::program p;
    auto l  = p.add_literal(1);
    auto p1 = p.add_instruction(pass_op{}, l);
    auto p2 = p.add_instruction(pass_op{}, p1);
    auto p3 = p.add_instruction(pass_op{}, p2);
    EXPECT(bool{migraph::instruction::get_output_alias(l) == l});
    EXPECT(bool{migraph::instruction::get_output_alias(p1) == l});
    EXPECT(bool{migraph::instruction::get_output_alias(p2) == l});
    EXPECT(bool{migraph::instruction::get_output_alias(p3) == l});
}

void no_alias()
{
    migraph::program p;
    auto x   = p.add_literal(1);
    auto y   = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, x, y);
    EXPECT(bool{migraph::instruction::get_output_alias(sum) == sum});
}

int main()
{
    simple_alias();
    cascade_alias();
    no_alias();
}
