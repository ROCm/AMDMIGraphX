#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <test.hpp>
#include <basic_ops.hpp>

TEST_CASE(simple_alias)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(1);
    auto p1  = mm->add_instruction(pass_op{}, l);
    EXPECT(bool{migraphx::instruction::get_output_alias(l) == l});
    EXPECT(bool{migraphx::instruction::get_output_alias(p1) == l});
}

TEST_CASE(cascade_alias)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(1);
    auto p1  = mm->add_instruction(pass_op{}, l);
    auto p2  = mm->add_instruction(pass_op{}, p1);
    auto p3  = mm->add_instruction(pass_op{}, p2);
    EXPECT(bool{migraphx::instruction::get_output_alias(l) == l});
    EXPECT(bool{migraphx::instruction::get_output_alias(p1) == l});
    EXPECT(bool{migraphx::instruction::get_output_alias(p2) == l});
    EXPECT(bool{migraphx::instruction::get_output_alias(p3) == l});
}

TEST_CASE(no_alias)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(1);
    auto y   = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, x, y);
    EXPECT(bool{migraphx::instruction::get_output_alias(sum) == sum});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
