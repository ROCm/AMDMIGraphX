
#include <migraphx/program.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <sstream>
#include "test.hpp"
#include <basic_ops.hpp>

migraphx::program create_program()
{
    migraphx::program p;

    auto x = p.add_parameter("x", {migraphx::shape::int64_type});
    auto y = p.add_parameter("y", {migraphx::shape::int64_type});

    auto sum = p.add_instruction(sum_op{}, x, y);
    auto one = p.add_literal(1);
    p.add_instruction(sum_op{}, sum, one);

    return p;
}

TEST_CASE(program_equality)
{
    migraphx::program x = create_program();
    migraphx::program y = create_program();
    EXPECT(x == y);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
