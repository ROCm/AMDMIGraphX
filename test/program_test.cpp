
#include <migraph/program.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/instruction.hpp>
#include <sstream>
#include "test.hpp"
#include <basic_ops.hpp>

migraph::program create_program()
{
    migraph::program p;

    auto x = p.add_parameter("x", {migraph::shape::int64_type});
    auto y = p.add_parameter("y", {migraph::shape::int64_type});

    auto sum = p.add_instruction(sum_op{}, x, y);
    auto one  = p.add_literal(1);
    p.add_instruction(sum_op{}, sum, one);

    return p;
}

void program_equality() 
{
    migraph::program x = create_program();
    migraph::program y = create_program();
    EXPECT(x == y);
}

int main() {
    program_equality();
}
