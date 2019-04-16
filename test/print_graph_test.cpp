#include <migraphx/program.hpp>
#include <migraphx/ranges.hpp>
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

TEST_CASE(basic_graph_test)
{
    migraphx::program p = create_program();
    std::stringstream ss;
    p.print_graph(ss);
    std::string test = ss.str();
    std::cout << test << std::endl;
    EXPECT(test.find("[label=@literal]") != std::string::npos);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
