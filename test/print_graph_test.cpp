#include <migraphx/program.hpp>
#include <migraphx/ranges.hpp>
#include <sstream>
#include "test.hpp"
#include <basic_ops.hpp>

migraphx::program create_program()
{
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto x = mm->add_parameter("x", {migraphx::shape::int64_type});
    auto y = mm->add_parameter("y", {migraphx::shape::int64_type});

    auto sum = mm->add_instruction(sum_op{}, x, y);
    auto one = mm->add_literal(1);
    mm->add_instruction(sum_op{}, sum, one);

    return p;
}

TEST_CASE(basic_graph_test)
{
    migraphx::program p = create_program();

    std::stringstream ss;
    p.print_graph(ss);
    std::string test = ss.str();
    EXPECT(migraphx::contains(test, "digraph"));
    EXPECT(migraphx::contains(test, "rankdir=LR"));
    EXPECT(migraphx::contains(test, "\"@0\"[label=\"@literal\"]"));
    EXPECT(migraphx::contains(test, "\"y\"[label=\"@param:y\"]"));
    EXPECT(migraphx::contains(test, "\"x\"[label=\"@param:x\"]"));
    EXPECT(migraphx::contains(test, "\"@1\"[label=\"sum\"]"));
    EXPECT(migraphx::contains(test, "\"@2\"[label=\"sum\"]"));
    EXPECT(migraphx::contains(test, "\"x\" -> \"@1\""));
    EXPECT(migraphx::contains(test, "\"y\" -> \"@1\""));
    EXPECT(migraphx::contains(test, "\"@1\" -> \"@2\""));
    EXPECT(migraphx::contains(test, "\"@0\" -> \"@2\""));
    EXPECT(migraphx::contains(test, "[label=\"int64_type, {1}, {0}\"]"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
