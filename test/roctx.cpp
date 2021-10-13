#include <migraphx/program.hpp>
#include <migraphx/ref/target.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

#include "test.hpp"

TEST_CASE(perf_report)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::stringstream ss;
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(migraphx::make_op("add"), one, two);
    p.compile(migraphx::ref::target{});
    p.roctx();

    std::string output = ss.str();
    EXPECT(migraphx::contains(output, "rocTX:  Loading rocTX library..."));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
