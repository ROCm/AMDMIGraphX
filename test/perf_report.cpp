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
    p.perf_report(ss, 2, {});

    std::string output = ss.str();
    EXPECT(migraphx::contains(output, "Summary:"));
    EXPECT(migraphx::contains(output, "Batch size:"));
    EXPECT(migraphx::contains(output, "Rate:"));
    EXPECT(migraphx::contains(output, "Total time:"));
    EXPECT(migraphx::contains(output, "Total instructions time:"));
    EXPECT(migraphx::contains(output, "Overhead time:"));
    EXPECT(migraphx::contains(output, "Overhead:"));
    EXPECT(not migraphx::contains(output, "fast"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
