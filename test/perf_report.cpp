#include <migraphx/program.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/ranges.hpp>
#include "test.hpp"

TEST_CASE(perf_report)
{
    migraphx::program p;

    std::stringstream ss;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(migraphx::op::add{}, one, two);
    p.compile(migraphx::cpu::target{});
    p.perf_report(ss, 2, {});

    std::string output = ss.str();
    EXPECT(migraphx::contains(output, "Summary:"));
    EXPECT(migraphx::contains(output, "Rate:"));
    EXPECT(migraphx::contains(output, "Total time:"));
    EXPECT(migraphx::contains(output, "Total instructions time:"));
    EXPECT(migraphx::contains(output, "Overhead time:"));
    EXPECT(migraphx::contains(output, "Overhead:"));
    EXPECT(not migraphx::contains(output, "fast"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
