#include <migraphx/program.hpp>
#include <migraphx/ref/target.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/marker.hpp>
#include <migraphx/instruction.hpp>

#include "test.hpp"

struct mock_marker
{
    std::shared_ptr<std::stringstream> ss = std::make_shared<std::stringstream>();

    void mark_start(migraphx::instruction_ref ins_ref)
    {
        std::string text = "Mock marker instruction start:" + ins_ref->name();
        (*ss) << text;
    }
    void mark_stop(migraphx::instruction_ref)
    {
        std::string text = "Mock marker instruction stop.";
        (*ss) << text;
    }
    void mark_start(const migraphx::program&)
    {
        std::string text = "Mock marker program start.";
        (*ss) << text;
    }
    void mark_stop(const migraphx::program&)
    {
        std::string text = "Mock marker program stop.";
        (*ss) << text;
    }
};

TEST_CASE(roctx)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(migraphx::make_op("add"), one, two);
    p.compile(migraphx::ref::target{});

    mock_marker temp_marker;
    p.mark({}, temp_marker);

    std::string output = temp_marker.ss->str();
    EXPECT(migraphx::contains(output, "Mock marker instruction start:@literal"));
    EXPECT(migraphx::contains(output, "Mock marker instruction start:ref::op"));
    EXPECT(migraphx::contains(output, "Mock marker instruction stop."));
    EXPECT(migraphx::contains(output, "Mock marker program start."));
    EXPECT(migraphx::contains(output, "Mock marker program stop."));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
