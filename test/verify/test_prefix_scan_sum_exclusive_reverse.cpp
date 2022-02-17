#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_prefix_scan_sum_exclusive_reverse
    : verify_program<test_prefix_scan_sum_exclusive_reverse>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3, 3, 3}};
        auto x = mm->add_parameter("x", s);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum",
                              {{"axis", 0}, {"exclusive", true}, {"reverse", true}}),
            x);
        return p;
    }
};
