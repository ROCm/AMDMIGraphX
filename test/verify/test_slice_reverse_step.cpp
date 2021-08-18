
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_slice_reverse_step : verify_program<test_slice_reverse_step>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::int32_type, {7, 5}};
        auto x         = mm->add_parameter("x", s);
        auto slice_out = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {0, 2}}, {"ends", {2, -1}}}),
            x);
        auto step_out =
            mm->add_instruction(migraphx::make_op("reverse", {{"axes", {0, 1}}}), slice_out);
        mm->add_instruction(migraphx::make_op("step", {{"axes", {0, 1}}, {"steps", {2, 2}}}),
                            step_out);
        return p;
    }
};
