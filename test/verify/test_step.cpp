
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_step : verify_program<test_step>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s1{migraphx::shape::float_type, {2, 1, 4, 6}};
        auto l0 = mm->add_parameter("x", s1);
        auto r  = mm->add_instruction(
            migraphx::make_op("step", {{"axes", {0, 2, 3}}, {"steps", {2, 2, 3}}}), l0);
        mm->add_return({r});

        return p;
    }
};
