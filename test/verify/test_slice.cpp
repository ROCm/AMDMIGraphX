
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_slice : verify_program<test_slice>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::int32_type, {2, 2, 4}};
        auto x      = mm->add_parameter("x", s);
        auto y      = mm->add_parameter("y", {migraphx::shape::int32_type, {2, 2, 2}});
        auto slice0 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {2}}}), x);
        mm->add_instruction(migraphx::make_op("add"), y, slice0);

        return p;
    }
};
