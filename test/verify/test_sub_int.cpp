#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_sub_int : verify_program<test_sub_int>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x  = mm->add_parameter("x", {migraphx::shape::int16_type, {4, 5}});
        auto y  = mm->add_parameter("y", {migraphx::shape::int16_type, {2, 3, 4, 5}});
        auto xb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 4, 5}}}), x);
        auto diff = mm->add_instruction(migraphx::make_op("sub"), y, xb);
        mm->add_return({diff});
        return p;
    }
};
