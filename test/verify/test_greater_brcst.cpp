
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_greater_brcst : verify_program<test_greater_brcst>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s0{migraphx::shape::float_type, {3, 3}};
        auto l0 = mm->add_parameter("x", s0);
        migraphx::shape s1{migraphx::shape::float_type, {3, 1}};
        auto l1 = mm->add_parameter("y", s1);
        auto bl1 =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s0.lens()}}), l1);
        auto r = mm->add_instruction(migraphx::make_op("greater"), l0, bl1);
        mm->add_return({r});

        return p;
    };
};
