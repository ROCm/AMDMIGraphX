
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_equal_brcst : verify_program<test_equal_brcst>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s0{migraphx::shape::float_type, {3, 3}};
        auto l0 = p.add_parameter("x", s0);
        migraphx::shape s1{migraphx::shape::float_type, {3, 1}};
        auto l1  = p.add_parameter("y", s1);
        auto bl1 = p.add_instruction(migraphx::op::multibroadcast{s0.lens()}, l1);
        auto r   = p.add_instruction(migraphx::op::equal{}, l0, bl1);
        p.add_return({r});

        return p;
    };
};
