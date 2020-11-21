
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_concat_relu : verify_program<test_concat_relu>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        int axis = 0;
        migraphx::shape s0{migraphx::shape::float_type, {2, 2}};
        migraphx::shape s1{migraphx::shape::float_type, {3, 2}};
        migraphx::shape s2{migraphx::shape::float_type, {1, 2}};
        auto l0 = mm->add_parameter("x", s0);
        auto l1 = mm->add_parameter("y", s1);
        auto l2 = mm->add_parameter("z", s2);
        auto r0 = mm->add_instruction(migraphx::op::relu{}, l0);
        auto r1 = mm->add_instruction(migraphx::op::relu{}, l1);
        auto r2 = mm->add_instruction(migraphx::op::relu{}, l2);
        auto c0 = mm->add_instruction(migraphx::op::concat{axis}, r0, r1, r2);
        mm->add_instruction(migraphx::op::relu{}, c0);
        return p;
    }
};
