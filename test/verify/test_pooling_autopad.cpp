
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/op/pooling.hpp>

struct test_pooling_autopad : verify_program<test_pooling_autopad>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s0{migraphx::shape::float_type, {1, 3, 63, 63}};
        auto l0 = mm->add_parameter("x", s0);
        migraphx::op::pooling op{migraphx::op::pooling_mode::max};
        op.lengths = {2, 2};
        op.stride  = {2, 2};
        mm->add_instruction(op, l0);
        return p;
    }
};
