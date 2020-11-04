
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_pooling_autopad : verify_program<test_pooling_autopad>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s0{migraphx::shape::float_type, {1, 3, 63, 63}};
        auto l0 = p.add_parameter("x", s0);
        migraphx::op::pooling op{"max"};
        op.lengths = {2, 2};
        op.stride  = {2, 2};
        p.add_instruction(op, l0);
        return p;
    }
};
