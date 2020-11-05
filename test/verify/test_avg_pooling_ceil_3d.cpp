
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_avg_pooling_ceil_3d : verify_program<test_avg_pooling_ceil_3d>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto input =
            p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 3, 5, 5, 5}});
        auto op = migraphx::op::pooling{"average", {1, 1, 1}, {3, 3, 3}, {3, 3, 3}, true};
        p.add_instruction(op, input);
        return p;
    }
};
