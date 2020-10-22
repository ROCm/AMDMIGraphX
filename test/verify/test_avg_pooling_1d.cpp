
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_avg_pooling_1d : verify_program<test_avg_pooling_1d>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 3, 5}});
        auto op    = migraphx::op::pooling{"average", {0}, {1}, {3}};
        p.add_instruction(op, input);
        return p;
    }
};


