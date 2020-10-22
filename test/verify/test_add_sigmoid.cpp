
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_add_sigmoid : verify_program<test_add_sigmoid>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x   = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto y   = p.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto add = p.add_instruction(migraphx::op::add{}, x, y);
        p.add_instruction(migraphx::op::sigmoid{}, add);
        return p;
    }
};


