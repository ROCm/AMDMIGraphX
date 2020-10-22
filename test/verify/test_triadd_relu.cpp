
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_triadd_relu : verify_program<test_triadd_relu>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x   = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto y   = p.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto z   = p.add_parameter("z", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto sum = p.add_instruction(migraphx::op::add{}, x, y);
        auto triadd = p.add_instruction(migraphx::op::add{}, sum, z);
        p.add_instruction(migraphx::op::relu{}, triadd);
        return p;
    }
};


