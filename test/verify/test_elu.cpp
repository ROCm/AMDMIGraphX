
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_elu : verify_program<test_elu>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        p.add_instruction(migraphx::op::leaky_relu{1.0}, x);
        return p;
    }
};


