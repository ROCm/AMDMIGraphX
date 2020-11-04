
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_exp : verify_program<test_exp>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {6}};
        auto x = p.add_instruction(migraphx::op::abs{}, p.add_parameter("x", s));
        p.add_instruction(migraphx::op::exp{}, x);
        return p;
    }
};
