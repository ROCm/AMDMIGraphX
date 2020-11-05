
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_tan : verify_program<test_tan>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {16}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::tan{}, x);
        return p;
    }
};
