
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_div : verify_program<test_div>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x    = p.add_parameter("x", s);
        auto y    = p.add_parameter("y", s);
        auto z    = p.add_parameter("z", s);
        auto diff = p.add_instruction(migraphx::op::div{}, x, y);
        p.add_instruction(migraphx::op::div{}, diff, z);
        return p;
    }
};
