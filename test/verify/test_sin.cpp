
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_sin : verify_program<test_sin>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {10}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::sin{}, x);
        return p;
    }
};


