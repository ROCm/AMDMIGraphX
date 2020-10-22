
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_erf : verify_program<test_erf>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 6}};
        auto param = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::erf{}, param);
        return p;
    }
};


