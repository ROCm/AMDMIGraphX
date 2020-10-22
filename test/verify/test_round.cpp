
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_round : verify_program<test_round>
{
    migraphx::program create_program() const
    {
        migraphx::program p;

        migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 6}};
        auto param = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::round{}, param);
        return p;
    };
};


