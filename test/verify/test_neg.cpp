
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_neg : verify_program<test_neg>
{
    migraphx::program create_program() const
    {
        migraphx::program p;

        migraphx::shape s{migraphx::shape::double_type, {2, 3, 4, 6}};
        auto input = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::neg{}, input);
        return p;
    };
};
