
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_cos : verify_program<test_cos>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::double_type, {8}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::cos{}, x);
        return p;
    }
};


