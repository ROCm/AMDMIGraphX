
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_sign : verify_program<test_sign>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::double_type, {2, 3, 4, 6}};
        auto param = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::sign{}, param);
        return p;
    }
};


