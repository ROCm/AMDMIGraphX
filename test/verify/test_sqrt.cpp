
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_sqrt : verify_program<test_sqrt>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 6}};
        auto param     = p.add_parameter("x", s);
        auto param_abs = p.add_instruction(migraphx::op::abs{}, param);
        p.add_instruction(migraphx::op::sqrt{}, param_abs);
        return p;
    }
};
