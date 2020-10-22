
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_scale : verify_program<test_scale>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x     = p.add_parameter("x", s);
        auto y     = p.add_parameter("y", migraphx::shape::float_type);
        auto scale = p.add_instruction(migraphx::op::scalar{s.lens()}, y);
        p.add_instruction(migraphx::op::mul{}, x, scale);
        return p;
    }
};
