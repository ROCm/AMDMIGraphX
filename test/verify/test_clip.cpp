
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_clip : verify_program<test_clip>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x       = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3}});
        auto min_val = p.add_literal(0.0f);
        auto max_val = p.add_literal(6.0f);
        min_val      = p.add_instruction(migraphx::op::multibroadcast{{3}}, min_val);
        max_val      = p.add_instruction(migraphx::op::multibroadcast{{3}}, max_val);
        p.add_instruction(migraphx::op::clip{}, x, min_val, max_val);
        return p;
    }
};


