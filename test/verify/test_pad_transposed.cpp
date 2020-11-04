
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_pad_transposed : verify_program<test_pad_transposed>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::int32_type, {1, 224, 224, 3}};
        auto x = p.add_parameter("x", s);
        auto t = p.add_instruction(migraphx::op::transpose{{0, 3, 1, 2}}, x);
        p.add_instruction(migraphx::op::pad{{0, 0, 2, 2, 0, 0, 3, 3}}, t);
        return p;
    }
};
