
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_add_gelu : verify_program<test_add_gelu>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        std::vector<size_t> input_lens{1, 1, 5};
        auto x            = p.add_parameter("x", {migraphx::shape::float_type, input_lens});
        auto y            = p.add_parameter("y", {migraphx::shape::float_type, input_lens});
        auto half         = p.add_literal(0.5f);
        auto one          = p.add_literal(1.0f);
        auto sqrt2        = p.add_literal(static_cast<float>(M_SQRT2));
        auto add          = p.add_instruction(migraphx::op::add{}, x, y);
        auto half_mbcast  = p.add_instruction(migraphx::op::multibroadcast{input_lens}, half);
        auto mul_half     = p.add_instruction(migraphx::op::mul{}, add, half_mbcast);
        auto sqrt2_mbcast = p.add_instruction(migraphx::op::multibroadcast{input_lens}, sqrt2);
        auto div          = p.add_instruction(migraphx::op::div{}, add, sqrt2_mbcast);
        auto erf          = p.add_instruction(migraphx::op::erf{}, div);
        auto one_mbcast   = p.add_instruction(migraphx::op::multibroadcast{input_lens}, one);
        auto add_one      = p.add_instruction(migraphx::op::add{}, erf, one_mbcast);
        p.add_instruction(migraphx::op::mul{}, mul_half, add_one);
        return p;
    }
};
