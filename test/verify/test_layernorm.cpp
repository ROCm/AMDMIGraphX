
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

migraphx::instruction_ref add_layernorm(migraphx::program& p, std::vector<size_t> dims)
{
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, dims});
    auto scale =
        mm->add_parameter("scale", migraphx::shape{migraphx::shape::float_type, {dims.back()}});
    auto bias =
        mm->add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {dims.back()}});
    auto epsilon  = mm->add_literal(1e-12f);
    auto exponent = mm->add_literal(2.0f);

    auto mean            = mm->add_instruction(migraphx::op::reduce_mean({2}), x);
    auto mean_mbcast     = mm->add_instruction(migraphx::op::multibroadcast{{dims}}, mean);
    auto sub             = mm->add_instruction(migraphx::op::sub{}, x, mean_mbcast);
    auto exponent_mbcast = mm->add_instruction(migraphx::op::multibroadcast{{dims}}, exponent);
    auto pow             = mm->add_instruction(migraphx::op::pow{}, sub, exponent_mbcast);
    auto var             = mm->add_instruction(migraphx::op::reduce_mean({2}), pow);
    auto epsilon_mbcast =
        mm->add_instruction(migraphx::op::multibroadcast{{1, dims.at(1), 1}}, epsilon);
    auto add_epsilon  = mm->add_instruction(migraphx::op::add{}, var, epsilon_mbcast);
    auto sqrt         = mm->add_instruction(migraphx::op::sqrt{}, add_epsilon);
    auto sqrt_mbcast  = mm->add_instruction(migraphx::op::multibroadcast{dims}, sqrt);
    auto div          = mm->add_instruction(migraphx::op::div{}, sub, sqrt_mbcast);
    auto scale_mbcast = mm->add_instruction(migraphx::op::multibroadcast{dims}, scale);
    auto mul          = mm->add_instruction(migraphx::op::mul{}, scale_mbcast, div);
    auto bias_mbcast  = mm->add_instruction(migraphx::op::multibroadcast{dims}, bias);
    return mm->add_instruction(migraphx::op::add{}, mul, bias_mbcast);
}

struct test_layernorm : verify_program<test_layernorm>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        add_layernorm(p, {1, 1, 5});
        return p;
    }
};

struct test_layernorm2 : verify_program<test_layernorm2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        add_layernorm(p, {1, 4, 24});
        return p;
    }
};
