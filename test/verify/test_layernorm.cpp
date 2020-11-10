
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

migraphx::instruction_ref add_layernorm(migraphx::program& p, migraphx::instruction_ref x, std::vector<size_t> dims)
{
    auto scale =
        p.add_parameter("scale", migraphx::shape{migraphx::shape::float_type, {dims.back()}});
    auto bias =
        p.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {dims.back()}});
    auto epsilon  = p.add_literal(1e-12f);
    auto exponent = p.add_literal(2.0f);

    auto mean            = p.add_instruction(migraphx::op::reduce_mean({2}), x);
    auto mean_mbcast     = p.add_instruction(migraphx::op::multibroadcast{{dims}}, mean);
    auto sub             = p.add_instruction(migraphx::op::sub{}, x, mean_mbcast);
    auto exponent_mbcast = p.add_instruction(migraphx::op::multibroadcast{{dims}}, exponent);
    auto pow             = p.add_instruction(migraphx::op::pow{}, sub, exponent_mbcast);
    auto var             = p.add_instruction(migraphx::op::reduce_mean({2}), pow);
    auto epsilon_mbcast =
        p.add_instruction(migraphx::op::multibroadcast{{1, dims.at(1), 1}}, epsilon);
    auto add_epsilon  = p.add_instruction(migraphx::op::add{}, var, epsilon_mbcast);
    auto sqrt         = p.add_instruction(migraphx::op::sqrt{}, add_epsilon);
    auto sqrt_mbcast  = p.add_instruction(migraphx::op::multibroadcast{dims}, sqrt);
    auto div          = p.add_instruction(migraphx::op::div{}, sub, sqrt_mbcast);
    auto scale_mbcast = p.add_instruction(migraphx::op::multibroadcast{dims}, scale);
    auto mul          = p.add_instruction(migraphx::op::mul{}, scale_mbcast, div);
    auto bias_mbcast  = p.add_instruction(migraphx::op::multibroadcast{dims}, bias);
    return p.add_instruction(migraphx::op::add{}, mul, bias_mbcast);
}

struct test_layernorm : verify_program<test_layernorm>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        std::vector<size_t> dims = {1, 1, 5};
        auto x = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, dims});
        add_layernorm(p, x, dims);
        return p;
    }
};

struct test_layernorm2 : verify_program<test_layernorm2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        std::vector<size_t> dims = {1, 4, 24};
        auto x = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, dims});
        add_layernorm(p, x, dims);
        return p;
    }
};

struct test_layernorm_triadd : verify_program<test_layernorm_triadd>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        std::vector<size_t> dims = {1, 4, 24};
        auto x = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, dims});
        auto y = p.add_parameter("y", migraphx::shape{migraphx::shape::float_type, dims});
        auto z = p.add_parameter("z", migraphx::shape{migraphx::shape::float_type, dims});
        auto add1 = p.add_instruction(migraphx::op::add{}, x, y);
        auto add2 = p.add_instruction(migraphx::op::add{}, add1, z);
        add_layernorm(p, add2, dims);
        return p;
    }
};
