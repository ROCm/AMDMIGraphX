
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_conv_bn_relu_pooling : verify_program<test_conv_bn_relu_pooling>
{
    migraphx::program create_program() const
    {
        migraphx::program p;

        migraphx::shape xs{migraphx::shape::float_type, {1, 3, 224, 224}};
        migraphx::shape ws{migraphx::shape::float_type, {64, 3, 7, 7}};
        migraphx::shape vars{migraphx::shape::float_type, {64}};
        auto x        = p.add_parameter("x", xs);
        auto w        = p.add_parameter("w", ws);
        auto conv     = p.add_instruction(migraphx::op::convolution{{3, 3}, {2, 2}, {1, 1}}, x, w);
        auto scale    = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 1)));
        auto bias     = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 2)));
        auto mean     = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 3)));
        auto variance = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 4)));
        auto bn       = p.add_instruction(
            migraphx::op::batch_norm_inference{}, conv, scale, bias, mean, variance);
        auto relu = p.add_instruction(migraphx::op::relu{}, bn);
        p.add_instruction(migraphx::op::pooling{"average", {1, 1}, {2, 2}, {3, 3}}, relu);
        return p;
    }
};


