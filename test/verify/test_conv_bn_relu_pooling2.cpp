
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_conv_bn_relu_pooling2 : verify_program<test_conv_bn_relu_pooling2>
{
    static migraphx::instruction_ref
    add_bn(migraphx::program& p, migraphx::instruction_ref x, std::size_t channels)
    {
        migraphx::shape vars{migraphx::shape::float_type, {channels}};
        auto scale = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 1 + channels)));
        auto bias  = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 2 + channels)));
        auto mean  = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 3 + channels)));
        auto variance =
            p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 4 + channels)));
        return p.add_instruction(
            migraphx::op::batch_norm_inference{}, x, scale, bias, mean, variance);
    }
    migraphx::program create_program() const
    {
        migraphx::program p;

        migraphx::shape xs1{migraphx::shape::float_type, {1, 512, 7, 7}};
        migraphx::shape xs2{migraphx::shape::float_type, {1, 1024, 14, 14}};
        migraphx::shape ws1{migraphx::shape::float_type, {2048, 512, 1, 1}};
        migraphx::shape ws2{migraphx::shape::float_type, {2048, 1024, 1, 1}};
        auto x1    = p.add_parameter("x1", xs1);
        auto w1    = p.add_parameter("w1", ws1);
        auto conv1 = p.add_instruction(migraphx::op::convolution{{0, 0}, {1, 1}, {1, 1}}, x1, w1);
        auto bn1   = add_bn(p, conv1, 2048);
        auto x2    = p.add_parameter("x2", xs2);
        auto w2    = p.add_parameter("w2", ws2);
        auto conv2 = p.add_instruction(migraphx::op::convolution{{0, 0}, {2, 2}, {1, 1}}, x2, w2);
        auto bn2   = add_bn(p, conv2, 2048);
        auto add   = p.add_instruction(migraphx::op::add{}, bn1, bn2);
        auto relu  = p.add_instruction(migraphx::op::relu{}, add);
        p.add_instruction(migraphx::op::pooling{"average", {1, 1}, {2, 2}, {3, 3}}, relu);
        return p;
    }
};


