
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/common.hpp>

struct test_conv_bn_relu_pooling2 : verify_program<test_conv_bn_relu_pooling2>
{
    static migraphx::instruction_ref
    add_bn(migraphx::program& p, migraphx::instruction_ref x, std::size_t channels)
    {
        auto* mm = p.get_main_module();
        migraphx::shape vars{migraphx::shape::float_type, {channels}};
        auto scale = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 1 + channels)));
        auto bias  = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 2 + channels)));
        auto mean  = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 3 + channels)));
        auto variance =
            mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 4 + channels)));
        return mm->add_instruction(
            migraphx::make_op("batch_norm_inference"), x, scale, bias, mean, variance);
    }
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape xs1{migraphx::shape::float_type, {1, 512, 7, 7}};
        migraphx::shape xs2{migraphx::shape::float_type, {1, 1024, 14, 14}};
        migraphx::shape ws1{migraphx::shape::float_type, {2048, 512, 1, 1}};
        migraphx::shape ws2{migraphx::shape::float_type, {2048, 1024, 1, 1}};
        auto x1    = mm->add_parameter("x1", xs1);
        auto w1    = mm->add_parameter("w1", ws1);
        auto conv1 = mm->add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
            x1,
            w1);
        auto bn1   = add_bn(p, conv1, 2048);
        auto x2    = mm->add_parameter("x2", xs2);
        auto w2    = mm->add_parameter("w2", ws2);
        auto conv2 = mm->add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {0, 0}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x2,
            w2);
        auto bn2  = add_bn(p, conv2, 2048);
        auto add  = mm->add_instruction(migraphx::make_op("add"), bn1, bn2);
        auto relu = mm->add_instruction(migraphx::make_op("relu"), add);
        mm->add_instruction(migraphx::make_op("pooling",
                                              {{"mode", migraphx::op::pooling_mode::average},
                                               {"padding", {1, 1}},
                                               {"stride", {2, 2}},
                                               {"lengths", {3, 3}}}),
                            relu);
        return p;
    }
};
