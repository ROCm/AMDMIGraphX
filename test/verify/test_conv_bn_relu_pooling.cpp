
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_conv_bn_relu_pooling : verify_program<test_conv_bn_relu_pooling>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape xs{migraphx::shape::float_type, {1, 3, 224, 224}};
        migraphx::shape ws{migraphx::shape::float_type, {64, 3, 7, 7}};
        migraphx::shape vars{migraphx::shape::float_type, {64}};
        auto x    = mm->add_parameter("x", xs);
        auto w    = mm->add_parameter("w", ws);
        auto conv = mm->add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {3, 3}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        auto scale    = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 1)));
        auto bias     = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 2)));
        auto mean     = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 3)));
        auto variance = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 4)));
        auto bn       = mm->add_instruction(
            migraphx::make_op("batch_norm_inference"), conv, scale, bias, mean, variance);
        auto relu = mm->add_instruction(migraphx::make_op("relu"), bn);
        mm->add_instruction(migraphx::make_op("pooling",
                                              {{"mode", "average"},
                                               {"padding", {1, 1}},
                                               {"stride", {2, 2}},
                                               {"lengths", {3, 3}}}),
                            relu);
        return p;
    }
};
