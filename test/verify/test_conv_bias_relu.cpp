
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/instruction.hpp>

struct test_conv_bias_relu : verify_program<test_conv_bias_relu>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<float> bias_vals(64, 2.0f);
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 64, 56, 56}});
        auto weights =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {64, 64, 1, 1}});
        auto l0 = migraphx::literal{migraphx::shape{migraphx::shape::float_type, {64}}, bias_vals};
        auto bias      = mm->add_literal(l0);
        auto conv      = mm->add_instruction(migraphx::make_op("convolution"), input, weights);
        auto bcast_add = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv->get_shape().lens()}}),
            bias);
        auto bias_add = mm->add_instruction(migraphx::make_op("add"), conv, bcast_add);

        mm->add_instruction(migraphx::make_op("relu"), bias_add);
        return p;
    }
};
