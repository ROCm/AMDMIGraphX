
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/instruction.hpp>

struct test_conv_bias_clipped_relu : verify_program<test_conv_bias_clipped_relu>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<size_t> input_lens{4, 3, 3, 3};
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto weights =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto l0        = migraphx::literal{migraphx::shape{migraphx::shape::float_type, {4}},
                                    {2.0f, 2.0f, 2.0f, 2.0f}};
        auto bias      = mm->add_literal(l0);
        auto conv      = mm->add_instruction(migraphx::make_op("convolution"), input, weights);
        auto bcast_add = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv->get_shape().lens()}}),
            bias);
        auto bias_add = mm->add_instruction(migraphx::make_op("add"), conv, bcast_add);
        auto min_val  = mm->add_literal(0.0f);
        auto max_val  = mm->add_literal(6.0f);
        min_val       = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}), min_val);
        max_val = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}), max_val);
        mm->add_instruction(migraphx::make_op("clip"), bias_add, min_val, max_val);
        return p;
    }
};
