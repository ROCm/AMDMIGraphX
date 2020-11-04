
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>

struct test_conv_bias_clipped_relu : verify_program<test_conv_bias_clipped_relu>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        std::vector<size_t> input_lens{4, 3, 3, 3};
        auto input =
            p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto weights =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto l0   = migraphx::literal{migraphx::shape{migraphx::shape::float_type, {4}},
                                    {2.0f, 2.0f, 2.0f, 2.0f}};
        auto bias = p.add_literal(l0);
        auto conv = p.add_instruction(migraphx::op::convolution{}, input, weights);
        auto bcast_add =
            p.add_instruction(migraphx::op::broadcast{1, conv->get_shape().lens()}, bias);
        auto bias_add = p.add_instruction(migraphx::op::add{}, conv, bcast_add);
        auto min_val  = p.add_literal(0.0f);
        auto max_val  = p.add_literal(6.0f);
        min_val       = p.add_instruction(migraphx::op::multibroadcast{input_lens}, min_val);
        max_val       = p.add_instruction(migraphx::op::multibroadcast{input_lens}, max_val);
        p.add_instruction(migraphx::op::clip{}, bias_add, min_val, max_val);
        return p;
    }
};
