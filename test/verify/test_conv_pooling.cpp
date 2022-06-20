
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/common.hpp>

struct test_conv_pooling : verify_program<test_conv_pooling>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 32, 32}});
        auto weights =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto conv    = mm->add_instruction(migraphx::make_op("convolution"), input, weights);
        auto pooling = mm->add_instruction(
            migraphx::make_op("pooling", {{"mode", migraphx::op::pooling_mode::max}}), conv);
        mm->add_instruction(migraphx::make_op("relu"), pooling);
        return p;
    }
};
