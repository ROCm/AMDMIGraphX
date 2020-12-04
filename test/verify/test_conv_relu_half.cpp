
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_conv_relu_half : verify_program<test_conv_relu_half>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::half_type, {4, 3, 3, 3}});
        auto weights =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::half_type, {4, 3, 3, 3}});
        auto conv = mm->add_instruction(migraphx::make_op("convolution"), input, weights);
        mm->add_instruction(migraphx::make_op("relu"), conv);
        return p;
    }
};
