
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_conv2 : verify_program<test_conv2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 512, 28, 28}});
        auto weights =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {256, 512, 1, 1}});
        mm->add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
            input,
            weights);
        return p;
    }
};
