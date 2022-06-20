
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_deconv_1d : verify_program<test_deconv_1d>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 3}});
        auto weights =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {1, 1, 3}});
        mm->add_instruction(
            migraphx::make_op("deconvolution",
                              {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
            input,
            weights);
        return p;
    }
};
