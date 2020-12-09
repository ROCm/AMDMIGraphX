
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct quant_conv_padding : verify_program<quant_conv_padding>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape a_shape{migraphx::shape::int8_type, {2, 3, 4, 4}};
        auto pa = mm->add_parameter("a", a_shape);
        migraphx::shape c_shape{migraphx::shape::int8_type, {2, 3, 3, 3}};
        auto pc = mm->add_parameter("c", c_shape);
        mm->add_instruction(
            migraphx::make_op("quant_convolution", {{"padding", {1, 1}}, {"stride", {1, 1}}}),
            pa,
            pc);
        return p;
    }
};
