
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/op/quant_convolution.hpp>

struct quant_conv_int8x4_default : verify_program<quant_conv_int8x4_default>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape a_shape{migraphx::shape::int8_type, {16, 16, 4, 4}};
        auto pa = mm->add_parameter("a", a_shape);
        migraphx::shape c_shape{migraphx::shape::int8_type, {16, 16, 3, 3}};
        auto pc = mm->add_parameter("c", c_shape);
        mm->add_instruction(
            migraphx::op::quant_convolution{{{0, 0}}, {{1, 1}}, {{1, 1}}, migraphx::op::same},
            pa,
            pc);
        return p;
    }
};
