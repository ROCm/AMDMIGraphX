
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct quant_conv_valid_mode : verify_program<quant_conv_valid_mode>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape a_shape{migraphx::shape::int8_type, {2, 3, 4, 4}};
        auto pa = p.add_parameter("a", a_shape);
        migraphx::shape c_shape{migraphx::shape::int8_type, {2, 3, 3, 3}};
        auto pc = p.add_parameter("c", c_shape);
        p.add_instruction(
            migraphx::op::quant_convolution{{{0, 0}}, {{1, 1}}, {{1, 1}}, migraphx::op::valid},
            pa,
            pc);
        return p;
    }
};
