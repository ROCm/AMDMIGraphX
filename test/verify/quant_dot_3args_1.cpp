
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct quant_dot_3args_1 : verify_program<quant_dot_3args_1>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::int8_type, {2, 8}};
        migraphx::shape m2_shape{migraphx::shape::int8_type, {8, 7}};
        migraphx::shape m3_shape{migraphx::shape::int32_type, {2, 7}};

        auto l1 = p.add_parameter("a", m1_shape);
        auto l2 = p.add_parameter("b", m2_shape);
        auto l3 = p.add_parameter("c", m3_shape);
        p.add_instruction(migraphx::op::quant_dot{}, l1, l2, l3);
        return p;
    }
};


