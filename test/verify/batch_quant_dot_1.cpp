
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct batch_quant_dot_1 : verify_program<batch_quant_dot_1>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::int8_type, {3, 2, 8, 2}};
        migraphx::shape m2_shape{migraphx::shape::int8_type, {3, 2, 7, 8}};
        migraphx::shape m3_shape{migraphx::shape::int32_type, {3, 2, 2, 7}};

        auto l1  = p.add_parameter("a", m1_shape);
        auto tl1 = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, l1);
        auto l2  = p.add_parameter("b", m2_shape);
        auto tl2 = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, l2);
        auto l3  = p.add_parameter("c", m3_shape);
        p.add_instruction(migraphx::op::quant_dot{3, 2}, tl1, tl2, l3);
        return p;
    }
};
