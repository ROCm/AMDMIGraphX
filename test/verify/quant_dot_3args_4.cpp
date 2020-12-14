
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct quant_dot_3args_4 : verify_program<quant_dot_3args_4>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape m1_shape{migraphx::shape::int8_type, {8, 2}};
        migraphx::shape m2_shape{migraphx::shape::int8_type, {7, 8}};
        migraphx::shape m3_shape{migraphx::shape::int32_type, {2, 7}};

        auto l1  = mm->add_parameter("a", m1_shape);
        auto tl1 = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l1);
        auto l2  = mm->add_parameter("b", m2_shape);
        auto tl2 = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l2);
        auto l3  = mm->add_parameter("c", m3_shape);
        mm->add_instruction(
            migraphx::make_op("quant_dot", {{"alpha", 3}, {"beta", 2}}), tl1, tl2, l3);
        return p;
    }
};
