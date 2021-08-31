
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct batch_quant_dot_4 : verify_program<batch_quant_dot_4>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape m1_shape{migraphx::shape::int8_type, {2, 4, 6, 3}};
        migraphx::shape m2_shape{migraphx::shape::int8_type, {7, 2, 6, 3}};

        auto l1  = mm->add_parameter("a", m1_shape);
        auto l2  = mm->add_parameter("b", m2_shape);
        auto tl1 = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {3, 0, 1, 2}}}), l1);
        auto tl2 = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {3, 1, 2, 0}}}), l2);
        mm->add_instruction(migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 3}}), tl1, tl2);
        return p;
    }
};
