
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct batch_quant_dot_5 : verify_program<batch_quant_dot_5>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape m1_shape{migraphx::shape::int8_type, {3, 2, 7, 2}};
        migraphx::shape m2_shape{migraphx::shape::int8_type, {3, 2, 5, 7}};

        auto l1  = mm->add_parameter("a", m1_shape);
        auto l2  = mm->add_parameter("b", m2_shape);
        auto tl1 = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), l1);
        auto sl1 = mm->add_instruction(migraphx::make_op("add"), tl1, tl1);
        auto tl2 = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), l2);
        auto sl2 = mm->add_instruction(migraphx::make_op("add"), tl2, tl2);
        mm->add_instruction(migraphx::make_op("quant_dot", {{"alpha", 1}}), sl1, sl2);
        return p;
    }
};
