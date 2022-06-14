
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/apply_alpha_beta.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct quant_dot_3args_5 : verify_program<quant_dot_3args_5>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape m1_shape{migraphx::shape::int8_type, {6, 2}};
        migraphx::shape m2_shape{migraphx::shape::int8_type, {7, 6}};

        auto l1 = mm->add_parameter("a", m1_shape);
        auto tl1 =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l1);
        auto l2 = mm->add_parameter("b", m2_shape);
        auto tl2 =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l2);
        migraphx::add_apply_alpha_beta(*mm, {tl1, tl2}, migraphx::make_op("quant_dot"), 3);
        return p;
    }
};
