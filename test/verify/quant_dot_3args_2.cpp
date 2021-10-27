
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/apply_alpha_beta.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct quant_dot_3args_2 : verify_program<quant_dot_3args_2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape m1_shape{migraphx::shape::int8_type, {8, 2}};
        migraphx::shape m2_shape{migraphx::shape::int8_type, {8, 7}};
        migraphx::shape m3_shape{migraphx::shape::int32_type, {2, 7}};

        auto l1 = mm->add_parameter("a", m1_shape);
        auto tl1 =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l1);
        auto l2 = mm->add_parameter("b", m2_shape);
        auto l3 = mm->add_parameter("c", m3_shape);
        migraphx::add_apply_alpha_beta(*mm, {tl1, l2, l3}, migraphx::make_op("quant_dot"), 1, 3);
        return p;
    }
};
