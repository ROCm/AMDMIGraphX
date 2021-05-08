
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct batch_quant_dot_3 : verify_program<batch_quant_dot_3>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape m1_shape{migraphx::shape::int8_type, {3, 2, 2, 6}};
        migraphx::shape m2_shape{migraphx::shape::int8_type, {3, 2, 6, 7}};

        auto l1 = mm->add_parameter("a", m1_shape);
        auto l2 = mm->add_parameter("b", m2_shape);
        mm->add_instruction(migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 3}}), l1, l2);
        return p;
    }
};
