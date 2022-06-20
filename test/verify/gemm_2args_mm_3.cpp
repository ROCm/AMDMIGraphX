
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct gemm_2args_mm_3 : verify_program<gemm_2args_mm_3>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape m1_shape{migraphx::shape::float_type, {1, 2, 3}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {3, 3, 4}};
        auto l1 = mm->add_parameter("1", m1_shape);
        auto bl1 =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 2, 3}}}), l1);
        auto l2 = mm->add_parameter("2", m2_shape);

        mm->add_instruction(migraphx::make_op("dot"), bl1, l2);

        return p;
    }
};
