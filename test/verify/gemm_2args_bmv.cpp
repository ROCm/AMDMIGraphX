
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct gemm_2args_bmv : verify_program<gemm_2args_bmv>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape m1_shape{migraphx::shape::float_type, {2, 3, 3, 5}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {5}};
        auto l1   = mm->add_parameter("1", m1_shape);
        auto l2   = mm->add_parameter("2", m2_shape);
        auto ul2  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), l2);
        auto bul2 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 5, 1}}}), ul2);

        mm->add_instruction(migraphx::make_op("dot"), l1, bul2);

        return p;
    }
};
