
#include <migraphx/apply_alpha_beta.hpp>
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct gemm_multi_3args : verify_program<gemm_multi_3args>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape m1_shape{migraphx::shape::float_type, {2, 3, 2, 3}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {2, 3, 3, 2}};
        migraphx::shape m3_shape{migraphx::shape::float_type, {2, 3, 2, 2}};

        auto l1     = mm->add_parameter("1", m1_shape);
        auto l2     = mm->add_parameter("2", m2_shape);
        auto l3     = mm->add_parameter("3", m3_shape);
        float alpha = 0.35;
        float beta  = 0.41;
        migraphx::add_apply_alpha_beta(*mm, {l1, l2, l3}, migraphx::make_op("dot"), alpha, beta);
        return p;
    }
};
