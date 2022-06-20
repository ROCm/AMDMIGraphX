
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/apply_alpha_beta.hpp>
struct gemm_add : verify_program<gemm_add>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape m1_shape{migraphx::shape::float_type, {1, 2, 3}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {1, 3, 4}};
        migraphx::shape m3_shape{migraphx::shape::float_type, {1, 2, 4}};
        auto l1 = mm->add_parameter("1", m1_shape);
        auto l2 = mm->add_parameter("2", m2_shape);
        auto l3 = mm->add_parameter("3", m3_shape);

        auto dot = mm->add_instruction(migraphx::make_op("dot"), l1, l2);
        mm->add_instruction(migraphx::make_op("add"), dot, l3);
        return p;
    }
};
