
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct gemm_multi_transpose : verify_program<gemm_multi_transpose>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {2, 2, 3}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {3, 2, 4}};
        auto l1  = p.add_parameter("1", m1_shape);
        auto l2  = p.add_parameter("2", m2_shape);
        auto tl2 = p.add_instruction(migraphx::op::transpose{{1, 0, 2}}, l2);

        float alpha = 1.0f;
        float beta  = 1.0f;
        p.add_instruction(migraphx::op::dot{alpha, beta}, l1, tl2);

        return p;
    }
};
