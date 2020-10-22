
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct gemm_multi_dim_2_3 : verify_program<gemm_multi_dim_2_3>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {2, 3, 2, 3}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {2, 3, 3, 2}};
        auto l1 = p.add_parameter("1", m1_shape);
        auto l2 = p.add_parameter("2", m2_shape);

        p.add_instruction(migraphx::op::dot{}, l1, l2);

        return p;
    }
};


