
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct gemm_2args_mm_1 : verify_program<gemm_2args_mm_1>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {2, 2, 3}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {1, 3, 4}};
        auto l1  = p.add_parameter("1", m1_shape);
        auto l2  = p.add_parameter("2", m2_shape);
        auto bl2 = p.add_instruction(migraphx::op::multibroadcast{{2, 3, 4}}, l2);

        p.add_instruction(migraphx::op::dot{}, l1, bl2);

        return p;
    }
};


