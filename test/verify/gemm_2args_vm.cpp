
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct gemm_2args_vm : verify_program<gemm_2args_vm>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {5}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {5, 4}};
        auto l1  = p.add_parameter("1", m1_shape);
        auto ul1 = p.add_instruction(migraphx::op::unsqueeze{{0}}, l1);
        auto l2  = p.add_parameter("2", m2_shape);

        auto res = p.add_instruction(migraphx::op::dot{}, ul1, l2);
        p.add_instruction(migraphx::op::squeeze{{0}}, res);

        return p;
    }
};


