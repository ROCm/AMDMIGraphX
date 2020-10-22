
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct gemm_2args_vbm : verify_program<gemm_2args_vbm>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {5}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {2, 2, 5, 4}};
        auto l1   = p.add_parameter("1", m1_shape);
        auto ul1  = p.add_instruction(migraphx::op::unsqueeze{{0}}, l1);
        auto bul1 = p.add_instruction(migraphx::op::multibroadcast{{2, 2, 1, 5}}, ul1);

        auto l2 = p.add_parameter("2", m2_shape);

        auto res = p.add_instruction(migraphx::op::dot{}, bul1, l2);
        p.add_instruction(migraphx::op::squeeze{{2}}, res);

        return p;
    }
};


