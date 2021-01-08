
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct gemm_literal : verify_program<gemm_literal>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape a_shape{migraphx::shape::float_type, {2, 4}};
        migraphx::shape b_shape{migraphx::shape::float_type, {4, 4}};

        auto a = mm->add_literal(migraphx::generate_literal(a_shape));
        auto b = mm->add_parameter("b", b_shape);
        mm->add_instruction(migraphx::op::dot{}, a, b);

        return p;
    }
};
