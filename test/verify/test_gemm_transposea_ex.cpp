
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_gemm_transposea_ex : verify_program<test_gemm_transposea_ex>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto a  = p.add_parameter("a", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 4}});
        auto b  = p.add_parameter("b", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 3}});
        auto at = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, a);
        p.add_instruction(migraphx::op::dot{}, at, b);
        return p;
    }
};


