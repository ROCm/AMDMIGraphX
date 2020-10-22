
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_gemm_transposea : verify_program<test_gemm_transposea>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto a  = p.add_parameter("a", migraphx::shape{migraphx::shape::float_type, {5, 4}});
        auto b  = p.add_parameter("b", migraphx::shape{migraphx::shape::float_type, {5, 3}});
        auto at = p.add_instruction(migraphx::op::transpose{{1, 0}}, a);
        p.add_instruction(migraphx::op::dot{}, at, b);
        return p;
    }
};


