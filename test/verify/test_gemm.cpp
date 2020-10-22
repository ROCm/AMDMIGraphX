
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_gemm : verify_program<test_gemm>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto a = p.add_parameter("a", migraphx::shape{migraphx::shape::float_type, {4, 5}});
        auto b = p.add_parameter("b", migraphx::shape{migraphx::shape::float_type, {5, 3}});
        p.add_instruction(migraphx::op::dot{}, a, b);
        return p;
    }
};


