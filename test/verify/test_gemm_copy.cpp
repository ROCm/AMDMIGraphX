
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_gemm_copy : verify_program<test_gemm_copy>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);
        auto pc = mm->add_parameter("c", sc);
        auto dr = mm->add_instruction(migraphx::op::dot{}, pa, pb, pc);
        mm->add_instruction(migraphx::op::add{}, dr, dr);

        return p;
    }
};
