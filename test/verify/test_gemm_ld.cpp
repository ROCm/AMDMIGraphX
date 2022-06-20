
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_gemm_ld //: verify_program<test_gemm_ld>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto a =
            mm->add_parameter("a", migraphx::shape{migraphx::shape::float_type, {4, 5}, {10, 1}});
        auto b =
            mm->add_parameter("b", migraphx::shape{migraphx::shape::float_type, {5, 3}, {20, 1}});
        mm->add_instruction(migraphx::make_op("dot"), a, b);
        return p;
    }
};
