
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

// math op on half-precision float with odd size tensor can't fit half2 packing
struct test_sqrt_half1 : verify_program<test_sqrt_half1>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::half_type, {5}};
        auto param     = mm->add_parameter("x", s);
        auto param_abs = mm->add_instruction(migraphx::make_op("abs"), param);
        mm->add_instruction(migraphx::make_op("sqrt"), param_abs);
        return p;
    }
};
