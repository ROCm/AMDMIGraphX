
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_prelu_brcst : verify_program<test_prelu_brcst>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {6}};
        auto x   = mm->add_parameter("x", s);
        auto slp = mm->add_parameter("slp", s);
        auto r   = mm->add_instruction(migraphx::make_op("prelu"), x, slp);
        mm->add_return({r});

        return p;
    }
};
