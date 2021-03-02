
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_not : verify_program<test_not>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::bool_type, {4}};
        auto x = mm->add_parameter("x", s);
        mm->add_instruction(migraphx::make_op("not"), x);
        return p;
    }
};
