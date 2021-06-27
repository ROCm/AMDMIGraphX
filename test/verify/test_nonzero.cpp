
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_nonzero : verify_program<test_nonzero>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};
        auto x = mm->add_parameter("data", s);
        auto r = mm->add_instruction(migraphx::make_op("nonzero"), x);
        mm->add_return({r});

        return p;
    }
};
