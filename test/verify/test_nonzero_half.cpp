
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_nonzero_half : verify_program<test_nonzero_half>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::half_type, {3, 4, 3, 5}};
        auto x = mm->add_parameter("data", s);
        auto r = mm->add_instruction(migraphx::make_op("nonzero"), x);
        mm->add_return({r});

        return p;
    }
};
