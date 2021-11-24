#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_nonstd_nonzero : verify_program<test_nonstd_nonzero>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 3}};
        auto x = mm->add_parameter("data", s);
        auto xt =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {3, 1, 0, 2}}}), x);
        auto r = mm->add_instruction(migraphx::make_op("nonzero"), xt);
        mm->add_return({r});

        return p;
    }
};
