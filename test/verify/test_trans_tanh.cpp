
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_trans_tanh : verify_program<test_trans_tanh>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto tx =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), x);
        auto tanhx = mm->add_instruction(migraphx::make_op("tanh"), tx);
        auto r     = mm->add_instruction(migraphx::make_op("add"), tanhx, tanhx);
        mm->add_instruction(migraphx::make_op("contiguous"), r);

        return p;
    }
};
