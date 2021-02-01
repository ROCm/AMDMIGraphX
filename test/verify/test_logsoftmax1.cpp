
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_logsoftmax1 : verify_program<test_logsoftmax1>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {5, 3, 3, 4}});
        auto tx = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {2, 3, 0, 1}}}), x);
        auto r  = mm->add_instruction(migraphx::make_op("logsoftmax", {{"axis", 0}}), tx);
        mm->add_return({r});
        return p;
    }
};
