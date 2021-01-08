
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_equal : verify_program<test_equal>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape s{migraphx::shape::double_type, {2, 3, 4, 6}};
        auto input1 = mm->add_parameter("x", s);
        auto input2 = mm->add_parameter("y", s);
        auto r      = mm->add_instruction(migraphx::make_op("equal"), input1, input2);
        mm->add_return({r});
        return p;
    };
};
