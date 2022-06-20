
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_sub2 : verify_program<test_sub2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        migraphx::shape b{migraphx::shape::float_type, {3}};
        auto x  = mm->add_parameter("x", s);
        auto y  = mm->add_parameter("y", s);
        auto z  = mm->add_parameter("z", b);
        auto zb = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", s.lens()}}), z);
        auto diff = mm->add_instruction(migraphx::make_op("sub"), x, y);
        mm->add_instruction(migraphx::make_op("sub"), diff, zb);
        return p;
    }
};
