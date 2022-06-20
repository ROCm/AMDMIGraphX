
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_where2 : verify_program<test_where2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape sb{migraphx::shape::bool_type, {1, 3, 4, 5}};
        migraphx::shape sx{migraphx::shape::float_type, {1}};
        auto b   = mm->add_parameter("b", sb);
        auto x   = mm->add_parameter("x", sx);
        auto y   = mm->add_parameter("y", sx);
        auto mbx = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 3, 4, 5}}}), x);
        auto mby = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 3, 4, 5}}}), y);
        auto r = mm->add_instruction(migraphx::make_op("where"), b, mbx, mby);
        mm->add_return({r});
        return p;
    };
};
