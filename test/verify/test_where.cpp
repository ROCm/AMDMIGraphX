
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_where : verify_program<test_where>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape sb{migraphx::shape::bool_type, {1, 3, 4, 5}};
        migraphx::shape sx{migraphx::shape::float_type, {1, 3, 4, 5}};
        auto b = mm->add_parameter("b", sb);
        auto x = mm->add_parameter("x", sx);
        auto y = mm->add_parameter("y", sx);
        auto r = mm->add_instruction(migraphx::make_op("where"), b, x, y);
        mm->add_return({r});
        return p;
    };
};
