
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_add_relu : verify_program<test_add_relu>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto y = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto add = mm->add_instruction(migraphx::make_op("add"), x, y);
        mm->add_instruction(migraphx::make_op("relu"), add);
        return p;
    }
};
