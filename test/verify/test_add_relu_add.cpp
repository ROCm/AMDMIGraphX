
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_add_relu_add : verify_program<test_add_relu_add>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 5, 2, 2}});
        auto y = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, {1, 5, 2, 2}});
        auto z = mm->add_parameter("z", migraphx::shape{migraphx::shape::float_type, {1, 5, 2, 2}});
        auto a = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto b = mm->add_instruction(migraphx::make_op("relu"), a);
        auto c = mm->add_instruction(migraphx::make_op("add"), b, z);
        mm->add_instruction(migraphx::make_op("relu"), c);
        return p;
    }
};
