
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_conv_add : verify_program<test_conv_add>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {1, 8, 4, 4}});
        auto w   = mm->add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {2, 8, 3, 3}}, 1));
        auto y = mm->add_parameter("y", {migraphx::shape::float_type, {1, 8, 4, 4}});
        auto v = mm->add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {2, 8, 3, 3}}, 2));
        auto conv1 = mm->add_instruction(migraphx::make_op("convolution"), x, w);
        auto conv2 = mm->add_instruction(migraphx::make_op("convolution"), y, v);
        auto sum   = mm->add_instruction(migraphx::make_op("add"), conv1, conv2);
        mm->add_instruction(migraphx::make_op("exp"), sum);
        return p;
    }
};
