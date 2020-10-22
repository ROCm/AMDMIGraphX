
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_conv_add : verify_program<test_conv_add>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x = p.add_parameter("x", {migraphx::shape::float_type, {1, 8, 4, 4}});
        auto w = p.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {2, 8, 3, 3}}, 1));
        auto y = p.add_parameter("y", {migraphx::shape::float_type, {1, 8, 4, 4}});
        auto v = p.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {2, 8, 3, 3}}, 2));
        auto conv1 = p.add_instruction(migraphx::op::convolution{}, x, w);
        auto conv2 = p.add_instruction(migraphx::op::convolution{}, y, v);
        auto sum   = p.add_instruction(migraphx::op::add{}, conv1, conv2);
        p.add_instruction(migraphx::op::exp{}, sum);
        return p;
    }
};


