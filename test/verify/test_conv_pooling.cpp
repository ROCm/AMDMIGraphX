
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_conv_pooling : verify_program<test_conv_pooling>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto input =
            p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 32, 32}});
        auto weights =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto conv    = p.add_instruction(migraphx::op::convolution{}, input, weights);
        auto pooling = p.add_instruction(migraphx::op::pooling{"max"}, conv);
        p.add_instruction(migraphx::op::relu{}, pooling);
        return p;
    }
};


