
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_conv_pooling : verify_program<test_conv_pooling>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 32, 32}});
        auto weights =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto conv    = mm->add_instruction(migraphx::op::convolution{}, input, weights);
        auto pooling = mm->add_instruction(migraphx::op::pooling{"max"}, conv);
        mm->add_instruction(migraphx::op::relu{}, pooling);
        return p;
    }
};
