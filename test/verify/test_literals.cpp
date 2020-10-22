
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_literals : verify_program<test_literals>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto input = p.add_literal(
            generate_literal(migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}}));
        auto weights = p.add_literal(
            generate_literal(migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}}));
        auto conv = p.add_instruction(migraphx::op::convolution{}, input, weights);
        p.add_instruction(migraphx::op::relu{}, conv);
        return p;
    }
};


