
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_concat_pooling : verify_program<test_concat_pooling>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto input =
            p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 256, 8, 8}});
        auto transpose = p.add_instruction(migraphx::op::transpose{{0, 2, 3, 1}}, input);
        auto concat    = p.add_instruction(migraphx::op::concat{3}, transpose);
        auto concat_t  = p.add_instruction(migraphx::op::transpose{{0, 3, 1, 2}}, concat);

        auto pooling =
            p.add_instruction(migraphx::op::pooling{"average", {0, 0}, {1, 1}, {8, 8}}, concat_t);
        p.add_instruction(migraphx::op::relu{}, pooling);
        return p;
    }
};
