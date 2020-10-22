
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_relu_lrn : verify_program<test_relu_lrn>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 5, 2, 2}});
        auto y = p.add_instruction(migraphx::op::relu{}, x);
        p.add_instruction(migraphx::op::lrn{0.0001, 0.75, 1.0, 5}, y);
        return p;
    }
};
