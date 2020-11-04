
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_convert : verify_program<test_convert>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::float_type, {8, 24}};
        migraphx::shape sb{migraphx::shape::float_type, {24, 6}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);
        auto ia = p.add_instruction(migraphx::op::convert{migraphx::shape::int8_type}, pa);
        auto ib = p.add_instruction(migraphx::op::convert{migraphx::shape::int8_type}, pb);
        p.add_instruction(migraphx::op::quant_dot{}, ia, ib);

        return p;
    };
};
