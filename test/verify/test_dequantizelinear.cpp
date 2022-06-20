
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_dequantizelinear : verify_program<test_dequantizelinear>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape sx{migraphx::shape::int8_type, {2, 2, 2}};
        migraphx::shape ss{migraphx::shape::float_type, {2, 2, 2}};
        migraphx::shape sz{migraphx::shape::int8_type, {2, 2, 2}};
        auto input1 = mm->add_parameter("x", sx);
        auto input2 = mm->add_parameter("x_scale", ss);
        auto input3 = mm->add_parameter("x_zero_point", sz);
        auto r = mm->add_instruction(migraphx::make_op("dequantizelinear"), input1, input2, input3);
        mm->add_return({r});
        return p;
    };
};
