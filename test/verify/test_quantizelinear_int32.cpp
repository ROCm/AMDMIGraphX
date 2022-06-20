
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_quantizelinear_int32 : verify_program<test_quantizelinear_int32>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape sx{migraphx::shape::int32_type, {2, 2, 2}};
        migraphx::shape ss{migraphx::shape::float_type, {2, 2, 2}};
        migraphx::shape sz{migraphx::shape::int8_type, {2, 2, 2}};
        auto input1 = mm->add_parameter("x", sx);
        auto input2 = mm->add_parameter("y_scale", ss);
        auto input3 = mm->add_parameter("y_zero_point", sz);
        auto r = mm->add_instruction(migraphx::make_op("quantizelinear"), input1, input2, input3);
        mm->add_return({r});
        return p;
    };
};
