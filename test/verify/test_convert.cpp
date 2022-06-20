
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/serialize.hpp>

#include <migraphx/make_op.hpp>

struct test_convert : verify_program<test_convert>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::int8_type, {8, 24}};
        migraphx::shape sb{migraphx::shape::int8_type, {24, 6}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);
        auto ia = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            pa);
        auto ib = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            pb);
        mm->add_instruction(migraphx::make_op("dot"), ia, ib);

        return p;
    };
};
