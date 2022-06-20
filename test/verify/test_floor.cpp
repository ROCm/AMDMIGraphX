
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_floor : verify_program<test_floor>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 6}};
        auto param = mm->add_parameter("x", s);
        mm->add_instruction(migraphx::make_op("floor"), param);
        return p;
    };
};
