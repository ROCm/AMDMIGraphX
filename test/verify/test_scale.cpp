
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_scale : verify_program<test_scale>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x = mm->add_parameter("x", s);
        auto y = mm->add_parameter("y", migraphx::shape::float_type);
        auto scale =
            mm->add_instruction(migraphx::make_op("scalar", {{"scalar_bcst_dims", s.lens()}}), y);
        mm->add_instruction(migraphx::make_op("mul"), x, scale);
        return p;
    }
};
