#include <limits>
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_isnan_broadcast : verify_program<test_isnan_broadcast>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2}});
        auto l0  = mm->add_literal(std::numeric_limits<float>::quiet_NaN());
        x        = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, l0);
        auto s0  = migraphx::shape{migraphx::shape::float_type, {3, 2}};
        x        = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", s0.lens()}}), x);
        mm->add_instruction(migraphx::make_op("isnan"), x);
        return p;
    }
};
