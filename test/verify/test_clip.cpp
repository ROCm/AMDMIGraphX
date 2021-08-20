
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_clip : verify_program<test_clip>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto x       = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3}});
        auto min_val = mm->add_literal(0.0f);
        auto max_val = mm->add_literal(6.0f);
        min_val =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3}}}), min_val);
        max_val =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3}}}), max_val);
        mm->add_instruction(migraphx::make_op("clip"), x, min_val, max_val);
        return p;
    }
};
