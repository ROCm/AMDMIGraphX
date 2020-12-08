
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_relu_lrn : verify_program<test_relu_lrn>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 5, 2, 2}});
        auto y = mm->add_instruction(migraphx::make_op("relu"), x);
        mm->add_instruction(
            migraphx::make_op("lrn",
                              {{"alpha", 0.0001}, {"beta", 0.75}, {"bias", 1.0}, {"size", 5}}),
            y);
        return p;
    }
};
