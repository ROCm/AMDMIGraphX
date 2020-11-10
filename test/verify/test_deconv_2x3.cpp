
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_deconv_2x3 : verify_program<test_deconv_2x3>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 3, 6, 7}});
        auto weights =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {3, 4, 3, 3}});
        mm->add_instruction(migraphx::op::deconvolution{{1, 1}, {2, 3}, {1, 1}}, input, weights);
        return p;
    }
};
