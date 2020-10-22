
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_deconv_3d : verify_program<test_deconv_3d>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto input =
            p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 3, 3, 3}});
        auto weights =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {1, 1, 3, 3, 3}});
        p.add_instruction(
            migraphx::op::deconvolution{{0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, input, weights);
        return p;
    }
};


