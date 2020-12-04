
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_batchnorm_3d : verify_program<test_batchnorm_3d>
{
    const size_t d1       = 2;
    const size_t d2       = 2;
    const size_t d3       = 2;
    const size_t channels = 2;
    const size_t batches  = 2;

    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape s{migraphx::shape::float_type, {batches, channels, d1, d2, d3}};
        migraphx::shape vars{migraphx::shape::float_type, {channels}};
        auto x        = mm->add_parameter("x", s);
        auto scale    = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 1)));
        auto bias     = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 2)));
        auto mean     = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 3)));
        auto variance = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 4)));
        mm->add_instruction(
            migraphx::make_op("batch_norm_inference"), x, scale, bias, mean, variance);
        return p;
    }
};
