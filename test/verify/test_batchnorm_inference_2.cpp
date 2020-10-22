
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_batchnorm_inference_2 : verify_program<test_batchnorm_inference_2>
{
    const size_t width    = 14;
    const size_t height   = 14;
    const size_t channels = 256;
    const size_t batches  = 1;

    migraphx::program create_program() const
    {
        migraphx::program p;

        migraphx::shape s{migraphx::shape::float_type, {batches, channels, height, width}};
        migraphx::shape vars{migraphx::shape::float_type, {channels}};
        auto x        = p.add_parameter("x", s);
        auto scale    = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 1)));
        auto bias     = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 2)));
        auto mean     = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 3)));
        auto variance = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 4)));
        p.add_instruction(migraphx::op::batch_norm_inference{}, x, scale, bias, mean, variance);
        return p;
    }
};


