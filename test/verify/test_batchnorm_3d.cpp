
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

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

        migraphx::shape s{migraphx::shape::float_type, {batches, channels, d1, d2, d3}};
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


