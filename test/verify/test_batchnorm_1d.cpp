
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_batchnorm_1d : verify_program<test_batchnorm_1d>
{
    const size_t size     = 3;
    const size_t channels = 3;
    const size_t batches  = 4;

    migraphx::program create_program() const
    {
        migraphx::program p;

        migraphx::shape s{migraphx::shape::float_type, {batches, channels, size}};
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


