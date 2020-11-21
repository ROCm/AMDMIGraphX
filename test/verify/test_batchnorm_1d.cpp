
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
        auto* mm = p.get_main_module();

        migraphx::shape s{migraphx::shape::float_type, {batches, channels, size}};
        migraphx::shape vars{migraphx::shape::float_type, {channels}};
        auto x        = mm->add_parameter("x", s);
        auto scale    = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 1)));
        auto bias     = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 2)));
        auto mean     = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 3)));
        auto variance = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 4)));
        mm->add_instruction(migraphx::op::batch_norm_inference{}, x, scale, bias, mean, variance);
        return p;
    }
};
