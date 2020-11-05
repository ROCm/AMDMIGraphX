
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_batchnorm_1d_per_actv : verify_program<test_batchnorm_1d_per_actv>
{
    const size_t d1       = 5;
    const size_t channels = 2;
    const size_t batches  = 3;

    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape s{migraphx::shape::float_type, {batches, channels, d1}};
        migraphx::shape vars{migraphx::shape::float_type, {channels, d1}};
        auto x        = mm->add_parameter("x", s);
        auto scale    = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 1)));
        auto bias     = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 2)));
        auto mean     = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 3)));
        auto variance = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 4)));
        mm->add_instruction(
            migraphx::op::batch_norm_inference{
                1.0e-5, 0.96f, migraphx::op::batch_norm_inference::per_activation},
            x,
            scale,
            bias,
            mean,
            variance);
        return p;
    }
};
