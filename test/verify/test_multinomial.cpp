
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_multinomial : verify_program<test_multinomial>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm           = p.get_main_module();
        size_t sample_size = 10;
        float seed = 0.0f;
        std::mt19937 gen(seed);
        std::uniform_real_distribution<> dis(0.0, 1.0);
        std::vector<float> rand_samples(sample_size);
        std::transform(rand_samples.begin(), rand_samples.end(), rand_samples.begin(), [&](auto) {
            return dis(gen);
        });
        migraphx::shape rs{migraphx::shape::float_type, {1, sample_size}};
        auto rs_lit = mm->add_literal(migraphx::literal{rs, rand_samples});

        migraphx::shape s{migraphx::shape::float_type, {1, 5}};
        auto input = mm->add_parameter("input", s);

        auto maxes = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), input);
        auto mb_maxes =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 5}}}), maxes);
        auto cdf = mm->add_instruction(migraphx::make_op("sub"), input, mb_maxes);
        cdf      = mm->add_instruction(migraphx::make_op("exp"), cdf);
        cdf      = mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", false}}), cdf);

        mm->add_instruction(migraphx::make_op("multinomial"), cdf, rs_lit);
        return p;
    }
};
