#include <migraphx/make_op.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/gpu/target.hpp>
#include <test.hpp>

TEST_CASE(conv_bias_relu)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto lit = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}));
    auto weights = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 64, 1, 1}}));
    auto input = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1, 64, 56, 56}}));

    auto conv = mm->add_instruction(migraphx::make_op("convolution"), input, weights);
    auto bnorm = mm->add_instruction(migraphx::make_op("batch_norm_inference"), conv, lit, lit, lit ,lit);
    mm->add_instruction(migraphx::make_op("relu"), bnorm);

    EXPECT(not test::throws([&] { p.compile(migraphx::gpu::target{}); }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
