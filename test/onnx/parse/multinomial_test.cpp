
#include <onnx_test.hpp>


TEST_CASE(multinomial_test)
{
    migraphx::program p;
    auto* mm           = p.get_main_module();
    size_t sample_size = 13;
    size_t batch_size  = 3;
    size_t categories  = 10;
    float seed         = 0;

    auto input = mm->add_parameter(
        "input", migraphx::shape{migraphx::shape::float_type, {batch_size, categories}});
    auto maxes    = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), input);
    auto mb_maxes = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {batch_size, 10}}}), maxes);
    auto cdf = mm->add_instruction(migraphx::make_op("sub"), input, mb_maxes);
    cdf      = mm->add_instruction(migraphx::make_op("exp"), cdf);
    cdf      = mm->add_instruction(
        migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", false}}), cdf);

    migraphx::shape s{migraphx::shape::float_type, {1}};
    std::vector<float> seed_data = {seed};
    auto seed_input              = mm->add_literal(migraphx::literal(s, seed_data));
    auto rand_dummy              = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {batch_size, sample_size}},
                          std::vector<float>(batch_size * sample_size)});

    auto randoms = mm->add_instruction(migraphx::make_op("random_uniform"), seed_input, rand_dummy);
    mm->add_instruction(migraphx::make_op("multinomial"), cdf, randoms);
    auto prog = optimize_onnx("multinomial_test.onnx");

    EXPECT(p == prog);
}


