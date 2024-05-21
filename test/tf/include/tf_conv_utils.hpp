
#ifndef MIGRAPHX_GUARD_TEST_TF_TF_CONV_UTILS_HPP
#define MIGRAPHX_GUARD_TEST_TF_TF_CONV_UTILS_HPP


migraphx::program create_conv()
{
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    std::vector<float> weight_data(3 * 3 * 3 * 32);
    std::fill(weight_data.begin(), weight_data.end(), 1.0f);
    auto l1 =
        mm->add_literal(migraphx::shape{migraphx::shape::float_type, {3, 3, 3, 32}}, weight_data);

    migraphx::op::convolution op;
    op.padding  = {1, 1, 1, 1};
    op.stride   = {1, 1};
    op.dilation = {1, 1};
    auto l2 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {3, 2, 0, 1}}}), l1);
    mm->add_instruction(op, l0, l2);
    return p;
}


#endif
