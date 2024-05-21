
#include <tf_test.hpp>
#include <conv_test_utils.hpp>


TEST_CASE(depthwiseconv_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    std::vector<float> weight_data(3 * 3 * 3 * 1);
    std::fill(weight_data.begin(), weight_data.end(), 1.0f);
    auto l1 =
        mm->add_literal(migraphx::shape{migraphx::shape::float_type, {3, 3, 3, 1}}, weight_data);

    migraphx::op::convolution op;
    op.padding  = {1, 1};
    op.stride   = {1, 1};
    op.dilation = {1, 1};
    op.group    = 3;
    auto l3 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {3, 2, 0, 1}}}), l1);
    auto l4 = mm->add_instruction(migraphx::make_op("contiguous"), l3);
    auto l5 = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {3, 1, 3, 3}}}), l4);
    mm->add_instruction(op, l0, l5);
    auto prog = optimize_tf("depthwise_conv_test.pb", true);

    EXPECT(p == prog);
}


