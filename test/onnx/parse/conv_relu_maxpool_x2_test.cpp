
#include <onnx_test.hpp>
#include <migraphx/op/pooling.hpp>


TEST_CASE(conv_relu_maxpool_x2_test)
{
    migraphx::program p;
    auto* mm      = p.get_main_module();
    auto l0       = mm->add_parameter("0", {migraphx::shape::float_type, {1, 3, 32, 32}});
    auto l1       = mm->add_parameter("1", {migraphx::shape::float_type, {5, 3, 5, 5}});
    auto l2       = mm->add_parameter("2", {migraphx::shape::float_type, {5}});
    uint64_t axis = 1;
    auto l3 =
        mm->add_instruction(migraphx::make_op("convolution", {{"padding", {0, 0, 0, 0}}}), l0, l1);
    auto l4 = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", l3->get_shape().lens()}}), l2);
    auto l5 = mm->add_instruction(migraphx::make_op("add"), l3, l4);
    auto l6 = mm->add_instruction(migraphx::make_op("relu"), l5);
    auto l7 = mm->add_instruction(migraphx::make_op("pooling",
                                                    {{"mode", migraphx::op::pooling_mode::max},
                                                     {"padding", {0, 0, 0, 0}},
                                                     {"stride", {2, 2}},
                                                     {"lengths", {2, 2}},
                                                     {"dilations", {1, 1}}}),
                                  l6);

    auto l8 = mm->add_parameter("3", {migraphx::shape::float_type, {1, 5, 5, 5}});
    auto l9 = mm->add_parameter("4", {migraphx::shape::float_type, {1}});
    auto l10 =
        mm->add_instruction(migraphx::make_op("convolution", {{"padding", {0, 0, 0, 0}}}), l7, l8);
    auto l11 = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", l10->get_shape().lens()}}),
        l9);
    auto l12 = mm->add_instruction(migraphx::make_op("add"), l10, l11);
    auto l13 = mm->add_instruction(migraphx::make_op("relu"), l12);
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", migraphx::op::pooling_mode::max},
                                           {"padding", {0, 0, 0, 0}},
                                           {"stride", {2, 2}},
                                           {"lengths", {2, 2}},
                                           {"dilations", {1, 1}}}),
                        l13);

    auto prog = optimize_onnx("conv_relu_maxpool_x2_test.onnx");

    EXPECT(p == prog);
}


