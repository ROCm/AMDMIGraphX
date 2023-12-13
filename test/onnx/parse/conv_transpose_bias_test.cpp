
#include <onnx_test.hpp>


TEST_CASE(conv_transpose_bias_test)
{
    migraphx::program p;
    auto* mm      = p.get_main_module();
    auto l0       = mm->add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto l1       = mm->add_parameter("w", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto l2       = mm->add_parameter("b", {migraphx::shape::float_type, {1}});
    uint64_t axis = 1;
    auto l3       = mm->add_instruction(migraphx::make_op("convolution_backwards"), l0, l1);
    auto l4       = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", l3->get_shape().lens()}}), l2);
    mm->add_instruction(migraphx::make_op("add"), l3, l4);

    auto prog = optimize_onnx("conv_transpose_bias_test.onnx");
    EXPECT(p == prog);
}


