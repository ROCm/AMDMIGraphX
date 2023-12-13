
#include <onnx_test.hpp>

TEST_CASE(conv_transpose_output_shape_3d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3, 3}});
    auto l1  = mm->add_parameter("w", {migraphx::shape::float_type, {1, 2, 3, 3, 3}});
    auto l2  = mm->add_instruction(
        migraphx::make_op("convolution_backwards",
                           {{"padding", {0, 0, 0}}, {"stride", {3, 2, 2}}, {"dilation", {1, 1, 1}}}),
        l0,
        l1);
    mm->add_instruction(migraphx::make_op("pad", {{"pads", {0, 0, 0, 0, 0, 0, 0, 1, 1, 1}}}), l2);

    auto prog = optimize_onnx("conv_transpose_output_shape_3d_test.onnx");
    EXPECT(p == prog);
}
