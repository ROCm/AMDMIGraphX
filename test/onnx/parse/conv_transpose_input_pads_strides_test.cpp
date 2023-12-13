
#include <onnx_test.hpp>

TEST_CASE(conv_transpose_input_pads_strides_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto l1  = mm->add_parameter("w", {migraphx::shape::float_type, {1, 2, 3, 3}});
    mm->add_instruction(
        migraphx::make_op("convolution_backwards", {{"padding", {1, 1}}, {"stride", {3, 2}}}),
        l0,
        l1);

    auto prog = optimize_onnx("conv_transpose_input_pads_strides_test.onnx");
    EXPECT(p == prog);
}
