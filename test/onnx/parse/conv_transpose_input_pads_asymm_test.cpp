
#include <onnx_test.hpp>


TEST_CASE(conv_transpose_input_pads_asymm_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto l1  = mm->add_parameter("w", {migraphx::shape::float_type, {1, 2, 3, 3}});
    auto l2  = mm->add_instruction(
        migraphx::make_op("convolution_backwards", {{"padding", {0, 0}}, {"stride", {3, 2}}}),
        l0,
        l1);
    mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {2, 3}}, {"starts", {0, 0}}, {"ends", {8, 6}}}), l2);

    auto prog = optimize_onnx("conv_transpose_input_pads_asymm_test.onnx");
    EXPECT(p == prog);
}


