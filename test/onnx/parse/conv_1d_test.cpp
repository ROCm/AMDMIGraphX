
#include <onnx_test.hpp>


TEST_CASE(conv_1d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", {migraphx::shape::float_type, {1, 3, 5}});
    auto l1  = mm->add_parameter("1", {migraphx::shape::float_type, {1, 3, 3}});
    mm->add_instruction(
        migraphx::make_op("convolution", {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
        l0,
        l1);

    auto prog = optimize_onnx("conv_1d_test.onnx");
    EXPECT(p == prog);
}


