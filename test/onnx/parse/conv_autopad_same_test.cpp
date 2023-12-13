
#include <onnx_test.hpp>
#include <migraphx/op/convolution.hpp>


TEST_CASE(conv_autopad_same_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", {migraphx::shape::float_type, {1, 3, 32, 32}});
    auto l1  = mm->add_parameter("1", {migraphx::shape::float_type, {1, 3, 3, 3}});
    migraphx::op::convolution op;
    op.padding = {1, 1, 1, 1};
    mm->add_instruction(op, l0, l1);

    auto prog = optimize_onnx("conv_autopad_same_test.onnx");
    EXPECT(p == prog);
}


