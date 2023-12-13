
#include <onnx_test.hpp>
#include <migraphx/op/convolution.hpp>

TEST_CASE(group_conv_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 4, 16, 16}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {4, 1, 3, 3}});
    migraphx::op::convolution op;
    op.group = 4;
    mm->add_instruction(op, l0, l1);
    auto prog = optimize_onnx("group_conv_test.onnx");

    EXPECT(p == prog);
}
