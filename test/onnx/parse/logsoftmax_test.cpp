
#include <onnx_test.hpp>

TEST_CASE(logsoftmax_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    int axis = 1;
    mm->add_instruction(migraphx::make_op("logsoftmax", {{"axis", axis}}), l0);
    auto prog = optimize_onnx("logsoftmax_test.onnx");

    EXPECT(p == prog);
}
