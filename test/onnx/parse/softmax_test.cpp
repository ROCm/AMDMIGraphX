
#include <onnx_test.hpp>

TEST_CASE(softmax_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3}});
    mm->add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), l0);
    auto prog = optimize_onnx("softmax_test.onnx");

    EXPECT(p == prog);
}
