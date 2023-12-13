
#include <onnx_test.hpp>

TEST_CASE(softmax_nonstd_input_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {6, 8}});
    auto l1  = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {1, 0}}, {"ends", {4, 4}}}), l0);
    auto l2 = mm->add_instruction(migraphx::make_op("softmax", {{"axis", -1}}), l1);
    mm->add_return({l2});

    auto prog = migraphx::parse_onnx("softmax_nonstd_input_test.onnx");

    EXPECT(p == prog);
}
