
#include <onnx_test.hpp>

TEST_CASE(split_test_uneven)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {12, 15}});
    auto r1    = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {3}}}), input);
    auto r2 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {3}}, {"ends", {6}}}), input);
    auto r3 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {6}}, {"ends", {9}}}), input);
    auto r4 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {9}}, {"ends", {12}}}), input);
    auto r5 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {12}}, {"ends", {12}}}), input);
    mm->add_return({r1, r2, r3, r4, r5});

    auto prog = migraphx::parse_onnx("split_test_uneven.onnx");
    EXPECT(p == prog);
}
