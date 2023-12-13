
#include <onnx_test.hpp>

TEST_CASE(split_test_default)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10, 15}});
    auto r1    = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {5}}}), input);
    auto r2 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {5}}, {"ends", {10}}}), input);
    mm->add_return({r1, r2});

    auto prog = migraphx::parse_onnx("split_test_default.onnx");
    EXPECT(p == prog);
}
