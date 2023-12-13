
#include <onnx_test.hpp>


TEST_CASE(split_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10, 15}});
    auto r1    = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {7}}}), input);
    auto r2 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {7}}, {"ends", {11}}}), input);
    auto r3 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {11}}, {"ends", {15}}}), input);
    mm->add_return({r1, r2, r3});

    auto prog = migraphx::parse_onnx("split_test.onnx");
    EXPECT(p == prog);
}


