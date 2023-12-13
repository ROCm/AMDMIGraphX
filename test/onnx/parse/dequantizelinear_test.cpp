
#include <onnx_test.hpp>

TEST_CASE(dequantizelinear_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", {migraphx::shape::int8_type, {5}});
    auto l1  = mm->add_parameter("1", {migraphx::shape::float_type, {1}});
    auto l1_mbcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {5}}}), l1);
    auto dequant = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
        l0);
    mm->add_instruction(migraphx::make_op("mul"), dequant, l1_mbcast);

    auto prog = optimize_onnx("dequantizelinear_test.onnx", true);
    EXPECT(p.sort() == prog.sort());
}
