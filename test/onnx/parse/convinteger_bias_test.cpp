
#include <onnx_test.hpp>

TEST_CASE(convinteger_bias_test)
{
    migraphx::program p;
    auto* mm      = p.get_main_module();
    auto l0       = mm->add_parameter("0", {migraphx::shape::int8_type, {1, 3, 32, 32}});
    auto l1       = mm->add_parameter("1", {migraphx::shape::int8_type, {1, 3, 5, 5}});
    auto l2       = mm->add_parameter("2", {migraphx::shape::int32_type, {1}});
    uint64_t axis = 1;
    auto l3       = mm->add_instruction(migraphx::make_op("quant_convolution"), l0, l1);
    auto l4       = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", l3->get_shape().lens()}}), l2);
    mm->add_instruction(migraphx::make_op("add"), l3, l4);

    auto prog = optimize_onnx("convinteger_bias_test.onnx");
    EXPECT(p == prog);
}
