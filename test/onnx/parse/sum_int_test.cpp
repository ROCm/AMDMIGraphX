
#include <onnx_test.hpp>

TEST_CASE(sum_int_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto input0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::int16_type, {3}});
    auto input1 = mm->add_parameter("1", migraphx::shape{migraphx::shape::uint16_type, {3}});
    auto input2 = mm->add_parameter("2", migraphx::shape{migraphx::shape::uint32_type, {3}});
    auto cin0   = mm->add_instruction(
        migraphx::make_op("convert",
                            {{"target_type", migraphx::to_value(migraphx::shape::uint32_type)}}),
        input0);
    auto cin1 = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::uint32_type)}}),
        input1);
    auto l0 = mm->add_instruction(migraphx::make_op("add"), cin0, cin1);
    mm->add_instruction(migraphx::make_op("add"), l0, input2);

    auto prog = optimize_onnx("sum_int_test.onnx");
    EXPECT(p == prog);
}
