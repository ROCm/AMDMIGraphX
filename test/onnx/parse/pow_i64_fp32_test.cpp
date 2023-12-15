
#include <onnx_test.hpp>

TEST_CASE(pow_i64_fp32_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::int64_type, {2, 3, 4, 5}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l0f = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), l0);
    auto fr = mm->add_instruction(migraphx::make_op("pow"), l0f, l1);
    auto ir = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int64_type}}), fr);
    mm->add_return({ir});

    auto prog = migraphx::parse_onnx("pow_i64_fp32_test.onnx");

    EXPECT(p == prog);
}
