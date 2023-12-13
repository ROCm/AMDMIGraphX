
#include <onnx_test.hpp>


TEST_CASE(pow_fp32_i64_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::int64_type, {2, 3, 4, 5}});
    auto l1f = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), l1);
    auto ret = mm->add_instruction(migraphx::make_op("pow"), l0, l1f);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("pow_fp32_i64_test.onnx");

    EXPECT(p == prog);
}


