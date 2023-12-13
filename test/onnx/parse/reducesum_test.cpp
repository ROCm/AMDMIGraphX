
#include <onnx_test.hpp>

TEST_CASE(reducesum_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto l1  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), l0);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), l1);
    auto prog = optimize_onnx("reducesum_test.onnx");

    EXPECT(p == prog);
}
