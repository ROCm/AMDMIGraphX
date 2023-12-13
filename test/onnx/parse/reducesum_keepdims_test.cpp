
#include <onnx_test.hpp>

TEST_CASE(reducesum_keepdims_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2, 3}}}), l0);
    auto prog = optimize_onnx("reducesum_keepdims_test.onnx");

    EXPECT(p == prog);
}
