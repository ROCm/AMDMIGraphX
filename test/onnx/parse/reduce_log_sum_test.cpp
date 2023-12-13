
#include <onnx_test.hpp>

TEST_CASE(reduce_log_sum_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto sum_l0 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {-3}}}), l0);
    mm->add_instruction(migraphx::make_op("log"), sum_l0);
    auto prog = optimize_onnx("reduce_log_sum_test.onnx");

    EXPECT(p == prog);
}
