
#include <onnx_test.hpp>


TEST_CASE(reducel1_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto abs_l0 = mm->add_instruction(migraphx::make_op("abs"), l0);
    auto sum_l0 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {-2}}}), abs_l0);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {-2}}}), sum_l0);
    auto prog = optimize_onnx("reducel1_test.onnx");

    EXPECT(p == prog);
}


