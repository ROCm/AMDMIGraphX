
#include <onnx_test.hpp>


TEST_CASE(sum_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto input0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    auto input1 = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3}});
    auto input2 = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {3}});
    auto l0     = mm->add_instruction(migraphx::make_op("add"), input0, input1);
    mm->add_instruction(migraphx::make_op("add"), l0, input2);

    auto prog = optimize_onnx("sum_test.onnx");
    EXPECT(p == prog);
}


