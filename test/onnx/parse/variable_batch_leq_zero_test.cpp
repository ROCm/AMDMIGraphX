
#include <onnx_test.hpp>


TEST_CASE(variable_batch_leq_zero_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    mm->add_instruction(migraphx::make_op("add"), l0, l1);
    auto prog = optimize_onnx("variable_batch_leq_zero_test.onnx");

    EXPECT(p == prog);
}


