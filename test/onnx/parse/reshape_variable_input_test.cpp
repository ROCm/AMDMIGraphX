
#include <onnx_test.hpp>


TEST_CASE(reshape_variable_input_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto p0    = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {4, 2, 3}});
    auto p1    = mm->add_parameter("1", migraphx::shape{migraphx::shape::int64_type, {2}});
    auto alloc = mm->add_instruction(
        migraphx::make_op("allocate", {{"buf_type", migraphx::shape::float_type}}), p1);
    mm->add_instruction(migraphx::make_op("reshape"), p0, alloc);

    auto prog = optimize_onnx("reshape_variable_input_test.onnx");
    EXPECT(p == prog);
}


