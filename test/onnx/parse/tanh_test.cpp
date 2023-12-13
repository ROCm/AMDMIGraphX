
#include <onnx_test.hpp>


TEST_CASE(tanh_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1}});
    mm->add_instruction(migraphx::make_op("tanh"), input);

    auto prog = optimize_onnx("tanh_test.onnx");

    EXPECT(p == prog);
}


