
#include <onnx_test.hpp>

TEST_CASE(sqrt_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10, 15}});
    mm->add_instruction(migraphx::make_op("sqrt"), input);

    auto prog = optimize_onnx("sqrt_test.onnx");
    EXPECT(p == prog);
}
