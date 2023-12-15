
#include <onnx_test.hpp>

TEST_CASE(atanh_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    mm->add_instruction(migraphx::make_op("atanh"), input);

    auto prog = optimize_onnx("atanh_test.onnx");

    EXPECT(p == prog);
}
