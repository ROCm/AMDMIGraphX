
#include <onnx_test.hpp>

TEST_CASE(round_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::double_type, {10, 5}});
    mm->add_instruction(migraphx::make_op("nearbyint"), input);

    auto prog = optimize_onnx("round_test.onnx");
    EXPECT(p == prog);
}
