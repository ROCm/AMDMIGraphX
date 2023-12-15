
#include <onnx_test.hpp>

TEST_CASE(sign_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::double_type, {10, 5}});
    mm->add_instruction(migraphx::make_op("sign"), input);

    auto prog = optimize_onnx("sign_test.onnx");
    EXPECT(p == prog);
}
