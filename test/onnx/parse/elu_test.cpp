
#include <onnx_test.hpp>


TEST_CASE(elu_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    mm->add_instruction(migraphx::make_op("elu", {{"alpha", 0.01}}), input);

    auto prog = optimize_onnx("elu_test.onnx");

    EXPECT(p == prog);
}


