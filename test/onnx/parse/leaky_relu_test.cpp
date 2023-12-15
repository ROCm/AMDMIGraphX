
#include <onnx_test.hpp>


TEST_CASE(leaky_relu_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    float alpha = 0.01f;
    auto l0     = mm->add_parameter("0", {migraphx::shape::float_type, {3}});
    mm->add_instruction(migraphx::make_op("leaky_relu", {{"alpha", alpha}}), l0);

    auto prog = optimize_onnx("leaky_relu_test.onnx");

    EXPECT(p == prog);
}


