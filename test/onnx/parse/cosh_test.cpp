
#include <onnx_test.hpp>


TEST_CASE(cosh_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1}});
    mm->add_instruction(migraphx::make_op("cosh"), input);

    auto prog = optimize_onnx("cosh_test.onnx");

    EXPECT(p == prog);
}


