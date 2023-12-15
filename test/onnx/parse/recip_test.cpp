
#include <onnx_test.hpp>


TEST_CASE(recip_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3}});
    mm->add_instruction(migraphx::make_op("recip"), input);

    auto prog = optimize_onnx("recip_test.onnx");

    EXPECT(p == prog);
}


