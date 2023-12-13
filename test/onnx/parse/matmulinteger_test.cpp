
#include <onnx_test.hpp>


TEST_CASE(matmulinteger_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::int8_type, {3, 6, 16}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::int8_type, {3, 16, 8}});
    mm->add_instruction(migraphx::make_op("quant_dot"), l0, l1);

    auto prog = optimize_onnx("matmulinteger_test.onnx");

    EXPECT(p == prog);
}


