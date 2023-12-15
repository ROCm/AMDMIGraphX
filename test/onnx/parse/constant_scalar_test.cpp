
#include <onnx_test.hpp>


TEST_CASE(constant_scalar_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {1}}, {1}});
    auto prog = optimize_onnx("constant_scalar_test.onnx");

    EXPECT(p == prog);
}


