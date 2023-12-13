
#include <onnx_test.hpp>

TEST_CASE(constant_value_floats_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {3}}, {1.0f, 2.0f, 3.0f}});
    auto prog = optimize_onnx("constant_value_floats_test.onnx");

    EXPECT(p == prog);
}
