
#include <onnx_test.hpp>


TEST_CASE(constant_value_float_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {1.0f}});
    auto prog = optimize_onnx("constant_value_float_test.onnx");

    EXPECT(p == prog);
}


