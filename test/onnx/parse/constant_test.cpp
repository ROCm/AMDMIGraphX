
#include <onnx_test.hpp>

TEST_CASE(constant_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {3}}, {0, 1, 2}});
    auto prog = optimize_onnx("constant_test.onnx");

    EXPECT(p == prog);
}
