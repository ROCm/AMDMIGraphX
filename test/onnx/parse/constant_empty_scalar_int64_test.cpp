
#include <onnx_test.hpp>

TEST_CASE(constant_empty_scalar_int64_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(migraphx::literal{migraphx::shape::int64_type});
    auto prog = optimize_onnx("constant_empty_scalar_int64_test.onnx");

    EXPECT(p == prog);
}
