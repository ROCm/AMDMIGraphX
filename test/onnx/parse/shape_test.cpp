
#include <onnx_test.hpp>

TEST_CASE(shape_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 4, 5, 6}};
    auto l0 = mm->add_parameter("x", s);
    migraphx::shape s_shape{migraphx::shape::int64_type, {4}};
    mm->add_literal(s_shape, l0->get_shape().lens());
    auto prog = optimize_onnx("shape_test.onnx");

    EXPECT(p == prog);
}
