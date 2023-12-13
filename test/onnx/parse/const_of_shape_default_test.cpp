
#include <onnx_test.hpp>

TEST_CASE(const_of_shape_default_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape output_dims_shape(migraphx::shape::int64_type, {3});
    mm->add_literal(migraphx::literal(output_dims_shape, {2, 3, 4}));
    migraphx::shape output_shape{migraphx::shape::float_type, {2, 3, 4}};
    std::vector<float> vec(output_shape.elements(), 0.0);
    mm->add_literal(migraphx::literal(output_shape, vec));

    auto prog = optimize_onnx("const_of_shape_default_test.onnx");
    EXPECT(p == prog);
}
