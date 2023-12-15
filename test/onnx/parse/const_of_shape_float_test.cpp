
#include <onnx_test.hpp>


TEST_CASE(const_of_shape_float_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape ss(migraphx::shape::int64_type, {3});
    mm->add_literal(migraphx::literal(ss, {2, 3, 4}));
    migraphx::shape s(migraphx::shape::float_type, {2, 3, 4});
    std::vector<float> vec(s.elements(), 10.0f);
    mm->add_literal(migraphx::literal(s, vec));

    auto prog = optimize_onnx("const_of_shape_float_test.onnx");
    EXPECT(p == prog);
}


