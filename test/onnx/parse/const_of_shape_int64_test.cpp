
#include <onnx_test.hpp>


TEST_CASE(const_of_shape_int64_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    // output_dims
    migraphx::shape ss(migraphx::shape::int64_type, {3});
    mm->add_literal(migraphx::literal(ss, {2, 3, 4}));
    // constant shape literal
    migraphx::shape s(migraphx::shape::int64_type, {2, 3, 4});
    std::vector<int64_t> vec(s.elements(), 10);
    mm->add_literal(migraphx::literal(s, vec));

    auto prog = optimize_onnx("const_of_shape_int64_test.onnx");
    EXPECT(p == prog);
}


