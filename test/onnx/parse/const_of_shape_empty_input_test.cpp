
#include <onnx_test.hpp>

TEST_CASE(const_of_shape_empty_input_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(migraphx::literal(migraphx::shape::int64_type));
    migraphx::shape s(migraphx::shape::int64_type, {1}, {0});
    std::vector<int64_t> vec(s.elements(), 10);
    mm->add_literal(migraphx::literal(s, vec));

    auto prog = optimize_onnx("const_of_shape_empty_input_test.onnx");
    EXPECT(p == prog);
}
