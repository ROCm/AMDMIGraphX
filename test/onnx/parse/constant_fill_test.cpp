
#include <onnx_test.hpp>

TEST_CASE(constant_fill_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    std::vector<float> value(s.elements(), 1.0);
    mm->add_literal(migraphx::literal{s, value});
    auto prog = optimize_onnx("constant_fill_test.onnx");

    EXPECT(p == prog);
}
