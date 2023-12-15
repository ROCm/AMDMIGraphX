
#include <onnx_test.hpp>

TEST_CASE(size_float_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {2, 3, 4}};
    mm->add_parameter("x", s);
    mm->add_literal(migraphx::literal{migraphx::shape::int64_type, {s.elements()}});

    auto prog = optimize_onnx("size_float_test.onnx");
    EXPECT(p == prog);
}
