
#include <onnx_test.hpp>


TEST_CASE(size_half_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::half_type, {3, 1}};
    mm->add_parameter("x", s);
    mm->add_literal(migraphx::literal{migraphx::shape::int64_type, {s.elements()}});
    auto prog = optimize_onnx("size_half_test.onnx");
    EXPECT(p == prog);
}


