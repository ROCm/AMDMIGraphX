
#include <onnx_test.hpp>


TEST_CASE(size_int_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::int32_type, {8, 2, 3}};
    mm->add_parameter("x", s);
    mm->add_literal(migraphx::literal{migraphx::shape::int64_type, {s.elements()}});
    auto prog = optimize_onnx("size_int_test.onnx");
    EXPECT(p == prog);
}


