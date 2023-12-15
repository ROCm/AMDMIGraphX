
#include <onnx_test.hpp>

TEST_CASE(reducesum_noop_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(migraphx::literal{migraphx::shape::int64_type});
    auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    mm->add_return({x});
    auto prog = migraphx::parse_onnx("reducesum_noop_test.onnx");

    EXPECT(p == prog);
}
