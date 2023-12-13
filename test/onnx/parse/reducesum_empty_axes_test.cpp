
#include <onnx_test.hpp>

TEST_CASE(reducesum_empty_axes_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(migraphx::literal{migraphx::shape::int64_type});
    auto x  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto l1 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 1, 2, 3}}}), x);
    auto r  = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 1, 2, 3}}}), l1);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("reducesum_empty_axes_test.onnx");

    EXPECT(p == prog);
}
