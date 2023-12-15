
#include <onnx_test.hpp>

TEST_CASE(shape_gather_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {7, 3, 10}});
    migraphx::shape const_shape{migraphx::shape::int32_type, {1}};
    auto l2 = mm->add_literal(migraphx::literal{const_shape, {1}});
    auto l1 =
        mm->add_literal(migraphx::shape{migraphx::shape::int64_type, {3}}, l0->get_shape().lens());
    int axis = 0;
    mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), l1, l2);
    auto prog = optimize_onnx("shape_gather_test.onnx");

    EXPECT(p == prog);
}
