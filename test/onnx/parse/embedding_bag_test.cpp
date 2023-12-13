
#include <onnx_test.hpp>

TEST_CASE(embedding_bag_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("weight", migraphx::shape{migraphx::shape::float_type, {4, 2}});
    migraphx::literal l{migraphx::shape{migraphx::shape::int32_type, {3}}, {1, 0, 2}};
    auto l1 = mm->add_literal(l);
    mm->add_literal(0);
    auto l4 = mm->add_instruction(migraphx::make_op("gather"), l0, l1);
    auto r1 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), l4);
    auto l5 = mm->add_instruction(migraphx::make_op("gather"), l0, l1);
    auto r2 = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {0}}}), l5);
    auto l6 = mm->add_instruction(migraphx::make_op("gather"), l0, l1);
    auto r3 = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {0}}}), l6);
    mm->add_return({r1, r2, r3});

    auto prog = migraphx::parse_onnx("embedding_bag_test.onnx");

    EXPECT(p == prog);
}
