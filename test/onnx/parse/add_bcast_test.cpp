
#include <onnx_test.hpp>

TEST_CASE(add_bcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 4}});
    auto l2  = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", l0->get_shape().lens()}}), l1);
    mm->add_instruction(migraphx::make_op("add"), l0, l2);

    auto prog = optimize_onnx("add_bcast_test.onnx");

    EXPECT(p == prog);
}
