
#include <onnx_test.hpp>


TEST_CASE(sub_scalar_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {1}});
    auto m1 =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 4, 5}}}), l1);
    mm->add_instruction(migraphx::make_op("sub"), l0, m1);
    auto prog = optimize_onnx("sub_scalar_test.onnx");

    EXPECT(p == prog);
}


