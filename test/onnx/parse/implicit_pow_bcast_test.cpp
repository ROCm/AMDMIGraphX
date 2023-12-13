
#include <onnx_test.hpp>


TEST_CASE(implicit_pow_bcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 4, 1}});
    auto l3 =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 4, 5}}}), l1);
    mm->add_instruction(migraphx::make_op("pow"), l0, l3);

    auto prog = optimize_onnx("implicit_pow_bcast_test.onnx");

    EXPECT(p == prog);
}


