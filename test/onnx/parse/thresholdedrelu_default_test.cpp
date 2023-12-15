
#include <onnx_test.hpp>


TEST_CASE(thresholdedrelu_default_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2, 3}});
    auto lz  = mm->add_literal(migraphx::literal{migraphx::shape{x->get_shape().type()}, {0}});
    auto la  = mm->add_literal(migraphx::literal{migraphx::shape{x->get_shape().type()}, {1.0f}});
    auto mbz = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", x->get_shape().lens()}}), lz);
    auto mba = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", x->get_shape().lens()}}), la);
    auto condition = mm->add_instruction(migraphx::make_op("greater"), x, mba);
    mm->add_instruction(migraphx::make_op("where"), condition, x, mbz);

    auto prog = optimize_onnx("thresholdedrelu_default_test.onnx");

    EXPECT(p == prog);
}


