
#include <onnx_test.hpp>

TEST_CASE(clip_test_op11_min_only)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto min_val = mm->add_literal(0.0f);
    auto l0      = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    min_val =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3}}}), min_val);
    mm->add_instruction(migraphx::make_op("max"), l0, min_val);
    auto prog = optimize_onnx("clip_test_op11_min_only.onnx");

    EXPECT(p == prog);
}
