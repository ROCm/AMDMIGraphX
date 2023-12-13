
#include <onnx_test.hpp>

TEST_CASE(clip_test)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto l0      = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    auto min_val = mm->add_literal(0.0f);
    auto max_val = mm->add_literal(6.0f);
    min_val =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3}}}), min_val);
    max_val =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3}}}), max_val);
    mm->add_instruction(migraphx::make_op("clip"), l0, min_val, max_val);
    auto prog = optimize_onnx("clip_test.onnx");

    EXPECT(p == prog);
}
