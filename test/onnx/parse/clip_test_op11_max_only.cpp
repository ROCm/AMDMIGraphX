
#include <onnx_test.hpp>

TEST_CASE(clip_test_op11_max_only)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto max_val = mm->add_literal(0.0f);
    auto l0      = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    mm->add_instruction(migraphx::make_op("undefined"));
    max_val =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3}}}), max_val);
    auto r = mm->add_instruction(migraphx::make_op("min"), l0, max_val);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("clip_test_op11_max_only.onnx");

    EXPECT(p == prog);
}
