
#include <onnx_test.hpp>


TEST_CASE(clip_test_args_type_mismatch)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto min_val = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1, 3}}, {1.5, 2.5, 3.5}});
    auto max_val = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {3, 1}}, {2, 3, 4}});

    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3, 3}});
    min_val =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 3}}}), min_val);
    max_val =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 3}}}), max_val);
    max_val = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), max_val);
    auto r = mm->add_instruction(migraphx::make_op("clip"), l0, min_val, max_val);
    mm->add_return({r});
    auto prog = migraphx::parse_onnx("clip_test_args_type_mismatch.onnx");
    EXPECT(p == prog);
}


