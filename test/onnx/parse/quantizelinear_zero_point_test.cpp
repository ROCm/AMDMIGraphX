
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>


TEST_CASE(quantizelinear_zero_point_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", {migraphx::shape::float_type, {5}});
    auto l1  = mm->add_parameter("1", {migraphx::shape::float_type, {1}});
    auto l2  = mm->add_parameter("2", {migraphx::shape::int8_type, {1}});
    auto l1_mbcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {5}}}), l1);
    auto div   = mm->add_instruction(migraphx::make_op("div"), l0, l1_mbcast);
    auto round = mm->add_instruction(migraphx::make_op("nearbyint"), div);
    auto l2_mbcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {5}}}), l2);
    l2_mbcast = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
        l2_mbcast);
    auto add  = mm->add_instruction(migraphx::make_op("add"), round, l2_mbcast);
    auto s    = round->get_shape();
    auto clip = insert_quantizelinear_clip(*mm, div, add, s, -128, 127);
    mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::int8_type)}}),
        clip);

    auto prog = optimize_onnx("quantizelinear_zero_point_test.onnx", true);
    EXPECT(p.sort() == prog.sort());
}


