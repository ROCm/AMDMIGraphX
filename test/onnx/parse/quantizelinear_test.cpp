
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>


TEST_CASE(quantizelinear_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", {migraphx::shape::float_type, {5}});
    auto l1  = mm->add_parameter("1", {migraphx::shape::float_type, {1}});
    auto l1_mbcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {5}}}), l1);
    auto div   = mm->add_instruction(migraphx::make_op("div"), l0, l1_mbcast);
    auto nearbyint = mm->add_instruction(migraphx::make_op("nearbyint"), div);
    auto s         = nearbyint->get_shape();
    auto clip      = insert_quantizelinear_clip(*mm, div, nearbyint, s, 0, 255);
    mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::uint8_type)}}),
        clip);

    auto prog = optimize_onnx("quantizelinear_test.onnx", true);
    EXPECT(p.sort() == prog.sort());
}


