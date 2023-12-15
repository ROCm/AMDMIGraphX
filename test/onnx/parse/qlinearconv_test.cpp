
#include <onnx_test.hpp>

TEST_CASE(qlinearconv_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x = mm->add_parameter("X", {migraphx::shape::uint8_type, {1, 1, 7, 7}});

    auto sc_x   = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {0.00369204697}});
    auto z_pt_x = mm->add_literal(migraphx::literal{migraphx::shape::uint8_type, {132}});

    auto w = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::uint8_type, {1, 1, 1, 1}}, {0}});

    auto sc_w   = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {0.00172794575}});
    auto z_pt_w = mm->add_literal(migraphx::literal{migraphx::shape::uint8_type, {255}});

    auto sc_y   = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {0.00162681262}});
    auto z_pt_y = mm->add_literal(migraphx::literal{migraphx::shape::uint8_type, {123}});

    auto scale_x_bcast = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 7, 7}}}), sc_x);

    auto z_pt_x_bcast = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 7, 7}}}), z_pt_x);

    auto fp_x =
        mm->add_instruction(migraphx::make_op("dequantizelinear"), x, scale_x_bcast, z_pt_x_bcast);

    auto scale_w_bcast = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 1, 1}}}), sc_w);

    auto z_pt_w_bcast = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 1, 1}}}), z_pt_w);

    auto fp_w =
        mm->add_instruction(migraphx::make_op("dequantizelinear"), w, scale_w_bcast, z_pt_w_bcast);

    auto fp_y = mm->add_instruction(migraphx::make_op("convolution"), fp_x, fp_w);

    auto scale_y_bcast = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 7, 7}}}), sc_y);

    auto z_pt_y_bcast = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 7, 7}}}), z_pt_y);

    auto y =
        mm->add_instruction(migraphx::make_op("quantizelinear"), fp_y, scale_y_bcast, z_pt_y_bcast);

    mm->add_return({y});

    auto prog = migraphx::parse_onnx("qlinearconv_test.onnx");

    EXPECT(p.sort() == prog.sort());
}
