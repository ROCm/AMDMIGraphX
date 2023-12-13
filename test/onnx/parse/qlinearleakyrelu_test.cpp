
#include <onnx_test.hpp>

TEST_CASE(qlinearleakyrelu_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x = mm->add_parameter("X", {migraphx::shape::int8_type, {64}});

    auto sc_x   = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {0.05}});
    auto z_pt_x = mm->add_literal(migraphx::literal{migraphx::shape::int8_type, {0}});

    auto sc_y   = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {0.05}});
    auto z_pt_y = mm->add_literal(migraphx::literal{migraphx::shape::int8_type, {10}});

    auto scale_x_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {64}}}), sc_x);

    auto z_pt_x_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {64}}}), z_pt_x);

    auto fp_x =
        mm->add_instruction(migraphx::make_op("dequantizelinear"), x, scale_x_bcast, z_pt_x_bcast);

    auto fp_y = mm->add_instruction(migraphx::make_op("leaky_relu", {{"alpha", 1.1}}), fp_x);

    auto scale_y_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {64}}}), sc_y);

    auto z_pt_y_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {64}}}), z_pt_y);

    auto y =
        mm->add_instruction(migraphx::make_op("quantizelinear"), fp_y, scale_y_bcast, z_pt_y_bcast);

    mm->add_return({y});

    auto prog = migraphx::parse_onnx("qlinearleakyrelu_test.onnx");

    EXPECT(p.sort() == prog.sort());
}
