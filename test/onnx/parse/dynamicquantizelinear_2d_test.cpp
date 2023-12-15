
#include <onnx_test.hpp>


TEST_CASE(dynamicquantizelinear_2d_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto x_dims = {3, 4};
    auto x_type = migraphx::shape::float_type;
    auto x      = mm->add_parameter("x", {x_type, x_dims});

    auto l0         = mm->add_literal({0.f});
    auto x_reshaped = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {12}}}), x);
    x_reshaped = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x_reshaped, l0);

    auto q_range = mm->add_literal(
        migraphx::literal{migraphx::shape{x_type}, {std::numeric_limits<uint8_t>::max()}});

    auto max_x = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {0}}}), x_reshaped);
    auto min_x = mm->add_instruction(migraphx::make_op("reduce_min", {{"axes", {0}}}), x_reshaped);

    auto sub0    = mm->add_instruction(migraphx::make_op("sub"), max_x, min_x);
    auto y_scale = mm->add_instruction(migraphx::make_op("div"), sub0, q_range);

    auto q_min = mm->add_literal(
        migraphx::literal{migraphx::shape{x_type}, {std::numeric_limits<uint8_t>::min()}});
    auto q_max = mm->add_literal(
        migraphx::literal{migraphx::shape{x_type}, {std::numeric_limits<uint8_t>::max()}});
    auto sub1         = mm->add_instruction(migraphx::make_op("sub"), q_min, min_x);
    auto interm_zp    = mm->add_instruction(migraphx::make_op("div"), sub1, y_scale);
    auto saturate     = mm->add_instruction(migraphx::make_op("clip"), interm_zp, q_min, q_max);
    auto round        = mm->add_instruction(migraphx::make_op("nearbyint"), saturate);
    auto y_zero_point = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::uint8_type}}), round);

    auto scale_y_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", x_dims}}), y_scale);

    auto y_pt_c_bcast = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", x_dims}}), y_zero_point);

    mm->add_instruction(migraphx::make_op("quantizelinear"), x, scale_y_bcast, y_pt_c_bcast);

    auto prog = optimize_onnx("dynamicquantizelinear_2d_test.onnx");
    EXPECT(p == prog);
}


