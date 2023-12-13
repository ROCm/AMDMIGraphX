
#include <onnx_test.hpp>

TEST_CASE(qlinearadd_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto a = mm->add_parameter("A", {migraphx::shape::uint8_type, {64}});
    auto b = mm->add_parameter("B", {migraphx::shape::uint8_type, {64}});

    auto sc_a   = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {0.05}});
    auto z_pt_a = mm->add_literal(migraphx::literal{migraphx::shape::uint8_type, {0}});

    auto sc_b   = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {0.05}});
    auto z_pt_b = mm->add_literal(migraphx::literal{migraphx::shape::uint8_type, {128}});

    auto sc_c   = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {0.05}});
    auto z_pt_c = mm->add_literal(migraphx::literal{migraphx::shape::uint8_type, {64}});

    auto scale_a_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {64}}}), sc_a);

    auto z_pt_a_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {64}}}), z_pt_a);

    auto fp_a =
        mm->add_instruction(migraphx::make_op("dequantizelinear"), a, scale_a_bcast, z_pt_a_bcast);

    auto scale_b_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {64}}}), sc_b);

    auto z_pt_b_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {64}}}), z_pt_b);

    auto fp_b =
        mm->add_instruction(migraphx::make_op("dequantizelinear"), b, scale_b_bcast, z_pt_b_bcast);

    auto fp_c = mm->add_instruction(migraphx::make_op("add"), fp_a, fp_b);

    auto scale_c_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {64}}}), sc_c);

    auto z_pt_c_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {64}}}), z_pt_c);

    auto c =
        mm->add_instruction(migraphx::make_op("quantizelinear"), fp_c, scale_c_bcast, z_pt_c_bcast);

    mm->add_return({c});

    auto prog = migraphx::parse_onnx("qlinearadd_test.onnx");

    EXPECT(p.sort() == prog.sort());
}
