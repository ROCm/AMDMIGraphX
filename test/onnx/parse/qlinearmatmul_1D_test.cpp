
#include <onnx_test.hpp>

TEST_CASE(qlinearmatmul_1D_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto a = mm->add_parameter("A", {migraphx::shape::uint8_type, {8}});
    auto b = mm->add_parameter("B", {migraphx::shape::uint8_type, {8}});

    auto sc_a   = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {0.05}});
    auto z_pt_a = mm->add_literal(migraphx::literal{migraphx::shape::uint8_type, {0}});

    auto sc_b   = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {0.05}});
    auto z_pt_b = mm->add_literal(migraphx::literal{migraphx::shape::uint8_type, {128}});

    auto sc_c   = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {0.05}});
    auto z_pt_c = mm->add_literal(migraphx::literal{migraphx::shape::uint8_type, {64}});

    auto scale_a_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8}}}), sc_a);

    auto z_pt_a_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8}}}), z_pt_a);

    auto fp_a =
        mm->add_instruction(migraphx::make_op("dequantizelinear"), a, scale_a_bcast, z_pt_a_bcast);

    auto scale_b_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8}}}), sc_b);

    auto z_pt_b_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8}}}), z_pt_b);

    auto fp_b =
        mm->add_instruction(migraphx::make_op("dequantizelinear"), b, scale_b_bcast, z_pt_b_bcast);

    auto sq_a = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), fp_a);

    auto sq_b = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), fp_b);

    auto fp_c = mm->add_instruction(migraphx::make_op("dot"), sq_a, sq_b);

    auto sq_c = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), fp_c);

    auto scale_c_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1}}}), sc_c);

    auto z_pt_c_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1}}}), z_pt_c);

    auto c =
        mm->add_instruction(migraphx::make_op("quantizelinear"), sq_c, scale_c_bcast, z_pt_c_bcast);

    mm->add_return({c});

    auto prog = migraphx::parse_onnx("qlinearmatmul_1D_test.onnx");

    EXPECT(p.sort() == prog.sort());
}
