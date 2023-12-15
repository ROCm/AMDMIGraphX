
#include <onnx_test.hpp>
#include <migraphx/apply_alpha_beta.hpp>

TEST_CASE(gemm_brcst_C_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("A", migraphx::shape{migraphx::shape::float_type, {5, 6}});
    auto l1  = mm->add_parameter("B", migraphx::shape{migraphx::shape::float_type, {5, 7}});
    auto l2  = mm->add_parameter("C", migraphx::shape{migraphx::shape::float_type, {6, 1}});
    std::vector<std::size_t> out_lens{6, 7};
    auto alpha = 0.5f;
    auto beta  = 0.8f;
    auto a_l   = mm->add_literal(alpha);
    auto t_a   = add_common_op(*mm, migraphx::make_op("mul"), {a_l, l0});
    t_a      = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), t_a);
    auto dot = migraphx::add_apply_alpha_beta(*mm, {t_a, l1}, migraphx::make_op("dot"), 1.0f, 0.0f);
    auto b_l = mm->add_literal(beta);
    auto l2_b =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", out_lens}}), l2);
    auto b_b = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", l2_b->get_shape().lens()}}), b_l);
    auto l2_bb = mm->add_instruction(migraphx::make_op("mul"), l2_b, b_b);
    mm->add_instruction(migraphx::make_op("add"), dot, l2_bb);

    auto prog = optimize_onnx("gemm_brcst_C_test.onnx");
    EXPECT(p == prog);
}
