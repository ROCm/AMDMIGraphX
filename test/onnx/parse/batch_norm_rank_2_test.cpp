
#include <onnx_test.hpp>

TEST_CASE(batch_norm_rank_2_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x     = mm->add_parameter("x", {migraphx::shape::float_type, {2, 5}});
    auto scale = mm->add_parameter("scale", {migraphx::shape::float_type, {5}});
    auto bias  = mm->add_parameter("bias", {migraphx::shape::float_type, {5}});
    auto mean  = mm->add_parameter("mean", {migraphx::shape::float_type, {5}});
    auto var   = mm->add_parameter("variance", {migraphx::shape::float_type, {5}});

    auto eps = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {1e-6f}});

    auto x_sub_mean = add_common_op(*mm, migraphx::make_op("sub"), {x, mean});
    auto var_eps    = add_common_op(*mm, migraphx::make_op("add"), {var, eps});
    auto rsqrt      = mm->add_instruction(migraphx::make_op("rsqrt"), {var_eps});
    auto mul0       = add_common_op(*mm, migraphx::make_op("mul"), {scale, rsqrt});
    auto r0         = add_common_op(*mm, migraphx::make_op("mul"), {x_sub_mean, mul0});
    add_common_op(*mm, migraphx::make_op("add"), {r0, bias});

    auto prog = optimize_onnx("batch_norm_rank_2_test.onnx");
    EXPECT(p == prog);
}
