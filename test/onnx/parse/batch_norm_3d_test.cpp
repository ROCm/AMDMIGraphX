
#include <onnx_test.hpp>


TEST_CASE(batch_norm_3d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x     = mm->add_parameter("x", {migraphx::shape::half_type, {2, 2, 2, 2, 2}});
    auto scale = mm->add_parameter("scale", {migraphx::shape::half_type, {2}});
    auto bias  = mm->add_parameter("bias", {migraphx::shape::half_type, {2}});
    auto mean  = mm->add_parameter("mean", {migraphx::shape::half_type, {2}});
    auto var   = mm->add_parameter("variance", {migraphx::shape::half_type, {2}});

    auto eps = mm->add_literal(migraphx::literal{migraphx::shape::half_type, {1e-6f}});

    auto usq_scale =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2, 3}}}), scale);
    auto usq_bias =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2, 3}}}), bias);
    auto usq_mean =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2, 3}}}), mean);
    auto usq_var = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2, 3}}}), var);

    auto x_sub_mean = add_common_op(*mm, migraphx::make_op("sub"), {x, usq_mean});
    auto var_eps    = add_common_op(*mm, migraphx::make_op("add"), {usq_var, eps});
    auto rsqrt      = mm->add_instruction(migraphx::make_op("rsqrt"), var_eps);
    auto mul0       = add_common_op(*mm, migraphx::make_op("mul"), {usq_scale, rsqrt});
    auto r0         = add_common_op(*mm, migraphx::make_op("mul"), {x_sub_mean, mul0});
    add_common_op(*mm, migraphx::make_op("add"), {r0, usq_bias});

    auto prog = optimize_onnx("batch_norm_3d_test.onnx");

    EXPECT(p == prog);
}


