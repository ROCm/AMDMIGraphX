
#include <onnx_test.hpp>


TEST_CASE(instance_norm_dyn_batch_half_test)
{
    // instancenorm with half type, dynamic input in the 0'th (batch) dimension
    migraphx::shape s1{migraphx::shape::half_type, {{1, 2, {2}}, {2, 2}, {3, 3}, {3, 3}}};
    migraphx::shape s2{migraphx::shape::half_type, {2}};

    migraphx::program p;
    auto* mm        = p.get_main_module();
    auto x_fp16     = mm->add_parameter("0", s1);
    auto scale_fp16 = mm->add_parameter("1", s2);
    auto bias_fp16  = mm->add_parameter("2", s2);

    // conversion of half type to float is enabled by default
    auto x = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), x_fp16);
    auto scale = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), scale_fp16);
    auto bias = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), bias_fp16);

    auto mean = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2, 3}}}), x);
    auto l0   = add_common_op(*mm, migraphx::make_op("sub"), {x, mean});
    auto l1   = add_common_op(*mm, migraphx::make_op("sqdiff"), {x, mean});

    auto variance = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2, 3}}}), l1);
    // type of epsilon_literal is same as 0'th input; convert instruction will be added by
    // add_common_op
    auto epsilon_literal =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {1e-5}});
    auto l2 = add_common_op(*mm, migraphx::make_op("add"), {variance, epsilon_literal});

    auto l3 = mm->add_instruction(migraphx::make_op("rsqrt"), l2);
    auto l4 = add_common_op(*mm, migraphx::make_op("mul"), {l0, l3});

    auto scale_bcast = mm->add_instruction(migraphx::make_op("broadcast", {{"axis", 1}}), scale, x);
    auto bias_bcast  = mm->add_instruction(migraphx::make_op("broadcast", {{"axis", 1}}), bias, x);
    auto l5          = mm->add_instruction(migraphx::make_op("mul"), l4, scale_bcast);
    auto instance_norm_fp32 = mm->add_instruction(migraphx::make_op("add"), l5, bias_bcast);
    auto ret                = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
        instance_norm_fp32);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 2, {2}};
    auto prog = migraphx::parse_onnx("instance_norm_dyn_batch_half_test.onnx", options);
    EXPECT(p == prog);
}


