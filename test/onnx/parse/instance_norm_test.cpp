
#include <onnx_test.hpp>

TEST_CASE(instance_norm_test)
{
    std::vector<size_t> dims{1, 2, 3, 3};
    migraphx::shape s1{migraphx::shape::float_type, dims};
    migraphx::shape s2{migraphx::shape::float_type, {2}};

    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto x     = mm->add_parameter("0", s1);
    auto scale = mm->add_parameter("1", s2);
    auto bias  = mm->add_parameter("2", s2);

    auto mean = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2, 3}}}), x);
    auto l1   = add_common_op(*mm, migraphx::make_op("sub"), {x, mean});
    auto l0   = add_common_op(*mm, migraphx::make_op("sqdiff"), {x, mean});

    auto variance = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2, 3}}}), l0);

    auto epsilon_literal =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {1e-5}});
    auto l2 = add_common_op(*mm, migraphx::make_op("add"), {variance, epsilon_literal});

    auto l3 = mm->add_instruction(migraphx::make_op("rsqrt"), l2);
    auto l4 = add_common_op(*mm, migraphx::make_op("mul"), {l1, l3});

    auto scale_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", dims}}), scale);
    auto bias_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", dims}}), bias);
    auto l5  = mm->add_instruction(migraphx::make_op("mul"), l4, scale_bcast);
    auto ret = mm->add_instruction(migraphx::make_op("add"), l5, bias_bcast);
    mm->add_return({ret});

    migraphx::onnx_options options;
    auto prog = migraphx::parse_onnx("instance_norm_test.onnx", options);

    EXPECT(p == prog);
}
