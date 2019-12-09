#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/tf.hpp>
#include "test.hpp"

migraphx::program optimize_tf(const std::string& name, bool is_nhwc)
{
    auto prog = migraphx::parse_tf(name, is_nhwc);
    if(is_nhwc)
        migraphx::run_passes(prog,
                             {migraphx::simplify_reshapes{},
                              migraphx::dead_code_elimination{},
                              migraphx::eliminate_identity{}});
    return prog;
}

TEST_CASE(add_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    p.add_instruction(migraphx::op::add{}, l0, l1);
    auto prog = optimize_tf("add_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(add_bcast_test)
{

    migraphx::program p;
    migraphx::shape s0{migraphx::shape::float_type, {2, 3}};
    auto l0 = p.add_parameter("0", s0);
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {2, 1}});
    auto l2 = p.add_instruction(migraphx::op::multibroadcast{s0.lens()}, l0);
    auto l3 = p.add_instruction(migraphx::op::multibroadcast{s0.lens()}, l1);
    p.add_instruction(migraphx::op::add{}, l2, l3);
    auto prog = optimize_tf("add_bcast_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(argmax_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    p.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {2}});
    auto ins = p.add_instruction(migraphx::op::argmax{2}, l0);
    p.add_instruction(migraphx::op::squeeze{{2}}, ins);
    auto prog = migraphx::parse_tf("argmax_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(assert_less_equal_test)
{
    migraphx::program p;
    migraphx::shape s0{migraphx::shape::float_type, {2, 3}};
    auto l0 = p.add_parameter("0", s0);
    auto l1 = p.add_parameter("1", s0);
    migraphx::literal l{migraphx::shape{migraphx::shape::int32_type, {2}}, {0, 1}};
    auto l2 = p.add_literal(l);
    p.add_instruction(migraphx::op::add{}, l0, l1);
    auto l3 = p.add_instruction(migraphx::op::identity{}, l0, l1);
    p.add_instruction(migraphx::op::identity{}, l3, l2);
    auto prog = optimize_tf("assert_less_equal_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(batchmatmul_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 8, 4}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 4, 8}});

    auto trans_l0 = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, l0);
    auto trans_l1 = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, l1);

    p.add_instruction(migraphx::op::dot{}, trans_l0, trans_l1);
    auto prog = optimize_tf("batchmatmul_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(batchnorm_test)
{
    float epsilon  = 1.001e-5f;
    float momentum = 0.9f;

    migraphx::program p;
    migraphx::op::batch_norm_inference op{
        epsilon, momentum, migraphx::op::batch_norm_inference::spatial};
    migraphx::shape s0{migraphx::shape::float_type, {32}};
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 32, 16, 16}});
    std::vector<float> const_vals(32);
    std::fill(const_vals.begin(), const_vals.end(), 1.0f);

    auto l2 = p.add_parameter("2", s0);
    auto l3 = p.add_parameter("3", s0);
    auto l4 = p.add_parameter("4", s0);
    auto l1 = p.add_literal(migraphx::literal{s0, const_vals});
    p.add_instruction(op, l0, l1, l2, l3, l4);
    auto prog = optimize_tf("batchnorm_test.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(biasadd_test)
{
    migraphx::program p;
    migraphx::shape s0{migraphx::shape::float_type, {1, 500, 1, 1}};
    uint64_t axis = 1;
    auto l0       = p.add_parameter("0", s0);
    auto l1       = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {500}});
    auto l2       = p.add_instruction(migraphx::op::broadcast{axis, l0->get_shape().lens()}, l1);
    p.add_instruction(migraphx::op::add{}, l0, l2);
    auto prog = optimize_tf("biasadd_test.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(cast_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    p.add_instruction(migraphx::op::convert{migraphx::shape::int32_type}, l0);
    auto prog = optimize_tf("cast_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(concat_test)
{
    migraphx::program p;

    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {4, 7, 3}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {4, 2, 3}});

    int axis = 1;
    // tf uses axis as the third input, and it is in int32 format
    // add the literal using a vector in order to set stride to 1 (like in tf parser)
    p.add_literal(migraphx::shape{migraphx::shape::int32_type}, std::vector<int>{axis});

    p.add_instruction(migraphx::op::concat{axis}, l0, l1);
    auto prog = optimize_tf("concat_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(const_test)
{
    migraphx::program p;
    p.add_literal(migraphx::shape{migraphx::shape::float_type}, std::vector<float>{1.0f});
    auto prog = optimize_tf("constant_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(conv_test)
{
    migraphx::program p;

    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    std::vector<float> weight_data(3 * 3 * 3 * 32);
    std::fill(weight_data.begin(), weight_data.end(), 1.0f);
    auto l1 =
        p.add_literal(migraphx::shape{migraphx::shape::float_type, {3, 3, 3, 32}}, weight_data);

    migraphx::op::convolution op;
    op.padding_mode = migraphx::op::padding_mode_t::same;
    op.padding      = {1, 1};
    op.stride       = {1, 1};
    op.dilation     = {1, 1};
    auto l2         = p.add_instruction(migraphx::op::transpose{{3, 2, 0, 1}}, l1);
    p.add_instruction(op, l0, l2);
    auto prog = optimize_tf("conv_test.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(depthwiseconv_test)
{
    migraphx::program p;

    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    std::vector<float> weight_data(3 * 3 * 3 * 1);
    std::fill(weight_data.begin(), weight_data.end(), 1.0f);
    auto l1 =
        p.add_literal(migraphx::shape{migraphx::shape::float_type, {3, 3, 3, 1}}, weight_data);

    migraphx::op::convolution op;
    op.padding_mode = migraphx::op::padding_mode_t::same;
    op.padding      = {1, 1};
    op.stride       = {1, 1};
    op.dilation     = {1, 1};
    op.group        = 3;
    auto l3         = p.add_instruction(migraphx::op::transpose{{3, 2, 0, 1}}, l1);
    auto l4         = p.add_instruction(migraphx::op::contiguous{}, l3);
    auto l5         = p.add_instruction(migraphx::op::reshape{{3, 1, 3, 3}}, l4);
    p.add_instruction(op, l0, l5);
    auto prog = optimize_tf("depthwise_conv_test.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(expanddims_test)
{
    migraphx::program p;

    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4}});
    p.add_literal(0);
    p.add_instruction(migraphx::op::reshape{{1, 2, 3, 4}}, l0);
    auto prog = optimize_tf("expanddims_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(expanddims_test_neg_dims)
{
    // this check makes sure the pb parses negative dim value correctly
    migraphx::program p;

    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4}});
    p.add_literal(-1);
    p.add_instruction(migraphx::op::reshape{{2, 3, 4, 1}}, l0);
    auto prog = optimize_tf("expanddims_neg_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(gather_test)
{
    migraphx::program p;

    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 4}});
    auto l1 =
        p.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {2}}, {1, 1}});
    p.add_literal(1);

    int axis = 1;
    p.add_instruction(migraphx::op::gather{axis}, l0, l1);
    auto prog = optimize_tf("gather_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(identity_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    p.add_instruction(migraphx::op::identity{}, l0);
    auto prog = optimize_tf("identity_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(matmul_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {8, 4}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {4, 8}});

    auto trans_l0 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l0);
    auto trans_l1 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l1);

    p.add_instruction(migraphx::op::dot{}, trans_l0, trans_l1);
    auto prog = optimize_tf("matmul_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(mean_test)
{
    migraphx::program p;
    migraphx::literal l{migraphx::shape{migraphx::shape::int32_type, {2}}, {2, 3}};
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    p.add_literal(l);
    p.add_literal(l);
    migraphx::op::reduce_mean op{{2, 3}};
    p.add_instruction(op, l0);
    auto l3 = p.add_instruction(op, l0);
    p.add_instruction(migraphx::op::squeeze{{2, 3}}, l3);
    auto prog = optimize_tf("mean_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(mean_test_nhwc)
{
    migraphx::program p;
    migraphx::literal l{migraphx::shape{migraphx::shape::int32_type, {2}}, {1, 2}};
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    auto l1 = p.add_instruction(migraphx::op::transpose{{0, 2, 3, 1}}, l0);
    migraphx::op::reduce_mean op{{1, 2}};
    auto l2 = p.add_instruction(op, l1);
    p.add_instruction(migraphx::op::squeeze{{1, 2}}, l2);
    auto prog = optimize_tf("mean_test_nhwc.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(mul_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 1, 1, 16}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 1, 1, 16}});

    p.add_instruction(migraphx::op::mul{}, l0, l1);
    auto prog = optimize_tf("mul_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(onehot_test)
{
    migraphx::program p;
    auto l0 = p.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {5}}, {1, 1, 1, 1, 1}});
    p.add_literal(2);
    p.add_literal(1.0f);
    p.add_literal(0.0f);
    auto l1 = p.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {2, 2}}, {1, 0, 0, 1}});
    int axis = 0;
    p.add_instruction(migraphx::op::gather{axis}, l1, l0);
    auto prog = optimize_tf("onehot_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(pack_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {2}});
    auto l2 = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {2}});
    std::vector<migraphx::instruction_ref> args{l0, l1, l2};
    std::vector<migraphx::instruction_ref> unsqueezed_args;
    int64_t axis = 1;

    std::transform(args.begin(),
                   args.end(),
                   std::back_inserter(unsqueezed_args),
                   [&](migraphx::instruction_ref arg) {
                       return p.add_instruction(migraphx::op::unsqueeze{{axis}}, arg);
                   });
    p.add_instruction(migraphx::op::concat{static_cast<int>(axis)}, unsqueezed_args);
    auto prog = optimize_tf("pack_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(pack_test_nhwc)
{
    migraphx::program p;
    auto l0  = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 1, 1}});
    auto lt0 = p.add_instruction(migraphx::op::transpose{{0, 2, 3, 1}}, l0);
    auto l1  = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 1, 1}});
    auto lt1 = p.add_instruction(migraphx::op::transpose{{0, 2, 3, 1}}, l1);
    auto l2  = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {1, 2, 1, 1}});
    auto lt2 = p.add_instruction(migraphx::op::transpose{{0, 2, 3, 1}}, l2);
    std::vector<migraphx::instruction_ref> args{lt0, lt1, lt2};
    std::vector<migraphx::instruction_ref> unsqueezed_args;
    int64_t nchw_axis = 3;

    std::transform(args.begin(),
                   args.end(),
                   std::back_inserter(unsqueezed_args),
                   [&](migraphx::instruction_ref arg) {
                       return p.add_instruction(migraphx::op::unsqueeze{{nchw_axis}}, arg);
                   });
    p.add_instruction(migraphx::op::concat{static_cast<int>(nchw_axis)}, unsqueezed_args);
    auto prog = optimize_tf("pack_test_nhwc.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(pooling_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    migraphx::op::pooling avg_pool_op{"average"};
    migraphx::op::pooling max_pool_op{"max"};
    avg_pool_op.padding_mode = migraphx::op::padding_mode_t::valid;
    max_pool_op.padding_mode = migraphx::op::padding_mode_t::valid;
    avg_pool_op.stride       = {2, 2};
    max_pool_op.stride       = {2, 2};
    avg_pool_op.lengths      = {2, 2};
    max_pool_op.lengths      = {2, 2};
    p.add_instruction(max_pool_op, l0);
    auto prog = optimize_tf("pooling_test.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(pow_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    p.add_instruction(migraphx::op::pow{}, l0, l1);
    auto prog = optimize_tf("pow_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(relu_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    p.add_instruction(migraphx::op::relu{}, l0);
    auto prog = optimize_tf("relu_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(relu6_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    p.add_instruction(migraphx::op::clip{6.0, 0.0}, l0);
    auto prog = optimize_tf("relu6_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(reshape_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {16}});
    migraphx::shape s0{migraphx::shape::int32_type, {4}};
    // in tf, the second arg is a literal that contains new dimensions
    p.add_literal(migraphx::literal{s0, {1, 1, 1, 16}});
    p.add_instruction(migraphx::op::reshape{{1, 1, 1, 16}}, l0);
    auto prog = optimize_tf("reshape_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(rsqrt_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    p.add_instruction(migraphx::op::rsqrt{}, l0);
    auto prog = optimize_tf("rsqrt_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(shape_test)
{
    migraphx::program p;
    p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    p.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {4}}, {1, 3, 16, 16}});
    auto prog = optimize_tf("shape_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(slice_test)
{
    migraphx::program p;
    std::size_t num_axes = 2;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 10}});
    migraphx::shape s0{migraphx::shape::int32_type, {num_axes}};
    p.add_literal(migraphx::literal{s0, {1, 0}});
    p.add_literal(migraphx::literal{s0, {2, -1}});

    migraphx::op::slice op;
    op.starts = {1, 0};
    op.ends   = {3, 10};
    op.axes   = std::vector<int64_t>(num_axes);
    std::iota(op.axes.begin(), op.axes.end(), 0);
    p.add_instruction(op, l0);
    auto prog = optimize_tf("slice_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(softmax_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3}});
    p.add_instruction(migraphx::op::softmax{1}, l0);
    auto prog = optimize_tf("softmax_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(split_test)
{
    migraphx::program p;
    std::vector<int64_t> axes{0, 1};
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 30}});
    p.add_literal(3); // num_splits
    p.add_literal(1); // split axis
    p.add_literal(1); // concat axis
    p.add_literal(1); // concat axis
    auto l1 = p.add_instruction(migraphx::op::slice{axes, {0, 0}, {5, 10}}, l0);
    auto l2 = p.add_instruction(migraphx::op::slice{axes, {0, 10}, {5, 20}}, l0);
    auto l3 = p.add_instruction(migraphx::op::slice{axes, {0, 20}, {5, 30}}, l0);
    p.add_instruction(migraphx::op::concat{1}, l1, l2);
    p.add_instruction(migraphx::op::concat{1}, l2, l3);

    auto prog = migraphx::parse_tf("split_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(split_test_one_output)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 30}});
    p.add_literal(1); // num_splits
    p.add_literal(1); // split axis
    p.add_instruction(migraphx::op::identity{}, l0);

    auto prog = migraphx::parse_tf("split_test_one_output.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(split_test_vector_as_input)
{
    migraphx::program p;
    std::vector<int64_t> axes{0, 1};
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 30}});
    // split sizes
    p.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {3}}, {4, 15, 11}});
    p.add_literal(1); // split axis
    p.add_literal(1); // concat axis
    p.add_literal(1); // concat axis
    auto l1 = p.add_instruction(migraphx::op::slice{axes, {0, 0}, {5, 4}}, l0);
    auto l2 = p.add_instruction(migraphx::op::slice{axes, {0, 4}, {5, 19}}, l0);
    auto l3 = p.add_instruction(migraphx::op::slice{axes, {0, 19}, {5, 30}}, l0);
    p.add_instruction(migraphx::op::concat{1}, l1, l2);
    p.add_instruction(migraphx::op::concat{1}, l2, l3);

    auto prog = migraphx::parse_tf("split_test_vector_as_input.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(sqdiff_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    p.add_instruction(migraphx::op::sqdiff{}, l0, l1);
    auto prog = optimize_tf("sqdiff_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(squeeze_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 1}});
    p.add_instruction(migraphx::op::squeeze{{0, 3}}, l0);
    auto prog = optimize_tf("squeeze_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(stopgradient_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    p.add_instruction(migraphx::op::identity{}, l0);
    auto prog = optimize_tf("stopgradient_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(stridedslice_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 10, 1, 1}});
    auto l1 = p.add_instruction(migraphx::op::transpose{{0, 2, 3, 1}}, l0);
    std::size_t num_axes = 4;
    migraphx::op::slice op;
    op.starts = {0, 0, 0, 0};
    op.ends   = {1, 1, 1, 5};
    op.axes   = std::vector<int64_t>(num_axes);
    std::iota(op.axes.begin(), op.axes.end(), 0);
    auto l2          = p.add_instruction(op, l1);
    auto shrink_axis = 1;
    p.add_instruction(migraphx::op::squeeze{{shrink_axis}}, l2);
    auto prog = optimize_tf("stridedslice_test.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(stridedslice_masks_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 10, 3, 3}});
    std::size_t num_axes = 4;
    migraphx::op::slice op;
    op.starts = {0, 1, 1, 0};
    op.ends   = {1, 3, 3, 10};
    op.axes   = std::vector<int64_t>(num_axes);
    std::iota(op.axes.begin(), op.axes.end(), 0);
    // add literals for starts, ends, and strides in tf (NHWC format)
    p.add_literal(migraphx::shape{migraphx::shape::int32_type, {4}}, std::vector<int>{0, 1, 1, 0});
    p.add_literal(migraphx::shape{migraphx::shape::int32_type, {4}}, std::vector<int>{0, 0, 0, 0});
    p.add_literal(migraphx::shape{migraphx::shape::int32_type, {4}}, std::vector<int>{1, 1, 1, 1});

    auto l1 = p.add_instruction(migraphx::op::transpose{{0, 2, 3, 1}}, l0);
    auto l2 = p.add_instruction(op, l1);
    p.add_instruction(migraphx::op::transpose{{0, 3, 1, 2}}, l2);
    auto prog = migraphx::parse_tf("stridedslice_masks_test.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(sub_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    p.add_instruction(migraphx::op::sub{}, l0, l1);
    auto prog = migraphx::parse_tf("sub_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(tanh_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    p.add_instruction(migraphx::op::sub{}, l0, l1);
    auto prog = migraphx::parse_tf("sub_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(transpose_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    migraphx::shape s0{migraphx::shape::int32_type, {4}};
    p.add_literal(migraphx::literal{s0, {0, 2, 3, 1}});
    p.add_instruction(migraphx::op::transpose{{0, 2, 3, 1}}, l0);
    auto prog = optimize_tf("transpose_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(variable_batch_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    p.add_instruction(migraphx::op::identity{}, l0);
    auto prog = optimize_tf("variable_batch_test.pb", false);

    EXPECT(p == prog);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
