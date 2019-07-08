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

TEST_CASE(concat_test)
{
    migraphx::program p;

    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {4, 7, 3}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {4, 2, 3}});

    int axis = 1;
    // tf uses axis as the third input, and it is in int32 format
    // add the literal using a vector in order to set stride to 1 (like in tf parser)
    p.add_literal(migraphx::shape{migraphx::shape::int32_type}, std::vector<int>{axis});

    p.add_instruction(migraphx::op::concat{static_cast<std::size_t>(axis)}, l0, l1);
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
    p.add_instruction(migraphx::op::reshape{{1, 2, 3, 4}}, l0);
    auto prog = optimize_tf("expanddims_test.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(expanddims_test_neg_dims)
{
    // this check makes sure the pb parses negative dim value correctly
    migraphx::program p;

    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4}});
    p.add_instruction(migraphx::op::reshape{{2, 3, 4, 1}}, l0);
    auto prog = optimize_tf("expanddims_neg_test.pb", true);

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
    migraphx::op::pooling op;
    op.lengths = {16, 16};
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
    migraphx::op::pooling op;
    op.lengths = {16, 16};
    auto l3    = p.add_instruction(op, l0);
    p.add_instruction(migraphx::op::squeeze{{2, 3}}, l3);
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
    p.add_instruction(migraphx::op::concat{static_cast<size_t>(axis)}, unsqueezed_args);
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
    p.add_instruction(migraphx::op::concat{static_cast<size_t>(nchw_axis)}, unsqueezed_args);
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
    // p.add_instruction(avg_pool_op, l0);
    auto prog = optimize_tf("pooling_test.pb", true);

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

TEST_CASE(softmax_test)
{
    migraphx::program p;
    auto l0   = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3}});
    auto dims = l0->get_shape().lens();
    auto r    = p.add_instruction(migraphx::op::reshape{{long(dims[0]), long(dims[1]), 1, 1}}, l0);
    auto s    = p.add_instruction(migraphx::op::softmax{}, r);
    p.add_instruction(migraphx::op::reshape{{long(dims[0]), long(dims[1])}}, s);
    auto prog = optimize_tf("softmax_test.pb", false);

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

TEST_CASE(stridedslice_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 10, 1, 1}});
    std::size_t num_axes = 4;
    migraphx::op::slice op;
    op.starts = {0, 0, 0, 0};
    op.ends   = {1, 1, 1, 5};
    op.axes   = std::vector<int64_t>(num_axes);
    std::iota(op.axes.begin(), op.axes.end(), 0);
    auto l1          = p.add_instruction(op, l0);
    auto shrink_axis = 1;
    p.add_instruction(migraphx::op::squeeze{{shrink_axis}}, l1);
    auto prog = optimize_tf("stridedslice_test.pb", true);

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

int main(int argc, const char* argv[]) { test::run(argc, argv); }
