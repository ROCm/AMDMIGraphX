/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <iostream>
#include <vector>
#include <unordered_map>
#include <migraphx/common.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/tf.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/reduce_mean.hpp>
#include <migraphx/op/pooling.hpp>

#include <migraphx/serialize.hpp>

#include "test.hpp"

migraphx::program
parse_tf(const std::string& name,
         bool is_nhwc,
         const std::unordered_map<std::string, std::vector<std::size_t>>& dim_params = {},
         const std::vector<std::string>& output_node_names                           = {})
{
    return migraphx::parse_tf(name,
                              migraphx::tf_options{is_nhwc, 1, dim_params, output_node_names});
}

migraphx::program optimize_tf(const std::string& name, bool is_nhwc)
{
    auto prog = migraphx::parse_tf(name, migraphx::tf_options{is_nhwc, 1});
    auto* mm  = prog.get_main_module();
    if(is_nhwc)
        migraphx::run_passes(*mm,
                             {migraphx::simplify_reshapes{},
                              migraphx::dead_code_elimination{},
                              migraphx::eliminate_identity{}});

    // remove the last return instruction
    auto last_ins = std::prev(mm->end());
    if(last_ins != mm->end())
    {
        if(last_ins->name() == "@return")
        {
            mm->remove_instruction(last_ins);
        }
    }
    return prog;
}

TEST_CASE(add_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    mm->add_instruction(migraphx::make_op("add"), l0, l1);
    auto prog = optimize_tf("add_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(addv2_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    mm->add_instruction(migraphx::make_op("add"), l0, l1);
    auto prog = optimize_tf("addv2_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(add_bcast_test)
{

    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::float_type, {2, 3}};
    auto l0 = mm->add_parameter("0", s0);
    auto l1 = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {2, 1}});
    auto l2 =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s0.lens()}}), l1);
    mm->add_instruction(migraphx::make_op("add"), l0, l2);
    auto prog = optimize_tf("add_bcast_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(argmax_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {4, 5, 6, 7}});
    mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {2}});
    auto ins = mm->add_instruction(migraphx::make_op("argmax", {{"axis", 2}}), l0);
    auto l1  = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), ins);
    mm->add_return({l1});
    auto prog = parse_tf("argmax_test.pb", false, {{"0", {4, 5, 6, 7}}});

    EXPECT(p == prog);
}

TEST_CASE(argmin_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {2}});
    auto ins = mm->add_instruction(migraphx::make_op("argmin", {{"axis", 2}}), l0);
    auto l1  = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), ins);
    mm->add_return({l1});
    auto prog = parse_tf("argmin_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(assert_less_equal_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::float_type, {2, 3}};
    auto l0 = mm->add_parameter("0", s0);
    auto l1 = mm->add_parameter("1", s0);
    migraphx::literal l{migraphx::shape{migraphx::shape::int32_type, {2}}, {0, 1}};
    auto l2 = mm->add_literal(l);
    mm->add_instruction(migraphx::make_op("add"), l0, l1);
    auto l3 = mm->add_instruction(migraphx::make_op("identity"), l0, l1);
    mm->add_instruction(migraphx::make_op("identity"), l3, l2);
    auto prog = optimize_tf("assert_less_equal_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(batchmatmul_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 8, 4}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 4, 8}});

    auto trans_l0 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), l0);
    auto trans_l1 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), l1);

    mm->add_instruction(migraphx::make_op("dot"), trans_l0, trans_l1);
    auto prog = optimize_tf("batchmatmul_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(batchnorm_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x    = mm->add_parameter("x", {migraphx::shape::float_type, {1, 32, 16, 16}});
    auto bias = mm->add_parameter("bias", {migraphx::shape::float_type, {32}});
    auto mean = mm->add_parameter("mean", {migraphx::shape::float_type, {32}});
    auto var  = mm->add_parameter("variance", {migraphx::shape::float_type, {32}});

    std::vector<float> scale_data(32, 1.0);
    auto scale = mm->add_literal(migraphx::shape{migraphx::shape::float_type, {32}}, scale_data);
    auto eps   = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {1e-4f}});

    auto usq_scale = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), scale);
    auto usq_bias  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), bias);
    auto usq_mean  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), mean);
    auto usq_var   = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), var);

    auto x_sub_mean = add_common_op(*mm, migraphx::make_op("sub"), {x, usq_mean});
    auto var_eps    = add_common_op(*mm, migraphx::make_op("add"), {usq_var, eps});
    auto rsqrt      = mm->add_instruction(migraphx::make_op("rsqrt"), var_eps);
    auto mul0       = add_common_op(*mm, migraphx::make_op("mul"), {usq_scale, rsqrt});
    auto r0         = add_common_op(*mm, migraphx::make_op("mul"), {x_sub_mean, mul0});
    add_common_op(*mm, migraphx::make_op("add"), {r0, usq_bias});

    auto prog = optimize_tf("batchnorm_test.pb", true);
    EXPECT(p == prog);
}

TEST_CASE(batchnorm_half_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x    = mm->add_parameter("x", {migraphx::shape::half_type, {1, 32, 16, 16}});
    auto bias = mm->add_parameter("bias", {migraphx::shape::float_type, {32}});
    auto mean = mm->add_parameter("mean", {migraphx::shape::float_type, {32}});
    auto var  = mm->add_parameter("variance", {migraphx::shape::float_type, {32}});

    std::vector<float> scale_data(32, 1.0);
    auto scale = mm->add_literal(migraphx::shape{migraphx::shape::float_type, {32}}, scale_data);
    auto eps   = mm->add_literal(migraphx::literal{migraphx::shape::half_type, {1e-4f}});

    auto usq_scale = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), scale);
    auto usq_bias  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), bias);
    auto usq_mean  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), mean);
    auto usq_var   = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), var);

    auto x_sub_mean = add_common_op(*mm, migraphx::make_op("sub"), {x, usq_mean});
    auto var_eps    = add_common_op(*mm, migraphx::make_op("add"), {usq_var, eps});
    auto rsqrt      = mm->add_instruction(migraphx::make_op("rsqrt"), var_eps);
    auto mul0       = add_common_op(*mm, migraphx::make_op("mul"), {usq_scale, rsqrt});
    auto r0         = add_common_op(*mm, migraphx::make_op("mul"), {x_sub_mean, mul0});
    add_common_op(*mm, migraphx::make_op("add"), {r0, usq_bias});

    auto prog = optimize_tf("batchnorm_half_test.pb", true);
    EXPECT(p == prog);
}

TEST_CASE(batchnormv3_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x    = mm->add_parameter("x", {migraphx::shape::float_type, {1, 32, 16, 16}});
    auto bias = mm->add_parameter("bias", {migraphx::shape::float_type, {32}});
    auto mean = mm->add_parameter("mean", {migraphx::shape::float_type, {32}});
    auto var  = mm->add_parameter("variance", {migraphx::shape::float_type, {32}});

    std::vector<float> scale_data(32, 1.0);
    auto scale = mm->add_literal(migraphx::shape{migraphx::shape::float_type, {32}}, scale_data);
    auto eps   = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {1e-6f}});

    auto usq_scale = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), scale);
    auto usq_bias  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), bias);
    auto usq_mean  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), mean);
    auto usq_var   = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), var);

    auto x_sub_mean = add_common_op(*mm, migraphx::make_op("sub"), {x, usq_mean});
    auto var_eps    = add_common_op(*mm, migraphx::make_op("add"), {usq_var, eps});
    auto rsqrt      = mm->add_instruction(migraphx::make_op("rsqrt"), var_eps);
    auto mul0       = add_common_op(*mm, migraphx::make_op("mul"), {usq_scale, rsqrt});
    auto r0         = add_common_op(*mm, migraphx::make_op("mul"), {x_sub_mean, mul0});
    add_common_op(*mm, migraphx::make_op("add"), {r0, usq_bias});

    auto prog = optimize_tf("batchnormv3_test.pb", true);
    EXPECT(p == prog);
}

TEST_CASE(biasadd_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::float_type, {1, 500, 1, 1}};
    uint64_t axis = 1;
    auto l0       = mm->add_parameter("0", s0);
    auto l1       = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {500}});
    auto l2       = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", l0->get_shape().lens()}}), l1);
    mm->add_instruction(migraphx::make_op("add"), l0, l2);
    auto prog = optimize_tf("biasadd_test.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(biasadd_scalar_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::float_type, {1, 1}};
    uint64_t axis = 1;
    auto l0       = mm->add_parameter("0", s0);
    auto l1       = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}, {0}}, {1.0}});
    auto l2 = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", l0->get_shape().lens()}}), l1);
    mm->add_instruction(migraphx::make_op("add"), l0, l2);
    auto prog = optimize_tf("biasadd_scalar_test.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(cast_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::int32_type)}}),
        l0);
    auto prog = optimize_tf("cast_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(concat_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {4, 7, 3}});
    auto l1 = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {4, 2, 3}});

    int axis = 1;
    // tf uses axis as the third input, and it is in int32 format
    // add the literal using a vector in order to set stride to 1 (like in tf parser)
    mm->add_literal(migraphx::shape{migraphx::shape::int32_type}, std::vector<int>{axis});

    mm->add_instruction(migraphx::make_op("concat", {{"axis", axis}}), l0, l1);
    auto prog = optimize_tf("concat_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(const_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    mm->add_literal(migraphx::shape{migraphx::shape::float_type}, std::vector<float>{1.0f});
    auto prog = optimize_tf("constant_test.pb", false);

    EXPECT(p == prog);
}

migraphx::program create_conv()
{
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    std::vector<float> weight_data(3 * 3 * 3 * 32);
    std::fill(weight_data.begin(), weight_data.end(), 1.0f);
    auto l1 =
        mm->add_literal(migraphx::shape{migraphx::shape::float_type, {3, 3, 3, 32}}, weight_data);

    migraphx::op::convolution op;
    op.padding  = {1, 1, 1, 1};
    op.stride   = {1, 1};
    op.dilation = {1, 1};
    auto l2 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {3, 2, 0, 1}}}), l1);
    mm->add_instruction(op, l0, l2);
    return p;
}

TEST_CASE(conv_test)
{
    migraphx::program p = create_conv();
    auto prog           = optimize_tf("conv_test.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(conv_add_test)
{
    migraphx::program p = create_conv();
    auto* mm            = p.get_main_module();
    auto l0             = std::prev(mm->end());
    mm->add_instruction(migraphx::make_op("add"), l0, l0);
    auto prog = optimize_tf("conv_add_test.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(conv_nchw_test)
{
    migraphx::program p = create_conv();
    auto prog           = optimize_tf("conv_nchw_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(conv_relu_test)
{
    migraphx::program p = create_conv();
    auto* mm            = p.get_main_module();
    auto l0             = std::prev(mm->end());
    mm->add_instruction(migraphx::make_op("relu"), l0);
    auto prog = optimize_tf("conv_relu_test.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(conv_relu6_test)
{
    migraphx::program p = create_conv();
    auto* mm            = p.get_main_module();
    std::vector<size_t> input_lens{1, 32, 16, 16};
    auto l0      = std::prev(mm->end());
    auto min_val = mm->add_literal(0.0f);
    auto max_val = mm->add_literal(6.0f);
    min_val = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                                  min_val);
    max_val = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                                  max_val);
    mm->add_instruction(migraphx::make_op("clip"), l0, min_val, max_val);
    auto prog = optimize_tf("conv_relu6_test.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(depthwiseconv_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    std::vector<float> weight_data(3 * 3 * 3 * 1);
    std::fill(weight_data.begin(), weight_data.end(), 1.0f);
    auto l1 =
        mm->add_literal(migraphx::shape{migraphx::shape::float_type, {3, 3, 3, 1}}, weight_data);

    migraphx::op::convolution op;
    op.padding  = {1, 1};
    op.stride   = {1, 1};
    op.dilation = {1, 1};
    op.group    = 3;
    auto l3 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {3, 2, 0, 1}}}), l1);
    auto l4 = mm->add_instruction(migraphx::make_op("contiguous"), l3);
    auto l5 = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {3, 1, 3, 3}}}), l4);
    mm->add_instruction(op, l0, l5);
    auto prog = optimize_tf("depthwise_conv_test.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(expanddims_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4}});
    mm->add_literal(0);
    mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 3, 4}}}), l0);
    auto prog = optimize_tf("expanddims_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(expanddims_test_neg_dims)
{
    // this check makes sure the pb parses negative dim value correctly
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4}});
    mm->add_literal(-1);
    mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 4, 1}}}), l0);
    auto prog = optimize_tf("expanddims_neg_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(gather_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 4}});
    auto l1 = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {2}}, {1, 1}});
    mm->add_literal(1);

    int axis = 1;
    mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), l0, l1);
    auto prog = optimize_tf("gather_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(identity_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    mm->add_instruction(migraphx::make_op("identity"), l0);
    auto prog = optimize_tf("identity_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(matmul_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {8, 4}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {4, 8}});

    auto trans_l0 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l0);
    auto trans_l1 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l1);

    mm->add_instruction(migraphx::make_op("dot"), trans_l0, trans_l1);
    auto prog = optimize_tf("matmul_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(mean_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::literal l{migraphx::shape{migraphx::shape::int32_type, {2}}, {2, 3}};
    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    mm->add_literal(l);
    mm->add_literal(l);
    migraphx::op::reduce_mean op{{2, 3}};
    mm->add_instruction(op, l0);
    auto l3 = mm->add_instruction(op, l0);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2, 3}}}), l3);
    auto prog = optimize_tf("mean_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(mean_test_nhwc)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::literal l{migraphx::shape{migraphx::shape::int32_type, {2}}, {1, 2}};
    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    auto l1 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l0);
    migraphx::op::reduce_mean op{{1, 2}};
    auto l2 = mm->add_instruction(op, l1);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1, 2}}}), l2);
    auto prog = optimize_tf("mean_test_nhwc.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(mul_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 1, 1, 16}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 1, 1, 16}});

    mm->add_instruction(migraphx::make_op("mul"), l0, l1);
    auto prog = optimize_tf("mul_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(multi_output_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    auto l1  = mm->add_instruction(migraphx::make_op("relu"), l0);
    auto l2  = mm->add_instruction(migraphx::make_op("tanh"), l0);
    mm->add_return({l1, l2});

    EXPECT(test::throws([&] { parse_tf("multi_output_test.pb", false, {}, {"relu", "relu6"}); }));
    auto prog = parse_tf("multi_output_test.pb", false, {}, {"relu", "tanh"});

    EXPECT(p == prog);
}

TEST_CASE(onehot_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {5}}, {1, 1, 1, 1, 1}});
    mm->add_literal(2);
    mm->add_literal(1.0f);
    mm->add_literal(0.0f);
    auto l1 = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {2, 2}}, {1, 0, 0, 1}});
    int axis = 0;
    mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), l1, l0);
    auto prog = optimize_tf("onehot_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(noop_test)
{
    migraphx::program p;
    auto prog = optimize_tf("noop_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(pack_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {2}});
    auto l2  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {2}});
    std::vector<migraphx::instruction_ref> args{l0, l1, l2};
    std::vector<migraphx::instruction_ref> unsqueezed_args;
    int64_t axis = 1;

    std::transform(
        args.begin(),
        args.end(),
        std::back_inserter(unsqueezed_args),
        [&](migraphx::instruction_ref arg) {
            return mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {axis}}}), arg);
        });
    mm->add_instruction(migraphx::make_op("concat", {{"axis", static_cast<int>(axis)}}),
                        unsqueezed_args);
    auto prog = optimize_tf("pack_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(pack_test_nhwc)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 1, 1}});
    auto lt0 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l0);
    auto l1 = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 1, 1}});
    auto lt1 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l1);
    auto l2 = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {1, 2, 1, 1}});
    auto lt2 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l2);
    std::vector<migraphx::instruction_ref> args{lt0, lt1, lt2};
    std::vector<migraphx::instruction_ref> unsqueezed_args;
    int64_t nchw_axis = 3;

    std::transform(args.begin(),
                   args.end(),
                   std::back_inserter(unsqueezed_args),
                   [&](migraphx::instruction_ref arg) {
                       return mm->add_instruction(
                           migraphx::make_op("unsqueeze", {{"axes", {nchw_axis}}}), arg);
                   });
    mm->add_instruction(migraphx::make_op("concat", {{"axis", static_cast<int>(nchw_axis)}}),
                        unsqueezed_args);
    auto prog = optimize_tf("pack_test_nhwc.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(pad_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 4}});
    std::vector<int> pad_literals{1, 1, 2, 2};
    std::vector<int> pads{1, 2, 1, 2};
    mm->add_literal(migraphx::shape{migraphx::shape::int32_type, {2, 2}}, pad_literals);

    mm->add_instruction(migraphx::make_op("pad", {{"pads", pads}}), l0);
    auto prog = optimize_tf("pad_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(pooling_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    migraphx::op::pooling avg_pool_op{migraphx::op::pooling_mode::average};
    migraphx::op::pooling max_pool_op{migraphx::op::pooling_mode::max};
    avg_pool_op.stride  = {2, 2};
    max_pool_op.stride  = {2, 2};
    avg_pool_op.lengths = {2, 2};
    max_pool_op.lengths = {2, 2};
    mm->add_instruction(avg_pool_op, l0);
    mm->add_instruction(max_pool_op, l0);
    auto prog = optimize_tf("pooling_test.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(pow_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    mm->add_instruction(migraphx::make_op("pow"), l0, l1);
    auto prog = optimize_tf("pow_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(relu_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    mm->add_instruction(migraphx::make_op("relu"), l0);
    auto prog = optimize_tf("relu_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(relu6_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    std::vector<size_t> input_lens{1, 3, 16, 16};
    auto l0      = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, input_lens});
    auto min_val = mm->add_literal(0.0f);
    auto max_val = mm->add_literal(6.0f);
    min_val = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                                  min_val);
    max_val = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                                  max_val);
    mm->add_instruction(migraphx::make_op("clip"), l0, min_val, max_val);
    auto prog = optimize_tf("relu6_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(relu6_half_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    std::vector<size_t> input_lens{1, 3, 16, 16};
    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::half_type, input_lens});
    auto min_val =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::half_type}, {0.0f}});
    auto max_val =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::half_type}, {6.0f}});
    min_val = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                                  min_val);
    max_val = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                                  max_val);
    mm->add_instruction(migraphx::make_op("clip"), l0, min_val, max_val);
    auto prog = optimize_tf("relu6_half_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(reshape_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {16}});
    migraphx::shape s0{migraphx::shape::int32_type, {4}};
    // in tf, the second arg is a literal that contains new dimensions
    mm->add_literal(migraphx::literal{s0, {1, 1, 1, 16}});
    mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 1, 16}}}), l0);
    auto prog = optimize_tf("reshape_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(rsqrt_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    mm->add_instruction(migraphx::make_op("rsqrt"), l0);
    auto prog = optimize_tf("rsqrt_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(shape_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {4}}, {1, 3, 16, 16}});
    auto prog = optimize_tf("shape_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(slice_test)
{
    migraphx::program p;

    auto* mm             = p.get_main_module();
    std::size_t num_axes = 2;
    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 10}});
    migraphx::shape s0{migraphx::shape::int32_type, {num_axes}};
    mm->add_literal(migraphx::literal{s0, {1, 0}});
    mm->add_literal(migraphx::literal{s0, {2, -1}});

    mm->add_instruction(
        migraphx::make_op("slice", {{"starts", {1, 0}}, {"ends", {3, 10}}, {"axes", {0, 1}}}), l0);
    auto prog = optimize_tf("slice_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(softmax_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3}});
    mm->add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), l0);
    auto prog = optimize_tf("softmax_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(split_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    std::vector<int64_t> axes{0, 1};
    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 30}});
    mm->add_literal(3); // num_splits
    mm->add_literal(1); // split axis
    mm->add_literal(1); // concat axis
    mm->add_literal(1); // concat axis
    auto l1 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", axes}, {"starts", {0, 0}}, {"ends", {5, 10}}}), l0);
    auto l2 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", axes}, {"starts", {0, 10}}, {"ends", {5, 20}}}), l0);
    auto l3 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", axes}, {"starts", {0, 20}}, {"ends", {5, 30}}}), l0);
    auto l4 = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), l1, l2);
    auto l5 = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), l2, l3);
    mm->add_return({l4, l5});
    auto prog = parse_tf("split_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(split_test_one_output)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 30}});
    mm->add_literal(1); // num_splits
    mm->add_literal(1); // split axis
    auto l1 = mm->add_instruction(migraphx::make_op("identity"), l0);
    mm->add_return({l1});
    auto prog = parse_tf("split_test_one_output.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(split_test_vector_as_input)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    std::vector<int64_t> axes{0, 1};
    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 30}});
    // split sizes
    mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {3}}, {4, 15, 11}});
    mm->add_literal(1); // split axis
    mm->add_literal(1); // concat axis
    mm->add_literal(1); // concat axis
    auto l1 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", axes}, {"starts", {0, 0}}, {"ends", {5, 4}}}), l0);
    auto l2 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", axes}, {"starts", {0, 4}}, {"ends", {5, 19}}}), l0);
    auto l3 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", axes}, {"starts", {0, 19}}, {"ends", {5, 30}}}), l0);
    auto l4 = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), l1, l2);
    auto l5 = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), l2, l3);
    mm->add_return({l4, l5});
    auto prog = parse_tf("split_test_vector_as_input.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(sqdiff_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    mm->add_instruction(migraphx::make_op("sqdiff"), l0, l1);
    auto prog = optimize_tf("sqdiff_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(squeeze_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 1}});
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 3}}}), l0);
    auto prog = optimize_tf("squeeze_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(stopgradient_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    mm->add_instruction(migraphx::make_op("identity"), l0);
    auto prog = optimize_tf("stopgradient_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(stridedslice_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 10, 1, 1}});
    auto l1 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l0);
    auto l2 = mm->add_instruction(
        migraphx::make_op(
            "slice", {{"starts", {0, 0, 0, 0}}, {"ends", {1, 1, 1, 5}}, {"axes", {0, 1, 2, 3}}}),
        l1);
    auto shrink_axis = 1;
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {shrink_axis}}}), l2);
    auto prog = optimize_tf("stridedslice_test.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(stridedslice_masks_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 10, 3, 3}});
    // add literals for starts, ends, and strides in tf (NHWC format)
    mm->add_literal(migraphx::shape{migraphx::shape::int32_type, {4}},
                    std::vector<int>{0, 1, 1, 0});
    mm->add_literal(migraphx::shape{migraphx::shape::int32_type, {4}},
                    std::vector<int>{0, 0, 0, 0});
    mm->add_literal(migraphx::shape{migraphx::shape::int32_type, {4}},
                    std::vector<int>{1, 1, 1, 1});

    auto l1 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l0);
    auto l2 = mm->add_instruction(
        migraphx::make_op(
            "slice", {{"starts", {0, 1, 1, 0}}, {"ends", {1, 3, 3, 10}}, {"axes", {0, 1, 2, 3}}}),
        l1);
    auto l3 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), l2);
    mm->add_return({l3});
    auto prog = parse_tf("stridedslice_masks_test.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(sub_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    auto l2  = mm->add_instruction(migraphx::make_op("sub"), l0, l1);
    mm->add_return({l2});
    auto prog = parse_tf("sub_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(tanh_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    auto l1  = mm->add_instruction(migraphx::make_op("tanh"), l0);
    mm->add_return({l1});
    auto prog = parse_tf("tanh_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(transpose_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    migraphx::shape s0{migraphx::shape::int32_type, {4}};
    mm->add_literal(migraphx::literal{s0, {0, 2, 3, 1}});
    mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l0);
    auto prog = optimize_tf("transpose_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(variable_batch_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    mm->add_instruction(migraphx::make_op("identity"), l0);
    auto prog = optimize_tf("variable_batch_test.pb", false);

    EXPECT(p == prog);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
