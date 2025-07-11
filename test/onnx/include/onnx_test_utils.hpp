/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef MIGRAPHX_GUARD_TEST_ONNX_ONNX_TEST_UTILS_HPP
#define MIGRAPHX_GUARD_TEST_ONNX_ONNX_TEST_UTILS_HPP

#include <onnx_test.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/common.hpp>
#include <migraphx/env.hpp>

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_CK_WORKAROUNDS);

inline migraphx::program
make_attention_program(const uint64_t batch,
                       const uint64_t sequence_length,
                       const uint64_t num_heads,
                       const uint64_t embedding_size,
                       bool bias_arg            = false,
                       bool key_pad_mask        = false,
                       const int64_t mask_value = -10000, // Default based on OnnxRT spec
                       const float scale_value  = std::numeric_limits<float>::quiet_NaN(),
                       const migraphx::shape::type_t dtype = migraphx::shape::float_type)
{
    // Also known as "head size" in literature
    uint64_t query_size = embedding_size / num_heads;
    // Assumes K=Q=V sizes for now (some cases V can be different)
    uint64_t weight_size = 3 * embedding_size;

    migraphx::program p;
    auto* mm = p.get_main_module();

    auto input = mm->add_parameter(
        "input", migraphx::shape{dtype, {batch, sequence_length, embedding_size}});
    auto weights =
        mm->add_parameter("weights", migraphx::shape{dtype, {embedding_size, weight_size}});
    auto bias = input;
    if(bias_arg)
    {
        bias = mm->add_parameter("bias", migraphx::shape{dtype, {3 * embedding_size}});
    }

    // Masking depeends on what type of masked used. Currently have key_pad raw masking here
    // Others down the line can be either left/right padded, or 3d masking (masking per batch)
    auto mask = input;
    if(key_pad_mask)
    {
        mask = mm->add_parameter(
            "mask_index", migraphx::shape{migraphx::shape::int32_type, {batch, sequence_length}});
    }

    // Input Projection
    auto unsq_weights =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), weights);
    auto bc_weights = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {batch, embedding_size, weight_size}}}),
        unsq_weights);
    auto pre_qkv = mm->add_instruction(migraphx::make_op("dot"), input, bc_weights);

    auto qkv_biased = pre_qkv;
    if(bias_arg)
    {
        auto bc_bias = mm->add_instruction(
            migraphx::make_op("multibroadcast",
                              {{"out_lens", {batch, sequence_length, weight_size}}}),
            bias);
        qkv_biased = mm->add_instruction(migraphx::make_op("add"), pre_qkv, bc_bias);
    }

    // Extract out QKV matrcies after input projection add in head dimension
    auto q = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {embedding_size}}}),
        qkv_biased);
    auto k = mm->add_instruction(
        migraphx::make_op(
            "slice", {{"axes", {2}}, {"starts", {embedding_size}}, {"ends", {embedding_size * 2}}}),
        qkv_biased);
    auto v = mm->add_instruction(
        migraphx::make_op(
            "slice",
            {{"axes", {2}}, {"starts", {embedding_size * 2}}, {"ends", {embedding_size * 3}}}),
        qkv_biased);

    auto attention_mask = input;
    if(key_pad_mask)
    {
        auto zero = mm->add_literal(migraphx::literal(migraphx::shape{dtype, {1}}, {0}));
        auto mask_lit =
            mm->add_literal(migraphx::literal(migraphx::shape{dtype, {1}}, {mask_value}));

        auto bc_pass = mm->add_instruction(
            migraphx::make_op("multibroadcast",
                              {{"out_lens", {batch, num_heads, sequence_length, sequence_length}}}),
            zero);
        auto bc_mask = mm->add_instruction(
            migraphx::make_op("multibroadcast",
                              {{"out_lens", {batch, num_heads, sequence_length, sequence_length}}}),
            mask_lit);

        // For raw masks we just need to mask out key value padding thus the 3d mask isn't needed
        // here.
        auto raw_mask = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {batch, 1, 1, sequence_length}}}), mask);
        raw_mask = mm->add_instruction(
            migraphx::make_op("multibroadcast",
                              {{"out_lens", {batch, num_heads, sequence_length, sequence_length}}}),
            raw_mask);
        raw_mask = mm->add_instruction(
            migraphx::make_op("reshape",
                              {{"dims", {batch, num_heads, sequence_length, sequence_length}}}),
            raw_mask);

        // Reuse "0" broadcasted converted to int32 to check if input mask is greater than 0 for
        // where condition
        auto in_pass = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), bc_pass);
        auto in_bool = mm->add_instruction(migraphx::make_op("equal"), raw_mask, in_pass);
        // Need this for mlir to allow us to use "Where"
        in_bool = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), in_bool);
        attention_mask = mm->add_instruction(migraphx::make_op("where"), in_bool, bc_mask, bc_pass);
    }

    migraphx::instruction_ref scale;
    if(not std::isnan(scale_value))
    { // No Need for sqrt now
        scale = mm->add_literal(migraphx::literal(migraphx::shape{dtype, {1}, {0}}, {scale_value}));
    }
    else
    {
        auto sl_literal =
            mm->add_literal(migraphx::literal(migraphx::shape{dtype, {1}, {0}}, {query_size}));
        scale = mm->add_instruction(migraphx::make_op("sqrt"), sl_literal);
        scale = mm->add_instruction(migraphx::make_op("recip"), scale);
    }

    q = mm->add_instruction(
        migraphx::make_op("reshape", {{"dims", {batch, sequence_length, num_heads, query_size}}}),
        q);
    k = mm->add_instruction(
        migraphx::make_op("reshape", {{"dims", {batch, sequence_length, num_heads, query_size}}}),
        k);
    v = mm->add_instruction(
        migraphx::make_op("reshape", {{"dims", {batch, sequence_length, num_heads, query_size}}}),
        v);

    // Get this into (batch, head, sequence_length, query_size)
    auto q_rsh =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), q);
    auto k_rsh =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), k);
    auto v_rsh =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), v);

    // Block for scale dot attention
    auto k_trans =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k_rsh);
    auto qk = mm->add_instruction(migraphx::make_op("dot"), q_rsh, k_trans);

    // Apply mask before scale and softmax
    if(key_pad_mask)
    {
        qk = mm->add_instruction(migraphx::make_op("add"), qk, attention_mask);
    }

    // Scale before softmax
    auto bc_scale = mm->add_instruction(
        migraphx::make_op("multibroadcast",
                          {{"out_lens", {batch, num_heads, sequence_length, sequence_length}}}),
        scale);
    auto qk_scaled = mm->add_instruction(migraphx::make_op("mul"), qk, bc_scale);

    auto smax_score = mm->add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), qk_scaled);
    auto score      = mm->add_instruction(migraphx::make_op("dot"), smax_score, v_rsh);

    // Get back into final shape of batch, sequence_length, embedding_size
    score =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), score);
    mm->add_instruction(
        migraphx::make_op("reshape", {{"dims", {batch, sequence_length, embedding_size}}}), score);

    return p;
}

inline void add_celu_instruction(migraphx::module* mm, const migraphx::shape& s, float alpha)
{
    auto x                 = mm->add_parameter("x", s);
    const auto& input_lens = s.lens();
    const auto& input_type = s.type();
    auto zero_lit =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                            mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {0.}}));
    auto one_lit =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                            mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {1.}}));
    auto alpha_lit = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
        mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {alpha}}));
    auto linear_part = mm->add_instruction(migraphx::make_op("max"), zero_lit, x);
    auto divi        = mm->add_instruction(migraphx::make_op("div"), x, alpha_lit);
    auto expo        = mm->add_instruction(migraphx::make_op("exp"), divi);
    auto sub         = mm->add_instruction(migraphx::make_op("sub"), expo, one_lit);
    auto mul         = mm->add_instruction(migraphx::make_op("mul"), alpha_lit, sub);
    auto exp_part    = mm->add_instruction(migraphx::make_op("min"), zero_lit, mul);
    mm->add_instruction(migraphx::make_op("add"), linear_part, exp_part);
}

inline std::vector<double> make_r_eyelike(size_t num_rows, size_t num_cols, size_t k)
{
    std::vector<double> eyelike_mat(num_rows * num_cols, 0);
    for(size_t i = 0; i < num_rows; ++i)
    {
        if(i + k < num_cols)
            eyelike_mat[(num_cols + 1) * i + k] = 1.;
    }
    return eyelike_mat;
}

inline migraphx::program make_dequantizelinear_axis_prog()
{
    migraphx::program p;
    std::vector<size_t> input_lens{1, 1, 5, 1};
    int axis      = 2;
    auto* mm      = p.get_main_module();
    auto l0       = mm->add_parameter("0", {migraphx::shape::int8_type, input_lens});
    auto l1       = mm->add_parameter("1", {migraphx::shape::float_type, {5}});
    auto l2       = mm->add_parameter("2", {migraphx::shape::int8_type, {5}});
    auto l1_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", input_lens}}), l1);
    auto l2_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", input_lens}}), l2);
    l2_bcast = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
        l2_bcast);
    l0 = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
        l0);
    auto sub = mm->add_instruction(migraphx::make_op("sub"), l0, l2_bcast);

    mm->add_instruction(migraphx::make_op("mul"), sub, l1_bcast);
    return p;
}

inline migraphx::program create_external_data_prog()
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s(migraphx::shape::float_type, {1, 1, 224, 224});
    migraphx::shape s2(migraphx::shape::float_type, {10, 1, 11, 11});
    std::vector<float> weight_data(1210, 1);
    std::vector<float> bias_data(10, 1);
    auto bias = mm->add_literal(migraphx::literal({migraphx::shape::float_type, {10}}, bias_data));
    auto weights = mm->add_literal(migraphx::literal(s2, weight_data));
    auto param   = mm->add_parameter("input", s);
    auto conv    = mm->add_instruction(
        migraphx::make_op("convolution", {{"padding", {0, 0, 0, 0}}}), param, weights);
    auto bias_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 10, 214, 214}}}), bias);
    mm->add_instruction(migraphx::make_op("add"), conv, bias_bcast);
    return p;
}

inline migraphx::program make_group_norm(
    const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& scale_dims,
    const std::vector<int64_t>& bias_dims,
    const std::vector<int64_t>& reshape_dims,
    const std::vector<int64_t>& reduce_axes,
    const float eps_value                                         = 1e-5f,
    const migraphx::shape::type_t dtype                           = migraphx::shape::float_type,
    const std::pair<std::string, migraphx::shape::type_t>& param1 = {"scale",
                                                                     migraphx::shape::float_type},
    const std::pair<std::string, migraphx::shape::type_t>& param2 = {"bias",
                                                                     migraphx::shape::float_type})
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x     = mm->add_parameter("x", {dtype, input_dims});
    auto scale = mm->add_parameter(param1.first, {param1.second, scale_dims});
    auto bias  = mm->add_parameter(param2.first, {param2.second, bias_dims});

    auto x_dims = x->get_shape().lens();

    auto eps = mm->add_literal(migraphx::literal{dtype, {eps_value}});

    if(scale->get_shape().type() != dtype)
        scale = mm->add_instruction(migraphx::make_op("convert", {{"target_type", dtype}}), scale);
    if(bias->get_shape().type() != dtype)
        bias = mm->add_instruction(migraphx::make_op("convert", {{"target_type", dtype}}), bias);

    auto x_reshaped =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", reshape_dims}}), x);
    auto mean =
        mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", reduce_axes}}), x_reshaped);
    auto x_sub_mean    = add_common_op(*mm, migraphx::make_op("sub"), {x_reshaped, mean});
    auto x_sqdiff_mean = add_common_op(*mm, migraphx::make_op("sqdiff"), {x_reshaped, mean});
    auto var     = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", reduce_axes}}),
                                   x_sqdiff_mean);
    auto var_eps = add_common_op(*mm, migraphx::make_op("add"), {var, eps});
    auto rsqrt   = mm->add_instruction(migraphx::make_op("rsqrt"), {var_eps});
    auto result  = add_common_op(*mm, migraphx::make_op("mul"), {x_sub_mean, rsqrt});
    auto result_reshaped =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", x_dims}}), result);
    auto scale_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", x_dims}}), scale);
    auto bias_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", x_dims}}), bias);
    auto scaled = mm->add_instruction(migraphx::make_op("mul"), {result_reshaped, scale_bcast});
    mm->add_instruction(migraphx::make_op("add"), {scaled, bias_bcast});

    return p;
}

inline migraphx::program
make_layer_norm(const std::vector<int64_t>& input_shape,
                const std::vector<int64_t>& scale_bias_shape,
                const std::vector<int64_t>& reduce_axes,
                size_t skipped_axis,
                bool skip_bias                      = false,
                const float eps_value               = 1e-5f,
                const migraphx::shape::type_t dtype = migraphx::shape::float_type)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto x     = mm->add_parameter("x", {dtype, input_shape});
    auto scale = mm->add_parameter("scale", {dtype, scale_bias_shape});
    migraphx::instruction_ref bias;
    if(not skip_bias)
    {
        bias = mm->add_parameter("bias", {dtype, scale_bias_shape});
    }
    auto eps  = mm->add_literal(migraphx::literal{dtype, {eps_value}});
    auto mean = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", reduce_axes}}), x);
    auto x_sub_mean    = add_common_op(*mm, migraphx::make_op("sub"), {x, mean});
    auto x_sqdiff_mean = add_common_op(*mm, migraphx::make_op("sqdiff"), {x, mean});
    auto var     = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", reduce_axes}}),
                                   x_sqdiff_mean);
    auto var_eps = add_common_op(*mm, migraphx::make_op("add"), {var, eps});
    auto rsqrt   = mm->add_instruction(migraphx::make_op("rsqrt"), {var_eps});
    auto result  = add_common_op(*mm, migraphx::make_op("mul"), {x_sub_mean, rsqrt});
    migraphx::instruction_ref scale_bcast = scale;
    migraphx::instruction_ref bias_bcast  = bias;
    if(skipped_axis > 0)
    {
        if(scale_bias_shape.size() == 1)
        {
            scale_bcast = mm->add_instruction(
                migraphx::make_op("broadcast", {{"axis", skipped_axis}, {"out_lens", input_shape}}),
                scale);
        }

        if(not skip_bias)
        {
            if(scale_bias_shape.size() == 1)
            {
                bias_bcast = mm->add_instruction(
                    migraphx::make_op("broadcast",
                                      {{"axis", skipped_axis}, {"out_lens", input_shape}}),
                    bias);
            }
        }
    }
    auto scaled = add_common_op(*mm, migraphx::make_op("mul"), {result, scale_bcast});
    if(not skip_bias)
    {
        add_common_op(*mm, migraphx::make_op("add"), {scaled, bias_bcast});
    }
    return p;
}

inline migraphx::program
make_skip_layer_norm(const std::vector<int64_t>& input_dims,
                     const std::vector<int64_t>& skip_dims,
                     const std::vector<int64_t>& gamma_dims,
                     const std::vector<int64_t>& beta_dims,
                     const std::vector<int64_t>& bias_dims,
                     const int axes,
                     const float eps_value               = 1e-5f,
                     const migraphx::shape::type_t dtype = migraphx::shape::half_type)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto x     = mm->add_parameter("x", {dtype, input_dims});
    auto skip  = mm->add_parameter("skip", {dtype, skip_dims});
    auto scale = mm->add_parameter("gamma", {dtype, gamma_dims});

    migraphx::instruction_ref beta;
    migraphx::instruction_ref bias;
    if(not beta_dims.empty())
    {
        beta = mm->add_parameter("beta", {dtype, beta_dims});
    }

    if(not bias_dims.empty())
    {
        bias = mm->add_parameter("bias", {dtype, bias_dims});
    }

    x = add_common_op(*mm, migraphx::make_op("add"), {x, skip});
    if(not bias_dims.empty())
        x = add_common_op(*mm, migraphx::make_op("add"), {x, bias});

    auto eps  = mm->add_literal(migraphx::literal{migraphx::shape{dtype}, {eps_value}});
    auto mean = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {axes}}}), x);
    auto x_sqdiff_mean = add_common_op(*mm, migraphx::make_op("sqdiff"), {x, mean});
    auto var =
        mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {axes}}}), x_sqdiff_mean);

    auto var_eps = add_common_op(*mm, migraphx::make_op("add"), {var, eps});
    auto rsqrt   = mm->add_instruction(migraphx::make_op("rsqrt"), {var_eps});

    auto x_sub_mean = add_common_op(*mm, migraphx::make_op("sub"), {x, mean});
    auto result     = add_common_op(*mm, migraphx::make_op("mul"), {x_sub_mean, rsqrt});
    result          = add_common_op(*mm, migraphx::make_op("mul"), {result, scale});

    if(not beta_dims.empty())
    {
        result = add_common_op(*mm, migraphx::make_op("add"), {result, beta});
    }

    return p;
}

inline migraphx::program
make_simplified_layer_norm(const std::vector<int64_t>& input_shape,
                           const std::vector<int64_t>& skip_shape,
                           const std::vector<int64_t>& scale_shape,
                           const int axis,
                           const float eps_value               = 1e-5f,
                           const migraphx::shape::type_t dtype = migraphx::shape::half_type)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", {dtype, input_shape});
    migraphx::instruction_ref skip;
    migraphx::instruction_ref scale;
    if(skip_shape.empty())
    {
        scale = mm->add_parameter("scale", {dtype, scale_shape});
    }
    else
    {
        skip  = mm->add_parameter("skip", {dtype, skip_shape});
        scale = mm->add_parameter("gamma", {dtype, scale_shape});
        x     = add_common_op(*mm, migraphx::make_op("add"), {x, skip});
    }

    auto eps = mm->add_literal(migraphx::literal{dtype, {eps_value}});

    auto float_x = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), x);
    auto x_sq      = add_common_op(*mm, migraphx::make_op("mul"), {float_x, float_x});
    auto norm_axis = axis < 0 ? axis + x->get_shape().lens().size() : axis;
    auto rms = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {norm_axis}}}), x_sq);
    rms         = mm->add_instruction(migraphx::make_op("convert", {{"target_type", dtype}}), rms);
    rms      = add_common_op(*mm, migraphx::make_op("add"), {rms, eps});
    auto rrms   = mm->add_instruction(migraphx::make_op("rsqrt"), {rms});
    auto result = add_common_op(*mm, migraphx::make_op("mul"), {x, rrms});
    result      = add_common_op(*mm, migraphx::make_op("mul"), {result, scale});
    return p;
}

inline void mvn_n_rank_test(std::vector<int64_t> axes,
                            std::vector<size_t> input_shape,
                            const migraphx::program& prog)
{
    using migraphx::make_op;

    migraphx::program p;
    auto* mm = p.get_main_module();

    auto data = mm->add_parameter("data", {migraphx::shape::float_type, std::move(input_shape)});
    auto data_mean         = mm->add_instruction(make_op("reduce_mean", {{"axes", axes}}), data);
    auto data_mean_squared = add_common_op(*mm, make_op("mul"), {data_mean, data_mean});

    auto data_squared = add_common_op(*mm, make_op("mul"), {data, data});
    auto data_squared_mean =
        mm->add_instruction(make_op("reduce_mean", {{"axes", axes}}), data_squared);

    auto mean_sub = add_common_op(*mm, make_op("sub"), {data_squared_mean, data_mean_squared});
    auto std      = add_common_op(*mm, make_op("sqrt"), {mean_sub});

    auto dividend = add_common_op(*mm, make_op("sub"), {data, data_mean});
    auto epsilon  = mm->add_literal({migraphx::shape::float_type, {1e-9}});
    auto divisor  = add_common_op(*mm, make_op("add"), {std, epsilon});
    add_common_op(*mm, make_op("div"), {dividend, divisor});

    EXPECT(p == prog);
}

inline migraphx::instruction_ref insert_quantizelinear_clip(migraphx::module& m,
                                                            const migraphx::instruction_ref ins,
                                                            const migraphx::instruction_ref round,
                                                            const migraphx::shape s,
                                                            const int64_t min_quant,
                                                            const int64_t max_quant)
{
    migraphx::instruction_ref min_arg;
    migraphx::instruction_ref max_arg;
    if(migraphx::enabled(MIGRAPHX_ENABLE_CK_WORKAROUNDS{}))
    {
        std::vector<int> min_data(s.elements(), min_quant);
        std::vector<int> max_data(s.elements(), max_quant);
        min_arg = m.add_literal(migraphx::literal(s, min_data));
        max_arg = m.add_literal(migraphx::literal(s, max_data));
    }
    else
    {
        min_arg = m.add_literal(migraphx::literal{migraphx::shape{s.type()}, {min_quant}});
        max_arg = m.add_literal(migraphx::literal{migraphx::shape{s.type()}, {max_quant}});
    }

    return migraphx::insert_common_op(m, ins, migraphx::make_op("clip"), {round, min_arg, max_arg});
}

inline migraphx::program make_quantizelinear_axis_prog()
{
    migraphx::program p;
    std::vector<size_t> input_lens{1, 1, 5, 1};
    int axis = 2;
    auto* mm = p.get_main_module();

    auto l0       = mm->add_parameter("0", {migraphx::shape::float_type, input_lens});
    auto l1       = mm->add_parameter("1", {migraphx::shape::float_type, {5}});
    auto l2       = mm->add_parameter("2", {migraphx::shape::int8_type, {5}});
    auto l1_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", input_lens}}), l1);

    auto div      = mm->add_instruction(migraphx::make_op("div"), l0, l1_bcast);
    auto round    = mm->add_instruction(migraphx::make_op("nearbyint"), div);
    auto l2_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", input_lens}}), l2);
    l2_bcast = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
        l2_bcast);
    auto add  = mm->add_instruction(migraphx::make_op("add"), round, l2_bcast);
    auto s    = round->get_shape();
    auto clip = insert_quantizelinear_clip(*mm, div, add, s, -128, 127);
    mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::int8_type)}}),
        clip);
    return p;
}

/*  Parsed IR equivalent of create_upsample_linear_prog()
module: "main"
@0 = @literal{ ... } -> float_type, {1, 1, 4, 4}, {16, 16, 4, 1}
@1 = @literal{ ... } -> float_type, {2, 1, 4, 4}, {16, 16, 4, 1}
@2 = @literal{ ... } -> int32_type, {4, 1, 4, 4}, {16, 16, 4, 1}
X = @param:X -> float_type, {1, 1, 2, 2}, {4, 4, 2, 1}
@4 = @literal{1, 1, 2, 2} -> float_type, {4}, {1}
@5 = undefined -> float_type, {}, {}
@6 = reshape[dims={4}](X) -> float_type, {4}, {1}
@7 = gather[axis=0](@6,@2) -> float_type, {4, 1, 4, 4}, {16, 16, 4, 1}
@8 = slice[axes={0},starts={0},ends={2}](@7) -> float_type, {2, 1, 4, 4}, {16, 16, 4, 1}
@9 = slice[axes={0},starts={2},ends={4}](@7) -> float_type, {2, 1, 4, 4}, {16, 16, 4, 1}
@10 = sub(@9,@8) -> float_type, {2, 1, 4, 4}, {16, 16, 4, 1}
@11 = mul(@10,@1) -> float_type, {2, 1, 4, 4}, {16, 16, 4, 1}
@12 = add(@11,@8) -> float_type, {2, 1, 4, 4}, {16, 16, 4, 1}
@13 = slice[axes={0},starts={0},ends={1}](@12) -> float_type, {1, 1, 4, 4}, {16, 16, 4, 1}
@14 = slice[axes={0},starts={1},ends={2}](@12) -> float_type, {1, 1, 4, 4}, {16, 16, 4, 1}
@15 = sub(@14,@13) -> float_type, {1, 1, 4, 4}, {16, 16, 4, 1}
@16 = mul(@15,@0) -> float_type, {1, 1, 4, 4}, {16, 16, 4, 1}
@17 = add(@16,@13) -> float_type, {1, 1, 4, 4}, {16, 16, 4, 1}
@18 = @return(@17)
*/

inline auto create_upsample_linear_prog()
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape ss{migraphx::shape::float_type, {4}};
    std::vector<float> ds = {1, 1, 2, 2};
    mm->add_literal(migraphx::literal(ss, ds));

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto x = mm->add_parameter("X", sx);
    migraphx::shape s_ind{migraphx::shape::int32_type, {4, 1, 4, 4}};

    std::vector<int> d_ind = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 3, 0, 0, 0, 1, 2, 2,
                              2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1,
                              2, 3, 3, 3, 0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3};

    auto l_ind = mm->add_literal(migraphx::literal(s_ind, d_ind));

    migraphx::shape s2{migraphx::shape::float_type, {2, 1, 4, 4}};

    std::vector<float> d2 = {-0.25, 0.25, 0.75, 0.25, -0.25, 0.25, 0.75, 0.25,
                             -0.25, 0.25, 0.75, 0.25, -0.25, 0.25, 0.75, 0.25,
                             -0.25, 0.25, 0.75, 0.25, -0.25, 0.25, 0.75, 0.25,
                             -0.25, 0.25, 0.75, 0.25, -0.25, 0.25, 0.75, 0.25};

    auto l2 = mm->add_literal(migraphx::literal(s2, d2));

    migraphx::shape s1{migraphx::shape::float_type, {1, 1, 4, 4}};

    std::vector<float> d1 = {-0.25,
                             -0.25,
                             -0.25,
                             -0.25,
                             0.25,
                             0.25,
                             0.25,
                             0.25,
                             0.75,
                             0.75,
                             0.75,
                             0.75,
                             0.25,
                             0.25,
                             0.25,
                             0.25};

    auto l1 = mm->add_literal(migraphx::literal(s1, d1));

    mm->add_instruction(migraphx::make_op("undefined"));
    auto rsp   = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), x);
    auto data  = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), rsp, l_ind);
    auto slc20 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {2}}}), data);
    auto slc21 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {4}}}), data);
    auto diff2 = mm->add_instruction(migraphx::make_op("sub"), slc21, slc20);
    auto mul2  = mm->add_instruction(migraphx::make_op("mul"), diff2, l2);
    auto add2  = mm->add_instruction(migraphx::make_op("add"), mul2, slc20);
    auto slc10 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), add2);
    auto slc11 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), add2);
    auto diff1 = mm->add_instruction(migraphx::make_op("sub"), slc11, slc10);
    auto mul1  = mm->add_instruction(migraphx::make_op("mul"), diff1, l1);
    auto add1  = mm->add_instruction(migraphx::make_op("add"), mul1, slc10);
    mm->add_return({add1});

    return p;
}

// the ScatterElements op has 3 reduction modes, which map to separate reference ops
inline void scatter_test_base(const std::string& reduction, int axis, const std::string& onnx_file)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0 = mm->add_parameter("data", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto l1 =
        mm->add_parameter("indices", migraphx::shape{migraphx::shape::int32_type, {2, 3, 4, 5}});
    auto l2 =
        mm->add_parameter("update", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto r = mm->add_instruction(
        migraphx::make_op("scatter_" + reduction, {{"axis", axis}}), l0, l1, l2);
    mm->add_return({r});
    auto prog = read_onnx(onnx_file);

    EXPECT(p == prog);
}

#endif
