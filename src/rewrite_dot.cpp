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
 *
 */
#include <migraphx/rewrite_dot.hpp>
#include <migraphx/module.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/op/common.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

MIGRAPHX_PRED_MATCHER(conv_1x1, instruction_ref ins)
{
    if(ins->name() != "convolution")
        return false;
    auto v = ins->get_operator().to_value();
    if(v.at("group").to<int>() != 1)
        return false;
    if(not all_of(v.at("stride"), [](const value& x) { return x.to<std::size_t>() == 1; }))
        return false;
    if(not all_of(v.at("padding"), [](const value& x) { return x.to<std::size_t>() == 0; }))
        return false;
    if(not all_of(v.at("dilation"), [](const value& x) { return x.to<std::size_t>() == 1; }))
        return false;
    auto w = ins->inputs().at(1)->get_shape();
    return std::all_of(w.lens().begin() + 2, w.lens().end(), [](std::size_t i) { return i == 1; });
}

MIGRAPHX_PRED_MATCHER(depthwise_conv_1x1, instruction_ref ins)
{
    if(ins->name() != "convolution")
        return false;
    auto v     = ins->get_operator().to_value();
    auto group = v.at("group").to<int>();
    if(group == 1)
        return false;
    if(not all_of(v.at("stride"), [](const value& x) { return x.to<std::size_t>() == 1; }))
        return false;
    if(not all_of(v.at("padding"), [](const value& x) { return x.to<std::size_t>() == 0; }))
        return false;
    if(not all_of(v.at("dilation"), [](const value& x) { return x.to<std::size_t>() == 1; }))
        return false;
    auto w = ins->inputs().at(1)->get_shape();
    // Check 1x1 kernel
    if(not std::all_of(w.lens().begin() + 2, w.lens().end(), [](std::size_t i) { return i == 1; }))
        return false;
    // Check depthwise: group == input channels
    auto x_shape = ins->inputs().at(0)->get_shape();
    if(static_cast<std::size_t>(group) != x_shape.lens().at(1))
        return false;
    // Check multiplier == 1: output channels == group
    return static_cast<std::size_t>(group) == w.lens().at(0);
}

MIGRAPHX_PRED_MATCHER(conv_c1_1x1, instruction_ref ins)
{
    if(ins->name() != "convolution")
        return false;
    auto v = ins->get_operator().to_value();
    if(v.at("group").to<int>() != 1)
        return false;
    if(not all_of(v.at("stride"), [](const value& x) { return x.to<std::size_t>() == 1; }))
        return false;
    if(not all_of(v.at("padding"), [](const value& x) { return x.to<std::size_t>() == 0; }))
        return false;
    if(not all_of(v.at("dilation"), [](const value& x) { return x.to<std::size_t>() == 1; }))
        return false;
    // Check C_in == 1
    auto x_shape = ins->inputs().at(0)->get_shape();
    if(x_shape.lens().at(1) != 1)
        return false;
    auto w = ins->inputs().at(1)->get_shape();
    // Check 1x1 kernel
    return std::all_of(w.lens().begin() + 2, w.lens().end(), [](std::size_t i) { return i == 1; });
}

// Matches convolution where each filter has 1 input channel:
// - depthwise (group == C_in, multiplier == 1) with non-1x1 kernel
// - group=1 with C_in=1 and non-1x1 kernel
MIGRAPHX_PRED_MATCHER(conv_channelwise, instruction_ref ins)
{
    if(ins->name() != "convolution")
        return false;
    auto v     = ins->get_operator().to_value();
    auto group = v.at("group").to<int>();
    if(not all_of(v.at("stride"), [](const value& x) { return x.to<std::size_t>() == 1; }))
        return false;
    if(not all_of(v.at("padding"), [](const value& x) { return x.to<std::size_t>() == 0; }))
        return false;
    if(not all_of(v.at("dilation"), [](const value& x) { return x.to<std::size_t>() == 1; }))
        return false;
    auto w = ins->inputs().at(1)->get_shape();
    // Weight must have 1 input channel per group
    if(w.lens().at(1) != 1)
        return false;
    // Exclude 1x1 kernel (handled by 1x1 matchers)
    if(std::all_of(w.lens().begin() + 2, w.lens().end(), [](std::size_t i) { return i == 1; }))
        return false;
    auto x_shape = ins->inputs().at(0)->get_shape();
    auto c_in    = x_shape.lens().at(1);
    auto c_out   = w.lens().at(0);
    if(group == 1)
        return c_in == 1;
    // group > 1: depthwise with multiplier == 1
    return static_cast<std::size_t>(group) == c_in and static_cast<std::size_t>(group) == c_out;
}

struct find_c1_1x1_convolution
{
    auto matcher() const { return conv_c1_1x1(match::arg(1)(match::is_constant())); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;

        auto input    = ins->inputs().front();
        auto weights  = ins->inputs().back();
        auto out_lens = ins->get_shape().lens();

        // Squeeze weight [C_out, 1, 1, ...] -> [C_out]
        std::vector<int64_t> sq_axes(weights->get_shape().ndim() - 1);
        std::iota(sq_axes.begin(), sq_axes.end(), 1);
        auto sq_weights =
            m.insert_instruction(ins, make_op("squeeze", {{"axes", sq_axes}}), weights);

        // Broadcast weight [C_out] -> [N, C_out, H, W] along axis 1
        auto bcast_weights = m.insert_instruction(
            ins, make_op("broadcast", {{"axis", 1}, {"out_lens", out_lens}}), sq_weights);

        // Broadcast input [N, 1, H, W] -> [N, C_out, H, W]
        auto bcast_input =
            m.insert_instruction(ins, make_op("multibroadcast", {{"out_lens", out_lens}}), input);

        // Replace with elementwise multiply
        m.replace_instruction(ins, make_op("mul"), bcast_input, bcast_weights);
    }
};

struct find_1x1_depthwise
{
    auto matcher() const { return depthwise_conv_1x1(match::arg(1)(match::is_constant())); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;

        auto input   = ins->inputs().front();
        auto weights = ins->inputs().back();

        // Squeeze all dimensions except first: [C, 1, 1, ...] -> [C]
        std::vector<int64_t> sq_axes(weights->get_shape().ndim() - 1);
        std::iota(sq_axes.begin(), sq_axes.end(), 1);
        auto sq_weights =
            m.insert_instruction(ins, make_op("squeeze", {{"axes", sq_axes}}), weights);

        // Broadcast [C] -> output shape along channel axis (axis 1)
        auto bcast_weights = m.insert_instruction(
            ins,
            make_op("broadcast", {{"axis", 1}, {"out_lens", ins->get_shape().lens()}}),
            sq_weights);

        // Replace convolution with elementwise multiply
        m.replace_instruction(ins, make_op("mul"), input, bcast_weights);
    }
};

struct find_channelwise_convolution
{
    auto matcher() const { return conv_channelwise(match::arg(1)(match::is_constant())); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;

        auto input   = ins->inputs().front();
        auto weights = ins->inputs().back();

        auto w_lens      = weights->get_shape().lens();
        auto x_lens      = input->get_shape().lens();
        auto ndim        = ins->get_shape().ndim();
        auto num_spatial = ndim - 2;

        // Compute kernel_elements for the average pooling scale factor
        std::size_t kernel_elements = 1;
        for(std::size_t d = 2; d < ndim; ++d)
            kernel_elements *= w_lens[d];

        // Pre-scale weights by kernel_elements to compensate for average pooling
        auto scale_lit = m.add_literal(literal{shape{weights->get_shape().type(), {1}},
                                               {static_cast<double>(kernel_elements)}});
        auto scale_bcast =
            m.insert_instruction(ins, make_op("multibroadcast", {{"out_lens", w_lens}}), scale_lit);
        auto scaled_weights = m.insert_instruction(ins, make_op("mul"), weights, scale_bcast);

        // Build interleaved product shape: [N, C_out, k0, s0, k1, s1, ...]
        std::vector<std::size_t> prod_lens;
        prod_lens.push_back(x_lens[0]);
        prod_lens.push_back(w_lens[0]);
        for(std::size_t d = 0; d < num_spatial; ++d)
        {
            prod_lens.push_back(w_lens[2 + d]);
            prod_lens.push_back(x_lens[2 + d]);
        }

        // Unsqueeze input: [N, C_in, s0, s1, ...] -> [N, C_in, 1, s0, 1, s1, ...]
        // Insert singletons at positions 2, 4, 6, ... for the kernel dims
        std::vector<int64_t> input_unsq_axes;
        for(std::size_t d = 0; d < num_spatial; ++d)
            input_unsq_axes.push_back(static_cast<int64_t>(2 + 2 * d));
        auto unsq_input =
            m.insert_instruction(ins, make_op("unsqueeze", {{"axes", input_unsq_axes}}), input);
        auto bcast_input = m.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", prod_lens}}), unsq_input);

        // Squeeze weight axis 1: [C_out, 1, k0, ...] -> [C_out, k0, ...]
        auto sq_weights =
            m.insert_instruction(ins, make_op("squeeze", {{"axes", {1}}}), scaled_weights);

        // Unsqueeze weight: [C_out, k0, k1, ...] -> [1, C_out, k0, 1, k1, 1, ...]
        // Add batch dim at 0 and spatial singletons at 3, 5, 7, ...
        std::vector<int64_t> w_unsq_axes;
        w_unsq_axes.push_back(0);
        for(std::size_t d = 0; d < num_spatial; ++d)
            w_unsq_axes.push_back(static_cast<int64_t>(3 + 2 * d));
        auto unsq_weights =
            m.insert_instruction(ins, make_op("unsqueeze", {{"axes", w_unsq_axes}}), sq_weights);
        auto bcast_weights = m.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", prod_lens}}), unsq_weights);

        // Multiply in interleaved broadcast space: [N, C_out, k0, s0, k1, s1, ...]
        auto product = m.insert_instruction(ins, make_op("mul"), bcast_input, bcast_weights);

        // Reshape to flatten paired dims: [N, C_out, k0*s0, k1*s1, ...]
        std::vector<int64_t> flat_dims;
        flat_dims.push_back(static_cast<int64_t>(x_lens[0]));
        flat_dims.push_back(static_cast<int64_t>(w_lens[0]));
        for(std::size_t d = 0; d < num_spatial; ++d)
            flat_dims.push_back(static_cast<int64_t>(w_lens[2 + d] * x_lens[2 + d]));
        auto reshaped =
            m.insert_instruction(ins, make_op("reshape", {{"dims", flat_dims}}), product);

        // Dilated average pooling: the anti-diagonal sum becomes a standard pooling
        // Window of size k_d with dilation (s_d + 1) in each flattened spatial dim
        std::vector<std::size_t> pool_lengths;
        std::vector<std::size_t> pool_dilations;
        std::vector<std::size_t> pool_stride;
        std::vector<std::size_t> pool_padding;
        for(std::size_t d = 0; d < num_spatial; ++d)
        {
            pool_lengths.push_back(w_lens[2 + d]);
            pool_dilations.push_back(x_lens[2 + d] + 1);
            pool_stride.push_back(1);
            pool_padding.push_back(0);
        }

        m.replace_instruction(ins,
                              make_op("pooling",
                                      {{"mode", op::pooling_mode::average},
                                       {"lengths", pool_lengths},
                                       {"dilations", pool_dilations},
                                       {"stride", pool_stride},
                                       {"padding", pool_padding},
                                       {"count_include_pad", true}}),
                              reshaped);
    }
};

struct find_1x1_convolution
{
    auto matcher() const { return conv_1x1(match::arg(1)(match::is_constant())); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;

        auto input   = ins->inputs().front();
        auto weights = ins->inputs().back();

        std::vector<int64_t> sq_axes(ins->get_shape().ndim() - 2);
        std::iota(sq_axes.begin(), sq_axes.end(), 2);
        auto sq_weights =
            m.insert_instruction(ins, make_op("squeeze", {{"axes", sq_axes}}), weights);

        if(ins->get_shape().transposed())
        {
            std::vector<int64_t> aperm(ins->get_shape().ndim());
            std::iota(aperm.begin(), aperm.end(), 0);
            std::rotate(aperm.begin() + 1, aperm.begin() + 2, aperm.end());
            auto a_mat =
                m.insert_instruction(ins, make_op("transpose", {{"permutation", aperm}}), input);

            auto transpose = m.insert_instruction(
                ins, make_op("transpose", {{"permutation", {1, 0}}}), sq_weights);
            auto b_lens = a_mat->get_shape().lens();
            copy(transpose->get_shape().lens(), b_lens.end() - 2);
            auto b_mat = m.insert_instruction(
                ins, make_op("multibroadcast", {{"out_lens", b_lens}}), transpose);

            auto dot = m.insert_instruction(ins, make_op("dot"), a_mat, b_mat);
            m.replace_instruction(
                ins, make_op("transpose", {{"permutation", invert_permutation(aperm)}}), dot);
        }
        else
        {
            auto batch_dim = ins->get_shape().lens().front();
            auto m_dim     = std::accumulate(input->get_shape().lens().begin() + 2,
                                         input->get_shape().lens().end(),
                                         1,
                                         std::multiplies<>{});
            auto n_dim     = weights->get_shape().lens()[0];
            auto k_dim     = weights->get_shape().lens()[1];
            auto a_mat     = m.insert_instruction(
                ins,
                make_op("multibroadcast", {{"out_lens", {batch_dim, n_dim, k_dim}}}),
                sq_weights);
            auto b_mat = m.insert_instruction(
                ins, make_op("reshape", {{"dims", {batch_dim, k_dim, m_dim}}}), input);
            auto dot = m.insert_instruction(ins, make_op("dot"), a_mat, b_mat);
            m.replace_instruction(
                ins, make_op("reshape", {{"dims", ins->get_shape().lens()}}), dot);
        }
    }
};

} // namespace

void rewrite_dot::apply(module& m) const
{
    match::find_matches(m,
                        find_c1_1x1_convolution{},
                        find_channelwise_convolution{},
                        find_1x1_convolution{},
                        find_1x1_depthwise{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
