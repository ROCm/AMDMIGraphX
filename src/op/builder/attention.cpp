/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/common.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/builder/insert.hpp>
#include <optional>
#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

struct attention : op_builder<attention>
{
    // Attributes
    bool do_rotary                   = false;
    bool past_present_share_buffer   = false;
    bool unidirectional              = false;
    std::size_t num_heads            = 1;
    std::size_t rotary_embedding_dim = 0;
    std::vector<std::size_t> qkv_hidden_sizes{0, 0, 0};
    float scale           = 0.0f;
    float mask_filter_val = -10000.0f;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.do_rotary, "do_rotary"),
                    f(self.past_present_share_buffer, "past_present_share_buffer"),
                    f(self.unidirectional, "unidirectional"),
                    f(self.num_heads, "num_heads"),
                    f(self.rotary_embedding_dim, "rotary_embedding_dim"),
                    f(self.qkv_hidden_sizes, "qkv_hidden_sizes"),
                    f(self.scale, "scale"),
                    f(self.mask_filter_val, "mask_filter_val"));
    }

    enum class mask_pad
    {
        no_pad,
        raw,
        right_padding,
        left_padding,
    };

    // Helper struct to hold computed values during insertion
    struct attention_context
    {
        instruction_ref input;
        instruction_ref weights;
        std::optional<instruction_ref> projection_bias;
        std::optional<instruction_ref> mask_index;
        std::vector<std::size_t> qkv_sizes;
        std::size_t num_heads_val;
        float scale_val;
        float mask_filter_val_val;
        bool scale_is_set;

        std::size_t batch_size() const { return input->get_shape().lens().at(0); }
        std::size_t sequence_length() const { return input->get_shape().lens().at(1); }
        std::size_t hidden_size() const { return input->get_shape().lens().at(2); }
        std::size_t total_sequence_length() const { return sequence_length(); }
        std::size_t query_size() const { return hidden_size() / num_heads_val; }

        mask_pad padding_mode() const
        {
            if(mask_index.has_value())
            {
                auto mask_shape = mask_index.value()->get_shape();
                if(mask_shape.ndim() == 1)
                {
                    if(mask_shape.lens().at(0) == batch_size())
                        return mask_pad::right_padding;
                    else if(mask_shape.lens().at(0) == (batch_size() * 2))
                        return mask_pad::left_padding;
                }
                else if(mask_shape.ndim() > 1)
                {
                    return mask_pad::raw;
                }
            }
            return mask_pad::no_pad;
        }

        float get_scale_value() const
        {
            if(scale_is_set)
                return scale_val;
            else
                return static_cast<float>(query_size());
        }
    };

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        if(args.size() < 2 or args.size() > 7)
        {
            MIGRAPHX_THROW("attention: Wrong number of inputs provided");
        }

        // Validate unsupported features
        if(do_rotary)
            MIGRAPHX_THROW("attention: Rotary Embedding not supported");
        if(unidirectional)
            MIGRAPHX_THROW("attention: unidirectional attr not supported");

        // Setup context
        attention_context ctx;
        ctx.input               = args.at(0);
        ctx.weights             = args.at(1);
        ctx.num_heads_val       = num_heads;
        ctx.scale_val           = scale;
        ctx.scale_is_set        = (scale != 0.0f);
        ctx.mask_filter_val_val = mask_filter_val;

        // Validate input shape
        if(ctx.input->get_shape().ndim() != 3)
        {
            MIGRAPHX_THROW(
                "attention: Input must have shape (batch, sequence_length, hidden_size)");
        }

        // Setup QKV hidden sizes
        ctx.qkv_sizes     = qkv_hidden_sizes;
        auto weight_shape = ctx.weights->get_shape();
        if(weight_shape.lens().at(1) % 3 == 0 &&
           std::any_of(ctx.qkv_sizes.begin(), ctx.qkv_sizes.end(), [](auto i) { return i <= 0; }))
        {
            std::size_t size = weight_shape.lens().at(1) / 3;
            ctx.qkv_sizes    = {size, size, size};
        }

        // Optional inputs
        if(args.size() > 2)
            ctx.projection_bias = args.at(2);
        if(args.size() > 3)
            ctx.mask_index = args.at(3);
        if(args.size() > 4)
            MIGRAPHX_THROW("attention: Past Not supported");
        if(args.size() > 5)
            MIGRAPHX_THROW("attention: attention_bias Not supported");
        if(args.size() > 6)
            MIGRAPHX_THROW("attention: past_sequence_length not supported");

        // Apply linear stage to QKV mats from weight matrix
        auto qkv_mats = input_linear_to_qkv(m, ins, ctx);

        // Set attention mask when detected
        std::optional<instruction_ref> attn_mask;
        if(ctx.mask_index.has_value() && ctx.padding_mode() == mask_pad::raw)
            attn_mask = generate_raw_mask_per_batch(m, ins, ctx);

        // Scale factor
        auto scale_factor = m.add_literal(migraphx::literal{
            migraphx::shape{qkv_mats.at(0)->get_shape().type()}, {ctx.get_scale_value()}});

        if(not ctx.scale_is_set)
        {
            scale_factor = m.insert_instruction(ins, make_op("sqrt"), scale_factor);
            scale_factor = m.insert_instruction(ins, make_op("recip"), scale_factor);
        }

        // Split QKV per head
        auto split_qkv = qkv_split_per_head(m, ins, qkv_mats, ctx.num_heads_val);

        return {scale_dot_attention_head(m, ins, split_qkv, scale_factor, attn_mask)};
    }

    private:
    std::vector<instruction_ref>
    input_linear_to_qkv(module& m, instruction_ref ins, const attention_context& ctx) const
    {
        auto input            = ctx.input;
        auto stacked_weights  = ctx.weights;
        const auto& qkv_sizes = ctx.qkv_sizes;

        auto input_lens = input->get_shape().lens();
        auto stacked_weights_unsq =
            m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {0}}}), stacked_weights);
        auto w_lens                     = stacked_weights_unsq->get_shape().lens();
        w_lens.at(0)                    = input_lens.at(0);
        auto stacked_weights_unsq_bcast = m.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", w_lens}}), stacked_weights_unsq);

        auto stacked_result =
            m.insert_instruction(ins, make_op("dot"), input, stacked_weights_unsq_bcast);

        if(ctx.projection_bias.has_value())
        {
            stacked_result =
                insert_common_op(m, ins, "add", stacked_result, ctx.projection_bias.value());
        }

        auto q = m.insert_instruction(ins,
                                      make_op("slice",
                                              {{"axes", {2}},
                                               {"starts", {0}},
                                               {"ends", {static_cast<int64_t>(qkv_sizes.at(0))}}}),
                                      stacked_result);
        auto k = m.insert_instruction(
            ins,
            make_op("slice",
                    {{"axes", {2}},
                     {"starts", {static_cast<int64_t>(qkv_sizes.at(0))}},
                     {"ends", {static_cast<int64_t>(qkv_sizes.at(1) + qkv_sizes.at(0))}}}),
            stacked_result);
        auto v = m.insert_instruction(
            ins,
            make_op(
                "slice",
                {{"axes", {2}},
                 {"starts", {static_cast<int64_t>(qkv_sizes.at(0) + qkv_sizes.at(1))}},
                 {"ends",
                  {static_cast<int64_t>(qkv_sizes.at(0) + qkv_sizes.at(1) + qkv_sizes.at(2))}}}),
            stacked_result);

        return {q, k, v};
    }

    std::vector<instruction_ref> qkv_split_per_head(module& m,
                                                    instruction_ref ins,
                                                    const std::vector<instruction_ref>& qkv_mats,
                                                    std::size_t num_heads_val) const
    {
        auto q_lens = qkv_mats.at(0)->get_shape().lens();
        auto k_lens = qkv_mats.at(1)->get_shape().lens();
        auto v_lens = qkv_mats.at(2)->get_shape().lens();

        auto split_q = m.insert_instruction(
            ins,
            make_op("reshape",
                    {{"dims",
                      {q_lens.at(0), q_lens.at(1), num_heads_val, q_lens.at(2) / num_heads_val}}}),
            qkv_mats.at(0));
        auto split_k = m.insert_instruction(
            ins,
            make_op("reshape",
                    {{"dims",
                      {k_lens.at(0), k_lens.at(1), num_heads_val, k_lens.at(2) / num_heads_val}}}),
            qkv_mats.at(1));
        auto split_v = m.insert_instruction(
            ins,
            make_op("reshape",
                    {{"dims",
                      {v_lens.at(0), v_lens.at(1), num_heads_val, v_lens.at(2) / num_heads_val}}}),
            qkv_mats.at(2));

        split_q = m.insert_instruction(
            ins, make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), split_q);
        split_k = m.insert_instruction(
            ins, make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), split_k);
        split_v = m.insert_instruction(
            ins, make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), split_v);

        return {split_q, split_k, split_v};
    }

    instruction_ref scale_dot_attention_head(module& m,
                                             instruction_ref ins,
                                             const std::vector<instruction_ref>& qkv,
                                             const instruction_ref& scale_factor,
                                             const std::optional<instruction_ref>& mask) const
    {
        auto q = qkv.at(0);
        auto k = qkv.at(1);
        auto v = qkv.at(2);

        auto k_trans =
            m.insert_instruction(ins, make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k);
        auto qk_out = m.insert_instruction(ins, make_op("dot"), q, k_trans);

        auto qk_masked = qk_out;
        if(mask.has_value())
        {
            qk_masked = insert_common_op(m, ins, "add", qk_masked, mask.value());
        }

        auto qk_scaled   = insert_common_op(m, ins, "mul", qk_masked, scale_factor);
        auto softmax_out = m.insert_instruction(ins, make_op("softmax", {{"axis", 3}}), qk_scaled);
        auto output      = m.insert_instruction(ins, make_op("dot"), softmax_out, v);

        output = m.insert_instruction(
            ins, make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), output);

        auto lens = output->get_shape().lens();
        output    = m.insert_instruction(
            ins,
            make_op("reshape", {{"dims", {lens.at(0), lens.at(1), lens.at(2) * lens.at(3)}}}),
            output);

        return output;
    }

    instruction_ref
    generate_raw_mask_per_batch(module& m, instruction_ref ins, const attention_context& ctx) const
    {
        auto batch_size    = ctx.batch_size();
        auto total_seq_len = ctx.total_sequence_length();
        auto num_heads_val = ctx.num_heads_val;

        auto pass_value_lit = m.add_literal(
            migraphx::literal{migraphx::shape{ctx.input->get_shape().type(), {1}, {1}}, {0}});
        auto mask_value_lit = m.add_literal(migraphx::literal{
            migraphx::shape{ctx.input->get_shape().type(), {1}, {1}}, {ctx.mask_filter_val_val}});

        auto bc_pass = m.insert_instruction(
            ins,
            make_op("multibroadcast",
                    {{"out_lens", {batch_size, num_heads_val, total_seq_len, total_seq_len}}}),
            pass_value_lit);
        auto bc_mask = m.insert_instruction(
            ins,
            make_op("multibroadcast",
                    {{"out_lens", {batch_size, num_heads_val, total_seq_len, total_seq_len}}}),
            mask_value_lit);

        auto raw_mask = ctx.mask_index.value();
        raw_mask      = m.insert_instruction(
            ins, make_op("reshape", {{"dims", {batch_size, 1, 1, total_seq_len}}}), raw_mask);
        raw_mask = m.insert_instruction(
            ins,
            make_op("multibroadcast",
                    {{"out_lens", {batch_size, num_heads_val, total_seq_len, total_seq_len}}}),
            raw_mask);
        raw_mask = m.insert_instruction(
            ins,
            make_op("reshape",
                    {{"dims", {batch_size, num_heads_val, total_seq_len, total_seq_len}}}),
            raw_mask);

        auto in_pass = m.insert_instruction(
            ins,
            make_op("convert", {{"target_type", ctx.mask_index.value()->get_shape().type()}}),
            bc_pass);
        auto in_bool = m.insert_instruction(ins, make_op("equal"), raw_mask, in_pass);
        in_bool      = m.insert_instruction(
            ins, make_op("convert", {{"target_type", migraphx::shape::bool_type}}), in_bool);

        return m.insert_instruction(ins, make_op("where"), in_bool, bc_mask, bc_pass);
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
