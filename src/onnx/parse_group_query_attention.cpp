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
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/reshape_helpers.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/op/builder/insert.hpp>
#include <cassert>

#include <algorithm>
#include <optional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_group_query_attention : op_parser<parse_group_query_attention>
{
    std::vector<op_desc> operators() const { return {{"GroupQueryAttention"}}; }

    static void append_shape_source(std::vector<instruction_ref>& sources, instruction_ref source)
    {
        if(std::find(sources.begin(), sources.end(), source) == sources.end())
            sources.push_back(source);
    }

    static std::optional<std::vector<std::size_t>>
    fixed_lens(const std::vector<shape::dynamic_dimension>& dims)
    {
        std::vector<std::size_t> result;
        result.reserve(dims.size());
        for(const auto& dim : dims)
        {
            auto value = sym::fixed_value(dim.sym_expr);
            if(not value.has_value())
                return std::nullopt;
            result.push_back(sym::to<std::size_t>(*value));
        }
        return result;
    }

    static instruction_ref add_reshape(const onnx_parser::node_info& info,
                                       instruction_ref input,
                                       const std::vector<shape::dynamic_dimension>& dims,
                                       const std::vector<instruction_ref>& extra_shape_sources = {})
    {
        auto lens = fixed_lens(dims);
        if(lens.has_value())
            return info.add_instruction(make_op("reshape", {{"dims", *lens}}), input);
        return add_symbolic_reshape(info, input, dims, extra_shape_sources);
    }

    static instruction_ref
    add_multibroadcast(const onnx_parser::node_info& info,
                       instruction_ref input,
                       const std::vector<shape::dynamic_dimension>& dims,
                       const std::vector<instruction_ref>& extra_shape_sources = {})
    {
        auto lens = fixed_lens(dims);
        if(lens.has_value() and not input->get_shape().dynamic())
            return info.add_instruction(make_op("multibroadcast", {{"out_lens", *lens}}), input);

        auto shape_sources = info.mod->get_parameters();
        auto output        = migraphx::try_add_runtime_allocation(
            *info.mod, input->get_shape().type(), dims, shape_sources);
        if(not output.has_value())
        {
            append_shape_source(shape_sources, input);
            for(auto source : extra_shape_sources)
                append_shape_source(shape_sources, source);
            output = migraphx::try_add_runtime_allocation(
                *info.mod, input->get_shape().type(), dims, shape_sources);
        }
        if(output.has_value())
            return info.add_instruction(
                make_op("multibroadcast", {{"out_dyn_dims", to_value(dims)}}), input, *output);
        MIGRAPHX_THROW("GroupQueryAttention: runtime broadcast dimensions have no shape source");
    }

    static bool has_input(const std::vector<instruction_ref>& args, std::size_t index)
    {
        return args.size() > index and not args.at(index)->is_undefined();
    }

    static instruction_ref insert_rotary(module& m,
                                         bool interleaved,
                                         std::size_t sequence_length,
                                         std::vector<instruction_ref> args)
    {
        // GQA position semantics: prefill starts from 0, decode uses seqlens_k
        auto& pos_ids = args.at(1);
        if(sequence_length > 1)
        {
            pos_ids = m.add_literal(literal{shape{pos_ids->get_shape().type(), {1}}, {0}});
        }
        return op::builder::add("rotary_embedding", m, args, {{"interleaved", interleaved}}).at(0);
    }

    static instruction_ref add_attention_mask(const onnx_parser::node_info& info,
                                              instruction_ref q,
                                              instruction_ref k,
                                              instruction_ref past_length,
                                              int local_window_size,
                                              instruction_ref logits,
                                              instruction_ref ninf,
                                              instruction_ref scale)
    {
        const auto q_dims     = q->get_shape().to_symbolic().dyn_dims();
        const auto cache_dims = k->get_shape().to_symbolic().dyn_dims();
        std::vector<shape::dynamic_dimension> attention_dims{
            q_dims.at(0), q_dims.at(1), q_dims.at(2), cache_dims.at(2)};
        const auto attention_lens = fixed_lens(attention_dims);
        const bool runtime_shape  = not attention_lens.has_value();

        instruction_ref zero;
        instruction_ref one;
        instruction_ref range;
        if(runtime_shape)
        {
            zero =
                info.add_literal(literal{shape{shape::int64_type, {1}}, std::vector<int64_t>{0}});
            one = info.add_literal(literal{shape{shape::int64_type, {1}}, std::vector<int64_t>{1}});
            auto cache_length =
                info.add_instruction(make_op("dimensions_of", {{"start", 2}, {"end", 3}}), k);
            range = info.add_instruction(
                make_op("dynamic_range", {{"output_dim", to_value(cache_dims.at(2))}}),
                zero,
                cache_length,
                one);
        }
        else
        {
            std::vector<int> range_values(attention_lens->at(3));
            std::iota(range_values.begin(), range_values.end(), 0);
            range = info.add_literal(
                shape{past_length->get_shape().type(), {attention_lens->at(3)}}, range_values);
        }

        range  = add_multibroadcast(info, range, attention_dims, {logits});
        ninf   = add_multibroadcast(info, ninf, attention_dims, {logits});
        scale  = add_multibroadcast(info, scale, attention_dims, {logits});
        logits = info.add_instruction(make_op("mul"), logits, scale);

        instruction_ref sequence_range;
        instruction_ref mask_comp;
        if(runtime_shape)
        {
            auto sequence_length =
                info.add_instruction(make_op("dimensions_of", {{"start", 2}, {"end", 3}}), q);
            sequence_range = info.add_instruction(
                make_op("dynamic_range", {{"output_dim", to_value(q_dims.at(2))}}),
                zero,
                sequence_length,
                one);
            std::vector<shape::dynamic_dimension> sequence_range_dims{
                shape::dynamic_dimension{sym::lit(1)},
                shape::dynamic_dimension{sym::lit(1)},
                q_dims.at(2),
                shape::dynamic_dimension{sym::lit(1)}};
            sequence_range = add_reshape(info, sequence_range, sequence_range_dims);
            sequence_range = add_multibroadcast(info, sequence_range, attention_dims, {logits});

            past_length = info.add_instruction(
                make_op("convert", {{"target_type", shape::int64_type}}), past_length);
            std::vector<shape::dynamic_dimension> past_length_dims{
                q_dims.at(0),
                q_dims.at(1),
                shape::dynamic_dimension{sym::lit(1)},
                shape::dynamic_dimension{sym::lit(1)}};
            mask_comp = add_reshape(info, past_length, past_length_dims);
            mask_comp = add_multibroadcast(info, mask_comp, attention_dims, {logits});

            auto is_prompt = info.add_instruction(make_op("greater"), sequence_length, one);
            is_prompt      = info.add_instruction(
                make_op("convert", {{"target_type", shape::bool_type}}), is_prompt);
            is_prompt = add_multibroadcast(info, is_prompt, attention_dims, {logits});
            auto query_positions =
                info.add_instruction(make_op("where"), is_prompt, sequence_range, mask_comp);
            auto causal_mask = info.add_instruction(make_op("greater"), range, query_positions);
            causal_mask      = info.add_instruction(
                make_op("convert", {{"target_type", shape::bool_type}}), causal_mask);
            logits = info.add_instruction(make_op("where"), causal_mask, ninf, logits);
        }
        else
        {
            const auto sequence_length = attention_lens->at(2);
            if(sequence_length > 1)
            {
                std::vector<int> sequence_values(sequence_length);
                std::iota(sequence_values.begin(), sequence_values.end(), 0);
                sequence_range = info.add_literal(
                    shape{past_length->get_shape().type(), {sequence_length}}, sequence_values);
                std::vector<shape::dynamic_dimension> sequence_range_dims{
                    shape::dynamic_dimension{sym::lit(sequence_length)},
                    shape::dynamic_dimension{sym::lit(1)}};
                sequence_range = add_reshape(info, sequence_range, sequence_range_dims);
                sequence_range = add_multibroadcast(info, sequence_range, attention_dims, {logits});
                auto causal_mask = info.add_instruction(make_op("greater"), range, sequence_range);
                causal_mask      = info.add_instruction(
                    make_op("convert", {{"target_type", shape::bool_type}}), causal_mask);
                logits = info.add_instruction(make_op("where"), causal_mask, ninf, logits);
            }

            std::vector<shape::dynamic_dimension> past_length_dims{
                q_dims.at(0),
                q_dims.at(1),
                shape::dynamic_dimension{sym::lit(1)},
                shape::dynamic_dimension{sym::lit(1)}};
            mask_comp = add_reshape(info, past_length, past_length_dims);
            mask_comp = add_multibroadcast(info, mask_comp, attention_dims);

            if(local_window_size > 0)
            {
                const bool is_prompt = sequence_length > 1;
                auto window_offset   = info.add_literal(
                    literal{shape{past_length->get_shape().type(), {1}},
                            {is_prompt ? -local_window_size : -(local_window_size + 1)}});
                window_offset    = add_multibroadcast(info, window_offset, attention_dims);
                auto window_comp = info.add_instruction(
                    make_op("add"), is_prompt ? sequence_range : mask_comp, window_offset);
                auto window_mask = info.add_instruction(make_op("greater"), window_comp, range);
                window_mask      = info.add_instruction(
                    make_op("convert", {{"target_type", shape::bool_type}}), window_mask);
                logits = info.add_instruction(make_op("where"), window_mask, ninf, logits);
            }
        }

        auto length_mask = info.add_instruction(make_op("greater"), range, mask_comp);
        length_mask = info.add_instruction(make_op("convert", {{"target_type", shape::bool_type}}),
                                           length_mask);
        return info.add_instruction(make_op("where"), length_mask, ninf, logits);
    }

    struct gqa_attributes
    {
        bool do_rotary           = false;
        std::size_t kv_num_heads = 0;
        int local_window_size    = -1;
        std::size_t num_heads    = 0;
        bool rotary_interleaved  = false;
        float scale              = 0.0;
    };

    static gqa_attributes parse_attributes(const onnx_parser& parser,
                                           const onnx_parser::node_info& info)
    {
        gqa_attributes attrs;
        auto& [do_rotary, kv_num_heads, local_window_size, num_heads, rotary_interleaved, scale] =
            attrs;
        if(contains(info.attributes, "do_rotary"))
        {
            do_rotary = parser.parse_value(info.attributes.at("do_rotary")).at<bool>();
        }
        if(contains(info.attributes, "kv_num_heads"))
        {
            kv_num_heads = parser.parse_value(info.attributes.at("kv_num_heads")).at<std::size_t>();
        }
        else
        {
            MIGRAPHX_THROW(
                "GroupQueryAttention: Attribute 'kv_num_heads' is required but was not provided.");
        }
        if(contains(info.attributes, "local_window_size"))
        {
            local_window_size =
                parser.parse_value(info.attributes.at("local_window_size")).at<int>();
        }
        if(contains(info.attributes, "num_heads"))
        {
            num_heads = parser.parse_value(info.attributes.at("num_heads")).at<std::size_t>();
        }
        else
        {
            MIGRAPHX_THROW(
                "GroupQueryAttention: Attribute 'num_heads' is required but was not provided.");
        }
        if(contains(info.attributes, "rotary_interleaved"))
        {
            rotary_interleaved =
                parser.parse_value(info.attributes.at("rotary_interleaved")).at<bool>();
        }
        if(contains(info.attributes, "scale"))
        {
            scale = parser.parse_value(info.attributes.at("scale")).at<float>();
        }
        if(contains(info.attributes, "softcap") and
           not float_equal(parser.parse_value(info.attributes.at("softcap")).at<float>(), 0.0))
        {
            MIGRAPHX_THROW("GroupQueryAttention: non-zero softcap is not yet supported.");
        }
        return attrs;
    }

    static void validate_inputs(const std::vector<instruction_ref>& args)
    {
        if(args.size() < 7 or args.size() > 12)
        {
            MIGRAPHX_THROW("GroupQueryAttention: Wrong number of inputs provided");
        }
        if(has_input(args, 9))
        {
            MIGRAPHX_THROW("GroupQueryAttention: position_ids input is not yet supported.");
        }
        if(has_input(args, 10))
        {
            MIGRAPHX_THROW("GroupQueryAttention: attention_bias input is not yet supported.");
        }
    }

    // head_sink adds a per-head logit to the softmax denominator only (attention sinks):
    // append it as an extra score column, then drop that column after the softmax
    static instruction_ref insert_softmax_with_head_sink(const onnx_parser::node_info& info,
                                                         instruction_ref scores,
                                                         instruction_ref sink)
    {
        const auto& scores_shape = scores->get_shape();
        const auto nheads = fixed_dimension(scores_shape, 1, "GroupQueryAttention", "head count");
        if(sink->get_shape().dynamic() or sink->get_shape().elements() != nheads)
        {
            MIGRAPHX_THROW("GroupQueryAttention: head_sink must have num_heads elements");
        }
        if(sink->get_shape().type() != scores->get_shape().type())
        {
            sink = info.add_instruction(
                make_op("convert", {{"target_type", scores->get_shape().type()}}), sink);
        }
        sink = info.add_instruction(make_op("reshape", {{"dims", {1, nheads, 1, 1}}}), sink);
        const auto score_dims = scores_shape.to_symbolic().dyn_dims();
        std::vector<shape::dynamic_dimension> sink_dims{score_dims.at(0),
                                                        score_dims.at(1),
                                                        score_dims.at(2),
                                                        shape::dynamic_dimension{sym::lit(1)}};
        sink         = add_multibroadcast(info, sink, sink_dims, {scores});
        auto padded  = info.add_instruction(make_op("concat", {{"axis", 3}}), scores, sink);
        auto softmax = info.add_instruction(make_op("softmax", {{"axis", 3}}), padded);

        if(scores_shape.symbolic())
        {
            const auto& columns = scores_shape.dyn_dims().at(3).sym_expr;
            auto start =
                info.add_literal(literal{shape{shape::int64_type, {1}}, std::vector<int64_t>{0}});
            auto end =
                info.add_instruction(make_op("dimensions_of", {{"start", 3}, {"end", 4}}), scores);
            return info.add_instruction(
                make_op("dyn_slice",
                        {{"axes", {3}},
                         {"starts", to_value(std::vector<sym::expr>{sym::lit(0)})},
                         {"ends", to_value(std::vector<sym::expr>{columns})}}),
                softmax,
                start,
                end);
        }

        const auto columns = scores_shape.lens().at(3);
        return info.add_instruction(
            make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {columns}}}), softmax);
    }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       const std::vector<instruction_ref>& args) const
    {
        auto [do_rotary, kv_num_heads, local_window_size, num_heads, rotary_interleaved, scale] =
            parse_attributes(parser, info);
        validate_inputs(args);

        const bool symbolic = args.at(0)->get_shape().symbolic();
        if(symbolic and do_rotary)
            MIGRAPHX_THROW(
                "GroupQueryAttention: symbolic internal rotary embedding is not supported");
        if(symbolic and local_window_size > 0)
            MIGRAPHX_THROW("GroupQueryAttention: symbolic local window attention is not supported");
        if(kv_num_heads == 0 or num_heads == 0 or num_heads % kv_num_heads != 0)
            MIGRAPHX_THROW("GroupQueryAttention: num_heads must be divisible by kv_num_heads");

        auto qkv = args.at(0);
        if(args.at(1)->get_shape().ndim() > 1)
        {
            qkv = info.add_instruction(
                make_op("concat", {{"axis", 2}}), args.at(0), args.at(1), args.at(2));
        }

        const auto& q_shape = qkv->get_shape();
        const auto q_hidden_size =
            fixed_dimension(q_shape, 2, "GroupQueryAttention", "hidden size");
        const auto total_heads = num_heads + 2 * kv_num_heads;
        if(q_hidden_size % total_heads != 0)
            MIGRAPHX_THROW(
                "GroupQueryAttention: hidden size must be divisible by the total head count");
        std::size_t head_size = q_hidden_size / total_heads;

        const auto qkv_dims = q_shape.to_symbolic().dyn_dims();
        std::vector<shape::dynamic_dimension> bsnh{qkv_dims.at(0),
                                                   qkv_dims.at(1),
                                                   shape::dynamic_dimension{sym::lit(total_heads)},
                                                   shape::dynamic_dimension{sym::lit(head_size)}};
        auto transposed_qkv = add_reshape(info, qkv, bsnh);

        transposed_qkv = info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}),
                                              transposed_qkv);

        auto qk = info.add_instruction(
            make_op("slice",
                    {{"axes", {1}}, {"starts", {0}}, {"ends", {num_heads + kv_num_heads}}}),
            transposed_qkv);
        auto cur_v = info.add_instruction(make_op("slice",
                                                  {{"axes", {1}},
                                                   {"starts", {num_heads + kv_num_heads}},
                                                   {"ends", {num_heads + (2 * kv_num_heads)}}}),
                                          transposed_qkv);

        if(do_rotary)
        {
            qk = insert_rotary(*info.mod,
                               rotary_interleaved,
                               q_shape.lens().at(1),
                               {qk, args.at(5), args.at(7), args.at(8)});
        }

        auto q = info.add_instruction(
            make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {num_heads}}}), qk);
        auto cur_k = info.add_instruction(
            make_op("slice",
                    {{"axes", {1}}, {"starts", {num_heads}}, {"ends", {num_heads + kv_num_heads}}}),
            qk);

        auto k   = args.at(3);
        auto v   = args.at(4);
        auto slk = args.at(5);
        if(slk->get_shape().ndim() == 1)
            slk = info.add_instruction(make_op("unsqueeze", {{"axes", {1}}}), slk);
        std::vector<instruction_ref> concat_k_inputs{cur_k, slk, k};
        std::vector<instruction_ref> concat_v_inputs{cur_v, slk, v};
        if(symbolic)
        {
            auto sequence_length =
                info.add_instruction(make_op("dimensions_of", {{"start", 2}, {"end", 3}}), cur_k);
            cur_k           = info.add_instruction(make_op("contiguous"), cur_k);
            cur_v           = info.add_instruction(make_op("contiguous"), cur_v);
            concat_k_inputs = {cur_k, slk, sequence_length, k};
            concat_v_inputs = {cur_v, slk, sequence_length, v};
        }

        k = info.add_instruction(make_op("concat_past_present", {{"kv_num_heads", kv_num_heads}}),
                                 concat_k_inputs);
        v = info.add_instruction(make_op("concat_past_present", {{"kv_num_heads", kv_num_heads}}),
                                 concat_v_inputs);

        auto k_out = k;
        auto v_out = v;

        if(symbolic and not k->get_shape().dynamic())
        {
            const auto& q_dims = q->get_shape().dyn_dims();
            const auto& k_lens = k->get_shape().lens();
            if(fixed_dimension(q->get_shape(), 0, "GroupQueryAttention", "batch size") !=
               k_lens.at(0))
                MIGRAPHX_THROW(
                    "GroupQueryAttention: query and static cache batch sizes must match");
            std::vector<shape::dynamic_dimension> cache_dims{
                q_dims.at(0),
                shape::dynamic_dimension{sym::lit(k_lens.at(1))},
                shape::dynamic_dimension{sym::lit(k_lens.at(2))},
                shape::dynamic_dimension{sym::lit(k_lens.at(3))}};
            k = add_reshape(info, k, cache_dims, {q});
            v = add_reshape(info, v, cache_dims, {q});
        }

        auto kv_num_heads_factor    = num_heads / kv_num_heads;
        const auto q_attention_dims = q->get_shape().to_symbolic().dyn_dims();
        std::vector<shape::dynamic_dimension> past_dims{
            q_attention_dims.at(0), shape::dynamic_dimension{sym::lit(num_heads)}};
        auto past_sl = add_multibroadcast(info, slk, past_dims);

        if(kv_num_heads_factor != 1)
        {
            const auto kv_dims = k->get_shape().to_symbolic().dyn_dims();
            std::vector<shape::dynamic_dimension> expanded_dims{
                kv_dims.at(0),
                kv_dims.at(1),
                shape::dynamic_dimension{sym::lit(kv_num_heads_factor)},
                kv_dims.at(2),
                kv_dims.at(3)};
            std::vector<shape::dynamic_dimension> repeated_dims{
                kv_dims.at(0),
                shape::dynamic_dimension{sym::lit(num_heads)},
                kv_dims.at(2),
                kv_dims.at(3)};
            k = info.add_instruction(make_op("unsqueeze", {{"axes", {2}}}), k);
            v = info.add_instruction(make_op("unsqueeze", {{"axes", {2}}}), v);
            k = add_multibroadcast(info, k, expanded_dims);
            v = add_multibroadcast(info, v, expanded_dims);
            k = add_reshape(info, k, repeated_dims);
            v = add_reshape(info, v, repeated_dims);
        }
        auto kt    = info.add_instruction(make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k);
        auto gemm1 = info.add_instruction(make_op("dot"), q, kt);

        auto scalar_s = shape{transposed_qkv->get_shape().type(), {1}};
        auto ninf = info.add_literal(literal{scalar_s, {-std::numeric_limits<float>::infinity()}});

        if(float_equal(scale, 0.0))
        {
            scale = 1.0f / std::sqrt(static_cast<float>(head_size));
        }
        auto scale_ins = info.add_literal(literal{scalar_s, {scale}});
        auto where =
            add_attention_mask(info, q, k_out, past_sl, local_window_size, gemm1, ninf, scale_ins);
        auto softmax = has_input(args, 11)
                           ? insert_softmax_with_head_sink(info, where, args.at(11))
                           : info.add_instruction(make_op("softmax", {{"axis", 3}}), where);
        auto scores  = info.add_instruction(make_op("dot"), softmax, v);
        auto out =
            info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), scores);
        const auto out_dims = out->get_shape().to_symbolic().dyn_dims();
        std::vector<shape::dynamic_dimension> bsh{
            out_dims.at(0),
            out_dims.at(1),
            shape::dynamic_dimension{sym::lit(head_size * num_heads)}};
        out = add_reshape(info, out, bsh);

        return {out, k_out, v_out};
    }
};

// com.ryzenai.GQO: GroupQueryAttention fused with o_proj MatMulNBits.
// Inputs 0-11 match microsoft GroupQueryAttention, including optional
// position_ids, attention_bias, and head_sink. o_proj qweight/scales/qzeros
// stay at 12-14 when bias or the NPU-only packed input is omitted.
// Outputs: present_key, present_value, o_proj_output
struct parse_gqo : op_parser<parse_gqo>
{
    std::vector<op_desc> operators() const { return {{"GQO"}}; }

    std::vector<instruction_ref> parse(const op_desc& opd,
                                       const onnx_parser& parser,
                                       onnx_parser::node_info info,
                                       const std::vector<instruction_ref>& args) const
    {
        // 12 attention slots plus qweight, scales, and qzeros.
        constexpr std::size_t weight_index = 12;
        if(args.size() < weight_index + 3)
            MIGRAPHX_THROW("GQO: expects at least 15 inputs, got " + std::to_string(args.size()));

        std::vector<instruction_ref> gqa_args(args.begin(), args.begin() + weight_index);
        parse_group_query_attention gqa;
        auto gqa_outs = gqa.parse(opd, parser, info, gqa_args);
        assert(gqa_outs.size() == 3);

        auto attn_out = gqa_outs[0];
        auto k_out    = gqa_outs[1];
        auto v_out    = gqa_outs[2];

        if(not contains(info.attributes, "o_proj_N") or not contains(info.attributes, "o_proj_K") or
           not contains(info.attributes, "o_proj_bits") or
           not contains(info.attributes, "o_proj_block_size"))
            MIGRAPHX_THROW("GQO: o_proj_N/K/bits/block_size attributes are required");

        std::vector<instruction_ref> mm_args{
            attn_out, args[weight_index], args[weight_index + 1], args[weight_index + 2]};
        if(args.size() > weight_index + 3)
            mm_args.push_back(args[weight_index + 3]);
        // packed at weight_index + 4 is NPU-only and ignored.

        auto o_proj =
            op::builder::add(
                "matmulnbits",
                *info.mod,
                mm_args,
                {{"N", parser.parse_value(info.attributes.at("o_proj_N")).at<int>()},
                 {"K", parser.parse_value(info.attributes.at("o_proj_K")).at<int>()},
                 {"bits", parser.parse_value(info.attributes.at("o_proj_bits")).at<int>()},
                 {"block_size",
                  parser.parse_value(info.attributes.at("o_proj_block_size")).at<int>()}})
                .at(0);
        return {k_out, v_out, o_proj};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
