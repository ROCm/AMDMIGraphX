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
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/op/builder/insert.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_group_query_attention : op_parser<parse_group_query_attention>
{
    std::vector<op_desc> operators() const { return {{"GroupQueryAttention"}}; }

    static std::size_t
    fixed_dimension(const shape& s, std::size_t axis, const std::string& name)
    {
        if(not s.dynamic())
            return s.lens().at(axis);
        auto result = sym::fixed_value(s.dyn_dims().at(axis).sym_expr);
        if(not result.has_value())
            MIGRAPHX_THROW("GroupQueryAttention: " + name + " must be fixed");
        return sym::to<std::size_t>(*result);
    }

    static instruction_ref
    add_runtime_reshape(const onnx_parser::node_info& info,
                        instruction_ref input,
                        const std::vector<shape::dynamic_dimension>& dims,
                        const std::vector<instruction_ref>& shape_sources)
    {
        std::vector<sym::expr> expressions(dims.size());
        transform(dims, expressions.begin(), [](const auto& dim) { return dim.sym_expr; });
        auto resolved_dims = info.add_instruction(
            make_op("eval_expr_from_shape", {{"expressions", to_value(expressions)}}),
            shape_sources);
        auto allocation = info.add_instruction(
            make_op("allocate", {{"shape", to_value(shape{input->get_shape().type(), dims})}}),
            resolved_dims);
        return info.add_instruction(make_op("reshape"), input, allocation);
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

    static instruction_ref add_symbolic_attention_mask(const onnx_parser::node_info& info,
                                                       instruction_ref q,
                                                       instruction_ref k,
                                                       instruction_ref past_length,
                                                       instruction_ref logits,
                                                       instruction_ref ninf,
                                                       instruction_ref scale)
    {
        const auto& q_dims     = q->get_shape().dyn_dims();
        const auto& cache_dims = k->get_shape().dyn_dims();
        std::vector<shape::dynamic_dimension> attention_dims{
            q_dims.at(0), q_dims.at(1), q_dims.at(2), cache_dims.at(2)};
        auto zero = info.add_literal(
            literal{shape{shape::int64_type, {1}}, std::vector<int64_t>{0}});
        auto one = info.add_literal(
            literal{shape{shape::int64_type, {1}}, std::vector<int64_t>{1}});

        auto cache_length =
            info.add_instruction(make_op("dimensions_of", {{"start", 2}, {"end", 3}}), k);
        auto range = info.add_instruction(
            make_op("dynamic_range", {{"output_dim", to_value(cache_dims.at(2))}}),
            zero,
            cache_length,
            one);
        range = info.add_instruction(
            make_op("multibroadcast", {{"out_dyn_dims", to_value(attention_dims)}}),
            range,
            logits);
        ninf = info.add_instruction(
            make_op("multibroadcast", {{"out_dyn_dims", to_value(attention_dims)}}), ninf, logits);
        scale = info.add_instruction(
            make_op("multibroadcast", {{"out_dyn_dims", to_value(attention_dims)}}), scale, logits);
        logits = info.add_instruction(make_op("mul"), logits, scale);

        auto sequence_length =
            info.add_instruction(make_op("dimensions_of", {{"start", 2}, {"end", 3}}), q);
        auto sequence_range = info.add_instruction(
            make_op("dynamic_range", {{"output_dim", to_value(q_dims.at(2))}}),
            zero,
            sequence_length,
            one);
        std::vector<shape::dynamic_dimension> sequence_range_dims{
            shape::dynamic_dimension{sym::lit(1)},
            shape::dynamic_dimension{sym::lit(1)},
            q_dims.at(2),
            shape::dynamic_dimension{sym::lit(1)}};
        sequence_range =
            add_runtime_reshape(info, sequence_range, sequence_range_dims, {q});
        sequence_range = info.add_instruction(
            make_op("multibroadcast", {{"out_dyn_dims", to_value(attention_dims)}}),
            sequence_range,
            logits);

        past_length = info.add_instruction(
            make_op("convert", {{"target_type", shape::int64_type}}), past_length);
        std::vector<shape::dynamic_dimension> past_length_dims{
            q_dims.at(0),
            q_dims.at(1),
            shape::dynamic_dimension{sym::lit(1)},
            shape::dynamic_dimension{sym::lit(1)}};
        past_length = add_runtime_reshape(info, past_length, past_length_dims, {q});
        past_length = info.add_instruction(
            make_op("multibroadcast", {{"out_dyn_dims", to_value(attention_dims)}}),
            past_length,
            logits);
        auto sequence_offset = info.add_instruction(make_op("sub"), sequence_length, one);
        sequence_offset = info.add_instruction(
            make_op("multibroadcast", {{"out_dyn_dims", to_value(attention_dims)}}),
            sequence_offset,
            logits);
        auto query_positions =
            info.add_instruction(make_op("sub"), past_length, sequence_offset);
        query_positions =
            info.add_instruction(make_op("add"), sequence_range, query_positions);
        auto causal_mask = info.add_instruction(make_op("greater"), range, query_positions);
        causal_mask = info.add_instruction(
            make_op("convert", {{"target_type", shape::bool_type}}), causal_mask);
        logits = info.add_instruction(make_op("where"), causal_mask, ninf, logits);

        auto length_mask = info.add_instruction(make_op("greater"), range, past_length);
        length_mask = info.add_instruction(
            make_op("convert", {{"target_type", shape::bool_type}}), length_mask);
        return info.add_instruction(make_op("where"), length_mask, ninf, logits);
    }

    static instruction_ref add_static_attention_mask(const onnx_parser::node_info& info,
                                                     instruction_ref q,
                                                     instruction_ref k,
                                                     instruction_ref past_length,
                                                     std::size_t num_heads,
                                                     int local_window_size,
                                                     instruction_ref logits,
                                                     instruction_ref ninf,
                                                     instruction_ref scale)
    {
        const auto& q_lens         = q->get_shape().lens();
        const auto& cache_lens     = k->get_shape().lens();
        const auto batch_size      = q_lens.at(0);
        const auto sequence_length = q_lens.at(2);
        const auto max_seq_len     = cache_lens.at(2);
        std::vector<int> range_values(max_seq_len);
        std::iota(range_values.begin(), range_values.end(), 0);
        auto range =
            info.add_literal(shape{past_length->get_shape().type(), {max_seq_len}}, range_values);
        std::vector<std::size_t> attention_lens{
            batch_size, num_heads, sequence_length, max_seq_len};
        range = info.add_instruction(
            make_op("multibroadcast", {{"out_lens", attention_lens}}), range);
        ninf = info.add_instruction(
            make_op("multibroadcast", {{"out_lens", attention_lens}}), ninf);
        scale = info.add_instruction(
            make_op("multibroadcast", {{"out_lens", attention_lens}}), scale);
        logits = info.add_instruction(make_op("mul"), logits, scale);

        instruction_ref sequence_range;
        if(sequence_length > 1)
        {
            std::vector<int> sequence_values(sequence_length);
            std::iota(sequence_values.begin(), sequence_values.end(), 0);
            sequence_range = info.add_literal(
                shape{past_length->get_shape().type(), {sequence_length}}, sequence_values);
            sequence_range = info.add_instruction(
                make_op("reshape", {{"dims", {sequence_length, 1}}}), sequence_range);
            sequence_range = info.add_instruction(
                make_op("multibroadcast", {{"out_lens", attention_lens}}), sequence_range);
            auto causal_mask = info.add_instruction(make_op("greater"), range, sequence_range);
            causal_mask      = info.add_instruction(
                make_op("convert", {{"target_type", shape::bool_type}}), causal_mask);
            logits = info.add_instruction(make_op("where"), causal_mask, ninf, logits);
        }

        auto mask_comp = info.add_instruction(
            make_op("reshape", {{"dims", {batch_size, num_heads, 1, 1}}}), past_length);
        mask_comp = info.add_instruction(
            make_op("multibroadcast", {{"out_lens", attention_lens}}), mask_comp);
        if(local_window_size > 0)
        {
            const bool is_prompt = sequence_length > 1;
            auto window_offset   = info.add_literal(
                literal{shape{past_length->get_shape().type(), {1}},
                        {is_prompt ? -local_window_size : -(local_window_size + 1)}});
            window_offset = info.add_instruction(
                make_op("multibroadcast", {{"out_lens", attention_lens}}), window_offset);
            auto window_comp = info.add_instruction(
                make_op("add"), is_prompt ? sequence_range : mask_comp, window_offset);
            auto window_mask = info.add_instruction(make_op("greater"), window_comp, range);
            window_mask      = info.add_instruction(
                make_op("convert", {{"target_type", shape::bool_type}}), window_mask);
            logits = info.add_instruction(make_op("where"), window_mask, ninf, logits);
        }

        auto length_mask = info.add_instruction(make_op("greater"), range, mask_comp);
        length_mask = info.add_instruction(
            make_op("convert", {{"target_type", shape::bool_type}}), length_mask);
        return info.add_instruction(make_op("where"), length_mask, ninf, logits);
    }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       const std::vector<instruction_ref>& args) const
    {
        bool do_rotary           = false;
        std::size_t kv_num_heads = 0;
        int local_window_size    = -1;
        std::size_t num_heads    = 0;
        bool rotary_interleaved  = false;
        float scale              = 0.0;
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
        if(contains(info.attributes, "softcap"))
        {
            if(not float_equal(parser.parse_value(info.attributes.at("softcap")).at<float>(), 0.0))
            {
                MIGRAPHX_THROW("GroupQueryAttention: non-zero softcap is not yet supported.");
            }
        }

        if(args.size() < 7 or args.size() > 11)
        {
            MIGRAPHX_THROW("GroupQueryAttention: Wrong number of inputs provided");
        }

        const bool symbolic = args.at(0)->get_shape().symbolic();
        if(symbolic and do_rotary)
            MIGRAPHX_THROW(
                "GroupQueryAttention: symbolic internal rotary embedding is not supported");
        if(symbolic and local_window_size > 0)
            MIGRAPHX_THROW(
                "GroupQueryAttention: symbolic local window attention is not supported");
        if(kv_num_heads == 0 or num_heads == 0 or num_heads % kv_num_heads != 0)
            MIGRAPHX_THROW(
                "GroupQueryAttention: num_heads must be divisible by kv_num_heads");

        auto qkv = args.at(0);
        if(args.at(1)->get_shape().ndim() > 1)
        {
            qkv = info.add_instruction(
                make_op("concat", {{"axis", 2}}), args.at(0), args.at(1), args.at(2));
        }

        const auto& q_shape      = qkv->get_shape();
        const auto q_hidden_size = fixed_dimension(q_shape, 2, "hidden size");
        const auto total_heads   = num_heads + 2 * kv_num_heads;
        if(q_hidden_size % total_heads != 0)
            MIGRAPHX_THROW(
                "GroupQueryAttention: hidden size must be divisible by the total head count");
        std::size_t head_size = q_hidden_size / total_heads;

        instruction_ref transposed_qkv;
        if(symbolic)
        {
            const auto& q_dims = q_shape.dyn_dims();
            std::vector<shape::dynamic_dimension> bsnh{
                q_dims.at(0),
                q_dims.at(1),
                shape::dynamic_dimension{sym::lit(total_heads)},
                shape::dynamic_dimension{sym::lit(head_size)}};
            transposed_qkv = add_runtime_reshape(info, qkv, bsnh, {qkv});
        }
        else
        {
            const auto& q_lens = q_shape.lens();
            std::vector<std::size_t> bsnh{
                q_lens.at(0), q_lens.at(1), total_heads, head_size};
            transposed_qkv =
                info.add_instruction(make_op("reshape", {{"dims", bsnh}}), qkv);
        }

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
        std::vector<instruction_ref> concat_k_inputs{cur_k, slk, k};
        std::vector<instruction_ref> concat_v_inputs{cur_v, slk, v};

        k = info.add_instruction(make_op("concat_past_present", {{"kv_num_heads", kv_num_heads}}),
                                 concat_k_inputs);
        v = info.add_instruction(make_op("concat_past_present", {{"kv_num_heads", kv_num_heads}}),
                                 concat_v_inputs);

        auto k_out = k;
        auto v_out = v;

        auto kv_num_heads_factor = num_heads / kv_num_heads;
        instruction_ref past_sl;
        if(symbolic)
        {
            const auto& q_dims = q->get_shape().dyn_dims();
            std::vector<shape::dynamic_dimension> past_dims{
                q_dims.at(0), shape::dynamic_dimension{sym::lit(num_heads)}};
            past_sl = info.add_instruction(
                make_op("multibroadcast", {{"out_dyn_dims", to_value(past_dims)}}), slk, q);
        }
        else
        {
            past_sl = info.add_instruction(
                make_op("multibroadcast",
                        {{"out_lens", {q_shape.lens().at(0), num_heads}}}),
                slk);
        }

        if(kv_num_heads_factor != 1)
        {
            if(symbolic)
            {
                const auto& kv_dims = k->get_shape().dyn_dims();
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
                k = info.add_instruction(
                    make_op("multibroadcast", {{"out_dyn_dims", to_value(expanded_dims)}}),
                    k,
                    k_out);
                v = info.add_instruction(
                    make_op("multibroadcast", {{"out_dyn_dims", to_value(expanded_dims)}}),
                    v,
                    v_out);
                k = add_runtime_reshape(info, k, repeated_dims, {k});
                v = add_runtime_reshape(info, v, repeated_dims, {v});
            }
            else
            {
                auto kv_new_lens  = k->get_shape().lens();
                kv_new_lens.at(1) = num_heads;
                k = info.add_instruction(make_op("unsqueeze", {{"axes", {2}}}), k);
                v = info.add_instruction(make_op("unsqueeze", {{"axes", {2}}}), v);
                auto kv_unsqueezed_lens  = k->get_shape().lens();
                kv_unsqueezed_lens.at(2) = kv_num_heads_factor;
                k = info.add_instruction(
                    make_op("multibroadcast", {{"out_lens", kv_unsqueezed_lens}}), k);
                v = info.add_instruction(
                    make_op("multibroadcast", {{"out_lens", kv_unsqueezed_lens}}), v);
                k = info.add_instruction(make_op("reshape", {{"dims", kv_new_lens}}), k);
                v = info.add_instruction(make_op("reshape", {{"dims", kv_new_lens}}), v);
            }
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
            symbolic
                ? add_symbolic_attention_mask(info, q, k_out, past_sl, gemm1, ninf, scale_ins)
                : add_static_attention_mask(info,
                                            q,
                                            k_out,
                                            past_sl,
                                            num_heads,
                                            local_window_size,
                                            gemm1,
                                            ninf,
                                            scale_ins);
        auto softmax = info.add_instruction(make_op("softmax", {{"axis", 3}}), where);
        auto scores  = info.add_instruction(make_op("dot"), softmax, v);
        auto out =
            info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), scores);
        if(symbolic)
        {
            const auto& out_dims = out->get_shape().dyn_dims();
            std::vector<shape::dynamic_dimension> bsh{
                out_dims.at(0),
                out_dims.at(1),
                shape::dynamic_dimension{sym::lit(head_size * num_heads)}};
            out = add_runtime_reshape(info, out, bsh, {out});
        }
        else
        {
            const auto& out_lens = out->get_shape().lens();
            out = info.add_instruction(
                make_op("reshape",
                        {{"dims", {out_lens.at(0),
                                   out_lens.at(1),
                                   head_size * num_heads}}}),
                out);
        }

        return {out, k_out, v_out};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
