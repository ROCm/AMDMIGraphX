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
#include <migraphx/shape_for_each.hpp>
#include <algorithm>
#include <cmath>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

// Unified linear attention (ai.onnx opset 27). The recurrence
//   S_t = exp(g_t) * S_{t-1} + k_t (x) beta_t * (v_t - S_{t-1}^T k_t)
//   o_t = scale * q_t^T S_t
// is accumulated in float as the onnx reference implementation prescribes.
// The linear rule has the closed form
//   O = scale * (tril(Q K^T) V + Q S_0),  present = S_0 + K^T V
// emitted as batched matmuls with a causal mask in between so the backends
// can fuse it like attention; the gated and delta rules (and single-step
// decode) unroll the recurrence over the sequence axis instead.
struct parse_linear_attention : op_parser<parse_linear_attention>
{
    std::vector<op_desc> operators() const { return {{"LinearAttention"}}; }

    struct linear_attention_attributes
    {
        std::size_t q_num_heads  = 0;
        std::size_t kv_num_heads = 0;
        float scale              = 0.0f;
        std::string update_rule  = "gated_delta";

        bool gated() const { return update_rule == "gated" or update_rule == "gated_delta"; }
        bool delta() const { return update_rule == "delta" or update_rule == "gated_delta"; }
        std::size_t group_size() const { return q_num_heads / kv_num_heads; }
    };

    static bool has_input(const std::vector<instruction_ref>& args, std::size_t index)
    {
        return args.size() > index and not args.at(index)->is_undefined();
    }

    static linear_attention_attributes parse_attributes(const onnx_parser& parser,
                                                        const onnx_parser::node_info& info)
    {
        linear_attention_attributes attrs;
        if(not contains(info.attributes, "q_num_heads") or
           not contains(info.attributes, "kv_num_heads"))
        {
            MIGRAPHX_THROW("LinearAttention: q_num_heads and kv_num_heads attributes are required");
        }
        auto q_heads  = parser.parse_value(info.attributes.at("q_num_heads")).at<std::int64_t>();
        auto kv_heads = parser.parse_value(info.attributes.at("kv_num_heads")).at<std::int64_t>();
        if(q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads != 0)
        {
            MIGRAPHX_THROW("LinearAttention: q_num_heads (" + std::to_string(q_heads) +
                           ") must be a positive multiple of kv_num_heads (" +
                           std::to_string(kv_heads) + ")");
        }
        attrs.q_num_heads  = q_heads;
        attrs.kv_num_heads = kv_heads;
        if(contains(info.attributes, "scale"))
            attrs.scale = parser.parse_value(info.attributes.at("scale")).at<float>();
        if(contains(info.attributes, "update_rule"))
            attrs.update_rule = info.attributes.at("update_rule").s();
        static const std::vector<std::string> rules{"linear", "gated", "delta", "gated_delta"};
        if(not contains(rules, attrs.update_rule))
        {
            MIGRAPHX_THROW("LinearAttention: unsupported update_rule '" + attrs.update_rule +
                           "', expected one of: linear, gated, delta, gated_delta");
        }
        return attrs;
    }

    struct linear_attention_dims
    {
        std::size_t batch_size      = 0;
        std::size_t sequence_length = 0;
        std::size_t head_size_k     = 0;
        std::size_t head_size_v     = 0;
    };

    static void check_rank3(const shape& s, const std::string& name)
    {
        if(s.ndim() != 3)
            MIGRAPHX_THROW("LinearAttention: " + name +
                           " must be rank 3 (batch, sequence, channels)");
    }

    static void
    check_batch_seq(const shape& s, const linear_attention_dims& dims, const std::string& name)
    {
        if(s.lens().at(0) != dims.batch_size or s.lens().at(1) != dims.sequence_length)
            MIGRAPHX_THROW("LinearAttention: " + name +
                           " batch and sequence dims must match query");
    }

    static linear_attention_dims validate_inputs(const std::vector<instruction_ref>& args,
                                                 const linear_attention_attributes& attrs)
    {
        if(args.size() < 3 or args.size() > 6)
            MIGRAPHX_THROW("LinearAttention: expected 3 to 6 inputs, got " +
                           std::to_string(args.size()));
        if(std::any_of(args.begin(), args.end(), [](auto arg) {
               return not arg->is_undefined() and arg->get_shape().dynamic();
           }))
            MIGRAPHX_THROW("LinearAttention: dynamic shapes are not supported");
        check_rank3(args.at(0)->get_shape(), "query");
        check_rank3(args.at(1)->get_shape(), "key");
        check_rank3(args.at(2)->get_shape(), "value");
        const auto& q_lens = args.at(0)->get_shape().lens();
        const auto& k_lens = args.at(1)->get_shape().lens();
        const auto& v_lens = args.at(2)->get_shape().lens();

        linear_attention_dims dims;
        dims.batch_size      = q_lens.at(0);
        dims.sequence_length = q_lens.at(1);
        if(dims.sequence_length == 0)
            MIGRAPHX_THROW("LinearAttention: sequence length must be at least 1");
        if(q_lens.at(2) % attrs.q_num_heads != 0 or v_lens.at(2) % attrs.kv_num_heads != 0)
            MIGRAPHX_THROW("LinearAttention: hidden sizes must be divisible by the head counts");
        dims.head_size_k = q_lens.at(2) / attrs.q_num_heads;
        dims.head_size_v = v_lens.at(2) / attrs.kv_num_heads;

        check_batch_seq(args.at(1)->get_shape(), dims, "key");
        check_batch_seq(args.at(2)->get_shape(), dims, "value");
        if(k_lens.at(2) != attrs.kv_num_heads * dims.head_size_k)
            MIGRAPHX_THROW("LinearAttention: key hidden size must be kv_num_heads * head size");

        if(has_input(args, 3))
        {
            std::vector<std::size_t> state_lens{
                dims.batch_size, attrs.kv_num_heads, dims.head_size_k, dims.head_size_v};
            if(args.at(3)->get_shape().lens() != state_lens)
                MIGRAPHX_THROW("LinearAttention: past_state must have shape (batch, kv_num_heads, "
                               "k head size, v head size)");
        }
        if(attrs.gated() != has_input(args, 4))
        {
            MIGRAPHX_THROW("LinearAttention: update_rule '" + attrs.update_rule + "' " +
                           (attrs.gated() ? "requires" : "forbids") + " the decay input");
        }
        if(attrs.delta() != has_input(args, 5))
        {
            MIGRAPHX_THROW("LinearAttention: update_rule '" + attrs.update_rule + "' " +
                           (attrs.delta() ? "requires" : "forbids") + " the beta input");
        }
        if(has_input(args, 4))
        {
            const auto& decay_shape = args.at(4)->get_shape();
            check_rank3(decay_shape, "decay");
            check_batch_seq(decay_shape, dims, "decay");
            if(decay_shape.lens().at(2) != attrs.kv_num_heads and
               decay_shape.lens().at(2) != attrs.kv_num_heads * dims.head_size_k)
                MIGRAPHX_THROW("LinearAttention: decay last dim must be kv_num_heads or "
                               "kv_num_heads * k head size");
        }
        if(has_input(args, 5))
        {
            const auto& beta_shape = args.at(5)->get_shape();
            check_rank3(beta_shape, "beta");
            check_batch_seq(beta_shape, dims, "beta");
            if(beta_shape.lens().at(2) != attrs.kv_num_heads and beta_shape.lens().at(2) != 1)
                MIGRAPHX_THROW("LinearAttention: beta last dim must be kv_num_heads or 1");
        }
        return dims;
    }

    static instruction_ref to_float(const onnx_parser::node_info& info, instruction_ref ins)
    {
        if(ins->get_shape().type() == shape::float_type)
            return ins;
        return info.add_instruction(make_op("convert", {{"target_type", shape::float_type}}), ins);
    }

    static instruction_ref
    from_float(const onnx_parser::node_info& info, instruction_ref ins, shape::type_t target_type)
    {
        if(target_type == shape::float_type)
            return ins;
        return info.add_instruction(make_op("convert", {{"target_type", target_type}}), ins);
    }

    // (B, T, heads * d) -> (B, heads, d, T) so a per-step slice of the last
    // axis yields a (B, heads, d, 1) column
    static instruction_ref to_columns(const onnx_parser::node_info& info,
                                      instruction_ref ins,
                                      const linear_attention_dims& dims,
                                      std::size_t heads,
                                      std::size_t head_size)
    {
        std::vector<std::size_t> unpacked{dims.batch_size, dims.sequence_length, heads, head_size};
        auto r = info.add_instruction(make_op("reshape", {{"dims", unpacked}}), ins);
        return info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), r);
    }

    static instruction_ref slice_step(const onnx_parser::node_info& info,
                                      instruction_ref ins,
                                      std::int64_t axis,
                                      std::size_t i)
    {
        return info.add_instruction(
            make_op("slice", {{"axes", {axis}}, {"starts", {i}}, {"ends", {i + 1}}}), ins);
    }

    // (B, Hkv, x, y) -> (B, Hkv, group, x, y): broadcast kv-head tensors
    // across the query group axis so the batched dots pair every query head
    // with its kv head
    static instruction_ref
    broadcast_group(const onnx_parser::node_info& info, instruction_ref ins, std::size_t group)
    {
        auto r = info.add_instruction(make_op("unsqueeze", {{"axes", {2}}}), ins);
        if(group == 1)
            return r;
        auto lens  = r->get_shape().lens();
        lens.at(2) = group;
        return info.add_instruction(make_op("multibroadcast", {{"out_lens", lens}}), r);
    }

    // Zero the score entries where the key position is ahead of the query
    // position: where(upper_triangle, 0, scores) over the trailing (T, T)
    // axes, the same select form attention causal masks canonicalize to
    static instruction_ref
    apply_causal_mask(const onnx_parser::node_info& info, instruction_ref scores, std::size_t t)
    {
        const auto& lens = scores->get_shape().lens();
        shape mask_shape{shape::bool_type, {t, t}};
        std::vector<int8_t> upper(mask_shape.elements());
        shape_for_each(mask_shape,
                       [&](const auto& idx, std::size_t i) { upper[i] = idx[1] > idx[0] ? 1 : 0; });
        auto mask = info.add_literal(literal{mask_shape, upper});
        mask      = info.add_instruction(make_op("multibroadcast", {{"out_lens", lens}}), mask);
        auto zero = info.add_literal(literal{shape{scores->get_shape().type(), {1}}, {0.0f}});
        zero      = info.add_instruction(make_op("multibroadcast", {{"out_lens", lens}}), zero);
        return info.add_instruction(make_op("where"), mask, zero, scores);
    }

    // Parallel emission for the linear rule: two batched matmuls with a
    // causal mask in between. The matmuls run in the input type with their
    // results accumulated in float, so the score matrix is rounded to the
    // input type once between the two (a fused gemm-gemm kernel keeps it in
    // the accumulation type instead).
    std::vector<instruction_ref> parse_masked(const onnx_parser::node_info& info,
                                              const linear_attention_attributes& attrs,
                                              const linear_attention_dims& dims,
                                              const std::vector<instruction_ref>& args) const
    {
        const auto b     = dims.batch_size;
        const auto t     = dims.sequence_length;
        const auto d_k   = dims.head_size_k;
        const auto d_v   = dims.head_size_v;
        const auto heads = attrs.kv_num_heads;
        const auto group = attrs.group_size();

        const auto in_type    = args.at(0)->get_shape().type();
        const auto state_type = has_input(args, 3) ? args.at(3)->get_shape().type() : in_type;

        // query: (B, T, Hq*dk) -> (B, Hkv, group, T, dk)
        auto q = info.add_instruction(make_op("reshape", {{"dims", {b, t, heads, group, d_k}}}),
                                      args.at(0));
        q      = info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 3, 1, 4}}}), q);
        // key: (B, T, Hkv*dk) -> (B, Hkv, dk, T), value: (B, T, Hkv*dv) -> (B, Hkv, T, dv)
        auto k_cols = to_columns(info, args.at(1), dims, heads, d_k);
        auto v =
            info.add_instruction(make_op("reshape", {{"dims", {b, t, heads, d_v}}}), args.at(2));
        v = info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), v);

        // scores tril(Q K^T): (B, Hkv, group, T, T). The mask stage runs in
        // float; the convert back to the input type feeds the second matmul
        // (a fused gemm-gemm kernel elides this round trip and keeps the
        // scores in the accumulation type).
        auto scores = info.add_instruction(make_op("dot"), q, broadcast_group(info, k_cols, group));
        scores      = to_float(info, scores);
        scores      = apply_causal_mask(info, scores, t);
        scores      = from_float(info, scores, in_type);

        auto output = info.add_instruction(make_op("dot"), scores, broadcast_group(info, v, group));
        output      = to_float(info, output);

        // initial state contribution Q S_0
        if(has_input(args, 3))
        {
            auto s05  = broadcast_group(info, args.at(3), group);
            auto init = info.add_instruction(make_op("dot"), q, s05);
            init      = to_float(info, init);
            output    = info.add_instruction(make_op("add"), output, init);
        }

        auto scale_val = float_equal(attrs.scale, 0.0f) ? 1.0f / std::sqrt(static_cast<float>(d_k))
                                                        : attrs.scale;
        auto scale_lit = info.add_literal(literal{shape{shape::float_type, {1}}, {scale_val}});
        scale_lit      = info.add_instruction(
            make_op("multibroadcast", {{"out_lens", output->get_shape().lens()}}), scale_lit);
        output = info.add_instruction(make_op("mul"), output, scale_lit);

        // (B, Hkv, group, T, dv) -> (B, T, Hq*dv)
        output =
            info.add_instruction(make_op("transpose", {{"permutation", {0, 3, 1, 2, 4}}}), output);
        output = info.add_instruction(
            make_op("reshape", {{"dims", {b, t, attrs.q_num_heads * d_v}}}), output);
        output = from_float(info, output, in_type);

        // present state S_0 + K^T V: (B, Hkv, dk, T) x (B, Hkv, T, dv)
        auto state   = info.add_instruction(make_op("dot"), k_cols, v);
        auto present = state;
        if(has_input(args, 3))
        {
            auto state_f = to_float(info, state);
            auto s0      = to_float(info, args.at(3));
            auto sum     = info.add_instruction(make_op("add"), state_f, s0);
            present      = from_float(info, sum, state_type);
        }
        return {output, present};
    }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       const std::vector<instruction_ref>& args) const
    {
        auto attrs = parse_attributes(parser, info);
        auto dims  = validate_inputs(args, attrs);

        // the linear rule's recurrence has a closed form over the whole
        // sequence; a single step is cheaper as one state update
        if(attrs.update_rule == "linear" and dims.sequence_length > 1)
            return parse_masked(info, attrs, dims, args);

        const auto b     = dims.batch_size;
        const auto t     = dims.sequence_length;
        const auto d_k   = dims.head_size_k;
        const auto d_v   = dims.head_size_v;
        const auto heads = attrs.kv_num_heads;
        const auto group = attrs.group_size();

        const auto in_type    = args.at(0)->get_shape().type();
        const auto state_type = has_input(args, 3) ? args.at(3)->get_shape().type() : in_type;

        // query: (B, T, Hq*dk) -> (B, Hkv, group, T, dk); the group axis maps
        // each block of group consecutive query heads onto one kv head
        auto q = to_float(info, args.at(0));
        q      = info.add_instruction(make_op("reshape", {{"dims", {b, t, heads, group, d_k}}}), q);
        q      = info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 3, 1, 4}}}), q);

        auto k_cols = to_columns(info, to_float(info, args.at(1)), dims, heads, d_k);
        // value: (B, T, Hkv*dv) -> (B, Hkv, T, dv)
        auto v = to_float(info, args.at(2));
        v      = info.add_instruction(make_op("reshape", {{"dims", {b, t, heads, d_v}}}), v);
        v      = info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), v);

        // state: (B, Hkv, dk, dv); left unset until the first update when
        // there is no past_state, since S_0 = 0 makes that step a plain outer
        // product
        instruction_ref state;
        bool has_state = has_input(args, 3);
        if(has_state)
            state = to_float(info, args.at(3));
        const std::vector<std::size_t> state_lens{b, heads, d_k, d_v};

        // decay: (B, T, Hkv) or (B, T, Hkv*dk) -> exp(g) as (B, Hkv, c, T)
        instruction_ref decay_cols;
        if(attrs.gated())
        {
            auto decay_channels = args.at(4)->get_shape().lens().at(2) / heads;
            decay_cols = to_columns(info, to_float(info, args.at(4)), dims, heads, decay_channels);
            decay_cols = info.add_instruction(make_op("exp"), decay_cols);
        }

        // beta: (B, T, Hb) -> (B, Hb, 1, T) so a per-step slice yields a
        // (B, Hb, 1, 1) scalar per head
        instruction_ref beta;
        if(attrs.delta())
        {
            auto beta_heads = args.at(5)->get_shape().lens().at(2);
            beta            = to_columns(info, to_float(info, args.at(5)), dims, beta_heads, 1);
        }

        std::vector<instruction_ref> outputs;
        outputs.reserve(t);
        for(std::size_t i = 0; i < t; i++)
        {
            auto k_col = slice_step(info, k_cols, 3, i); // (B, Hkv, dk, 1)
            auto v_t   = slice_step(info, v, 2, i);      // (B, Hkv, 1, dv)

            if(attrs.gated() and has_state)
            {
                auto e_t = slice_step(info, decay_cols, 3, i); // (B, Hkv, c, 1)
                e_t   = info.add_instruction(make_op("multibroadcast", {{"out_lens", state_lens}}),
                                             e_t);
                state = info.add_instruction(make_op("mul"), state, e_t);
            }
            if(attrs.delta())
            {
                if(has_state)
                {
                    auto k_row = info.add_instruction(
                        make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k_col);
                    // S^T k as a row vector: (B, Hkv, 1, dk) x (B, Hkv, dk, dv)
                    auto retrieved = info.add_instruction(make_op("dot"), k_row, state);
                    v_t            = info.add_instruction(make_op("sub"), v_t, retrieved);
                }
                auto b_t = slice_step(info, beta, 3, i); // (B, Hb, 1, 1)
                b_t      = info.add_instruction(
                    make_op("multibroadcast", {{"out_lens", v_t->get_shape().lens()}}), b_t);
                v_t = info.add_instruction(make_op("mul"), b_t, v_t);
            }

            // outer product k_t (x) v_t: (B, Hkv, dk, 1) x (B, Hkv, 1, dv)
            auto outer = info.add_instruction(make_op("dot"), k_col, v_t);
            state      = has_state ? info.add_instruction(make_op("add"), state, outer) : outer;
            has_state  = true;

            auto q_t = slice_step(info, q, 3, i); // (B, Hkv, group, 1, dk)
            q_t      = info.add_instruction(make_op("squeeze", {{"axes", {3}}}), q_t);
            // read the updated state: (B, Hkv, group, dk) x (B, Hkv, dk, dv)
            auto o_t = info.add_instruction(make_op("dot"), q_t, state);
            outputs.push_back(info.add_instruction(make_op("unsqueeze", {{"axes", {3}}}), o_t));
        }

        auto output = outputs.size() == 1
                          ? outputs.front()
                          : info.add_instruction(make_op("concat", {{"axis", 3}}), outputs);

        auto scale_val = float_equal(attrs.scale, 0.0f) ? 1.0f / std::sqrt(static_cast<float>(d_k))
                                                        : attrs.scale;
        auto scale_lit = info.add_literal(literal{shape{shape::float_type, {1}}, {scale_val}});
        scale_lit      = info.add_instruction(
            make_op("multibroadcast", {{"out_lens", output->get_shape().lens()}}), scale_lit);
        output = info.add_instruction(make_op("mul"), output, scale_lit);

        // (B, Hkv, group, T, dv) -> (B, T, Hq*dv)
        output =
            info.add_instruction(make_op("transpose", {{"permutation", {0, 3, 1, 2, 4}}}), output);
        output = info.add_instruction(
            make_op("reshape", {{"dims", {b, t, attrs.q_num_heads * d_v}}}), output);
        output = from_float(info, output, in_type);

        auto present_state = from_float(info, state, state_type);
        return {output, present_state};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
