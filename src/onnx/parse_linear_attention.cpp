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
#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

// Unified linear attention (ai.onnx opset 27, com.microsoft opset 1). The
// recurrence
//   S~ = exp(g_t) * S_{t-1};  S_t = S~ + k_t (x) beta_t * (v_t - S~^T k_t)
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
        std::size_t chunk_size   = 64;

        bool gated() const { return update_rule == "gated" or update_rule == "gated_delta"; }
        bool delta() const { return update_rule == "delta" or update_rule == "gated_delta"; }
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
        // Standard GQA groups query heads onto one state head, inverse GQA
        // (Qwen3.5 gated delta nets) groups state heads onto one query head
        if(q_heads <= 0 or kv_heads <= 0 or (q_heads % kv_heads != 0 and kv_heads % q_heads != 0))
        {
            MIGRAPHX_THROW("LinearAttention: q_num_heads (" + std::to_string(q_heads) +
                           ") and kv_num_heads (" + std::to_string(kv_heads) +
                           ") must be positive with one a multiple of the other");
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
        // chunk_size only blocks the prefill differently; the result is the same
        if(contains(info.attributes, "chunk_size"))
        {
            auto chunk = parser.parse_value(info.attributes.at("chunk_size")).at<std::int64_t>();
            if(chunk <= 0)
                MIGRAPHX_THROW("LinearAttention: chunk_size must be positive, got " +
                               std::to_string(chunk));
            attrs.chunk_size = chunk;
        }
        // A state window makes both state tensors (W, B, H_kv, d_k, d_v) windows
        // of the last W per-token states instead of the final one
        if(contains(info.attributes, "state_window") and
           parser.parse_value(info.attributes.at("state_window")).at<std::int64_t>() != 0)
        {
            MIGRAPHX_THROW("LinearAttention: state_window is not supported");
        }
        return attrs;
    }

    // Head bookkeeping. The recurrence runs once per state head; the query and
    // key heads it reads can be coarser, with group_size() state heads sharing
    // one query head under inverse GQA (kv_num_heads > q_num_heads) and
    // group_size() query heads sharing one state head otherwise. Either way the
    // head axes form the (outer, group) pair that the emitted layouts carry, and
    // the operator's output has one head per pair.
    struct linear_attention_dims
    {
        std::size_t batch_size      = 0;
        std::size_t sequence_length = 0;
        std::size_t head_size_k     = 0;
        std::size_t head_size_v     = 0;
        std::size_t q_heads         = 0;
        std::size_t k_heads         = 0;
        std::size_t state_heads     = 0;

        bool inverse() const { return state_heads > q_heads; }
        std::size_t outer_heads() const { return std::min(q_heads, state_heads); }
        std::size_t out_heads() const { return std::max(q_heads, state_heads); }
        std::size_t group_size() const { return out_heads() / outer_heads(); }
        // Inverse GQA splits the state's head axis into the (outer, group) pair,
        // so the state and everything multiplied into it gains one axis
        std::size_t state_ndim() const { return inverse() ? 5 : 4; }
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
        dims.q_heads         = attrs.q_num_heads;
        dims.state_heads     = attrs.kv_num_heads;
        if(dims.sequence_length == 0)
            MIGRAPHX_THROW("LinearAttention: sequence length must be at least 1");
        if(q_lens.at(2) == 0 or k_lens.at(2) == 0 or v_lens.at(2) == 0)
            MIGRAPHX_THROW("LinearAttention: hidden sizes must be positive");
        if(q_lens.at(2) % attrs.q_num_heads != 0 or v_lens.at(2) % attrs.kv_num_heads != 0)
            MIGRAPHX_THROW("LinearAttention: hidden sizes must be divisible by the head counts");
        dims.head_size_k = q_lens.at(2) / attrs.q_num_heads;
        dims.head_size_v = v_lens.at(2) / attrs.kv_num_heads;

        check_batch_seq(args.at(1)->get_shape(), dims, "key");
        check_batch_seq(args.at(2)->get_shape(), dims, "value");
        // The key head size comes from the query; its head count is read back
        // from the hidden size since the key may follow either head count
        if(k_lens.at(2) % dims.head_size_k != 0)
            MIGRAPHX_THROW("LinearAttention: key hidden size must be a multiple of the query head "
                           "size");
        dims.k_heads = k_lens.at(2) / dims.head_size_k;
        // The keys are read either per state head or once per query head group,
        // so any other count would need the head axis itself broadcast
        if(dims.k_heads != dims.outer_heads() and dims.k_heads != dims.state_heads)
            MIGRAPHX_THROW("LinearAttention: key head count (" + std::to_string(dims.k_heads) +
                           ") must be kv_num_heads (" + std::to_string(dims.state_heads) +
                           "), or q_num_heads (" + std::to_string(dims.q_heads) +
                           ") when kv_num_heads is larger");

        if(has_input(args, 3))
        {
            std::vector<std::size_t> state_lens{
                dims.batch_size, dims.state_heads, dims.head_size_k, dims.head_size_v};
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
            if(decay_shape.lens().at(2) != dims.state_heads and
               decay_shape.lens().at(2) != dims.state_heads * dims.head_size_k)
                MIGRAPHX_THROW("LinearAttention: decay last dim must be kv_num_heads or "
                               "kv_num_heads * k head size");
        }
        if(has_input(args, 5))
        {
            const auto& beta_shape = args.at(5)->get_shape();
            check_rank3(beta_shape, "beta");
            check_batch_seq(beta_shape, dims, "beta");
            if(beta_shape.lens().at(2) != dims.state_heads and beta_shape.lens().at(2) != 1)
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

    // Head axes of a tensor with one entry per state head: inverse GQA splits
    // them into the (outer, group) pair, which is exact because the group's
    // state heads are adjacent in the packed layout
    static std::vector<std::size_t> state_head_axes(const linear_attention_dims& dims,
                                                    std::size_t heads)
    {
        if(dims.inverse() and heads == dims.state_heads)
            return {dims.outer_heads(), dims.group_size()};
        return {heads};
    }

    // The query is indexed by output head: standard GQA splits it into the
    // (state head, group) pair so the group's reads become the rows of one gemm,
    // while inverse GQA shares one query head with the whole group
    static std::vector<std::size_t> query_head_axes(const linear_attention_dims& dims)
    {
        if(dims.inverse())
            return {dims.q_heads};
        return {dims.outer_heads(), dims.group_size()};
    }

    // (B, prod(seq), prod(heads) * d) -> (B, seq..., heads..., d); the
    // sequence splits into (chunks, chunk length) for the chunked prefill
    static instruction_ref unpack_heads(const onnx_parser::node_info& info,
                                        instruction_ref ins,
                                        const linear_attention_dims& dims,
                                        const std::vector<std::size_t>& seq,
                                        const std::vector<std::size_t>& heads,
                                        std::size_t head_size)
    {
        std::vector<std::size_t> unpacked{dims.batch_size};
        unpacked.insert(unpacked.end(), seq.begin(), seq.end());
        unpacked.insert(unpacked.end(), heads.begin(), heads.end());
        unpacked.push_back(head_size);
        return info.add_instruction(make_op("reshape", {{"dims", unpacked}}), ins);
    }

    // (B, T, prod(heads) * d) -> (B, heads..., d, T) so a per-step slice of the
    // last axis yields a (B, heads..., d, 1) column
    static instruction_ref to_columns(const onnx_parser::node_info& info,
                                      instruction_ref ins,
                                      const linear_attention_dims& dims,
                                      const std::vector<std::size_t>& heads,
                                      std::size_t head_size)
    {
        auto r    = unpack_heads(info, ins, dims, {dims.sequence_length}, heads, head_size);
        auto rank = r->get_shape().ndim();
        std::vector<std::int64_t> perm(rank);
        perm.front() = 0;
        std::iota(perm.begin() + 1, perm.end() - 1, 2);
        perm.back() = 1;
        return info.add_instruction(make_op("transpose", {{"permutation", perm}}), r);
    }

    // (B, prod(seq), prod(heads) * d) -> (B, heads..., seq..., d) so a slice of
    // the second to last axis yields a (B, heads..., 1, d) row
    static instruction_ref rows_layout(const onnx_parser::node_info& info,
                                       instruction_ref ins,
                                       const linear_attention_dims& dims,
                                       const std::vector<std::size_t>& seq,
                                       const std::vector<std::size_t>& heads,
                                       std::size_t head_size)
    {
        auto r    = unpack_heads(info, ins, dims, seq, heads, head_size);
        auto rank = r->get_shape().ndim();
        std::vector<std::int64_t> perm(rank);
        perm.front() = 0;
        auto it      = std::next(perm.begin());
        std::iota(it, std::next(it, heads.size()), 1 + seq.size());
        std::advance(it, heads.size());
        std::iota(it, std::next(it, seq.size()), 1);
        perm.back() = rank - 1;
        return info.add_instruction(make_op("transpose", {{"permutation", perm}}), r);
    }

    static instruction_ref to_rows(const onnx_parser::node_info& info,
                                   instruction_ref ins,
                                   const linear_attention_dims& dims,
                                   const std::vector<std::size_t>& heads,
                                   std::size_t head_size)
    {
        return rows_layout(info, ins, dims, {dims.sequence_length}, heads, head_size);
    }

    // The state keeps its (B, H_kv, d_k, d_v) layout at the operator boundary;
    // inverse GQA views the state heads as the (outer, group) pair in between
    static instruction_ref unpack_state(const onnx_parser::node_info& info,
                                        instruction_ref ins,
                                        const linear_attention_dims& dims)
    {
        if(not dims.inverse())
            return ins;
        return info.add_instruction(make_op("reshape",
                                            {{"dims",
                                              {dims.batch_size,
                                               dims.outer_heads(),
                                               dims.group_size(),
                                               dims.head_size_k,
                                               dims.head_size_v}}}),
                                    ins);
    }

    static instruction_ref pack_state(const onnx_parser::node_info& info,
                                      instruction_ref ins,
                                      const linear_attention_dims& dims)
    {
        if(not dims.inverse())
            return ins;
        return info.add_instruction(
            make_op("reshape",
                    {{"dims",
                      {dims.batch_size, dims.state_heads, dims.head_size_k, dims.head_size_v}}}),
            ins);
    }

    // Give a gate shared by a whole group (a single-head beta) the state's rank
    // so the elementwise state updates broadcast axis by axis
    static instruction_ref align_state_rank(const onnx_parser::node_info& info,
                                            instruction_ref ins,
                                            const linear_attention_dims& dims)
    {
        if(ins->get_shape().ndim() >= dims.state_ndim())
            return ins;
        return info.add_instruction(make_op("unsqueeze", {{"axes", {2}}}), ins);
    }

    // Scale the float output and pack (B, outer, group, T, dv) back to the
    // operator's (B, T, max(H_q, H_kv) * dv) layout in the input type
    static instruction_ref scale_and_pack_output(const onnx_parser::node_info& info,
                                                 const linear_attention_attributes& attrs,
                                                 const linear_attention_dims& dims,
                                                 instruction_ref output,
                                                 shape::type_t in_type)
    {
        auto scale_val = float_equal(attrs.scale, 0.0f)
                             ? 1.0f / std::sqrt(static_cast<float>(dims.head_size_k))
                             : attrs.scale;
        auto scale_lit = info.add_literal(literal{shape{shape::float_type, {1}}, {scale_val}});
        output         = info.add_common_op("mul", output, scale_lit);
        output =
            info.add_instruction(make_op("transpose", {{"permutation", {0, 3, 1, 2, 4}}}), output);
        output = info.add_instruction(
            make_op(
                "reshape",
                {{"dims",
                  {dims.batch_size, dims.sequence_length, dims.out_heads() * dims.head_size_v}}}),
            output);
        return from_float(info, output, in_type);
    }

    static instruction_ref slice_step(const onnx_parser::node_info& info,
                                      instruction_ref ins,
                                      std::int64_t axis,
                                      std::size_t i)
    {
        return info.add_instruction(
            make_op("slice", {{"axes", {axis}}, {"starts", {i}}, {"ends", {i + 1}}}), ins);
    }

    // The sequence axis is the last axis of a column layout and the second to
    // last of a row layout, whatever head axes precede it
    static instruction_ref
    slice_column(const onnx_parser::node_info& info, instruction_ref ins, std::size_t i)
    {
        return slice_step(info, ins, ins->get_shape().ndim() - 1, i);
    }

    static instruction_ref
    slice_row(const onnx_parser::node_info& info, instruction_ref ins, std::size_t i)
    {
        return slice_step(info, ins, ins->get_shape().ndim() - 2, i);
    }

    // Swap the trailing matrix axes, leaving the batch and head axes in place
    static instruction_ref transpose_matrix(const onnx_parser::node_info& info, instruction_ref ins)
    {
        std::vector<std::int64_t> perm(ins->get_shape().ndim());
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(perm.back(), *std::prev(perm.end(), 2));
        return info.add_instruction(make_op("transpose", {{"permutation", perm}}), ins);
    }

    // (B, H, x, y) -> (B, H, group, x, y): broadcast a tensor shared by a head
    // group across the group axis so the batched dots pair every output head
    // with the state it reads
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

    // Batched dot over the head axes: whichever operand is shared by the head
    // group carries one axis less and is broadcast over the group first
    static instruction_ref grouped_dot(const onnx_parser::node_info& info,
                                       const linear_attention_dims& dims,
                                       instruction_ref a,
                                       instruction_ref b)
    {
        auto rank = std::max(a->get_shape().ndim(), b->get_shape().ndim());
        if(a->get_shape().ndim() < rank)
            a = broadcast_group(info, a, dims.group_size());
        if(b->get_shape().ndim() < rank)
            b = broadcast_group(info, b, dims.group_size());
        return info.add_instruction(make_op("dot"), a, b);
    }

    // where(mask, 0, scores) over the trailing (t, t) axes for a (t, t) mask
    static instruction_ref zero_masked(const onnx_parser::node_info& info,
                                       instruction_ref scores,
                                       const shape& mask_shape,
                                       const std::vector<int8_t>& mask_data)
    {
        const auto& lens = scores->get_shape().lens();
        auto mask        = info.add_literal(literal{mask_shape, mask_data});
        mask      = info.add_instruction(make_op("multibroadcast", {{"out_lens", lens}}), mask);
        auto zero = info.add_literal(literal{shape{scores->get_shape().type(), {1}}, {0.0f}});
        zero      = info.add_instruction(make_op("multibroadcast", {{"out_lens", lens}}), zero);
        return info.add_instruction(make_op("where"), mask, zero, scores);
    }

    // Zero the score entries where the key position is ahead of the query
    // position: where(upper_triangle, 0, scores) over the trailing (T, T)
    // axes, the same select form attention causal masks canonicalize to
    static instruction_ref
    apply_causal_mask(const onnx_parser::node_info& info, instruction_ref scores, std::size_t t)
    {
        shape mask_shape{shape::bool_type, {t, t}};
        std::vector<int8_t> upper(mask_shape.elements());
        shape_for_each(mask_shape,
                       [&](const auto& idx, std::size_t i) { upper[i] = idx[1] > idx[0] ? 1 : 0; });
        return zero_masked(info, scores, mask_shape, upper);
    }

    // Also zero the diagonal, leaving only keys strictly before the query
    static instruction_ref apply_strict_causal_mask(const onnx_parser::node_info& info,
                                                    instruction_ref scores,
                                                    std::size_t t)
    {
        shape mask_shape{shape::bool_type, {t, t}};
        std::vector<int8_t> upper(mask_shape.elements());
        shape_for_each(mask_shape, [&](const auto& idx, std::size_t i) {
            upper[i] = idx[1] >= idx[0] ? 1 : 0;
        });
        return zero_masked(info, scores, mask_shape, upper);
    }

    // Elementwise op between tensors that may differ in carrying the group
    // axis; add_common_op aligns ranks from the right, so insert it first
    static instruction_ref group_common_op(const onnx_parser::node_info& info,
                                           const std::string& name,
                                           instruction_ref a,
                                           instruction_ref b)
    {
        if(a->get_shape().ndim() < b->get_shape().ndim())
            a = info.add_instruction(make_op("unsqueeze", {{"axes", {2}}}), a);
        else if(b->get_shape().ndim() < a->get_shape().ndim())
            b = info.add_instruction(make_op("unsqueeze", {{"axes", {2}}}), b);
        return info.add_common_op(name, a, b);
    }

    // Zero-pad the sequence axis of a packed (B, T, X) input up to t_pad
    static instruction_ref
    pad_sequence(const onnx_parser::node_info& info, instruction_ref ins, std::size_t t_pad)
    {
        auto t = ins->get_shape().lens().at(1);
        if(t == t_pad)
            return ins;
        std::vector<std::int64_t> pads{0, 0, 0, 0, static_cast<std::int64_t>(t_pad - t), 0};
        return info.add_instruction(make_op("pad", {{"pads", pads}}), ins);
    }

    // A packed input as float rows (B, heads..., n, c, d) of n chunks of c
    // steps, zero padded to whole chunks: a padded step has k = v = beta = 0
    // and g = 0, which leaves the state untouched
    static instruction_ref chunk_rows(const onnx_parser::node_info& info,
                                      const linear_attention_dims& dims,
                                      instruction_ref ins,
                                      std::size_t n,
                                      std::size_t c,
                                      const std::vector<std::size_t>& heads,
                                      std::size_t head_size)
    {
        auto padded = pad_sequence(info, to_float(info, ins), n * c);
        return rows_layout(info, padded, dims, {n, c}, heads, head_size);
    }

    // Chunk i of a (B, heads..., n, c, d) tensor as (B, heads..., c, d)
    static instruction_ref
    slice_chunk(const onnx_parser::node_info& info, instruction_ref ins, std::size_t i)
    {
        std::int64_t axis = ins->get_shape().ndim() - 3;
        if(ins->get_shape().lens().at(axis) > 1)
            ins = slice_step(info, ins, axis, i);
        return info.add_instruction(make_op("squeeze", {{"axes", {axis}}}), ins);
    }

    // (I + M)^-1 for a strictly lower triangular (c, c) M. M is nilpotent, so
    // the Neumann series is finite and factors into log2(c) squarings:
    //   (I + M)^-1 = (I - M)(I + M^2)(I + M^4)...(I + M^(2^(k-1))),  2^k >= c
    static instruction_ref
    invert_unit_lower(const onnx_parser::node_info& info, instruction_ref m, std::size_t c)
    {
        shape eye_shape{shape::float_type, {c, c}};
        std::vector<float> eye_data(eye_shape.elements());
        shape_for_each(eye_shape, [&](const auto& idx, std::size_t i) {
            eye_data[i] = idx[0] == idx[1] ? 1.0f : 0.0f;
        });
        auto eye = info.add_literal(literal{eye_shape, eye_data});
        eye = info.add_instruction(make_op("multibroadcast", {{"out_lens", m->get_shape().lens()}}),
                                   eye);
        auto result = info.add_instruction(make_op("sub"), eye, m);
        auto power  = m;
        for(std::size_t width = 2; width < c; width *= 2)
        {
            power       = info.add_instruction(make_op("dot"), power, power);
            auto factor = info.add_instruction(make_op("add"), eye, power);
            result      = info.add_instruction(make_op("dot"), result, factor);
        }
        return result;
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
        const auto t   = dims.sequence_length;
        const auto d_k = dims.head_size_k;
        const auto d_v = dims.head_size_v;

        const auto in_type = args.at(0)->get_shape().type();

        auto q      = to_rows(info, args.at(0), dims, query_head_axes(dims), d_k);
        auto k_cols = to_columns(info, args.at(1), dims, state_head_axes(dims, dims.k_heads), d_k);
        auto v      = to_rows(info, args.at(2), dims, state_head_axes(dims, dims.state_heads), d_v);

        instruction_ref s0;
        if(has_input(args, 3))
            s0 = unpack_state(info, args.at(3), dims);

        // scores tril(Q K^T): (B, outer, group, T, T)
        auto scores = grouped_dot(info, dims, q, k_cols);
        scores      = to_float(info, scores);
        scores      = apply_causal_mask(info, scores, t);
        scores      = from_float(info, scores, in_type);

        auto output = grouped_dot(info, dims, scores, v);
        output      = to_float(info, output);

        // initial state contribution Q S_0
        if(has_input(args, 3))
        {
            auto init = grouped_dot(info, dims, q, s0);
            init      = to_float(info, init);
            output    = info.add_instruction(make_op("add"), output, init);
        }

        output = scale_and_pack_output(info, attrs, dims, output, in_type);

        // present state S_0 + K^T V: (B, .., dk, T) x (B, .., T, dv)
        auto present = grouped_dot(info, dims, k_cols, v);
        if(has_input(args, 3))
        {
            auto state_f = to_float(info, present);
            auto sum     = info.add_instruction(make_op("add"), state_f, to_float(info, s0));
            present      = from_float(info, sum, args.at(3)->get_shape().type());
        }
        return {output, pack_state(info, present, dims)};
    }

    // Chunked emission of the gated and delta rules (gated delta networks,
    // Yang et al. 2024). Over a chunk of c steps with gc the within-chunk
    // cumulative log decay, the recurrence has the closed form
    //   D[t,s] = exp(gc_t - gc_s)                     (causal)
    //   M      = beta (K K^T) D                       (strictly lower)
    //   W      = (I + M)^-1 beta exp(gc) K
    //   U      = (I + M)^-1 beta V - W S_0
    //   O      = (Q K^T D) U + (Q exp(gc)) S_0
    //   S_1    = exp(gc_c) S_0 + (K exp(gc_c - gc))^T U
    // Everything but the state carry is batched over the chunks, so the
    // sequential part shrinks from T steps to T / c, each a few dots on the
    // state. Decays only ever appear as within-chunk differences, which keeps
    // the exponents bounded.
    std::vector<instruction_ref> parse_chunked(const onnx_parser::node_info& info,
                                               const linear_attention_attributes& attrs,
                                               const linear_attention_dims& dims,
                                               const std::vector<instruction_ref>& args) const
    {
        const auto t   = dims.sequence_length;
        const auto d_k = dims.head_size_k;
        const auto d_v = dims.head_size_v;
        const auto c   = std::min(attrs.chunk_size, t);
        const auto n   = (t + c - 1) / c;

        const auto in_type    = args.at(0)->get_shape().type();
        const auto state_type = has_input(args, 3) ? args.at(3)->get_shape().type() : in_type;

        auto q = chunk_rows(info, dims, args.at(0), n, c, query_head_axes(dims), d_k);
        auto k = chunk_rows(info, dims, args.at(1), n, c, state_head_axes(dims, dims.k_heads), d_k);
        auto v =
            chunk_rows(info, dims, args.at(2), n, c, state_head_axes(dims, dims.state_heads), d_v);
        auto kt = transpose_matrix(info, k);

        auto scores = grouped_dot(info, dims, q, kt); // Q K^T: (B, .., n, c, c)

        // Decay factors: q and the write-side k scaled by exp(gc), the read-side
        // k by the decay to the chunk end, and the pair matrices by D
        instruction_ref gamma; // exp(gc_c): (B, .., n, 1, 1)
        instruction_ref decay; // D, causal
        auto q_gated  = q;
        auto k_gated  = k;
        auto k_to_end = k;
        if(attrs.gated())
        {
            auto g = chunk_rows(
                info, dims, args.at(4), n, c, state_head_axes(dims, dims.state_heads), 1);
            std::int64_t step_axis = g->get_shape().ndim() - 2;
            auto gc = info.add_instruction(make_op("prefix_scan_sum", {{"axis", step_axis}}), g);
            auto gc_end = slice_step(info, gc, step_axis, c - 1);
            auto exp_gc = info.add_instruction(make_op("exp"), gc);
            gamma       = info.add_instruction(make_op("exp"), gc_end);
            auto to_end = info.add_common_op("sub", gc_end, gc);
            to_end      = info.add_instruction(make_op("exp"), to_end);
            q_gated     = group_common_op(info, "mul", q, exp_gc);
            k_gated     = group_common_op(info, "mul", k, exp_gc);
            k_to_end    = group_common_op(info, "mul", k, to_end);
            // the upper triangle holds the decays' negatives, which exp could
            // overflow, so it is zeroed before as well as after
            auto pair = info.add_common_op("sub", gc, transpose_matrix(info, gc));
            pair      = apply_causal_mask(info, pair, c);
            decay     = info.add_instruction(make_op("exp"), pair);
            decay     = apply_causal_mask(info, decay, c);
            scores    = group_common_op(info, "mul", scores, decay);
        }
        else
        {
            scores = apply_causal_mask(info, scores, c);
        }

        // Delta rule: pseudo values U_0 = (I + M)^-1 beta V and W; the plain
        // gated rule writes V as is
        auto u0 = v;
        instruction_ref w;
        if(attrs.delta())
        {
            auto beta_heads = args.at(5)->get_shape().lens().at(2);
            auto beta =
                chunk_rows(info, dims, args.at(5), n, c, state_head_axes(dims, beta_heads), 1);
            auto gram = grouped_dot(info, dims, k, kt); // K K^T
            if(attrs.gated())
                gram = group_common_op(info, "mul", gram, decay);
            auto m    = group_common_op(info, "mul", beta, gram);
            m         = apply_strict_causal_mask(info, m, c);
            auto ninv = invert_unit_lower(info, m, c);
            auto ubar = group_common_op(info, "mul", beta, v);
            auto wbar = group_common_op(info, "mul", beta, k_gated);
            u0        = grouped_dot(info, dims, ninv, ubar);
            w         = grouped_dot(info, dims, ninv, wbar);
        }

        // Intra-chunk output and the query that reads the incoming state:
        // O = P U_0 + (Q exp(gc) - P W) S_0
        auto pu0     = grouped_dot(info, dims, scores, u0);
        auto q_state = q_gated;
        if(attrs.delta())
            q_state = group_common_op(info, "sub", q_gated, grouped_dot(info, dims, scores, w));
        auto k_to_end_t = transpose_matrix(info, k_to_end);

        instruction_ref state;
        bool has_state = has_input(args, 3);
        if(has_state)
            state = to_float(info, unpack_state(info, args.at(3), dims));

        std::vector<instruction_ref> outputs;
        outputs.reserve(n);
        for(std::size_t i = 0; i < n; i++)
        {
            auto o_i = slice_chunk(info, pu0, i); // (B, outer, group, c, dv)
            auto u_i = slice_chunk(info, u0, i);  // (B, .., c, dv)
            if(has_state)
            {
                if(attrs.delta())
                {
                    auto ws = grouped_dot(info, dims, slice_chunk(info, w, i), state);
                    u_i     = group_common_op(info, "sub", u_i, ws);
                }
                auto qs = grouped_dot(info, dims, slice_chunk(info, q_state, i), state);
                o_i     = info.add_instruction(make_op("add"), o_i, qs);
            }
            auto update = grouped_dot(info, dims, slice_chunk(info, k_to_end_t, i), u_i);
            if(has_state)
            {
                if(attrs.gated())
                    state = group_common_op(info, "mul", state, slice_chunk(info, gamma, i));
                state = group_common_op(info, "add", state, update);
            }
            else
            {
                state = update;
            }
            has_state = true;
            outputs.push_back(o_i);
        }

        auto output = outputs.size() == 1
                          ? outputs.front()
                          : info.add_instruction(make_op("concat", {{"axis", 3}}), outputs);
        if(n * c != t)
            output = info.add_instruction(
                make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {t}}}), output);
        output = scale_and_pack_output(info, attrs, dims, output, in_type);

        auto present_state = pack_state(info, from_float(info, state, state_type), dims);
        return {output, present_state};
    }

    // Read the state updated by step i with that step's query. Standard GQA
    // makes the group axis the rows of one gemm per state head; inverse GQA
    // shares one query row across the group, so there the group stays a batch
    // axis of the dot. Both give a (B, outer, group, 1, dv) step output.
    static instruction_ref read_state(const onnx_parser::node_info& info,
                                      const linear_attention_dims& dims,
                                      instruction_ref q,
                                      instruction_ref state,
                                      std::size_t i)
    {
        auto q_t = slice_row(info, q, i);
        if(dims.inverse())
            return grouped_dot(info, dims, q_t, state);
        q_t      = info.add_instruction(make_op("squeeze", {{"axes", {3}}}), q_t);
        auto o_t = grouped_dot(info, dims, q_t, state);
        return info.add_instruction(make_op("unsqueeze", {{"axes", {3}}}), o_t);
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
        // A single step is cheapest as one state update (fewer kernels than the
        // chunk form); a per-key-dimension decay makes the chunk decay a vector
        // per token pair, which only the step-by-step recurrence handles
        bool vector_decay =
            attrs.gated() and args.at(4)->get_shape().lens().at(2) != dims.state_heads;
        if(dims.sequence_length > 1 and not vector_decay)
            return parse_chunked(info, attrs, dims, args);

        const auto t   = dims.sequence_length;
        const auto d_k = dims.head_size_k;
        const auto d_v = dims.head_size_v;

        const auto in_type    = args.at(0)->get_shape().type();
        const auto state_type = has_input(args, 3) ? args.at(3)->get_shape().type() : in_type;

        auto q      = to_rows(info, to_float(info, args.at(0)), dims, query_head_axes(dims), d_k);
        auto k_cols = to_columns(
            info, to_float(info, args.at(1)), dims, state_head_axes(dims, dims.k_heads), d_k);
        auto v = to_rows(
            info, to_float(info, args.at(2)), dims, state_head_axes(dims, dims.state_heads), d_v);

        // state: (B, Hkv, dk, dv), group-split under inverse GQA; left unset
        // until the first update when there is no past_state, since S_0 = 0
        // makes that step a plain outer product
        instruction_ref state;
        bool has_state = has_input(args, 3);
        if(has_state)
            state = to_float(info, unpack_state(info, args.at(3), dims));

        // decay: (B, T, Hkv) or (B, T, Hkv*dk) -> exp(g) as (B, .., c, T)
        instruction_ref decay_cols;
        if(attrs.gated())
        {
            auto decay_channels = args.at(4)->get_shape().lens().at(2) / dims.state_heads;
            decay_cols          = to_columns(info,
                                             to_float(info, args.at(4)),
                                             dims,
                                             state_head_axes(dims, dims.state_heads),
                                             decay_channels);
            decay_cols          = info.add_instruction(make_op("exp"), decay_cols);
        }

        // beta: (B, T, Hkv) or (B, T, 1) -> (B, .., 1, T) so a per-step slice
        // yields a (B, .., 1, 1) scalar per head
        instruction_ref beta;
        if(attrs.delta())
        {
            auto beta_heads = args.at(5)->get_shape().lens().at(2);
            beta            = to_columns(
                info, to_float(info, args.at(5)), dims, state_head_axes(dims, beta_heads), 1);
            beta = align_state_rank(info, beta, dims);
        }

        std::vector<instruction_ref> outputs;
        outputs.reserve(t);
        for(std::size_t i = 0; i < t; i++)
        {
            auto k_col = slice_column(info, k_cols, i); // (B, .., dk, 1)
            auto v_t   = slice_row(info, v, i);         // (B, .., 1, dv)

            if(attrs.gated() and has_state)
            {
                auto e_t = slice_column(info, decay_cols, i); // (B, .., c, 1)
                state    = info.add_common_op("mul", state, e_t);
            }
            if(attrs.delta())
            {
                if(has_state)
                {
                    auto k_row = transpose_matrix(info, k_col);
                    // S^T k as a row vector: (B, .., 1, dk) x (B, .., dk, dv)
                    auto retrieved = grouped_dot(info, dims, k_row, state);
                    v_t            = info.add_instruction(make_op("sub"), v_t, retrieved);
                }
                auto b_t = slice_column(info, beta, i); // (B, .., 1, 1)
                v_t      = info.add_common_op("mul", b_t, v_t);
            }

            // outer product k_t (x) v_t: (B, .., dk, 1) x (B, .., 1, dv)
            auto outer = grouped_dot(info, dims, k_col, v_t);
            state      = has_state ? info.add_instruction(make_op("add"), state, outer) : outer;
            has_state  = true;

            outputs.push_back(read_state(info, dims, q, state, i));
        }

        auto output = outputs.size() == 1
                          ? outputs.front()
                          : info.add_instruction(make_op("concat", {{"axis", 3}}), outputs);

        output = scale_and_pack_output(info, attrs, dims, output, in_type);

        auto present_state = pack_state(info, from_float(info, state, state_type), dims);
        return {output, present_state};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
