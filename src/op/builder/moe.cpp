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
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/builder/insert.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

// Mixture of experts block following the com.microsoft MoE/QMoE contract.
//
// Argument layout (optional arguments can be passed as "undefined" instructions
// or omitted from the end):
//   0: input            [num_tokens, hidden] or [batch, seq, hidden]
//   1: router_probs     [num_tokens, num_experts] routing logits
//   2: fc1_weights      [num_experts, fusion_size * inter, hidden(/pack)]
//   3: fc1_scales
//   4: fc1_bias         [num_experts, fusion_size * inter]
//   5: fc2_weights      [num_experts, hidden, inter(/pack)]
//   6: fc2_scales
//   7: fc2_bias         [num_experts, hidden]
//   8: fc3_weights      [num_experts, inter, hidden(/pack)]
//   9: fc3_scales
//  10: fc3_bias         [num_experts, inter]
//  11: fc1_zero_points
//  12: fc2_zero_points
//  13: fc3_zero_points
//
// Weights are stored as [num_experts, out_features, in_features]. When scales
// are provided for an fc, its weights are quantized with expert_weight_bits
// bits packed along the in_features axis. Scales (and packed zero points) are
// per output column when 2D, or blockwise along in_features when 3D. When zero
// points are absent, the default zero point is 2^(bits - 1).
//
// Routing follows onnxruntime: softmax over the logits, top-k selection, and
// optional renormalization of the selected weights. Only the top-k experts are
// gathered (and dequantized) per token before running the FFN.
struct moe : op_builder<moe>
{
    std::string activation_type    = "relu";
    float activation_alpha         = 1.0f;
    float activation_beta          = 0.0f;
    int64_t k                      = 1;
    bool normalize_routing_weights = false;
    int64_t swiglu_fusion          = 0;
    float swiglu_limit             = std::numeric_limits<float>::infinity();
    int64_t expert_weight_bits     = 4;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.activation_type, "activation_type"),
                    f(self.activation_alpha, "activation_alpha"),
                    f(self.activation_beta, "activation_beta"),
                    f(self.k, "k"),
                    f(self.normalize_routing_weights, "normalize_routing_weights"),
                    f(self.swiglu_fusion, "swiglu_fusion"),
                    f(self.swiglu_limit, "swiglu_limit"),
                    f(self.expert_weight_bits, "expert_weight_bits"));
    }

    enum arg_slot : std::size_t
    {
        slot_input = 0,
        slot_router,
        slot_fc1_weights,
        slot_fc1_scales,
        slot_fc1_bias,
        slot_fc2_weights,
        slot_fc2_scales,
        slot_fc2_bias,
        slot_fc3_weights,
        slot_fc3_scales,
        slot_fc3_bias,
        slot_fc1_zero_points,
        slot_fc2_zero_points,
        slot_fc3_zero_points
    };

    static std::optional<instruction_ref> arg(const std::vector<instruction_ref>& args,
                                              std::size_t i)
    {
        if(i >= args.size() or args[i]->is_undefined())
            return std::nullopt;
        return args[i];
    }

    static instruction_ref
    required(const std::vector<instruction_ref>& args, std::size_t i, const std::string& name)
    {
        auto r = arg(args, i);
        if(not r.has_value())
            MIGRAPHX_THROW("moe: missing required argument " + name);
        return *r;
    }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto x      = required(args, slot_input, "input");
        auto router = required(args, slot_router, "router_probs");
        auto fc1_w  = required(args, slot_fc1_weights, "fc1_weights");

        if(std::any_of(args.begin(), args.end(), [](instruction_ref a) {
               return a->get_shape().dynamic();
           }))
            MIGRAPHX_THROW("moe: dynamic shapes are not supported");

        const auto in_lens       = x->get_shape().lens();
        const std::size_t hidden = in_lens.back();
        if(hidden == 0)
            MIGRAPHX_THROW("moe: input hidden dimension must be non-zero");
        const std::size_t tokens = x->get_shape().elements() / hidden;
        if(fc1_w->get_shape().ndim() != 3)
            MIGRAPHX_THROW(
                "moe: fc1_weights must have shape [num_experts, out_features, in_features]");
        const std::size_t num_experts = fc1_w->get_shape().lens().front();
        const std::size_t fc1_out     = fc1_w->get_shape().lens().at(1);
        const std::size_t fusion_size =
            (activation_type == "swiglu" and swiglu_fusion != 0) ? 2 : 1;

        if(expert_weight_bits != 4 and expert_weight_bits != 8)
            MIGRAPHX_THROW("moe: only 4 and 8 bit expert weights are supported, got " +
                           std::to_string(expert_weight_bits));
        if(swiglu_fusion < 0 or swiglu_fusion > 2)
            MIGRAPHX_THROW("moe: invalid swiglu_fusion value " + std::to_string(swiglu_fusion));
        if(fc1_out % fusion_size != 0)
            MIGRAPHX_THROW("moe: fc1 output size must be divisible by the swiglu fusion size");
        const std::size_t inter = fc1_out / fusion_size;

        if(router->get_shape().lens() != std::vector<std::size_t>{tokens, num_experts})
            MIGRAPHX_THROW("moe: router_probs must have shape [num_tokens, num_experts]");
        if(k < 1)
            MIGRAPHX_THROW("moe: k must be at least 1");
        const std::size_t top_k = k;
        if(top_k > num_experts)
            MIGRAPHX_THROW("moe: k must not exceed the number of experts");

        // Routing: softmax over logits, then select the top-k experts per token
        auto probs = m.insert_instruction(ins, make_op("softmax", {{"axis", 1}}), router);
        auto top   = m.insert_instruction(
            ins, make_op("topk", {{"k", k}, {"axis", 1}, {"largest", true}}), probs);
        auto routing_weights =
            m.insert_instruction(ins, make_op("get_tuple_elem", {{"index", 0}}), top);
        auto expert_indices =
            m.insert_instruction(ins, make_op("get_tuple_elem", {{"index", 1}}), top);
        if(normalize_routing_weights)
        {
            auto sum =
                m.insert_instruction(ins, make_op("reduce_sum", {{"axes", {1}}}), routing_weights);
            routing_weights = insert_common_op(m, ins, "div", routing_weights, sum);
        }

        const std::size_t rows = tokens * top_k;
        auto selected =
            m.insert_instruction(ins, make_op("reshape", {{"dims", {rows}}}), expert_indices);

        // Replicate each token for its k selected experts: [rows, 1, hidden]
        auto xr = m.insert_instruction(ins, make_op("reshape", {{"dims", {tokens, 1, hidden}}}), x);
        xr      = m.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", {tokens, top_k, hidden}}}), xr);
        xr = m.insert_instruction(ins, make_op("reshape", {{"dims", {rows, 1, hidden}}}), xr);

        auto h1 = fc(m, ins, xr, selected, args, 0, num_experts, hidden, fc1_out);

        std::optional<instruction_ref> h3;
        if(arg(args, slot_fc3_weights).has_value())
            h3 = fc(m, ins, xr, selected, args, 2, num_experts, hidden, inter);

        auto activated = apply_activation(m, ins, h1, h3, inter);

        auto h2 = fc(m, ins, activated, selected, args, 1, num_experts, inter, hidden);

        // Weight each expert output by its routing weight and sum per token
        auto wr = m.insert_instruction(
            ins, make_op("reshape", {{"dims", {rows, 1, 1}}}), routing_weights);
        if(wr->get_shape().type() != h2->get_shape().type())
            wr = m.insert_instruction(
                ins, make_op("convert", {{"target_type", h2->get_shape().type()}}), wr);
        auto weighted = insert_common_op(m, ins, "mul", h2, wr);
        weighted      = m.insert_instruction(
            ins, make_op("reshape", {{"dims", {tokens, top_k, hidden}}}), weighted);
        auto summed = m.insert_instruction(ins, make_op("reduce_sum", {{"axes", {1}}}), weighted);
        auto out    = m.insert_instruction(ins, make_op("reshape", {{"dims", in_lens}}), summed);
        return {out};
    }

    private:
    // Validate the shapes and types of one fc's weight/scale/bias/zero point arguments
    void validate_fc(const std::string& name,
                     instruction_ref w,
                     const std::optional<instruction_ref>& scales,
                     const std::optional<instruction_ref>& bias,
                     const std::optional<instruction_ref>& zero_points,
                     std::size_t num_experts,
                     std::size_t in_features,
                     std::size_t out_features) const
    {
        const bool quantized = scales.has_value();
        const auto w_lens    = w->get_shape().lens();
        if(w_lens.size() != 3)
            MIGRAPHX_THROW("moe: " + name +
                           "_weights must have shape [num_experts, out_features, in_features]");
        if(w_lens.front() != num_experts)
            MIGRAPHX_THROW("moe: " + name + "_weights num_experts must be " +
                           std::to_string(num_experts));
        if(w_lens.at(1) != out_features)
            MIGRAPHX_THROW("moe: " + name + "_weights out_features must be " +
                           std::to_string(out_features));
        const std::size_t packed_in =
            (quantized and expert_weight_bits == 4) ? (in_features + 1) / 2 : in_features;
        if(w_lens.back() != packed_in)
            MIGRAPHX_THROW("moe: " + name + "_weights in_features must be " +
                           std::to_string(packed_in));
        if(quantized and w->get_shape().type() != shape::uint8_type)
            MIGRAPHX_THROW("moe: quantized " + name + "_weights must be uint8");
        if(scales.has_value())
        {
            const auto s_lens = (*scales)->get_shape().lens();
            if((s_lens.size() != 2 and s_lens.size() != 3) or s_lens.front() != num_experts or
               s_lens.at(1) != out_features)
                MIGRAPHX_THROW("moe: " + name +
                               "_scales must have shape [num_experts, out_features] or "
                               "[num_experts, out_features, nblocks]");
        }
        if(bias.has_value() and
           (*bias)->get_shape().lens() != std::vector<std::size_t>{num_experts, out_features})
            MIGRAPHX_THROW("moe: " + name + "_bias must have shape [num_experts, out_features]");
        if(zero_points.has_value())
        {
            if(not quantized)
                MIGRAPHX_THROW("moe: " + name + "_zero_points require " + name + "_scales");
            const auto z_lens = (*zero_points)->get_shape().lens();
            // 2D zero points pack the columns, 3D pack the blocks
            const std::size_t z_cols = (expert_weight_bits == 4 and z_lens.size() == 2)
                                           ? (out_features + 1) / 2
                                           : out_features;
            if((z_lens.size() != 2 and z_lens.size() != 3) or z_lens.front() != num_experts or
               z_lens.at(1) != z_cols)
                MIGRAPHX_THROW("moe: " + name + "_zero_points columns must be " +
                               std::to_string(z_cols) + " with " + std::to_string(num_experts) +
                               " experts");
        }
    }

    // Gathered expert GEMM for fc(n + 1): x_rows is [rows, 1, in_features],
    // returns [rows, 1, out_features]. Fetches the fc's weight/scale/bias/zero
    // point slots from args.
    instruction_ref fc(module& m,
                       instruction_ref ins,
                       instruction_ref x_rows,
                       instruction_ref selected,
                       const std::vector<instruction_ref>& args,
                       std::size_t n,
                       std::size_t num_experts,
                       std::size_t in_features,
                       std::size_t out_features) const
    {
        const auto name  = "fc" + std::to_string(n + 1);
        auto w           = required(args, slot_fc1_weights + 3 * n, name + "_weights");
        auto scales      = arg(args, slot_fc1_scales + 3 * n);
        auto bias        = arg(args, slot_fc1_bias + 3 * n);
        auto zero_points = arg(args, slot_fc1_zero_points + n);

        const bool quantized = scales.has_value();
        validate_fc(name, w, scales, bias, zero_points, num_experts, in_features, out_features);

        auto wg = m.insert_instruction(ins, make_op("gather", {{"axis", 0}}), w, selected);
        if(quantized)
        {
            auto sg =
                m.insert_instruction(ins, make_op("gather", {{"axis", 0}}), *scales, selected);
            std::optional<instruction_ref> zg;
            if(zero_points.has_value())
                zg = m.insert_instruction(
                    ins, make_op("gather", {{"axis", 0}}), *zero_points, selected);
            wg = dequantize(m, ins, wg, sg, zg, in_features);
        }
        auto x_type = x_rows->get_shape().type();
        if(wg->get_shape().type() != x_type)
            wg = m.insert_instruction(ins, make_op("convert", {{"target_type", x_type}}), wg);
        wg = m.insert_instruction(ins, make_op("transpose", {{"permutation", {0, 2, 1}}}), wg);
        auto out = m.insert_instruction(ins, make_op("dot"), x_rows, wg);
        if(bias.has_value())
        {
            auto bg = m.insert_instruction(ins, make_op("gather", {{"axis", 0}}), *bias, selected);
            bg      = m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {1}}}), bg);
            if(bg->get_shape().type() != x_type)
                bg = m.insert_instruction(ins, make_op("convert", {{"target_type", x_type}}), bg);
            out = m.insert_instruction(ins, make_op("add"), out, bg);
        }
        return out;
    }

    // Dequantize gathered expert weights [rows, n, in_features/pack] -> [rows, n, in_features]
    instruction_ref dequantize(module& m,
                               instruction_ref ins,
                               instruction_ref w,
                               instruction_ref scales,
                               const std::optional<instruction_ref>& zero_points,
                               std::size_t in_features) const
    {
        if(expert_weight_bits == 4)
            w = unpack(m, ins, w, in_features);

        auto sb = expand_to_in_features(m, ins, scales, in_features);
        instruction_ref zb;
        if(zero_points.has_value())
        {
            auto z = *zero_points;
            if(expert_weight_bits == 4)
            {
                auto z_lens = z->get_shape().lens();
                if(z_lens.size() == 3 and scales->get_shape().ndim() != 3)
                    MIGRAPHX_THROW("moe: blockwise zero points require blockwise scales");
                // 2D zero points are packed along the columns, 3D along the blocks
                auto expected = z_lens.size() == 3 ? scales->get_shape().lens().at(2)
                                                   : w->get_shape().lens().at(1);
                z             = unpack(m, ins, z, expected);
            }
            zb = expand_to_in_features(m, ins, z, in_features);
        }
        else
        {
            auto default_zp = m.add_literal(
                literal{shape{w->get_shape().type(), {1}}, {expert_weight_bits == 4 ? 8 : 128}});
            zb = m.insert_instruction(
                ins, make_op("multibroadcast", {{"out_lens", w->get_shape().lens()}}), default_zp);
        }
        return m.insert_instruction(ins, make_op("dequantizelinear"), w, sb, zb);
    }

    // Unpack two 4-bit values per byte along the last axis, trimming any pad
    static instruction_ref
    unpack(module& m, instruction_ref ins, instruction_ref x, std::size_t last_dim)
    {
        x = m.insert_instruction(ins, make_op("unpack_int4"), x);
        if(x->get_shape().lens().back() > last_dim)
        {
            auto axis = x->get_shape().ndim() - 1;
            x         = m.insert_instruction(
                ins,
                make_op("slice", {{"axes", {axis}}, {"starts", {0}}, {"ends", {last_dim}}}),
                x);
        }
        return x;
    }

    // Broadcast per-column [rows, n] or blockwise [rows, n, nblocks] quantization
    // parameters to the full weight shape [rows, n, in_features]
    static instruction_ref expand_to_in_features(module& m,
                                                 instruction_ref ins,
                                                 instruction_ref s,
                                                 std::size_t in_features)
    {
        auto lens = s->get_shape().lens();
        if(lens.size() != 2 and lens.size() != 3)
            MIGRAPHX_THROW("moe: quantization scales and zero points must be rank 2 or 3");
        if(lens.size() == 2)
        {
            s = m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {2}}}), s);
            return m.insert_instruction(
                ins, make_op("multibroadcast", {{"out_lens", {lens[0], lens[1], in_features}}}), s);
        }
        auto nblocks = lens.at(2);
        if(nblocks == in_features)
            return s;
        if(nblocks == 0 or in_features % nblocks != 0)
            MIGRAPHX_THROW("moe: in_features must be divisible by the number of "
                           "quantization blocks");
        auto block_size = in_features / nblocks;
        s               = m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {3}}}), s);
        s               = m.insert_instruction(
            ins,
            make_op("multibroadcast", {{"out_lens", {lens[0], lens[1], nblocks, block_size}}}),
            s);
        return m.insert_instruction(
            ins, make_op("reshape", {{"dims", {lens[0], lens[1], in_features}}}), s);
    }

    // h1 is the fc1 output [rows, 1, fusion_size * inter]; h3 is the optional
    // fc3 output [rows, 1, inter]. Returns the activated [rows, 1, inter].
    instruction_ref apply_activation(module& m,
                                     instruction_ref ins,
                                     instruction_ref h1,
                                     const std::optional<instruction_ref>& h3,
                                     std::size_t inter) const
    {
        if(activation_type == "swiglu")
        {
            instruction_ref g;
            instruction_ref l;
            if(swiglu_fusion == 0)
            {
                if(not h3.has_value())
                    MIGRAPHX_THROW("moe: swiglu without fusion requires fc3 inputs");
                g = h1;
                l = *h3;
            }
            else if(swiglu_fusion == 1)
            {
                // g and l are interleaved along the last axis
                auto rows = h1->get_shape().lens().front();
                auto p    = m.insert_instruction(
                    ins, make_op("reshape", {{"dims", {rows, 1, inter, 2}}}), h1);
                g = m.insert_instruction(
                    ins, make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {1}}}), p);
                g = m.insert_instruction(ins, make_op("squeeze", {{"axes", {3}}}), g);
                l = m.insert_instruction(
                    ins, make_op("slice", {{"axes", {3}}, {"starts", {1}}, {"ends", {2}}}), p);
                l = m.insert_instruction(ins, make_op("squeeze", {{"axes", {3}}}), l);
            }
            else
            {
                // g and l are concatenated along the last axis
                g = m.insert_instruction(
                    ins, make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {inter}}}), h1);
                l = m.insert_instruction(
                    ins,
                    make_op("slice", {{"axes", {2}}, {"starts", {inter}}, {"ends", {2 * inter}}}),
                    h1);
            }
            return swiglu(m, ins, g, l);
        }

        auto a = h1;
        if(activation_type == "relu")
            a = m.insert_instruction(ins, make_op("relu"), h1);
        else if(activation_type == "gelu")
            a = op::builder::insert("gelu_erf", m, ins, {h1}).at(0);
        else if(activation_type == "silu")
        {
            auto sig = m.insert_instruction(ins, make_op("sigmoid"), h1);
            a        = m.insert_instruction(ins, make_op("mul"), h1, sig);
        }
        else if(activation_type != "identity")
            MIGRAPHX_THROW("moe: unsupported activation type " + activation_type);

        // A separate fc3 acts as the linear branch of a gated activation
        if(h3.has_value())
            a = m.insert_instruction(ins, make_op("mul"), a, *h3);
        return a;
    }

    static instruction_ref scalar_literal(module& m, shape::type_t t, float v)
    {
        return m.add_literal(literal{shape{t}, {v}});
    }

    // swiglu = gc * sigmoid(alpha * gc) * (lc + beta), where gc = min(g, limit) and
    // lc = clamp(l, -limit, limit); g is only clamped from above, per onnxruntime
    instruction_ref
    swiglu(module& m, instruction_ref ins, instruction_ref g, instruction_ref l) const
    {
        auto dtype = g->get_shape().type();
        if(not std::isinf(swiglu_limit))
        {
            auto limit = scalar_literal(m, dtype, swiglu_limit);
            g          = insert_common_op(m, ins, "min", g, limit);
            l          = insert_common_op(
                m, ins, make_op("clip"), {l, scalar_literal(m, dtype, -swiglu_limit), limit});
        }
        auto ag  = insert_common_op(m, ins, "mul", g, scalar_literal(m, dtype, activation_alpha));
        auto sig = m.insert_instruction(ins, make_op("sigmoid"), ag);
        auto gs  = m.insert_instruction(ins, make_op("mul"), g, sig);
        auto lb  = insert_common_op(m, ins, "add", l, scalar_literal(m, dtype, activation_beta));
        return m.insert_instruction(ins, make_op("mul"), gs, lb);
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
