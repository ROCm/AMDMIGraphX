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
 *
 */

#include <algorithm>
#include <limits>
#include <migraphx/common.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/builder/insert.hpp>
#include <migraphx/op/builder/kit.hpp>
#include <migraphx/op/common.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

struct torch_lstm : op_builder<torch_lstm>
{
    std::size_t hidden_size = 1;
    std::vector<operation> actv_funcs{};
    rnn_direction direction = rnn_direction::forward;
    float clip              = 0.0f;
    int input_forget        = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.hidden_size, "hidden_size"),
                    f(self.actv_funcs, "actv_func"),
                    f(self.direction, "direction"),
                    f(self.clip, "clip"),
                    f(self.input_forget, "input_forget"));
    }

    static std::vector<std::string> names() { return {"tm::lstm"}; }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto self = *this;
        if(self.actv_funcs.empty())
        {
            self.actv_funcs = {make_op("sigmoid"), make_op("tanh"), make_op("tanh")};
            if(self.direction == rnn_direction::bidirectional)
            {
                self.actv_funcs.insert(self.actv_funcs.end(),
                                       {make_op("sigmoid"), make_op("tanh"), make_op("tanh")});
            }
        }
        auto hidden_states =
            m.insert_instruction(ins, make_op("lstm", migraphx::to_value(self)), args);
        auto last_hs   = m.insert_instruction(ins, make_op("rnn_last_hs_output"), hidden_states);
        auto last_cell = m.insert_instruction(ins, make_op("rnn_last_cell_output"), hidden_states);
        return {hidden_states, last_hs, last_cell};
    }
};

// linear reuses the gemm builder; ND inputs are flattened to rank 2.
struct torch_linear : op_builder<torch_linear>
{
    static std::vector<std::string> names() { return {"tm::linear"}; }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        const value gemm_opts{{"transB", true}};
        auto lens = args[0]->get_shape().lens();
        if(lens.size() == 2)
            return op::builder::insert("gemm", m, ins, args, gemm_opts);

        std::size_t rows = 1;
        for(std::size_t i = 0; i + 1 < lens.size(); ++i)
            rows *= lens[i];
        std::vector<int64_t> flat = {static_cast<int64_t>(rows),
                                     static_cast<int64_t>(lens.back())};
        auto x2d = m.insert_instruction(ins, make_op("reshape", {{"dims", flat}}), args[0]);

        auto gemm_args = args;
        gemm_args[0]   = x2d;
        auto out       = op::builder::insert("gemm", m, ins, gemm_args, gemm_opts).front();

        std::vector<int64_t> out_dims(lens.begin(), lens.end() - 1);
        out_dims.push_back(static_cast<int64_t>(out->get_shape().lens().back()));
        return {m.insert_instruction(ins, make_op("reshape", {{"dims", out_dims}}), out)};
    }
};

// nan_to_num has no native op: replace NaN/+inf/-inf with the given values.
struct torch_nan_to_num : op_builder<torch_nan_to_num>
{
    float nan    = 0.0f;
    float posinf = std::numeric_limits<float>::max();
    float neginf = std::numeric_limits<float>::lowest();

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.nan, "nan"), f(self.posinf, "posinf"), f(self.neginf, "neginf"));
    }

    static std::vector<std::string> names() { return {"tm::nan_to_num"}; }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto x    = args[0];
        auto type = x->get_shape().type();

        auto nan_lit    = m.add_literal({type, {nan}});
        auto zero       = m.add_literal({type, {0.0f}});
        auto posinf_lit = m.add_literal({type, {posinf}});
        auto neginf_lit = m.add_literal({type, {neginf}});

        // where selects per-element, so inputs are broadcast but not type-promoted
        const common_options no_promote{.common_type = false};
        auto select = [&](instruction_ref cond, instruction_ref val, instruction_ref other) {
            return insert_common_op(m, ins, make_op("where"), {cond, val, other}, no_promote);
        };

        auto is_nan   = m.insert_instruction(ins, make_op("isnan"), x);
        auto result   = select(is_nan, nan_lit, x);
        auto is_inf   = m.insert_instruction(ins, make_op("isinf"), x);
        auto less     = insert_common_op(m, ins, "less", x, zero);
        auto greater  = insert_common_op(m, ins, "greater", x, zero);
        auto neg_mask = insert_common_op(m, ins, "logical_and", less, is_inf);
        auto pos_mask = insert_common_op(m, ins, "logical_and", greater, is_inf);
        result        = select(neg_mask, neginf_lit, result);
        return {select(pos_mask, posinf_lit, result)};
    }
};

// conv_transpose has no native op: run convolution_backwards, apply the asymmetric
// output_padding crop, then broadcast the channel bias and add.
struct torch_conv_transpose : op_builder<torch_conv_transpose>
{
    std::vector<std::size_t> stride;
    std::vector<std::size_t> padding;
    std::vector<std::size_t> dilation;
    std::vector<std::size_t> output_padding;
    int group = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.stride, "stride"),
                    f(self.padding, "padding"),
                    f(self.dilation, "dilation"),
                    f(self.output_padding, "output_padding"),
                    f(self.group, "group"));
    }

    static std::vector<std::string> names() { return {"tm::conv_transpose"}; }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        // output_padding cannot be expressed by the op: run it unpadded, then crop
        bool crop = std::any_of(
            output_padding.begin(), output_padding.end(), [](std::size_t o) { return o != 0; });
        auto pad = crop ? std::vector<std::size_t>(padding.size(), 0) : padding;
        auto out = m.insert_instruction(ins,
                                        make_op("convolution_backwards",
                                                {{"stride", stride},
                                                 {"padding", pad},
                                                 {"dilation", dilation},
                                                 {"group", group}}),
                                        args[0],
                                        args[1]);

        if(crop)
        {
            auto spatial = out->get_shape().lens();
            std::vector<int64_t> axes(output_padding.size());
            std::vector<int64_t> starts(output_padding.size());
            std::vector<int64_t> ends(output_padding.size());
            for(std::size_t i = 0; i < output_padding.size(); ++i)
            {
                axes[i]   = static_cast<int64_t>(i + 2);
                starts[i] = static_cast<int64_t>(padding[i]);
                ends[i]   = static_cast<int64_t>(spatial[i + 2] - padding[i] + output_padding[i]);
            }
            out = m.insert_instruction(
                ins, make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}), out);
        }

        if(args.size() < 3)
            return {out};

        auto out_lens = out->get_shape().lens();
        auto bias     = m.insert_instruction(
            ins, make_op("broadcast", {{"axis", 1}, {"out_lens", out_lens}}), args[2]);
        return {m.insert_instruction(ins, make_op("add"), out, bias)};
    }
};

// std has no native op: sqrt of the corrected variance reduced over axes.
struct torch_std : op_builder<torch_std>
{
    std::vector<int64_t> axes = {};
    bool keepdim              = false;
    float correction          = 1.0f;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(
            f(self.axes, "axes"), f(self.keepdim, "keepdim"), f(self.correction, "correction"));
    }

    static std::vector<std::string> names() { return {"tm::std"}; }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto x    = args[0];
        auto lens = x->get_shape().lens();
        auto rank = static_cast<int64_t>(lens.size());
        auto type = x->get_shape().type();

        std::size_t n = 1;
        for(auto a : axes)
            n *= lens[a < 0 ? a + rank : a];

        auto mean  = m.insert_instruction(ins, make_op("reduce_mean", {{"axes", axes}}), x);
        auto sub   = insert_common_op(m, ins, "sub", x, mean);
        auto sq    = insert_common_op(m, ins, "mul", sub, sub);
        auto sum   = m.insert_instruction(ins, make_op("reduce_sum", {{"axes", axes}}), sq);
        auto denom = m.add_literal({type, {static_cast<float>(n) - correction}});
        auto var   = insert_common_op(m, ins, "div", sum, denom);
        auto out   = m.insert_instruction(ins, make_op("sqrt"), var);
        if(not keepdim)
            out = m.insert_instruction(ins, make_op("squeeze", {{"axes", axes}}), out);
        return {out};
    }
};

struct torch_kit : kit<torch_kit>
{
    std::string prefix() const { return "tm::"; }
    void apply() const
    {
        this->common_ops({
            "abs",   "acos",        "add",     "asin",    "atan",  "bitwise_and", "ceil", "convert",
            "cos",   "cosh",        "div",     "elu",     "equal", "erf",         "exp",
            "floor", "fmod",        "greater", "isinf",   "isnan", "leaky_relu",  "less", "log",
            "log2",  "logical_and", "max",     "min",     "mul",   "neg",         "not",  "pow",
            "recip", "relu",        "rsqrt",   "sigmoid", "sign",  "sin",         "sinh", "sqrt",
            "sub",   "tan",         "tanh",
        });
        this->common_ops({"where"}, {.common_type = false});

        this->ops({
            "argmax",
            "argmin",
            "broadcast",
            "concat",
            "contiguous",
            "convolution_backwards",
            "dequantizelinear",
            "gather",
            "gathernd",
            "get_tuple_elem",
            "logsoftmax",
            "multibroadcast",
            "pad",
            "pooling",
            "prefix_scan_sum",
            "quantizelinear",
            "reduce_all",
            "reduce_any",
            "reduce_max",
            "reduce_mean",
            "reduce_min",
            "reduce_prod",
            "reduce_sum",
            "reshape",
            "scatter_none",
            "slice",
            "softmax",
            "squeeze",
            "step",
            "topk",
            "transpose",
            "undefined",
            "unsqueeze",
        });

        // Composite builders (bias fusion, broadcasting, etc.), not plain ops.
        this->builders({"batchnorm",
                        "clip",
                        "convolution",
                        "dot",
                        "floor_div",
                        "gelu_erf",
                        "glu",
                        "group_norm",
                        "hardsigmoid",
                        "instance_norm",
                        "layer_norm",
                        "selu",
                        "softsign",
                        "vector_norm"});
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
