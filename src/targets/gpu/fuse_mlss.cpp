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
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/env.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/gpu/fuse_mlss.hpp>
#include <migraphx/gpu/mlss_conv_op.hpp>
#ifdef MIGRAPHX_USE_AMDMLSS
#include <amdmlss/amdmlss_api.h>
#include <iostream>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

/*
 * @brief Declares a new MIGraphX environment variable which forces to delegate
 * only specific operations to AMDMLSS.
 *
 * The variable, if defined, forces MIGraphX to use only specific operations
 * with AMDMLSS regardless of the underlying GPU architecture. The variable
 * accepts a list of operations separated by comma. The variable recognizes the
 * following operations: "conv". If the variable is not defined MIGraphX will
 * decide by itself which operations to delegate to AMDMLSS.
 */
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_MLSS_USE_SPECIFIC_OPS);

#ifdef MIGRAPHX_USE_AMDMLSS

static bool mlss_specific_op(std::string_view op_name)
{
    static const auto options =
        split_string(string_value_of(MIGRAPHX_MLSS_USE_SPECIFIC_OPS{}, ""), ',');
    return std::any_of(
        options.begin(), options.end(), [&](const auto& opt) { return opt == op_name; });
}

// ---------------------------------------------------------------------------
// Helper: create an mlss_conv_op intermediate node and replace the matched
// instruction(s). The JIT compiler (jit/mlss_conv.cpp) will later convert
// this into a code_object_op with the full kernarg layout.
// ---------------------------------------------------------------------------
static bool insert_mlss_conv(module& m,
                             context* ctx,
                             instruction_ref replace_ins,
                             instruction_ref act_ins,
                             instruction_ref wt_ins,
                             instruction_ref bias_ins, // end() if no bias
                             const shape& output_shape,
                             const std::vector<std::size_t>& cur_padding,
                             const std::vector<std::size_t>& cur_stride,
                             const std::vector<std::size_t>& cur_dilation,
                             std::size_t cur_group,
                             bool has_bias,
                             uint8_t act_mode,
                             float act_alpha)
{
    const auto act_lens = act_ins->get_shape().lens();
    const auto wt_lens  = wt_ins->get_shape().lens();
    const auto out_lens = output_shape.lens();
    const auto dtype    = act_ins->get_shape().type();

    // mlss_conv is 2D NCHW only — the kernel args, stride math, and
    // query_mlss_conv_binary all index [0..3]. A 3D/5D conv that slips past
    // the matcher's wt[2..3] guard would index out of bounds.
    if(act_lens.size() != 4 or wt_lens.size() != 4 or out_lens.size() != 4)
        return false;

    // mlss_conv kernel args only encode a single (pad_h, pad_w) pair (start
    // padding); the kernel does not support asymmetric start/end padding.
    // query_mlss_conv_binary may still return a binary for the symmetric case
    // it sees, so reject here before fusing.
    if(cur_padding.size() == 4 and
       (cur_padding[0] != cur_padding[2] or cur_padding[1] != cur_padding[3]))
        return false;

    // Check if AMDMLSS has a kernel for this configuration
    auto info = query_mlss_conv_binary(*ctx,
                                       act_lens,
                                       wt_lens,
                                       out_lens,
                                       cur_padding,
                                       cur_stride,
                                       cur_dilation,
                                       cur_group,
                                       has_bias,
                                       act_mode,
                                       dtype);
    if(info.empty())
        return false;

    mlss_conv_op op;
    op.padding          = cur_padding;
    op.stride           = cur_stride;
    op.dilation         = cur_dilation;
    op.group            = cur_group;
    op.has_bias         = has_bias;
    op.activation_mode  = act_mode;
    op.activation_alpha = act_alpha;
    op.output           = output_shape;

    std::vector<instruction_ref> args = {act_ins, wt_ins};
    if(has_bias)
        args.push_back(bias_ins);

    m.replace_instruction(replace_ins, op, args);
    return true;
}

// ---------------------------------------------------------------------------
// Conv matchers — each matches a pattern, validates attributes, and calls
// insert_mlss_conv to create the intermediate mlss_conv op.
// ---------------------------------------------------------------------------
struct find_mlss_conv
{
    context* ctx = nullptr;

    auto matcher() const
    {
        return match::name("convolution")(match::arg(0)(match::any()),
                                          match::arg(1)(match::name("@literal")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins    = r.result;
        auto inputs = ins->inputs();
        if(inputs.size() < 2)
            return;

        auto act_ins = inputs[0];
        auto wt_ins  = inputs[1];

        const auto wt_lens = wt_ins->get_shape().lens();
        if(wt_lens[2] == 1 and wt_lens[3] == 1)
            return;

        const auto dtype = act_ins->get_shape().type();
        if(dtype != shape::float_type and dtype != shape::half_type)
            return;
        if(wt_ins->get_shape().type() != dtype)
            return;
        if(ins->get_shape().type() != dtype)
            return;

        const auto& op_val = ins->get_operator().to_value();
        auto get_vec       = [&](const std::string& key) -> std::vector<std::size_t> {
            return op_val.get(key, std::vector<std::size_t>{});
        };
        const auto cur_padding  = get_vec("padding");
        const auto cur_stride   = get_vec("stride");
        const auto cur_dilation = get_vec("dilation");
        const auto cur_group    = op_val.get("group", std::size_t{1});

        auto& m = mpm.get_module();
        insert_mlss_conv(m,
                         ctx,
                         ins,
                         act_ins,
                         wt_ins,
                         m.end(),
                         ins->get_shape(),
                         cur_padding,
                         cur_stride,
                         cur_dilation,
                         cur_group,
                         false,
                         static_cast<uint8_t>(mlss_activation_mode::identity),
                         0.0f);
    }
};

struct find_mlss_conv_bias
{
    context* ctx = nullptr;

    auto matcher() const
    {
        auto conv_with_literal_weight = match::name("convolution")(
            match::arg(0)(match::any()), match::arg(1)(match::name("@literal")));

        auto broadcast_of_literal =
            match::name("broadcast")(match::arg(0)(match::name("@literal")));

        return match::name("add")(match::arg(0)(conv_with_literal_weight),
                                  match::arg(1)(broadcast_of_literal));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto add_ins   = r.result;
        auto conv_ins  = add_ins->inputs()[0];
        auto bcast_ins = add_ins->inputs()[1];
        auto bias_ins  = bcast_ins->inputs()[0];

        auto conv_inputs = conv_ins->inputs();
        if(conv_inputs.size() < 2)
            return;
        auto act_ins = conv_inputs[0];
        auto wt_ins  = conv_inputs[1];

        const auto wt_lens = wt_ins->get_shape().lens();
        if(wt_lens[2] == 1 and wt_lens[3] == 1)
            return;

        const auto dtype = act_ins->get_shape().type();
        if(dtype != shape::float_type and dtype != shape::half_type)
            return;
        if(wt_ins->get_shape().type() != dtype)
            return;
        if(conv_ins->get_shape().type() != dtype)
            return;

        const auto& op_val = conv_ins->get_operator().to_value();
        auto get_vec       = [&](const std::string& key) -> std::vector<std::size_t> {
            return op_val.get(key, std::vector<std::size_t>{});
        };
        const auto cur_padding  = get_vec("padding");
        const auto cur_stride   = get_vec("stride");
        const auto cur_dilation = get_vec("dilation");
        const auto cur_group    = op_val.get("group", std::size_t{1});

        auto& m = mpm.get_module();
        if(not insert_mlss_conv(m,
                                ctx,
                                add_ins,
                                act_ins,
                                wt_ins,
                                bias_ins,
                                add_ins->get_shape(),
                                cur_padding,
                                cur_stride,
                                cur_dilation,
                                cur_group,
                                true,
                                static_cast<uint8_t>(mlss_activation_mode::identity),
                                0.0f))
            return;

        if(bcast_ins->outputs().empty())
            m.remove_instruction(bcast_ins);
        if(conv_ins->outputs().empty())
            m.remove_instruction(conv_ins);
    }
};

struct find_mlss_conv_bias_relu
{
    context* ctx = nullptr;

    auto matcher() const
    {
        auto conv_with_literal_weight = match::name("convolution")(
            match::arg(0)(match::any()), match::arg(1)(match::name("@literal")));

        auto broadcast_of_literal =
            match::name("broadcast")(match::arg(0)(match::name("@literal")));

        auto add_conv_bias = match::name("add")(match::arg(0)(conv_with_literal_weight),
                                                match::arg(1)(broadcast_of_literal));

        return match::name("relu")(match::arg(0)(add_conv_bias));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto relu_ins  = r.result;
        auto add_ins   = relu_ins->inputs()[0];
        auto conv_ins  = add_ins->inputs()[0];
        auto bcast_ins = add_ins->inputs()[1];
        auto bias_ins  = bcast_ins->inputs()[0];

        auto conv_inputs = conv_ins->inputs();
        if(conv_inputs.size() < 2)
            return;
        auto act_ins = conv_inputs[0];
        auto wt_ins  = conv_inputs[1];

        const auto wt_lens = wt_ins->get_shape().lens();
        if(wt_lens[2] == 1 and wt_lens[3] == 1)
            return;

        const auto dtype = act_ins->get_shape().type();
        if(dtype != shape::float_type and dtype != shape::half_type)
            return;
        if(wt_ins->get_shape().type() != dtype)
            return;
        if(conv_ins->get_shape().type() != dtype)
            return;

        const auto& op_val = conv_ins->get_operator().to_value();
        auto get_vec       = [&](const std::string& key) -> std::vector<std::size_t> {
            return op_val.get(key, std::vector<std::size_t>{});
        };
        const auto cur_padding  = get_vec("padding");
        const auto cur_stride   = get_vec("stride");
        const auto cur_dilation = get_vec("dilation");
        const auto cur_group    = op_val.get("group", std::size_t{1});

        auto& m = mpm.get_module();
        if(not insert_mlss_conv(m,
                                ctx,
                                relu_ins,
                                act_ins,
                                wt_ins,
                                bias_ins,
                                relu_ins->get_shape(),
                                cur_padding,
                                cur_stride,
                                cur_dilation,
                                cur_group,
                                true,
                                static_cast<uint8_t>(mlss_activation_mode::relu),
                                0.0f))
            return;

        if(add_ins->outputs().empty())
            m.remove_instruction(add_ins);
        if(bcast_ins->outputs().empty())
            m.remove_instruction(bcast_ins);
        if(conv_ins->outputs().empty())
            m.remove_instruction(conv_ins);
    }
};

struct find_mlss_conv_bias_leaky_relu
{
    context* ctx = nullptr;

    auto matcher() const
    {
        auto conv_with_literal_weight = match::name("convolution")(
            match::arg(0)(match::any()), match::arg(1)(match::name("@literal")));

        auto broadcast_of_literal =
            match::name("broadcast")(match::arg(0)(match::name("@literal")));

        auto add_conv_bias = match::name("add")(match::arg(0)(conv_with_literal_weight),
                                                match::arg(1)(broadcast_of_literal));

        return match::name("leaky_relu")(match::arg(0)(add_conv_bias));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto lrelu_ins = r.result;
        auto add_ins   = lrelu_ins->inputs()[0];
        auto conv_ins  = add_ins->inputs()[0];
        auto bcast_ins = add_ins->inputs()[1];
        auto bias_ins  = bcast_ins->inputs()[0];

        auto conv_inputs = conv_ins->inputs();
        if(conv_inputs.size() < 2)
            return;
        auto act_ins = conv_inputs[0];
        auto wt_ins  = conv_inputs[1];

        const auto wt_lens = wt_ins->get_shape().lens();
        if(wt_lens[2] == 1 and wt_lens[3] == 1)
            return;

        const auto dtype = act_ins->get_shape().type();
        if(dtype != shape::float_type and dtype != shape::half_type)
            return;
        if(wt_ins->get_shape().type() != dtype)
            return;
        if(conv_ins->get_shape().type() != dtype)
            return;

        const auto& op_val = conv_ins->get_operator().to_value();
        auto get_vec       = [&](const std::string& key) -> std::vector<std::size_t> {
            return op_val.get(key, std::vector<std::size_t>{});
        };
        const auto cur_padding  = get_vec("padding");
        const auto cur_stride   = get_vec("stride");
        const auto cur_dilation = get_vec("dilation");
        const auto cur_group    = op_val.get("group", std::size_t{1});

        const auto& lrelu_val = lrelu_ins->get_operator().to_value();
        float alpha           = lrelu_val.get("alpha", 0.0f);

        auto& m = mpm.get_module();
        if(not insert_mlss_conv(m,
                                ctx,
                                lrelu_ins,
                                act_ins,
                                wt_ins,
                                bias_ins,
                                lrelu_ins->get_shape(),
                                cur_padding,
                                cur_stride,
                                cur_dilation,
                                cur_group,
                                true,
                                static_cast<uint8_t>(mlss_activation_mode::leaky_relu),
                                alpha))
            return;

        if(add_ins->outputs().empty())
            m.remove_instruction(add_ins);
        if(bcast_ins->outputs().empty())
            m.remove_instruction(bcast_ins);
        if(conv_ins->outputs().empty())
            m.remove_instruction(conv_ins);
    }
};
#endif // MIGRAPHX_USE_AMDMLSS

void fuse_mlss::apply(module_pass_manager& mpm) const
{
#ifdef MIGRAPHX_USE_AMDMLSS
    if(enable_conv or mlss_specific_op("conv"))
    {
        // Match most-specific patterns first to avoid partial consumption.
        match::find_matches(mpm, find_mlss_conv_bias_relu{ctx});
        match::find_matches(mpm, find_mlss_conv_bias_leaky_relu{ctx});
        match::find_matches(mpm, find_mlss_conv_bias{ctx});
        match::find_matches(mpm, find_mlss_conv{ctx});
    }
#else
    (void)mpm;
#endif // MIGRAPHX_USE_AMDMLSS
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
