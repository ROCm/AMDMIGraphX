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
#include <migraphx/matcher.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/gpu/prefuse_ops.hpp>
#include <migraphx/gpu/gemm_softmax_gemm.hpp>
#include <migraphx/match/layernorm.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#ifdef MIGRAPHX_USE_COMPOSABLEKERNEL
#include <migraphx/gpu/ck.hpp>
#endif
#include <migraphx/gpu/fuse_mlir.hpp>
#include <migraphx/op/convolution.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_LAYERNORM_FUSION);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_MLIR);

namespace {

template <class Derived, std::size_t N>
struct layernorm_base
{
    float epsilon = 1e-12f;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.epsilon, "epsilon"));
    }
    shape compute_shape(std::vector<shape> inputs, std::vector<module_ref> mods) const
    {
        std::size_t nargs = N;
        if(not mods.empty())
        {
            auto* pm = mods.front();
            nargs += pm->get_parameter_names().size() - 1;
        }
        check_shapes{inputs, static_cast<const Derived&>(*this)}.has(nargs);
        auto s = inputs.front();
        auto t = s.type();
        if(not mods.empty())
            t = mods.front()->get_output_shapes().front().type();

        // Scalar output if all inputs are scalar
        if(inputs.front().elements() == 1 and
           all_of(inputs, [](const auto& ss) { return ss.scalar(); }))
            return inputs.front();
        auto l_s = shape::from_permutation(
            t, s.lens(), find_permutation(std::vector<shape>(inputs.begin(), inputs.begin() + N)));
        // just prelayernorm or preadd_layernorm
        if(nargs <= N)
            return l_s;
        // else, layernorm + pointwise fusion, preserve layout of fused op
        std::vector<shape> lp_s(inputs.begin() + N, inputs.end());
        lp_s.insert(lp_s.begin(), l_s);
        return shape::from_permutation(t, s.lens(), find_permutation(lp_s));
    }
};

struct layernorm : layernorm_base<layernorm, 1>
{

    std::string name() const { return "gpu::prelayernorm"; }
};
MIGRAPHX_REGISTER_OP(layernorm);

struct add_layernorm : layernorm_base<add_layernorm, 2>
{
    std::string name() const { return "gpu::preadd_layernorm"; }
};
MIGRAPHX_REGISTER_OP(add_layernorm);

struct find_layernorm
{
    auto matcher() const { return match::layernorm(); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins   = r.result;
        auto x_ins = r.instructions["x"];
        float eps  = 0;
        if(contains(r.instructions, "eps"))
            eps = r.instructions["eps"]->eval().at<float>();

        m.replace_instruction(ins, layernorm{eps}, x_ins);
    }
};

struct find_add_layernorm
{
    auto matcher() const
    {
        return match::name("gpu::prelayernorm")(
            match::args(match::name("add")(match::used_once()).bind("add")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins     = r.result;
        auto add_ins = r.instructions["add"];
        auto op      = any_cast<layernorm>(ins->get_operator());

        m.replace_instruction(ins, add_layernorm{op.epsilon}, add_ins->inputs());
    }
};

struct pre_gemm_softmax_gemm : gemm_softmax_gemm
{
    std::string name() const { return "gpu::pre_gemm_softmax_gemm"; }
};
MIGRAPHX_REGISTER_OP(pre_gemm_softmax_gemm);

auto is_ck_gemm()
{
    return match::make_basic_pred_matcher([=](instruction_ref ins) {
#ifdef MIGRAPHX_USE_COMPOSABLEKERNEL
        if(not enabled(MIGRAPHX_ENABLE_CK{}))
            return false;
        if(ins->name() != "dot")
            return false;
        if(not pre_gemm_softmax_gemm::is_ck_supported_type(ins->get_shape().type()))
            return false;
        return true;
#else
        (void)ins;
        return false;
#endif
    });
}

auto is_test_gemm(bool enable_attention)
{
    return match::make_basic_pred_matcher([=](instruction_ref ins) {
        if(ins->name() != "dot")
            return false;
        return enable_attention;
    });
}

auto is_bias_supported()
{
    return match::make_basic_pred_matcher([=](instruction_ref) {
#ifdef MIGRAPHX_USE_COMPOSABLEKERNEL
        return not enabled(MIGRAPHX_ENABLE_CK{});
#else
        return true;
#endif
    });
}

auto is_conv1x1()
{
    return match::make_basic_pred_matcher([](instruction_ref ins) {
        if(ins->name() != "convolution")
            return false;

        auto input   = ins->inputs().at(0)->get_shape();
        auto weights = ins->inputs().at(1)->get_shape();

        // Dims exist
        if(weights.lens().size() < 4 || input.lens().size() < 4)
            return false;

        // Kernel is 1x1
        if(weights.lens()[2] != 1 || weights.lens()[3] != 1)
            return false;

        // Check convolution attributes
        auto conv_op = any_cast<op::convolution>(ins->get_operator());

        // Calculate output spatial dimensions
        // For 1x1 conv with no padding: out_h = (in_h - 1) / stride + 1
        auto out_h = (input.lens()[2] - 1) / conv_op.stride[0] + 1;
        auto out_w = (input.lens()[3] - 1) / conv_op.stride[1] + 1;

        // H <= 8 and W <= 8
        if(out_h > 8 || out_w > 8)
            return false;

        // Output channels >= 32
        if(weights.lens()[0] < 32)
            return false;

        // Batch_size = 1 (kernel limitation)
        if(input.lens()[0] != 1)
            return false;

        // No padding
        if(std::any_of(
               conv_op.padding.begin(), conv_op.padding.end(), [](auto p) { return p != 0; }))
            return false;

        // Dilation = 1
        if(std::any_of(
               conv_op.dilation.begin(), conv_op.dilation.end(), [](auto d) { return d != 1; }))
            return false;

        // Group = 1
        if(conv_op.group != 1)
            return false;

        return true;
    });
}

struct find_gemm_softmax_gemm
{
    bool enable_attention = false;

    auto matcher() const
    {
        auto gemm1 = match::skip(match::name("contiguous"))(match::name("dot")(
            match::any_of(is_ck_gemm(), is_test_gemm(enable_attention)).bind("gemm1")));
        auto mul   = match::name("mul")(
            match::nargs(2), match::either_arg(0, 1)(match::is_constant().bind("scale"), gemm1));
        auto where = match::name("where")(match::arg(2)(match::is_constant().bind("select_const")),
                                          match::arg(1)(mul),
                                          match::arg(0)(match::any().bind("select_cond")));
        auto add =
            match::name("add")(is_bias_supported(),
                               match::nargs(2),
                               match::either_arg(0, 1)(match::none_of(mul).bind("bias"), mul));
        auto softmax = match::name("softmax")(match::arg(0)(match::any_of(mul, add, gemm1, where)))
                           .bind("softmax");

        return match::name("dot")(
            match::any_of(is_ck_gemm(), is_test_gemm(enable_attention)).bind("gemm2"))(
            match::arg(0)(softmax));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins       = r.result;
        auto gemm2_ins = r.instructions["gemm2"];
        auto gemm1_ins = r.instructions["gemm1"];

        float scale = 1.0;
        if(contains(r.instructions, "scale"))
        {
            auto scale_lit = r.instructions["scale"];
            // CK only supports single-valued scale
            scale_lit->eval().visit([&](const auto s) {
                // CK only supports single-valued scale
                if(not std::all_of(
                       s.begin() + 1, s.end(), [&](auto v) { return float_equal(v, s.front()); }))
                    return;
                scale = s.front();
            });
        }

        auto inputs = gemm1_ins->inputs(); // A, B
        if(contains(r.instructions, "select_cond"))
        {
            inputs.push_back(r.instructions["select_cond"]);
            inputs.push_back(r.instructions["select_const"]);
        }
        if(contains(r.instructions, "bias"))
        {
            inputs.push_back(r.instructions["bias"]);
        }

        inputs.push_back(gemm2_ins->inputs().back()); // B1

        mpm.get_module().replace_instruction(
            ins, pre_gemm_softmax_gemm{gemm2_ins->get_operator(), scale}, inputs);
    }
};

void inline_group_sub_module(module_pass_manager& mpm)
{
    auto& m = mpm.get_module();
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "group")
            continue;

        const auto& mod_inputs = ins->module_inputs();
        auto inline_mod        = m.insert_inline(ins, *mod_inputs.at(0), ins->inputs());
        m.replace_instruction(ins, inline_mod.at(0));
    }
}

} // namespace



struct pre_conv1x1
{
    std::vector<std::size_t> strides = {1, 1};
    bool has_bias = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.strides, "strides"), f(self.has_bias, "has_bias"));
    }

    std::string name() const { return "gpu::pre_conv1x1"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        auto input_shape  = inputs[0];
        auto weight_shape = inputs[1];

        auto n     = input_shape.lens()[0];
        auto c_out = weight_shape.lens()[0];
        auto h_out = (input_shape.lens()[2] - 1) / strides[0] + 1;
        auto w_out = (input_shape.lens()[3] - 1) / strides[1] + 1;

        return shape{input_shape.type(), {n, c_out, h_out, w_out}};
    }
};
MIGRAPHX_REGISTER_OP(pre_conv1x1);


// Fused with bias
struct find_conv1x1_bias
{
    auto matcher() const
    {
        auto conv_match = match::name("convolution")(is_conv1x1()).bind("conv");
        auto broadcast_match = match::name("broadcast")(match::arg(0)(match::is_constant().bind("bias")));
        return match::name("add")(match::either_arg(0, 1)(conv_match, broadcast_match)).bind("add");
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto& m       = mpm.get_module();
        auto conv_ins = r.instructions["conv"];
        auto add_ins  = r.instructions["add"];
        auto bias_lit = r.instructions["bias"];

        auto conv_op = any_cast<op::convolution>(conv_ins->get_operator());

        auto input  = conv_ins->inputs()[0];
        auto weight = conv_ins->inputs()[1];

        auto weight_shape = weight->get_shape();
        auto c_out        = weight_shape.lens()[0];
        auto c_in         = weight_shape.lens()[1];

        auto reshape_weight = m.insert_instruction(
            add_ins,
            make_op("reshape", {{"dims", std::vector<std::size_t>{c_out, c_in}}}),
            weight);

        m.replace_instruction(add_ins, pre_conv1x1{conv_op.stride, true}, input, reshape_weight, bias_lit);
    }
};

// Standalone conv1x1
struct find_conv1x1
{
    auto matcher() const { return match::name("convolution")(is_conv1x1()).bind("conv"); }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto& m       = mpm.get_module();
        auto conv_ins = r.instructions["conv"];

        auto conv_op = any_cast<op::convolution>(conv_ins->get_operator());

        auto input  = conv_ins->inputs()[0];
        auto weight = conv_ins->inputs()[1];

        auto weight_shape = weight->get_shape();
        auto c_out        = weight_shape.lens()[0];
        auto c_in         = weight_shape.lens()[1];

        auto reshape_weight = m.insert_instruction(
            conv_ins,
            make_op("reshape", {{"dims", std::vector<std::size_t>{c_out, c_in}}}),
            weight);

        m.replace_instruction(conv_ins, pre_conv1x1{conv_op.stride, false}, input, reshape_weight);
    }
};

// Depthwise convolution operation
struct depthwise_conv
{
    std::vector<std::size_t> padding = {0, 0};
    std::vector<std::size_t> strides = {1, 1};
    bool has_bias = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.padding, "padding"), f(self.strides, "strides"), f(self.has_bias, "has_bias"));
    }

    std::string name() const { return "gpu::depthwise_conv"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        auto input_shape  = inputs[0];
        auto weight_shape = inputs[1];

        auto n  = input_shape.lens()[0];
        auto c  = input_shape.lens()[1];
        auto h  = input_shape.lens()[2];
        auto w  = input_shape.lens()[3];
        auto kh = weight_shape.lens()[2];
        auto kw = weight_shape.lens()[3];

        auto pad_h = padding.size() >= 1 ? padding[0] : 0;
        auto pad_w = padding.size() >= 2 ? padding[1] : 0;
        auto stride_h = strides.size() >= 1 ? strides[0] : 1;
        auto stride_w = strides.size() >= 2 ? strides[1] : 1;

        auto h_out = (h + 2 * pad_h - kh) / stride_h + 1;
        auto w_out = (w + 2 * pad_w - kw) / stride_w + 1;

        return shape{input_shape.type(), {n, c, h_out, w_out}};
    }
};
MIGRAPHX_REGISTER_OP(depthwise_conv);

auto is_depthwise_conv()
{
    return match::make_basic_pred_matcher([=](instruction_ref ins) {
        if(ins->name() != "convolution")
            return false;
        auto conv_op = any_cast<op::convolution>(ins->get_operator());
        auto weight_shape = ins->inputs()[1]->get_shape();
        auto input_shape = ins->inputs()[0]->get_shape();

        // groups == channels (depthwise)
        auto c = input_shape.lens()[1];
        if(conv_op.group != c)
            return false;

        // Weight shape is [C, 1, KH, KW]
        if(weight_shape.lens()[1] != 1)
            return false;

        // Only support float for now
        if(input_shape.type() != shape::float_type)
            return false;

        // stride 1 for now
        if(conv_op.stride[0] != 1 || conv_op.stride[1] != 1)
            return false;

        // dilation 1
        if(conv_op.dilation[0] != 1 || conv_op.dilation[1] != 1)
            return false;

        return true;
    });
}

// Matcher for depthwise convolution
struct find_depthwise_conv
{
    auto matcher() const { return match::name("convolution")(is_depthwise_conv()).bind("conv"); }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto& m       = mpm.get_module();
        auto conv_ins = r.instructions["conv"];

        auto conv_op = any_cast<op::convolution>(conv_ins->get_operator());

        auto input  = conv_ins->inputs()[0];
        auto weight = conv_ins->inputs()[1];

        // Padding: MIGraphX uses [padH_begin, padW_begin, padH_end, padW_end]
        // We assume symmetric padding
        std::vector<std::size_t> padding = {conv_op.padding[0], conv_op.padding[1]};

        m.replace_instruction(conv_ins, depthwise_conv{padding, conv_op.stride, false}, input, weight);
    }
};

void prefuse_ops::apply(module_pass_manager& mpm) const
{
    // First match depthwise convolutions
    match::find_matches(mpm, find_depthwise_conv{});
    mpm.run_pass(dead_code_elimination{});

    // Second match conv1x1+bias, then standalone conv1x1
    match::find_matches(mpm, find_conv1x1_bias{});
    mpm.run_pass(dead_code_elimination{});

    match::find_matches(mpm, find_conv1x1{});
    mpm.run_pass(dead_code_elimination{});

    if(enabled(MIGRAPHX_ENABLE_LAYERNORM_FUSION{}))
    {
        match::find_matches(mpm.get_module(), find_layernorm{});
        mpm.run_pass(dead_code_elimination{});
        match::find_matches(mpm.get_module(), find_add_layernorm{});
    }
    match::find_matches(mpm, find_gemm_softmax_gemm{enable_attention});

    if(enabled(MIGRAPHX_DISABLE_MLIR{}))
    {
        inline_group_sub_module(mpm);
        mpm.run_pass(dead_code_elimination{});
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
