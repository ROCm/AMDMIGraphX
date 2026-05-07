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
#include <migraphx/half.hpp>
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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_LAYERNORM_FUSION);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_MLIR);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_WINOGRAD);

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

struct channelwise_conv
{
    std::size_t num_spatial = 2;
    std::vector<std::size_t> padding;

    std::string name() const { return "gpu::channelwise_conv"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.num_spatial, "num_spatial"), f(self.padding, "padding"));
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2).same_ndims();
        auto x_lens = inputs[0].lens();
        auto w_lens = inputs[1].lens();
        std::vector<std::size_t> out_lens;
        out_lens.push_back(x_lens[0]);
        out_lens.push_back(w_lens[0]);
        for(std::size_t i = 0; i < num_spatial; i++)
        {
            std::size_t total_pad = 0;
            if(i < padding.size())
                total_pad += padding[i];
            if(i + num_spatial < padding.size())
                total_pad += padding[i + num_spatial];
            out_lens.push_back(x_lens[i + 2] + total_pad - w_lens[i + 2] + 1);
        }
        return inputs[0].with_lens(out_lens);
    }
};
MIGRAPHX_REGISTER_OP(channelwise_conv);

MIGRAPHX_PRED_MATCHER(conv_channelwise, instruction_ref ins)
{
    if(ins->name() != "convolution")
        return false;
    auto v = ins->get_operator().to_value();
    if(not all_of(v.at("stride"), [](const value& x) { return x.to<std::size_t>() == 1; }))
        return false;
    if(not all_of(v.at("dilation"), [](const value& x) { return x.to<std::size_t>() == 1; }))
        return false;
    auto w_lens = ins->inputs().back()->get_shape().lens();
    if(w_lens[1] != 1)
        return false;
    auto x_lens = ins->inputs().front()->get_shape().lens();
    auto c_in   = x_lens[1];
    auto group  = v.at("group").to<std::size_t>();
    return group == 1 or group == c_in;
}

struct find_channelwise_convolution
{
    auto matcher() const { return conv_channelwise(); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins         = r.result;
        auto input       = ins->inputs().front();
        auto weights     = ins->inputs().back();
        auto num_spatial = ins->get_shape().ndim() - 2;

        if(input->get_shape().type() != shape::float_type)
            return;

        auto v        = ins->get_operator().to_value();
        auto pad_vals = v.at("padding");
        std::vector<std::size_t> padding;
        std::transform(pad_vals.begin(),
                       pad_vals.end(),
                       std::back_inserter(padding),
                       [](const value& x) { return x.to<std::size_t>(); });

        m.replace_instruction(
            ins, channelwise_conv{num_spatial, std::move(padding)}, input, weights);
    }
};

struct winograd_conv
{
    std::vector<std::size_t> padding;

    std::string name() const { return "gpu::winograd_conv"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.padding, "padding"));
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        // inputs[0] = activation, NCHW [N, C, H, W]
        // inputs[1] = pre-transformed filter, [K, C, 4, 4]  (16 winograd elements)
        auto x_lens = inputs[0].lens();
        auto w_lens = inputs[1].lens();
        if(x_lens.size() != 4 or w_lens.size() != 4 or w_lens[2] != 4 or w_lens[3] != 4)
            MIGRAPHX_THROW("gpu::winograd_conv: expected NCHW input and [K,C,4,4] pre-transformed filter");
        std::vector<std::size_t> out_lens;
        out_lens.push_back(x_lens[0]);
        out_lens.push_back(w_lens[0]);
        // The original conv was 3x3 stride-1 padded; output spatial dim = input spatial dim.
        for(std::size_t i = 0; i < 2; i++)
        {
            std::size_t total_pad = 0;
            if(i < padding.size())
                total_pad += padding[i];
            if(i + 2 < padding.size())
                total_pad += padding[i + 2];
            // Original kernel was 3x3 (now stored pre-transformed as 4x4) - subtract 3.
            out_lens.push_back(x_lens[i + 2] + total_pad - 3 + 1);
        }
        return inputs[0].with_lens(out_lens);
    }
};
MIGRAPHX_REGISTER_OP(winograd_conv);

// Compute U = G * g * G^T for one 3x3 filter, returning 16 elements row-major.
// G = [[1, 0, 0], [.5, .5, .5], [.5, -.5, .5], [0, 0, 1]]
//
// When `miopen_signs` is true, six positions are negated to match MIOpen's
// DPP-cooperative input transform sign convention: V_miopen has the same six
// positions negated vs the standard B^T*d*B, so the negations cancel during
// the U·V dot product. The six positions are e ∈ {3, 7, 11, 12, 13, 14},
// i.e. (i,j) where (i==3) XOR (j==3) — the "row 3 and column 3 cross minus
// the (3,3) corner".
template <class T>
inline std::array<T, 16> winograd_filter_pretransform_3x3(const T* g, bool miopen_signs)
{
    const T half = T{0.5};
    // Step 1: t = G * g  (4x3) - declared uninitialized since we write all entries.
    std::array<T, 12> t;
    for(int j = 0; j < 3; ++j)
    {
        const T g0       = g[0 * 3 + j];
        const T g1       = g[1 * 3 + j];
        const T g2       = g[2 * 3 + j];
        t[0 * 3 + j]     = g0;
        t[1 * 3 + j]     = half * (g0 + g1 + g2);
        t[2 * 3 + j]     = half * (g0 - g1 + g2);
        t[3 * 3 + j]     = g2;
    }
    // Step 2: U = t * G^T  (4x4)
    std::array<T, 16> u;
    for(int i = 0; i < 4; ++i)
    {
        const T t0   = t[i * 3 + 0];
        const T t1   = t[i * 3 + 1];
        const T t2   = t[i * 3 + 2];
        u[i * 4 + 0] = t0;
        u[i * 4 + 1] = half * (t0 + t1 + t2);
        u[i * 4 + 2] = half * (t0 - t1 + t2);
        u[i * 4 + 3] = t2;
    }
    if(miopen_signs)
    {
        u[3]  = T{0} - u[3];
        u[7]  = T{0} - u[7];
        u[11] = T{0} - u[11];
        u[12] = T{0} - u[12];
        u[13] = T{0} - u[13];
        u[14] = T{0} - u[14];
    }
    return u;
}

// Pre-transform a [K, C, 3, 3] constant weight literal into a [K, C, 4, 4]
// literal containing G * g * G^T for each (k, c). The resulting 16 elements
// per filter are then loaded directly by the winograd kernel without any
// runtime filter_transform - mirrors MIOpen's offline-transformed filter.
//
// For half-type filters, MIOpen sign convention is applied (6 positions
// negated) to match the DPP-cooperative input transform's V layout.
inline literal pretransform_filter_literal(const literal& w_lit)
{
    const auto& s      = w_lit.get_shape();
    const std::size_t K = s.lens()[0];
    const std::size_t C = s.lens()[1];

    // Both fp16 and fp32 kernels use the 4-lane DPP-cooperative input transform,
    // which produces V_alt with 6 sign-flipped positions vs canonical
    // V = B^T*d*B (positions e ∈ {3, 7, 11, 12, 13, 14}).  The matching
    // pretransformed U_alt has the same 6 negations so the dot product
    // U_alt · V_alt = U · V leaves the inverse transform intact.
    const bool miopen_signs = (s.type() == shape::half_type or s.type() == shape::float_type);

    literal result;
    w_lit.visit([&](auto w_view) {
        using T = std::remove_cv_t<typename decltype(w_view)::value_type>;
        if constexpr(std::is_floating_point<T>::value or
                     std::is_same<T, migraphx::half>::value)
        {
            std::vector<T> out(K * C * 16);
            for(std::size_t k = 0; k < K; ++k)
            {
                for(std::size_t c = 0; c < C; ++c)
                {
                    std::array<T, 9> g;
                    for(std::size_t i = 0; i < 9; ++i)
                        g[i] = static_cast<T>(w_view(k, c, i / 3, i % 3));
                    const auto u =
                        winograd_filter_pretransform_3x3<T>(g.data(), miopen_signs);
                    std::copy(u.begin(), u.end(), out.begin() + (k * C + c) * 16);
                }
            }
            result = literal{shape{s.type(), {K, C, 4, 4}}, out};
        }
    });
    return result;
}

MIGRAPHX_PRED_MATCHER(conv_winograd_compatible, instruction_ref ins)
{
    if(ins->name() != "convolution")
        return false;
    auto v = ins->get_operator().to_value();
    if(not all_of(v.at("stride"), [](const value& x) { return x.to<std::size_t>() == 1; }))
        return false;
    if(not all_of(v.at("dilation"), [](const value& x) { return x.to<std::size_t>() == 1; }))
        return false;
    if(v.at("group").to<std::size_t>() != 1)
        return false;
    auto w_lens = ins->inputs().back()->get_shape().lens();
    if(w_lens.size() != 4 or w_lens[2] != 3 or w_lens[3] != 3)
        return false;
    if(not all_of(v.at("padding"), [](const value& x) { return x.to<std::size_t>() == 1; }))
        return false;
    return true;
}

struct find_winograd_convolution
{
    auto matcher() const { return conv_winograd_compatible(); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins     = r.result;
        auto input   = ins->inputs().front();
        auto weights = ins->inputs().back();

        const auto t = input->get_shape().type();
        if(t != shape::half_type and t != shape::float_type)
            return;
        if(input->get_shape().lens().size() != 4)
            return;

        // Filter must be a compile-time constant so we can pre-transform
        // (G * g * G^T) offline and skip filter_transform at kernel runtime
        // - matches MIOpen's offline-transformed filter strategy.
        if(not weights->can_eval())
            return;

        auto w_arg = weights->eval();
        if(w_arg.empty())
            return;
        literal w_lit{w_arg.get_shape(), w_arg.data()};
        literal w_pre = pretransform_filter_literal(w_lit);

        auto v        = ins->get_operator().to_value();
        auto pad_vals = v.at("padding");
        std::vector<std::size_t> padding;
        std::transform(pad_vals.begin(),
                       pad_vals.end(),
                       std::back_inserter(padding),
                       [](const value& x) { return x.to<std::size_t>(); });

        auto pre_w_ins = m.add_literal(std::move(w_pre));
        m.replace_instruction(ins, winograd_conv{std::move(padding)}, input, pre_w_ins);
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

void prefuse_ops::apply(module_pass_manager& mpm) const
{
    const auto& device_name = ctx == nullptr ? "" : ctx->get_current_device().get_gfx_name();
    const bool is_navi = starts_with(device_name, "gfx11") or starts_with(device_name, "gfx12");
    if(enabled(MIGRAPHX_ENABLE_LAYERNORM_FUSION{}))
    {
        match::find_matches(mpm.get_module(), find_layernorm{});
        mpm.run_pass(dead_code_elimination{});
        match::find_matches(mpm.get_module(), find_add_layernorm{});
    }
    match::find_matches(mpm, find_gemm_softmax_gemm{enable_attention});
    if(is_navi)
        match::find_matches(mpm.get_module(), find_channelwise_convolution{});
    if(enabled(MIGRAPHX_ENABLE_WINOGRAD{}))
        match::find_matches(mpm.get_module(), find_winograd_convolution{});
    if(enabled(MIGRAPHX_DISABLE_MLIR{}))
    {
        inline_group_sub_module(mpm);
        mpm.run_pass(dead_code_elimination{});
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
