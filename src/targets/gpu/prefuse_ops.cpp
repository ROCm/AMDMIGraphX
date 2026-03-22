/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

struct pre_winograd_conv
{
    int group           = 1;
    bool pretransformed = false;
    // K stored for compute_shape when weights are pretransformed (flat shape)
    std::size_t num_filters = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.group, "group"),
                    f(self.pretransformed, "pretransformed"),
                    f(self.num_filters, "num_filters"));
    }

    std::string name() const { return "gpu::pre_winograd_conv"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        auto in_lens = inputs[0].lens();
        auto k       = pretransformed ? num_filters : inputs[1].lens()[0];
        return inputs[0].with_lens({in_lens[0], k, in_lens[2], in_lens[3]});
    }
};
MIGRAPHX_REGISTER_OP(pre_winograd_conv);

auto is_winograd_eligible()
{
    return match::make_basic_pred_matcher([](instruction_ref ins) {
        if(ins->get_shape().dynamic())
            return false;
        if(ins->get_shape().type() != shape::float_type)
            return false;

        auto v        = ins->get_operator().to_value();
        auto padding  = v.at("padding").to_vector<std::size_t>();
        auto stride   = v.at("stride").to_vector<std::size_t>();
        auto dilation = v.at("dilation").to_vector<std::size_t>();

        if(stride != std::vector<std::size_t>{1, 1})
            return false;
        if(dilation != std::vector<std::size_t>{1, 1})
            return false;
        if(padding != std::vector<std::size_t>{1, 1, 1, 1})
            return false;

        auto inputs = ins->inputs();
        if(inputs.size() != 2)
            return false;
        auto in_shape = inputs[0]->get_shape();
        auto w_shape  = inputs[1]->get_shape();
        // Must be 4D NCHW
        if(in_shape.ndim() != 4 or w_shape.ndim() != 4)
            return false;
        // Must be 3x3 kernel
        if(w_shape.lens()[2] != 3 or w_shape.lens()[3] != 3)
            return false;
        // Both inputs must be standard (packed NCHW) layout
        if(not in_shape.standard() or not w_shape.standard())
            return false;
#if 0
        // Eligibility based on empirical benchmarks across top-20 configs.
        // Winograd F(2x2,3x3) wins when:
        //   1. Enough tiles for the LDS-GEMM to amortize overhead (tiles ≥ 49)
        //   2. Enough compute per tile for arithmetic reduction to matter (C*K ≥ 32K)
        auto c     = in_shape.lens()[1];
        auto k     = w_shape.lens()[0];
        auto h     = in_shape.lens()[2];
        auto w_dim = in_shape.lens()[3];
        std::size_t tiles = ((h + 1) / 2) * ((w_dim + 1) / 2);
        if(tiles < 49)
            return false;
        // Too many tiles: Phase 1 tile loading dominates over GEMM savings
        if(tiles > 512)
            return false;
        if(c * k < 32768)
            return false;
#endif
        return true;
    });
}

struct find_winograd_conv
{
    auto matcher() const { return match::name("convolution")(is_winograd_eligible()); }

    // Winograd F(2x2,3x3) filter transform G * g * G^T on CPU
    static literal transform_filters_cpu(const argument& w_arg, std::size_t k, std::size_t cpg)
    {
        std::vector<float> result(k * cpg * 16);
        const auto* w_data = reinterpret_cast<const float*>(w_arg.data());
        for(std::size_t i = 0; i < k * cpg; i++)
        {
            const float* g = w_data + i * 9;
            float* u       = result.data() + i * 16;
            // G column transform (3→4)
            float t[12];
            for(int j = 0; j < 3; j++)
            {
                float g0 = g[j], g1 = g[3 + j], g2 = g[6 + j];
                float s = (g0 + g2) * 0.5f, d = g1 * 0.5f;
                t[j]     = g0;
                t[3 + j] = s + d;
                t[6 + j] = s - d;
                t[9 + j] = g2;
            }
            // G^T row transform (3→4)
            for(int r = 0; r < 4; r++)
            {
                float t0 = t[r * 3], t1 = t[r * 3 + 1], t2 = t[r * 3 + 2];
                float s = (t0 + t2) * 0.5f, dd = t1 * 0.5f;
                u[r * 4]     = t0;
                u[r * 4 + 1] = s + dd;
                u[r * 4 + 2] = s - dd;
                u[r * 4 + 3] = t2;
            }
        }
        shape out_shape{shape::float_type, {k * cpg * 16}};
        return literal{out_shape, result};
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins        = r.result;
        auto v          = ins->get_operator().to_value();
        int group       = v.at("group").to<int>();
        auto weight_ins = ins->inputs()[1];
        auto w_shape    = weight_ins->get_shape();
        auto k          = w_shape.lens()[0];
        auto cpg        = w_shape.lens()[1];

        if(weight_ins->can_eval())
        {
            // Precompute filter transform at compile time
            auto w_arg       = weight_ins->eval();
            auto xformed_lit = transform_filters_cpu(w_arg, k, cpg);
            auto lit_ins     = m.add_literal(xformed_lit);
            m.replace_instruction(
                ins, pre_winograd_conv{group, true, k}, ins->inputs()[0], lit_ins);
        }
        else
        {
            m.replace_instruction(ins, pre_winograd_conv{group, false, k}, ins->inputs());
        }
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
    if(enabled(MIGRAPHX_ENABLE_WINOGRAD{}))
    {
        match::find_matches(mpm.get_module(), find_winograd_conv{});
        mpm.run_pass(dead_code_elimination{});
    }
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
