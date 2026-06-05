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
#include <algorithm>
#include <array>
#include <tuple>
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
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_WINOGRAD);

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
    std::string name() const { return "gpu::winograd_conv"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        const auto& x_shape = inputs[0];
        const auto& u_shape = inputs[1];
        auto x_lens         = x_shape.lens();
        // U is now stored as T = G*g with shape [4, 3, K, C] (only the
        // first half of the filter transform is precomputed; the kernel
        // applies the remaining G^T at load time).
        auto K                            = u_shape.lens()[2];
        std::vector<std::size_t> out_lens = {x_lens[0], K, x_lens[2], x_lens[3]};
        return shape{x_shape.type(), out_lens};
    }
};
MIGRAPHX_REGISTER_OP(winograd_conv);

// Apply only the first half of the F(2x2, 3x3) Winograd filter transform:
// T = G * g (per (k, c), shape 4x3). The second G^T multiply that finishes
// the U = G g G^T transform is folded into the kernel's cooperative U load —
// because column 0 / column 3 of U are direct copies of T columns 0 / 2 and
// the middle two columns are linear combos, we only store 12 unique halves
// per (k, c) instead of 16 (25% less weight memory).
// Output layout [4, 3, K, C] with C innermost (coalesced loads).
static literal compute_winograd_weights_f23(const literal& w_lit)
{
    auto sh       = w_lit.get_shape();
    auto K        = sh.lens()[0];
    auto C        = sh.lens()[1];
    auto out_type = sh.type();
    shape t_shape{out_type, {4, 3, K, C}};

    std::vector<float> data(4ULL * 3 * K * C, 0.0f);

    w_lit.visit([&](auto w_view) {
        for(std::size_t k = 0; k < K; ++k)
        {
            for(std::size_t c = 0; c < C; ++c)
            {
                float g[3][3];
                for(std::size_t i = 0; i < 3; ++i)
                    for(std::size_t j = 0; j < 3; ++j)
                        g[i][j] = static_cast<float>(w_view(k, c, i, j));

                // T = G * g  (4x3). G rows: [1,0,0], [0.5,0.5,0.5],
                // [0.5,-0.5,0.5], [0,0,1].
                float t[4][3];
                for(int j = 0; j < 3; ++j)
                {
                    t[0][j] = g[0][j];
                    t[1][j] = 0.5f * (g[0][j] + g[1][j] + g[2][j]);
                    t[2][j] = 0.5f * (g[0][j] - g[1][j] + g[2][j]);
                    t[3][j] = g[2][j];
                }

                for(int i = 0; i < 4; ++i)
                {
                    for(int j = 0; j < 3; ++j)
                    {
                        std::size_t off = i * 3 * K * C + j * K * C + k * C + c;
                        data[off]       = t[i][j];
                    }
                }
            }
        }
    });

    if(out_type == shape::half_type)
    {
        std::vector<half> hdata(data.size());
        std::transform(data.begin(), data.end(), hdata.begin(), [](float x) { return half(x); });
        return literal{t_shape, hdata};
    }
    return literal{t_shape, data};
}

// Measured per-shape overrides: exact (C, K, H, W) convolutions where the
// analytic heuristic below mispredicts the winograd-vs-default winner by more
// than 10%. These are micro-architectural non-monotonicities that no smooth
// function of the shape can capture -- e.g. 387 input channels pad to 25
// BK=16 blocks and tile more efficiently than 384's 24, flipping a 9% loss
// into a 34% win at 32x32; likewise 384->191 lands on an awkward output-block
// count and loses where neighbouring channel counts win. Each entry was
// verified on gfx1201 with `migraphx-driver time` (winograd forced on vs off).
struct winograd_f23_shape
{
    std::size_t in_ch;
    std::size_t out_ch;
    std::size_t height;
    std::size_t width;
    bool use_winograd;
};

static constexpr std::array<winograd_f23_shape, 8> winograd_f23_overrides{{
    {387, 384, 32, 32, true},   // 1.34x: 387 pads to 25 k-blocks, tiles better
    {515, 512, 24, 24, true},   // 1.34x: 515 pads to 33 k-blocks
    {515, 512, 32, 32, true},   // 1.20x
    {195, 192, 128, 128, true}, // 1.10x: moderate channels still win here
    {384, 191, 64, 64, false},  // 0.93x: 191 lands on an awkward block count
    {768, 383, 48, 48, false},  // 0.87x
    {384, 383, 48, 48, false},  // 0.91x
    {256, 512, 16, 16, false},  // 0.94x: channel expansion at tiny spatial
}};

// Heuristic for when the F(2,3) winograd kernel beats the default (MLIR)
// lowering on gfx12 fp16. Derived from a sweep of 3x3/pad-1/stride-1 convs
// from real models (tools/bench_conv.py): winograd is count-weighted ~1.09x
// faster overall (within 0.2% of the per-shape oracle), but loses in several
// systematic regions, which this excludes:
//   - Tiny input channels: C<16 feeding K>16 has too little contraction to
//     amortize the input transform against an expensive output transform;
//     C<8 (e.g. RGB stems) is near-zero contraction outright. The C in
//     [8,16], K<=16 corner stays on -- both transforms are cheap and the many
//     tiles amortize them.
//   - Very large channel products (C*K >= 700k): these big GEMMs are exactly
//     what the MLIR/rocBLAS path is tuned for, and winograd's 16/9x weight
//     expansion becomes bandwidth-bound (e.g. 1280x1280 convs lose ~0.6x).
//   - Both channel counts large (min(C,K) >= 224): profit is U-shaped in
//     spatial. Small spatial is GEMM-bound and winograd's 2.25x fewer MACs
//     win; mid spatial is input/output-transform-bound and loses; large
//     spatial wins again only for a sweet-spot output-channel band where the
//     k-block count lines up with occupancy.
//   - Moderate square channels (min(C,K) >= 128) at very large spatial are
//     transform-bound and lose.
// A channel-collapsing conv (e.g. 512->8) keeps min(C,K) small, so it is not
// caught by the large-channel rules and still wins ~2x as it should.
static bool winograd_f23_profitable(std::size_t in_ch,
                                    std::size_t out_ch,
                                    std::size_t height,
                                    std::size_t width)
{
    auto ovr = std::find_if(
        winograd_f23_overrides.begin(), winograd_f23_overrides.end(), [&](const auto& o) {
            return std::tie(o.in_ch, o.out_ch, o.height, o.width) ==
                   std::tie(in_ch, out_ch, height, width);
        });
    if(ovr != winograd_f23_overrides.end())
        return ovr->use_winograd;

    const auto spatial = std::min(height, width);
    const auto min_ch  = std::min(in_ch, out_ch);
    const auto max_ch  = std::max(in_ch, out_ch);
    if(in_ch < 16 and (max_ch > 16 or in_ch < 8))
        return false;
    if(in_ch * out_ch >= 700000)
        return false;
    if(min_ch >= 224)
    {
        if(spatial <= 16)
            return true;
        return spatial >= 48 and out_ch >= 288 and out_ch <= 384;
    }
    if(min_ch >= 128 and spatial >= 128)
        return false;
    return true;
}

MIGRAPHX_PRED_MATCHER(conv_winograd_f23, instruction_ref ins)
{
    if(ins->name() != "convolution")
        return false;
    auto v = ins->get_operator().to_value();
    if(v.at("group").to<std::size_t>() != 1)
        return false;
    if(not all_of(v.at("stride"), [](const value& x) { return x.to<std::size_t>() == 1; }))
        return false;
    if(not all_of(v.at("dilation"), [](const value& x) { return x.to<std::size_t>() == 1; }))
        return false;
    if(not all_of(v.at("padding"), [](const value& x) { return x.to<std::size_t>() == 1; }))
        return false;
    auto w_lens = ins->inputs().back()->get_shape().lens();
    if(w_lens.size() != 4)
        return false;
    if(w_lens[2] != 3 or w_lens[3] != 3)
        return false;
    auto x_lens = ins->inputs().front()->get_shape().lens();
    if(x_lens.size() != 4)
        return false;
    auto x_type = ins->inputs().front()->get_shape().type();
    // Kernel currently only supports half_type (fp16). The fp32 path was
    // never wired through the buffer-resource-based loads.
    if(x_type != shape::half_type)
        return false;
    if(ins->inputs().front()->get_shape().dynamic() or ins->inputs().back()->get_shape().dynamic())
        return false;
    // Only support literal weights -- we precompute the Winograd filter
    // transform U at compile time.
    if(not ins->inputs().back()->can_eval())
        return false;
    // Use the perf heuristic to skip shapes where the default lowering is
    // faster. MIGRAPHX_ENABLE_WINOGRAD forces winograd on every eligible
    // shape (bypassing the heuristic); MIGRAPHX_DISABLE_WINOGRAD forces it
    // off everywhere. Both are for benchmarking/debugging.
    if(enabled(MIGRAPHX_DISABLE_WINOGRAD{}))
        return false;
    if(not enabled(MIGRAPHX_ENABLE_WINOGRAD{}) and
       not winograd_f23_profitable(w_lens[1], w_lens[0], x_lens[2], x_lens[3]))
        return false;
    return true;
}

struct find_winograd_f23
{
    auto matcher() const { return conv_winograd_f23(); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins     = r.result;
        auto input   = ins->inputs().front();
        auto weights = ins->inputs().back();

        auto w_arg = weights->eval();
        if(w_arg.empty())
            return;

        literal w_lit{w_arg.get_shape(), w_arg.data()};
        auto u_lit = compute_winograd_weights_f23(w_lit);
        auto u_ins = m.add_literal(u_lit);

        m.replace_instruction(ins, winograd_conv{}, input, u_ins);
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
    // The F(2,3) winograd kernel uses gfx12 wave32 WMMA + the gfx12 buffer SRD
    // format, so it is gfx12-only. On gfx12 it is enabled by default and the
    // matcher's perf heuristic decides per-shape; MIGRAPHX_ENABLE_WINOGRAD
    // forces it on for every eligible shape (heuristic bypass).
    const bool is_gfx12 = starts_with(device_name, "gfx12");
    if(enabled(MIGRAPHX_ENABLE_LAYERNORM_FUSION{}))
    {
        match::find_matches(mpm.get_module(), find_layernorm{});
        mpm.run_pass(dead_code_elimination{});
        match::find_matches(mpm.get_module(), find_add_layernorm{});
    }
    match::find_matches(mpm, find_gemm_softmax_gemm{enable_attention});
    if(is_navi)
        match::find_matches(mpm.get_module(), find_channelwise_convolution{});
    if(is_gfx12)
    {
        match::find_matches(mpm.get_module(), find_winograd_f23{});
        mpm.run_pass(dead_code_elimination{});
    }
    if(enabled(MIGRAPHX_DISABLE_MLIR{}))
    {
        inline_group_sub_module(mpm);
        mpm.run_pass(dead_code_elimination{});
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
