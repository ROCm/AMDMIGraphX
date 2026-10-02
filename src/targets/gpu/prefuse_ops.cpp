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
#include <migraphx/dfor.hpp>
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
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_WINOGRAD_FP32_SSTORE);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_WINOGRAD);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_WINOGRAD_FULL_TRANSFORM);

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
    // Minimum C*H*W for half_type to use channelwise kernel instead of MLIR
    std::size_t channelwise_half_min_chw = 48 * 1024;

    auto matcher() const { return conv_channelwise(); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins         = r.result;
        auto input       = ins->inputs().front();
        auto weights     = ins->inputs().back();
        auto num_spatial = ins->get_shape().ndim() - 2;

        const auto type = input->get_shape().type();
        if(type != shape::float_type and type != shape::half_type)
            return;

        if(type == shape::half_type)
        {
            const auto& lens = input->get_shape().lens();
            const auto chw =
                std::accumulate(lens.begin() + 1, lens.end(), std::size_t{1}, std::multiplies<>{});
            if(chw < channelwise_half_min_chw)
                return;
        }

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
    // When true the weight input is the raw filter g [3, 3, K, C] and the
    // kernel computes the full U = G g G^T transform at load time (9 halves
    // per (k,c), best for weight-bandwidth-bound large-channel convs). When
    // false the weight is the half-transformed T = G*g [4, 3, K, C] and the
    // kernel only applies G^T (12 halves, best when the kernel is VALU-bound).
    bool full_transform = false;
    // Output layout (permutation). Set to the layout layout_convolution chose
    // for the convolution this op replaces, so winograd is a drop-in: same
    // inputs, same output layout (e.g. NHWC in -> NHWC out). Defaults to NCHW.
    std::vector<int64_t> output_layout = {0, 1, 2, 3};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.full_transform, "full_transform"),
                    f(self.output_layout, "output_layout"));
    }

    std::string name() const { return "gpu::winograd_conv"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        const auto& x_shape = inputs[0];
        const auto& u_shape = inputs[1];
        auto x_lens         = x_shape.lens();
        const auto& u_lens  = u_shape.lens();
        // The weight encodes the output channel count K; its axis depends on the
        // store layout: fp16 [4|3, 3, K, C] and fp32 full-U / S-store [4|3, 4, K, C]
        // put K at dim 2, while the fp32 v-innermost store [4, K, C, 4] (the NHWC
        // coalesced weight load) puts K at dim 1. v-inner is the fp32 layout whose
        // dim 1 is not the 4-wide v axis (matches the kernel's lens[1] != 4 test).
        const bool vinner                 = x_shape.type() == shape::float_type and u_lens[1] != 4;
        const auto out_c                  = vinner ? u_lens[1] : u_lens[2];
        std::vector<std::size_t> out_lens = {x_lens[0], out_c, x_lens[2], x_lens[3]};
        return shape::from_permutation(x_shape.type(), out_lens, output_layout);
    }
};
MIGRAPHX_REGISTER_OP(winograd_conv);

// Precompute the F(2x2, 3x3) Winograd filter weight for the kernel. Two modes:
//   full_transform=false: store the half-transformed T = G*g, shape [4, 3, K, C]
//     (12 halves per (k,c)). The kernel finishes the U = T*G^T column transform
//     at load time. Fewer in-kernel ops -- best when the kernel is VALU-bound.
//   full_transform=true: store the raw filter g, shape [3, 3, K, C] (9 halves).
//     The kernel computes the full U = G g G^T at load time. 25% less weight
//     memory -- best for weight-bandwidth-bound large-channel convs.
// The first dim differs (4 vs 3) but the byte-offset formula is identical
// (both have a size-3 second dim). Output has C innermost (coalesced loads).
literal compute_winograd_weights_f23(const argument& w_arg, bool full_transform)
{
    auto sh                 = w_arg.get_shape();
    auto out_c              = sh.lens()[0];
    auto in_c               = sh.lens()[1];
    auto out_type           = sh.type();
    const std::size_t nrows = full_transform ? 3 : 4;
    shape w_shape{out_type, {nrows, 3, out_c, in_c}};

    std::vector<float> data(nrows * 3 * out_c * in_c, 0.0f);

    w_arg.visit([&](auto w_view) {
        dfor(out_c, in_c)([&](auto k, auto c) {
            float g[3][3];
            dfor(std::size_t{3},
                 std::size_t{3})([&](auto i, auto j) { g[i][j] = w_view(k, c, i, j); });

            if(full_transform)
            {
                dfor(std::size_t{3}, std::size_t{3})(
                    [&](auto i, auto j) { data[w_shape.index({i, j, k, c})] = g[i][j]; });
            }
            else
            {
                // T = G * g (4x3). G rows: [1,0,0], [0.5,0.5,0.5],
                // [0.5,-0.5,0.5], [0,0,1].
                dfor(std::size_t{3})([&](auto j) {
                    data[w_shape.index({0, j, k, c})] = g[0][j];
                    data[w_shape.index({1, j, k, c})] = 0.5f * (g[0][j] + g[1][j] + g[2][j]);
                    data[w_shape.index({2, j, k, c})] = 0.5f * (g[0][j] - g[1][j] + g[2][j]);
                    data[w_shape.index({3, j, k, c})] = g[2][j];
                });
            }
        });
    });

    if(out_type == shape::half_type)
    {
        std::vector<half> hdata(data.size());
        std::transform(data.begin(), data.end(), hdata.begin(), [](float x) { return half(x); });
        return literal{w_shape, hdata};
    }
    return literal{w_shape, data};
}

// Winograd F(2,3) filter matrix G (4x3): rows [1,0,0], [.5,.5,.5], [.5,-.5,.5],
// [0,0,1]. Shared by the fp32 full-U and S-store weight transforms below.
constexpr std::array<std::array<float, 3>, 4> winograd_f23_gmat{
    {{1.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}};

// Precompute the FULL Winograd filter transform U = G g G^T for the fp32 FMA/DPP
// kernel, stored as an [4, 4, K, C] literal (indices u, v, k, c; C innermost).
// The fp32 kernel does not transform the weight in-kernel (unlike the fp16
// path), so the whole 4x4 winograd weight is materialized here.
//
// The kernel realizes the input transform's v=3 column with a sign flip
// (d3-d1 instead of d1-d3) because that is the form a single quad DPP butterfly
// can produce; a matching negation of U[:,3,:,:] here makes the elementwise
// product U*V exact.
// vinner=true stores U packed as [u, k, c, v] (v physically INNERMOST) instead of
// the NCHW [u, v, k, c] (c innermost). In NHWC the fp32 kernel's 4 v_col lanes then
// read 4 *consecutive* floats -> the weight load coalesces (lane==v_col otherwise
// scatters the weight across v-slices K*C apart, thrashing the cache with the
// input). The kernel selects the v/k/c stride indices by layout (NHWC template).
literal compute_winograd_weights_f23_fp32(const argument& w_arg, bool vinner = false)
{
    const auto& sh = w_arg.get_shape();
    auto out_c     = sh.lens()[0];
    auto in_c      = sh.lens()[1];
    shape u_shape  = vinner ? shape{shape::float_type, {4, out_c, in_c, 4}}  // [u,k,c,v]
                            : shape{shape::float_type, {4, 4, out_c, in_c}}; // [u,v,k,c]
    auto widx      = [&](std::size_t u, std::size_t v, std::size_t k, std::size_t c) {
        return vinner ? u_shape.index({u, k, c, v}) : u_shape.index({u, v, k, c});
    };

    std::vector<float> data(u_shape.elements(), 0.0f);
    const auto& gmat = winograd_f23_gmat;
    w_arg.visit([&](auto w_view) {
        dfor(out_c, in_c)([&](auto k, auto c) {
            float g[3][3];
            dfor(std::size_t{3},
                 std::size_t{3})([&](auto i, auto j) { g[i][j] = w_view(k, c, i, j); });

            // Gg (4x3): (Gg)[u][j] = sum_i G[u][i] g[i][j].
            float gg[4][3];
            dfor(std::size_t{4}, std::size_t{3})([&](auto u, auto j) {
                gg[u][j] = gmat[u][0] * g[0][j] + gmat[u][1] * g[1][j] + gmat[u][2] * g[2][j];
            });
            // U[u][v] = sum_j (Gg)[u][j] G[v][j], with the v=3 column negated.
            dfor(std::size_t{4}, std::size_t{4})([&](auto u, auto v) {
                float uv = gg[u][0] * gmat[v][0] + gg[u][1] * gmat[v][1] + gg[u][2] * gmat[v][2];
                data[widx(u, v, k, c)] = (v == 3) ? -uv : uv;
            });
        });
    });
    return literal{u_shape, data};
}

// S-store: precompute the v-half-transformed winograd weight S = g G^T, stored as
// [3, 4, K, C] (i, v, k, c; C innermost). This is 12 values/(k,c) vs full U's 16
// (25% less weight DRAM), and each lane loads only its v_col's 3 values (vs 4 for
// U). The kernel finishes U = G S with a register-only u-transform (u lives in
// registers, so no cross-lane traffic -- unlike a u-half g-store, which would
// leave the kernel a cross-lane v-transform). The
// v=3 column is negated here to match the input butterfly's d3-d1 sign, exactly
// as the full-U store does.
literal compute_winograd_weights_f23_fp32_sstore(const argument& w_arg)
{
    const auto& sh = w_arg.get_shape();
    auto out_c     = sh.lens()[0];
    auto in_c      = sh.lens()[1];
    shape s_shape{shape::float_type, {3, 4, out_c, in_c}};

    std::vector<float> data(s_shape.elements(), 0.0f);
    const auto& gmat = winograd_f23_gmat;
    w_arg.visit([&](auto w_view) {
        dfor(out_c, in_c)([&](auto k, auto c) {
            float g[3][3];
            dfor(std::size_t{3},
                 std::size_t{3})([&](auto i, auto j) { g[i][j] = w_view(k, c, i, j); });
            // S[i][v] = sum_j g[i][j] G[v][j], with the v=3 column negated.
            dfor(std::size_t{3}, std::size_t{4})([&](auto i, auto v) {
                float sv = g[i][0] * gmat[v][0] + g[i][1] * gmat[v][1] + g[i][2] * gmat[v][2];
                data[s_shape.index({i, v, k, c})] = (v == 3) ? -sv : sv;
            });
        });
    });
    return literal{s_shape, data};
}

// Look up an exact (C, K, H, W) entry in a per-shape override table (each entry
// must expose in_ch/out_ch/height/width fields); returns the entry or nullptr.
// Shared by the fp16/fp32 profitability heuristics and the S-store table.
template <class Table>
const typename Table::value_type* find_shape_override(const Table& table,
                                                      std::size_t in_ch,
                                                      std::size_t out_ch,
                                                      std::size_t height,
                                                      std::size_t width)
{
    auto it = std::find_if(table.begin(), table.end(), [&](const auto& o) {
        return std::tie(o.in_ch, o.out_ch, o.height, o.width) ==
               std::tie(in_ch, out_ch, height, width);
    });
    return it == table.end() ? nullptr : &*it;
}

// Measured per-shape overrides: exact (C, K, H, W) convolutions where the
// analytic heuristic below mispredicts the winograd-vs-default winner by more
// than 10% (using the better of the two weight stores). These are
// micro-architectural non-monotonicities that no smooth function of the shape
// captures -- e.g. 768->383 at 32x32 lands on an awkward output-block count and
// loses where neighbouring channel counts win. The last group are very large
// (C*K >= 700k) convs that the bandwidth rule excludes but that the g weight
// store reclaims at the one spatial size where they cross 1.0. Each entry was
// verified on gfx1201 with `migraphx-driver time`.
struct winograd_f23_shape
{
    std::size_t in_ch;
    std::size_t out_ch;
    std::size_t height;
    std::size_t width;
    bool use_winograd;
};

constexpr std::array<winograd_f23_shape, 10> winograd_f23_overrides{{
    {195, 192, 128, 128, true}, // 1.12x: missed win (moderate ch, large spatial)
    {768, 383, 32, 32, false},  // 0.86x: wrong pick
    {768, 383, 48, 48, false},  // 0.90x
    {384, 383, 32, 32, false},  // 0.92x
    {384, 383, 48, 48, false},  // 0.91x
    {1024, 511, 24, 24, false}, // 0.86x
    {64, 112, 256, 256, false}, // 0.91x
    {1280, 1280, 30, 30, true}, // 1.04x: C*K>=700k reclaimed via g store
    {1920, 1280, 30, 30, true}, // 1.05x
    {640, 1280, 30, 30, true},  // 1.03x
}};

// Heuristic for when the F(2,3) winograd kernel beats the default (MLIR)
// lowering on gfx12 fp16. Derived from a sweep of 3x3/pad-1/stride-1 convs
// from real models (tools/bench_conv.py): winograd is count-weighted ~1.12x
// faster overall with the dual T/g weight store (see winograd_f23_full_transform),
// close to the per-shape oracle, but loses in several systematic regions, which
// this excludes:
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
bool winograd_f23_profitable(
    std::size_t in_ch, std::size_t out_ch, std::size_t height, std::size_t width, bool nhwc)
{
    const auto spatial = std::min(height, width);
    const auto min_ch  = std::min(in_ch, out_ch);
    const auto max_ch  = std::max(in_ch, out_ch);

    // NHWC layout gate. rocMLIR's channels-last convolution is heavily tuned
    // and becomes the GEMM-bound winner once both channel counts are large:
    // winograd's ~2.25x fewer MACs stop mattering (the kernel is not
    // compute-bound) and its transform overhead/weight expansion can only tie
    // or lose. Measured on gfx1201 fp16 (rigorous interleaved): 256ch@64 ~0.80x,
    // and the large-channel band that wins in NCHW (min_ch>=224 at small spatial
    // or out_ch in [288,384]) merely ties in NHWC. So gate the whole
    // large-channel regime for NHWC -- this is layout-specific; NCHW still wins
    // it and is unchanged. The override table below is NCHW-derived, so the NHWC
    // gate is applied first.
    if(nhwc)
    {
        if(min_ch >= 224)
            return false;
        if(min_ch >= 128 and spatial >= 48)
            return false;
        if(min_ch >= 64 and spatial >= 64)
            return false;
        if(max_ch >= 4 * min_ch and spatial >= 32)
            return false;
    }

    if(const auto* ovr = find_shape_override(winograd_f23_overrides, in_ch, out_ch, height, width))
        return ovr->use_winograd;

    if(in_ch < 16 and (max_ch > 16 or in_ch < 8))
        return false;
    if(in_ch * out_ch >= 700000)
        return false;
    if(min_ch >= 224)
    {
        if(spatial <= 32)
            return true;
        return spatial >= 48 and out_ch >= 288 and out_ch <= 384;
    }
    if(min_ch >= 128 and spatial >= 128)
        return false;
    return true;
}

// Whether to store the raw filter g (full in-kernel transform, 9 halves/KC)
// instead of the half-transformed T (12 halves/KC). Storing g saves 25% weight
// bandwidth -- a win for the bandwidth-bound large-channel convs -- but adds
// in-kernel transform VALU, which loses for VALU-bound large-spatial convs and
// for channel-collapsing convs (tiny K) whose weights aren't the bottleneck.
// Derived from the same gfx12 fp16 sweep as winograd_f23_profitable.
bool winograd_f23_full_transform(std::size_t in_ch,
                                 std::size_t out_ch,
                                 std::size_t height,
                                 std::size_t width)
{
    // Benchmarking override: force g storage on every winograd conv.
    if(enabled(MIGRAPHX_WINOGRAD_FULL_TRANSFORM{}))
        return true;
    const auto spatial = std::min(height, width);
    // g (full in-kernel transform) saves 25% weight bandwidth and wins on the
    // large majority of shapes. Fall back to the half-transformed T only where
    // the kernel is not bandwidth-bound and g's extra row-transform VALU costs
    // more than the bandwidth it saves:
    //   - Channel-collapsing convs (tiny output channel): weights are already
    //     small, so the saved bandwidth is negligible and the extra VALU loses.
    //   - VALU/transform-bound shapes: large min(C,K) * spatial, where the
    //     input/output transforms dominate -- EXCEPT very large channel
    //     products, which stay weight-bandwidth-bound and still prefer g.
    // Thresholds from the gfx12 fp16 sweep.
    if(out_ch <= 16)
        return false;
    if(in_ch * out_ch >= 700000)
        return true;
    if(std::min(in_ch, out_ch) * spatial >= 17000)
        return false;
    return true;
}

struct winograd_f23_sstore_shape
{
    std::size_t in_ch;
    std::size_t out_ch;
    std::size_t height;
    std::size_t width;
};

// Shapes in the spatial-16..64 high-channel band that prefer the S-store. In this
// band the S-vs-full-U preference is micro-architecturally non-monotonic --
// neighbouring channel counts flip it (512->512 vs 515->512 at 16x16, 192->191 vs
// 192->192 at 64x64, 768->383 vs 384->384 at 32x32) -- so a smooth rule cannot
// separate them without regressing the shapes that need the full U. Hence an
// explicit table, like the fp16 path.
constexpr std::array<winograd_f23_sstore_shape, 7> winograd_f23_sstore_overrides{{
    {512, 512, 16, 16},
    {512, 512, 24, 24},
    {195, 192, 64, 64},
    {768, 383, 32, 32},
    {384, 383, 32, 32},
    {384, 191, 64, 64},
    {192, 191, 64, 64},
}};

// Choose the fp32 winograd weight encoding: S-store (v-half-transformed g*G^T,
// [3,4,K,C]) vs the full U ([4,4,K,C]). S-store cuts weight loads AND bytes 25%
// and finishes U with a cheap register-only u-transform, so it suits the
// weight-load-dominated shapes: high channels with small spatial. Elsewhere its
// extra register FMA and k-outer's lower ILP cost more than the bandwidth saved.
//
// Two selection paths: (1) a smooth zone (very small spatial, high channels)
// where S-store is uniformly preferable; (2) the override table above for the
// non-monotonic spatial-16..64 band.
bool winograd_f23_use_sstore(std::size_t in_ch,
                             std::size_t out_ch,
                             std::size_t height,
                             std::size_t width)
{
    if(std::min(in_ch, out_ch) >= 256 and std::min(height, width) <= 12)
        return true;
    return find_shape_override(winograd_f23_sstore_overrides, in_ch, out_ch, height, width) !=
           nullptr;
}

// Per-shape overrides for the fp32 F(2,3) heuristic below: exact (C, K, H, W)
// convs where the smooth rule mispredicts winograd-vs-MLIR. Two groups need a
// table (like the fp16 path): high-channel square convs in the spatial-16..32
// band where the S-store flips some but not their neighbours, and awkward-out_ch
// convs at large spatial (see the per-entry notes below). Reuses
// winograd_f23_shape (same C/K/H/W + use_winograd fields as the fp16 table).
constexpr std::array<winograd_f23_shape, 4> winograd_f23_fp32_overrides{{
    // Exact-square high-channel convs at their native spatial: rocBLAS/MLIR is
    // tuned best exactly here, and the smooth min_ch>=224 rule (which holds for
    // the off-square and larger-channel neighbours) mispredicts these.
    {512, 512, 16, 16, false}, // square, the S-store cannot close the gap
    {256, 256, 32, 32, false}, // square
    // Awkward output-channel count (95, not a multiple of the KO tile) at large
    // spatial: the winograd kernel wastes a partial output-channel block that the
    // many tiles then pay for repeatedly, while MLIR does not. min(C,K)=95 < 128,
    // so the smooth rule would otherwise keep them (96->96 at the same spatial is
    // fine).
    {96, 95, 128, 128, false},
    {192, 95, 128, 128, false},
}};

// Heuristic for when the fp32 FMA/DPP F(2,3) winograd kernel beats the default
// (rocMLIR implicit-GEMM) lowering on gfx12, for 3x3/pad-1/stride-1 convs with
// the kernel's own weight-store selection (S-store / v-inner) active. Structure
// mirrors the fp16 winograd_f23_profitable, but the thresholds differ: the fp32
// kernel has 2.25x fewer MACs than MLIR yet a heavier input/weight scatter, so it
// wins the compute-bound low/mid-channel shapes and loses the memory-bandwidth-
// bound high-channel large-spatial ones.
//   - NHWC: rocMLIR's channels-last GEMM reads the input fully coalesced and wins
//     almost everywhere; the winograd kernel's C-strided input scatter only pays
//     off at tiny spatial + high channels (spatial<=16, min_ch>=256), so NHWC is
//     gated to that narrow zone.
//   - NCHW: winograd wins broadly. Excluded regions:
//       * C*K >= 700k: bandwidth-bound big GEMMs MLIR owns (1280-channel convs).
//       * min(C,K) >= 224: only small spatial (<=32) wins (2.25x fewer MACs);
//         mid/large spatial is input/output-transform + weight-expansion bound.
//       * min(C,K) >= 128 at spatial >= 128: transform-bound, loses.
//       * spatial >= 512 with out_ch > 32: the 4x input-tile re-read dominates
//         unless the output fits a single KO block.
// Output-collapse (out_ch <= 3) and tiny-input (in_ch < 16) shapes are handled
// layout-independently up front. MIGRAPHX_ENABLE/DISABLE_WINOGRAD override it.
bool winograd_f23_fp32_profitable(
    std::size_t in_ch, std::size_t out_ch, std::size_t height, std::size_t width, bool nhwc)
{
    const auto spatial = std::min(height, width);
    const auto min_ch  = std::min(in_ch, out_ch);
    const auto max_ch  = std::max(in_ch, out_ch);

    // The next three checks are layout-independent (they hold for both NCHW and
    // NHWC), so they run before the layout split.

    // Tiny input channels (RGB-style stems) have too little contraction to
    // amortize the winograd transforms, so they lose to a plain GEMM. (Mirrors
    // the fp16 rule.)
    if(in_ch < 16 and (max_ch > 16 or in_ch < 8))
        return false;

    // Output-collapse convs (out_ch <= 3, e.g. a segmentation/prediction head):
    // the winograd output transform is nearly free on so few output channels while
    // MLIR still runs a full small GEMM, so winograd wins in both layouts.
    if(out_ch <= 3)
        return true;

    // Per-shape overrides (each listed shape loses in both layouts).
    if(const auto* ovr =
           find_shape_override(winograd_f23_fp32_overrides, in_ch, out_ch, height, width))
        return ovr->use_winograd;

    // NHWC: rocMLIR's coalesced channels-last GEMM wins almost everywhere; the
    // winograd kernel's C-strided NHWC input scatter (no host layout freedom, so
    // it can't coalesce) only pays off at high channels with tiny spatial, where
    // the re-read footprint is small and cached.
    if(nhwc)
        return min_ch >= 256 and spatial <= 16;

    // NCHW: winograd wins broadly.
    if(in_ch * out_ch >= 700000)
        return false;
    if(min_ch >= 224)
        return spatial <= 32;
    if(min_ch >= 128 and spatial >= 128)
        return false;
    // Very large spatial (the 4x winograd input-tile re-read dominates): only a
    // single-KO-block output (out_ch <= 32) survives it. (Low/mid channel counts
    // still win up to 256x256; min(C,K)>=128 at large spatial is already excluded
    // above.)
    if(spatial >= 512 and out_ch > 32)
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
    // fp16 uses the WMMA kernel; fp32 uses the FMA/DPP kernel. Other types are
    // unsupported.
    if(x_type != shape::half_type and x_type != shape::float_type)
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
    // Channels-last (NHWC) when the conv input's channel axis is innermost --
    // the same test the kernel uses to pick its NHWC path. layout_convolution
    // runs before this pass, so the strides already reflect the chosen layout.
    const bool nhwc = ins->inputs().front()->get_shape().strides()[1] == 1;
    // MIGRAPHX_ENABLE_WINOGRAD forces winograd on every eligible shape (bypassing
    // the heuristic); otherwise the per-shape, per-dtype perf heuristic decides.
    // fp16 uses the WMMA kernel's heuristic, fp32 the FMA/DPP kernel's.
    if(enabled(MIGRAPHX_ENABLE_WINOGRAD{}))
        return true;
    if(x_type == shape::float_type)
        return winograd_f23_fp32_profitable(w_lens[1], w_lens[0], x_lens[2], x_lens[3], nhwc);
    return winograd_f23_profitable(w_lens[1], w_lens[0], x_lens[2], x_lens[3], nhwc);
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

        // Match the output layout layout_convolution chose for this conv, so
        // the op is a drop-in replacement (no surrounding transpose changes).
        auto out_layout = find_permutation(ins->get_shape());

        // fp32 uses the FMA/DPP kernel with the host-transformed weight (full U, or
        // the S-store half selected below); fp16 uses the WMMA kernel with the T or
        // g weight store. full_transform is unused for fp32 (precision is derived
        // from the input type); false is just the required ctor argument.
        if(input->get_shape().type() == shape::float_type)
        {
            // Pick the weight encoding: S-store (v-half g*G^T [3,4,K,C], 25% less
            // weight DRAM+loads) on the weight-load-significant shapes, else the
            // full U [4,4,K,C]. The JIT routes to the S path by the weight's first
            // dim (3 vs 4). MIGRAPHX_WINOGRAD_FP32_SSTORE forces S on.
            // NHWC uses the full U laid out v-innermost so the weight load coalesces
            // (S-store stays NCHW-only).
            const auto& x_lens = input->get_shape().lens();   // [N, C, H, W]
            const auto& w_lens = weights->get_shape().lens(); // [K, C, 3, 3]
            const bool nhwc    = input->get_shape().strides()[1] == 1;
            const bool use_s =
                not nhwc and (enabled(MIGRAPHX_WINOGRAD_FP32_SSTORE{}) or
                              winograd_f23_use_sstore(w_lens[1], w_lens[0], x_lens[2], x_lens[3]));
            // v-innermost weight coalesces the NHWC weight load, relieving the
            // input/weight cache thrash -- but its strided (b32) channel load adds
            // issue overhead that regresses shapes where the weight is small and
            // cached (low out_c) or the input dominates (channel-reducing). Gate to
            // where the weight is substantial and not channel-reducing.
            const auto out_c  = w_lens[0];
            const auto in_c   = w_lens[1];
            const bool vinner = nhwc and out_c >= 128 and in_c <= out_c;
            auto u_lit        = use_s ? compute_winograd_weights_f23_fp32_sstore(w_arg)
                                      : compute_winograd_weights_f23_fp32(w_arg, vinner);
            m.replace_instruction(
                ins, winograd_conv{false, out_layout}, input, m.add_literal(u_lit));
            return;
        }

        auto w_lens   = weights->get_shape().lens(); // [K, C, 3, 3]
        auto x_lens   = input->get_shape().lens();   // [N, C, H, W]
        const bool ft = winograd_f23_full_transform(w_lens[1], w_lens[0], x_lens[2], x_lens[3]);
        auto u_lit    = compute_winograd_weights_f23(w_arg, ft);
        auto u_ins    = m.add_literal(u_lit);

        m.replace_instruction(ins, winograd_conv{ft, out_layout}, input, u_ins);
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
    // forces it on for every eligible shape (heuristic bypass). Running here
    // (after layout_convolution) means winograd inherits the layout that pass
    // chose and replaces the convolution in place via its layout-matching
    // compute_shape.
    const bool supports_winograd = device_name == "gfx1151" or starts_with(device_name, "gfx12");
    if(enabled(MIGRAPHX_ENABLE_LAYERNORM_FUSION{}))
    {
        match::find_matches(mpm.get_module(), find_layernorm{});
        mpm.run_pass(dead_code_elimination{});
        match::find_matches(mpm.get_module(), find_add_layernorm{});
    }
    match::find_matches(mpm, find_gemm_softmax_gemm{enable_attention});
    if(is_navi)
        match::find_matches(mpm.get_module(), find_channelwise_convolution{});
    if(supports_winograd)
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
