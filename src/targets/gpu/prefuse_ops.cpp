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
    // When > 0, the conv's first resize_c input channels are a fused 2x
    // bilinear upsample (ONNX Resize linear/asymmetric) of the first input
    // tensor, computed in the kernel's input load; the logical conv input is
    // double the first input's H/W. resize_tail: a second input tensor
    // supplies the remaining (concat) channels at full resolution.
    std::size_t resize_c = 0;
    bool resize_tail     = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.full_transform, "full_transform"),
                    f(self.output_layout, "output_layout"),
                    f(self.resize_c, "resize_c"),
                    f(self.resize_tail, "resize_tail"));
    }

    std::string name() const { return "gpu::winograd_conv"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(resize_tail ? 3 : 2);
        const auto& x_shape = inputs[0];
        const auto& u_shape = inputs.back();
        auto x_lens         = x_shape.lens();
        // u_shape is [4 or 3, 3, K, C]; lens()[2] is the output channel count.
        auto out_c                        = u_shape.lens()[2];
        const auto scale                  = resize_c > 0 ? 2 : 1;
        std::vector<std::size_t> out_lens = {
            x_lens[0], out_c, scale * x_lens[2], scale * x_lens[3]};
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
bool winograd_f23_profitable(std::size_t in_ch,
                             std::size_t out_ch,
                             std::size_t height,
                             std::size_t width,
                             bool nhwc,
                             bool fused_resize)
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
    //
    // fused_resize shifts the economics: the conv absorbs the 2x bilinear
    // upsample of (most of) its input, which deletes the standalone resize
    // kernel and reads the low-resolution source (1/4 the bytes, 3x3 loads
    // per tile instead of 4x4). Measured on the u-net decoder shapes
    // (gfx1201 fp16 NHWC: 1024->512@12, 848->512@24, 704->288@48), that
    // turns the large-channel MLIR wins into winograd wins at small spatial.
    if(nhwc and min_ch >= 224)
    {
        if(fused_resize and spatial <= 48)
            return true;
        // Plain-conv exception: equal-channel convs at tiny spatial are
        // latency-bound in MLIR (a 6x6x512 conv gets ~3k threads) while
        // winograd's k-block grid keeps more CUs busy. Measured on gfx1201
        // fp16 NHWC (interleaved medians): 512x512@12 1.30x, 512x512@6 1.19x;
        // 512x512@24 and unequal-channel shapes still lose, and very large
        // channel products stay excluded (untested regime).
        return in_ch == out_ch and spatial <= 12 and in_ch * out_ch < 700000;
    }

    // NOLINTNEXTLINE(readability-qualified-auto)
    const auto ovr = std::find_if(
        winograd_f23_overrides.begin(), winograd_f23_overrides.end(), [&](const auto& o) {
            return std::tie(o.in_ch, o.out_ch, o.height, o.width) ==
                   std::tie(in_ch, out_ch, height, width);
        });
    if(ovr != winograd_f23_overrides.end())
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
    // MIGRAPHX_DISABLE_WINOGRAD forces winograd off everywhere (for
    // benchmarking/debugging). The per-shape perf heuristic is applied
    // separately by each finder (see winograd_profitable_matcher) since the
    // fused-resize variant changes the economics.
    return not enabled(MIGRAPHX_DISABLE_WINOGRAD{});
}

// Perf heuristic as a matcher: skip shapes where the default lowering is
// faster. MIGRAPHX_ENABLE_WINOGRAD forces winograd on every eligible shape
// (bypassing the heuristic). fused_resize: the conv would absorb a 2x
// upsample of its input, which shifts the winograd-vs-default tradeoff.
auto winograd_profitable_matcher(bool fused_resize)
{
    return match::make_basic_pred_matcher([=](instruction_ref ins) {
        if(enabled(MIGRAPHX_ENABLE_WINOGRAD{}))
            return true;
        auto w_lens = ins->inputs().back()->get_shape().lens();
        auto x_lens = ins->inputs().front()->get_shape().lens();
        // Channels-last (NHWC) when the conv input's channel axis is
        // innermost -- the same test the kernel uses to pick its NHWC path.
        // layout_convolution runs before this pass, so the strides already
        // reflect the chosen layout.
        const bool nhwc = ins->inputs().front()->get_shape().strides()[1] == 1;
        return winograd_f23_profitable(
            w_lens[1], w_lens[0], x_lens[2], x_lens[3], nhwc, fused_resize);
    });
}

// A resize that exactly doubles H and W via asymmetric bilinear interpolation
// (ONNX Resize mode=linear, coordinate_transformation_mode=asymmetric with an
// integer scale of 2). This is the form the winograd kernel fuses into its
// input load: even output rows/cols copy the source pixel, odd ones average
// two neighbours (the last odd one collapses onto the edge pixel).
MIGRAPHX_PRED_MATCHER(resize_linear_asym_2x, instruction_ref ins)
{
    if(ins->name() != "resize")
        return false;
    if(ins->inputs().size() != 1)
        return false;
    auto v = ins->get_operator().to_value();
    if(v.at("mode").to<std::string>() != "linear")
        return false;
    if(v.at("coordinate_transformation_mode").to<std::string>() != "asymmetric")
        return false;
    const auto& in_lens  = ins->inputs().front()->get_shape().lens();
    const auto& out_lens = ins->get_shape().lens();
    if(in_lens.size() != 4 or out_lens.size() != 4)
        return false;
    return out_lens[0] == in_lens[0] and out_lens[1] == in_lens[1] and
           out_lens[2] == 2 * in_lens[2] and out_lens[3] == 2 * in_lens[3];
}

// 4-d channels-last: the fused-resize load reads 8 contiguous channels per
// b128, so every tensor it touches must have the channel axis innermost.
MIGRAPHX_PRED_MATCHER(channels_last_4d, instruction_ref ins)
{
    const auto& s = ins->get_shape();
    return s.lens().size() == 4 and s.strides()[1] == 1;
}

// A 2-input channel concat whose first input's channel count is a multiple
// of 16 -- the kernel selects the source per 16-channel c-block, so the
// concat boundary must land on a block boundary.
MIGRAPHX_PRED_MATCHER(concat_2args_channels_16, instruction_ref ins)
{
    if(ins->name() != "concat")
        return false;
    if(ins->inputs().size() != 2)
        return false;
    if(ins->get_operator().to_value().at("axis").to<std::int64_t>() != 1)
        return false;
    return ins->inputs().front()->get_shape().lens()[1] % 16 == 0;
}

// Fused-resize variant: the conv input is a 2x bilinear upsample, optionally
// concatenated (on channels) with a full-resolution skip tensor -- the
// standard u-net decoder level. The kernel reads the upsampled channels
// straight from the low-resolution tensor (3x3 source loads per tile instead
// of 4x4, no intermediate resize kernel or buffer) and the skip channels
// directly. Runs before find_winograd_f23 so eligible convs fuse their
// resize; the plain matcher picks up whatever remains. All disqualifying
// conditions live in the matcher: a matched instruction is always replaced,
// and a non-fusible conv falls through to the plain winograd matcher.
struct find_winograd_f23_resize
{
    auto matcher() const
    {
        auto rz = match::name("resize")(resize_linear_asym_2x(),
                                        channels_last_4d(),
                                        match::arg(0)(channels_last_4d()))
                      .bind("resize");
        auto cat = concat_2args_channels_16()(
            channels_last_4d(), match::arg(0)(rz), match::arg(1)(channels_last_4d().bind("tail")));
        return conv_winograd_f23()(winograd_profitable_matcher(true),
                                   match::arg(0)(match::any_of(rz, cat)));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins            = r.result;
        auto input          = ins->inputs().front();
        auto weights        = ins->inputs().back();
        auto src            = r.instructions["resize"]->inputs().front();
        const bool has_tail = r.instructions.find("tail") != r.instructions.end();
        const auto split_c  = src->get_shape().lens()[1];

        auto w_arg      = weights->eval();
        auto w_lens     = weights->get_shape().lens(); // [K, C, 3, 3]
        auto x_lens     = input->get_shape().lens();   // logical [N, C, H, W]
        const bool ft   = winograd_f23_full_transform(w_lens[1], w_lens[0], x_lens[2], x_lens[3]);
        auto out_layout = find_permutation(ins->get_shape());

        auto u_ins = m.add_literal(compute_winograd_weights_f23(w_arg, ft));
        if(has_tail)
            m.replace_instruction(ins,
                                  winograd_conv{ft, out_layout, split_c, true},
                                  src,
                                  r.instructions["tail"],
                                  u_ins);
        else
            m.replace_instruction(ins, winograd_conv{ft, out_layout, split_c, false}, src, u_ins);
    }
};

struct find_winograd_f23
{
    auto matcher() const { return conv_winograd_f23()(winograd_profitable_matcher(false)); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins     = r.result;
        auto input   = ins->inputs().front();
        auto weights = ins->inputs().back();

        auto w_arg = weights->eval();

        auto w_lens   = weights->get_shape().lens(); // [K, C, 3, 3]
        auto x_lens   = input->get_shape().lens();   // [N, C, H, W]
        const bool ft = winograd_f23_full_transform(w_lens[1], w_lens[0], x_lens[2], x_lens[3]);
        // Match the output layout layout_convolution chose for this conv, so
        // the op is a drop-in replacement (no surrounding transpose changes).
        auto out_layout = find_permutation(ins->get_shape());

        auto u_lit = compute_winograd_weights_f23(w_arg, ft);
        auto u_ins = m.add_literal(u_lit);

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
        match::find_matches(mpm.get_module(), find_winograd_f23_resize{}, find_winograd_f23{});
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
