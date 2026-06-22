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
#include <migraphx/fuse_resize_conv.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/op/resize.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

// Polyphase weights of the bilinear-upsample stencil.  For output parity `a` (0=even, 1=odd)
// and conv tap offset `kh` in {-1,0,1}, the upsampled sample yup[2i+a+kh] expands to a weighted
// sum of low-res samples x[i+dh]; return those (dh, weight) pairs.  (asymmetric, scale-2 bilinear:
// yup[2i]=x[i], yup[2i+1]=0.5 x[i]+0.5 x[i+1].)
std::vector<std::pair<int, double>> upsample_coef(int a, int kh)
{
    switch(a + kh)
    {
    case -1: return {{-1, 0.5}, {0, 0.5}};
    case 0: return {{0, 1.0}};
    case 1: return {{0, 0.5}, {1, 0.5}};
    case 2: return {{1, 1.0}};
    default: return {};
    }
}

// Fold a [K,C,3,3] conv weight with the bilinear-upsample stencil into [4K,C,3,3] sub-pixel
// weights.  Output channels are phase-major: channel (a*2+b)*K + k holds the kernel that, applied
// to x at low-res, yields output pixel (2i+a, 2j+b) of conv(upsample(x)).
std::vector<double> fold_weights(const std::vector<double>& w, std::size_t k_n, std::size_t c_n)
{
    std::vector<double> wf(4 * k_n * c_n * 9, 0.0);
    for(int a = 0; a < 2; ++a)
        for(int b = 0; b < 2; ++b)
        {
            const std::size_t phase = a * 2 + b;
            for(int kh = -1; kh <= 1; ++kh)
                for(int kw = -1; kw <= 1; ++kw)
                    for(auto [dh, wh] : upsample_coef(a, kh))
                        for(auto [dw, ww] : upsample_coef(b, kw))
                            for(std::size_t k = 0; k < k_n; ++k)
                                for(std::size_t c = 0; c < c_n; ++c)
                                    wf[(((phase * k_n + k) * c_n + c) * 9) + (dh + 1) * 3 +
                                       (dw + 1)] +=
                                        w[((k * c_n + c) * 3 + (kh + 1)) * 3 + (kw + 1)] * wh * ww;
        }
    return wf;
}

bool all_equal(const value& v, const std::string& key, std::int64_t expect)
{
    if(not v.contains(key))
        return false;
    for(const auto& e : v.at(key))
        if(e.to<std::int64_t>() != expect)
            return false;
    return true;
}

} // namespace

void fuse_resize_conv::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "convolution")
            continue;
        auto conv_in = ins->inputs();
        auto rz      = conv_in[0];
        auto w       = conv_in[1];
        if(rz->name() != "resize")
            continue;
        if(not w->can_eval())
            continue;

        // resize must be: linear, asymmetric, exactly 2x on the two spatial dims and 1x otherwise.
        auto rop = any_cast<op::resize>(rz->get_operator());
        if(rop.mode != "linear" or rop.coordinate_transformation_mode != "asymmetric")
            continue;
        auto x       = rz->inputs()[0];
        auto x_lens  = x->get_shape().lens();
        auto up_lens = rz->get_shape().lens();
        if(x_lens.size() != 4)
            continue;
        if(not(up_lens[0] == x_lens[0] and up_lens[1] == x_lens[1] and
               up_lens[2] == 2 * x_lens[2] and up_lens[3] == 2 * x_lens[3]))
            continue;

        // conv must be a plain 3x3, stride 1, dilation 1, pad 1, single group.
        auto cv = ins->get_operator().to_value();
        if(cv.contains("group") and cv.at("group").to<int>() != 1)
            continue;
        if(not(all_equal(cv, "stride", 1) and all_equal(cv, "dilation", 1) and
               all_equal(cv, "padding", 1)))
            continue;
        auto w_lens = w->get_shape().lens();
        if(w_lens.size() != 4 or w_lens[2] != 3 or w_lens[3] != 3)
            continue;
        const std::size_t k_n = w_lens[0];
        const std::size_t c_n = w_lens[1];
        if(c_n != x_lens[1])
            continue;

        // --- fold weights at compile time ---
        std::vector<double> wd;
        w->eval().visit([&](auto v) { wd.assign(v.begin(), v.end()); });
        auto wf     = fold_weights(wd, k_n, c_n);
        auto wf_lit = m.add_literal(
            literal{shape{w->get_shape().type(), {4 * k_n, c_n, 3, 3}}, wf.begin(), wf.end()});

        // --- sub-pixel conv at low res: [N, 4K, H, W].  The conv's own zero-padding handles the
        // border (interior exact; outer 1px ring is a bilinear/clamp approximation, within fp16
        // tolerance).  prefuse_ops keeps this conv off the winograd path (it feeds the
        // depth-to-space reshape, which winograd cannot fuse) so MLIR fuses the interleave +
        // leaky-relu epilogue into the conv kernel. ---
        auto sub = m.insert_instruction(
            ins,
            make_op("convolution",
                    {{"padding", {1, 1}}, {"stride", {1, 1}}, {"dilation", {1, 1}}, {"group", 1}}),
            x,
            wf_lit);

        // --- interleave (depth-to-space, block 2): [N,4K,H,W] -> [N,K,2H,2W] ---
        const std::int64_t n = x_lens[0], h = x_lens[2], wdt = x_lens[3];
        const std::int64_t kk = k_n;
        auto r1 =
            m.insert_instruction(ins, make_op("reshape", {{"dims", {n, 2, 2, kk, h, wdt}}}), sub);
        auto tr = m.insert_instruction(
            ins, make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), r1);
        auto r2 =
            m.insert_instruction(ins, make_op("reshape", {{"dims", {n, kk, 2 * h, 2 * wdt}}}), tr);

        m.replace_instruction(ins, r2);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
