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
#ifndef MIGRAPHX_GUARD_KERNELS_KV_FLASH_DECODE_HPP
#define MIGRAPHX_GUARD_KERNELS_KV_FLASH_DECODE_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/dpp.hpp>
#include <migraphx/kernels/reduce.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/type_traits.hpp>

namespace migraphx {

// Split-KV flash attention for GQA decode (M=1). One workgroup per
// (batch, kv-head, sequence chunk); the R q-heads sharing a kv-head are
// processed together so each K/V row is streamed exactly once. Each of the
// four waves runs an online softmax over its slice of the chunk; the partial
// (m, l, acc) states are merged through LDS and written per chunk as the
// normalized partial output plus the log-sum-exp:
//   partials[b, qh, g, 0:D] = O'  and  partials[b, qh, g, D] = LSE
// Rows past seqlens[b] are masked by clipping the chunk range.
template <index_int QHeads,
          index_int KVHeads,
          index_int Groups,
          class QK,
          class K,
          class V,
          class SeqLens,
          class Partials>
__device__ void
kv_flash_decode_splitk(QK qk, K kcache, V vcache, SeqLens seqlens, Partials partials, float scale)
{
    auto idx = make_index();

    constexpr auto k_lens         = get_shape_c<K>{}.lens;
    constexpr index_int n_total   = k_lens[2];
    constexpr index_int d         = k_lens[3];
    constexpr index_int r         = QHeads / KVHeads;
    constexpr index_int wave_size = MIGRAPHX_WAVEFRONTSIZE;
    // Half a wave covers one row so each lane issues 8-byte loads, which the
    // memory fabric needs to reach full bandwidth; the two halves stream
    // interleaved rows as independent online-softmax accumulators.
    constexpr index_int row_lanes = wave_size / 2;
    constexpr index_int dpl       = d / row_lanes; // head dims per lane
    constexpr index_int nwaves    = 4;
    constexpr index_int chunk     = (n_total + Groups - 1) / Groups;
    constexpr auto qk_strides     = get_shape_c<QK>{}.strides;
    const float lowest            = numeric_lowest<float>();

    const index_int b    = idx.group / (KVHeads * Groups);
    const index_int kh   = (idx.group / Groups) % KVHeads;
    const index_int g    = idx.group % Groups;
    const index_int wave = idx.local / wave_size;
    const index_int half = (idx.local % wave_size) / row_lanes;
    const index_int lane = idx.local % row_lanes;

    const auto seqlen     = seqlens.data()[b];
    const index_int valid = seqlen < 0 ? 0 : min(n_total, static_cast<index_int>(seqlen) + 1);
    const index_int r0    = min(g * chunk, valid);
    const index_int r1    = min(r0 + chunk, valid);

    // q fragments for the r heads sharing this kv-head, distributed over lanes
    float qf[r][dpl];
    for(index_int j = 0; j < r; ++j)
        for(index_int i = 0; i < dpl; ++i)
            qf[j][i] = migraphx::convert<float>(
                qk.data()[b * qk_strides[0] + (kh * r + j) * qk_strides[1] +
                          (lane * dpl + i) * qk_strides[3]]);

    float m[r];
    float l[r];
    float acc[r][dpl];
    for(index_int j = 0; j < r; ++j)
    {
        m[j] = lowest;
        l[j] = 0.0f;
        for(index_int i = 0; i < dpl; ++i)
            acc[j][i] = 0.0f;
    }

    // each wave runs two interleaved online softmaxes over a contiguous
    // slice of the chunk, one per half-wave
    const index_int per_wave = (r1 - r0 + nwaves - 1) / nwaves;
    const index_int w0       = r0 + wave * per_wave;
    const index_int w1       = min(w0 + per_wave, r1);

    const auto* kbase = kcache.data() + (b * KVHeads + kh) * n_total * d;
    const auto* vbase = vcache.data() + (b * KVHeads + kh) * n_total * d;

    // Row pairs are processed in blocks of four with fully static loops
    // (runtime bounds would push the register arrays to scratch) so the loads
    // and the cross-lane reduction chains of independent rows pipeline, and
    // the online softmax rescale happens once per block instead of per row.
    constexpr index_int pairs_per_iter = 4;
    for(index_int n0 = w0; n0 < w1; n0 += 2 * pairs_per_iter)
    {
        float kf[pairs_per_iter][dpl];
        float vf[pairs_per_iter][dpl];
        bool ok[pairs_per_iter];
        for(index_int rr = 0; rr < pairs_per_iter; ++rr)
        {
            ok[rr]             = (n0 + 2 * rr + half) < w1;
            const index_int nn = ok[rr] ? (n0 + 2 * rr + half) : n0;
            for(index_int i = 0; i < dpl; ++i)
            {
                kf[rr][i] = migraphx::convert<float>(kbase[nn * d + lane * dpl + i]);
                vf[rr][i] = migraphx::convert<float>(vbase[nn * d + lane * dpl + i]);
            }
        }
        float part[pairs_per_iter][r];
        for(index_int rr = 0; rr < pairs_per_iter; ++rr)
        {
            for(index_int j = 0; j < r; ++j)
            {
                part[rr][j] = 0.0f;
                for(index_int i = 0; i < dpl; ++i)
                    part[rr][j] += qf[j][i] * kf[rr][i];
            }
        }
        for(index_int rr = 0; rr < pairs_per_iter; ++rr)
            for(index_int j = 0; j < r; ++j)
                dpp_reduce<row_lanes>(part[rr][j], op::sum{});
        for(index_int j = 0; j < r; ++j)
        {
            float s[pairs_per_iter];
            float m_new = m[j];
            for(index_int rr = 0; rr < pairs_per_iter; ++rr)
            {
                s[rr] = ok[rr] ? readlane<row_lanes - 1, row_lanes>(part[rr][j]) * scale : lowest;
                m_new = max(m_new, s[rr]);
            }
            const float alpha = __expf(m[j] - m_new);
            float p[pairs_per_iter];
            float psum = 0.0f;
            for(index_int rr = 0; rr < pairs_per_iter; ++rr)
            {
                p[rr] = __expf(s[rr] - m_new);
                psum += p[rr];
            }
            l[j] = l[j] * alpha + psum;
            for(index_int i = 0; i < dpl; ++i)
            {
                float av = acc[j][i] * alpha;
                for(index_int rr = 0; rr < pairs_per_iter; ++rr)
                    av += p[rr] * vf[rr][i];
                acc[j][i] = av;
            }
            m[j] = m_new;
        }
    }

    // merge the two half-wave states in-register: both halves hold the same
    // head dims in the same lane positions, so a cross-half shuffle pairs
    // them up and each half computes the identical merged state
    for(index_int j = 0; j < r; ++j)
    {
        const float mo = __shfl_xor(m[j], row_lanes, wave_size);
        const float lo = __shfl_xor(l[j], row_lanes, wave_size);
        const float mc = max(m[j], mo);
        const float aa = __expf(m[j] - mc);
        const float ao = __expf(mo - mc);
        l[j]           = aa * l[j] + ao * lo;
        for(index_int i = 0; i < dpl; ++i)
        {
            const float other = __shfl_xor(acc[j][i], row_lanes, wave_size);
            acc[j][i]         = aa * acc[j][i] + ao * other;
        }
        m[j] = mc;
    }

    // merge the per-wave partial softmax states
    __shared__ float lds_m[nwaves][r];
    __shared__ float lds_l[nwaves][r];
    __shared__ float lds_acc[nwaves][r][d];
    if(half == 0 and lane == 0)
    {
        for(index_int j = 0; j < r; ++j)
        {
            lds_m[wave][j] = m[j];
            lds_l[wave][j] = l[j];
        }
    }
    if(half == 0)
    {
        for(index_int j = 0; j < r; ++j)
            for(index_int i = 0; i < dpl; ++i)
                lds_acc[wave][j][lane * dpl + i] = acc[j][i];
    }
    __syncthreads();

    for(index_int o = idx.local; o < r * d; o += idx.nlocal())
    {
        const index_int j  = o / d;
        const index_int dd = o % d;
        float mw           = lowest;
        for(index_int w = 0; w < nwaves; ++w)
            mw = max(mw, lds_m[w][j]);
        float lw = 0.0f;
        float av = 0.0f;
        for(index_int w = 0; w < nwaves; ++w)
        {
            const float alpha = exp(lds_m[w][j] - mw);
            lw += alpha * lds_l[w][j];
            av += alpha * lds_acc[w][j][dd];
        }
        auto* orow = partials.data() + ((b * QHeads + kh * r + j) * Groups + g) * (d + 1);
        orow[dd]   = (lw > 0.0f) ? av / lw : 0.0f;
        if(dd == 0)
            orow[d] = (lw > 0.0f) ? mw + log(lw) : lowest;
    }
}

// Combine the per-chunk partial outputs with exp-normalized weights:
//   O[b, qh, :] = sum_g exp(LSE_g - max LSE) * O'_g / sum_g exp(LSE_g - max LSE)
// One workgroup per (batch, q-head) with one thread per head dim.
template <class Partials, class Output>
__device__ void kv_flash_decode_reduce(Partials partials, Output output)
{
    auto idx   = make_index();
    using type = typename Output::type;

    constexpr auto p_lens        = get_shape_c<Partials>{}.lens;
    constexpr index_int qh_total = p_lens[1];
    constexpr index_int groups   = p_lens[2];
    constexpr index_int d        = p_lens[3] - 1;

    const index_int b  = idx.group / qh_total;
    const index_int qh = idx.group % qh_total;
    const auto* base   = partials.data() + (b * qh_total + qh) * groups * (d + 1);

    float mx = numeric_lowest<float>();
    for(index_int g = 0; g < groups; ++g)
        mx = max(mx, base[g * (d + 1) + d]);
    float num = 0.0f;
    float den = 0.0f;
    for(index_int g = 0; g < groups; ++g)
    {
        const float w = exp(base[g * (d + 1) + d] - mx);
        num += w * base[g * (d + 1) + idx.local];
        den += w;
    }
    output.data()[(b * qh_total + qh) * d + idx.local] = migraphx::convert<type>(num / den);
}

} // namespace migraphx
#endif
