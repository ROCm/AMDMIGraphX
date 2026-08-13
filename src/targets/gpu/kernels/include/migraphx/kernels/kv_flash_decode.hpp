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
#include <migraphx/kernels/vec.hpp>
#include <migraphx/kernels/bit_cast.hpp>

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
    // A quarter of a wave covers one row so each lane issues 16-byte loads,
    // which the memory fabric needs to reach full bandwidth; the four
    // quarters stream interleaved rows as independent online-softmax
    // accumulators that are merged in-register at the end.
    constexpr index_int row_lanes = wave_size / 4;
    constexpr index_int dpl       = d / row_lanes; // head dims per lane
    constexpr index_int nwaves    = 4;
    constexpr auto qk_strides     = get_shape_c<QK>{}.strides;
    const float lowest            = numeric_lowest<float>();
    using kv_vec                  = vec<unsigned int, dpl / 2>;

    const index_int b    = idx.group / (KVHeads * Groups);
    const index_int kh   = (idx.group / Groups) % KVHeads;
    const index_int g    = idx.group % Groups;
    const index_int wave = idx.local / wave_size;
    const index_int part = (idx.local % wave_size) / row_lanes;
    const index_int lane = idx.local % row_lanes;

    const auto seqlen     = seqlens.data()[b];
    const index_int valid = seqlen < 0 ? 0 : min(n_total, static_cast<index_int>(seqlen) + 1);
    // split the rows that actually exist for this batch, not the cache
    // capacity, so every chunk stays busy at any sequence length
    const index_int chunk = (valid + Groups - 1) / Groups;
    const index_int r0    = min(g * chunk, valid);
    const index_int r1    = min(r0 + chunk, valid);

    // MFMA fragment geometry for v_mfma_f32_16x16x16bf16_1k:
    //   A: lane holds A[lane%16][4*(lane/16)+i]
    //   B: lane holds B[4*(lane/16)+i][lane%16]
    //   D: lane holds D[4*(lane/16)+i][lane%16]
    // Scores are computed transposed, S^T[row][head] = K-tile x Q^T, so the
    // result fragment leaves each lane holding four rows of one head column,
    // which is exactly the layout the P*V accumulation consumes: quarter
    // `part` owns tile rows 4*part+i and lane `lane` owns head dims
    // [lane*dq, lane*dq+dq).
    using bf16x4                = __bf16 __attribute__((ext_vector_type(4)));
    using floatx4               = float __attribute__((ext_vector_type(4)));
    constexpr index_int kblocks = d / 16; // MFMA steps over the head dim
    constexpr index_int dq      = dpl;    // head dims per lane in the P*V stage

    // Q as the B operand, one fragment per k-block; padding head columns are
    // zero so they never influence the real scores
    bf16x4 bq[kblocks];
    for(index_int kb = 0; kb < kblocks; ++kb)
    {
        for(index_int i = 0; i < 4; ++i)
        {
            bq[kb][i] = (lane < r) ? qk.data()[b * qk_strides[0] + (kh * r + lane) * qk_strides[1] +
                                               (16 * kb + 4 * part + i) * qk_strides[3]]
                                   : static_cast<__bf16>(0.0f);
        }
    }

    // Online softmax state per head, replicated on every lane and updated
    // from broadcast tile statistics so all lanes stay consistent; m_own
    // additionally tracks this lane's head column for the exponentials.
    float m[r];
    float l[r];
    float acc[r][dq];
    for(index_int j = 0; j < r; ++j)
    {
        m[j] = lowest;
        l[j] = 0.0f;
        for(index_int i = 0; i < dq; ++i)
            acc[j][i] = 0.0f;
    }
    float m_own = lowest;

    // each wave runs an online softmax over a contiguous slice of the chunk
    const index_int per_wave = (r1 - r0 + nwaves - 1) / nwaves;
    const index_int w0       = r0 + wave * per_wave;
    const index_int w1       = min(w0 + per_wave, r1);

    const auto* kbase = kcache.data() + (b * KVHeads + kh) * n_total * d;
    const auto* vbase = vcache.data() + (b * KVHeads + kh) * n_total * d;

    // every cache row is read by exactly one workgroup per token, so
    // nontemporal loads keep the KV stream from evicting reused data
    auto load_row = [&](const auto* base, index_int nn) {
        const auto* src = reinterpret_cast<const kv_vec*>(base + nn * d + lane * dpl);
        return __builtin_nontemporal_load(src);
    };
    auto unpack = [&](kv_vec v, float (&out)[dpl]) {
        for(index_int u = 0; u < dpl / 2; ++u)
        {
            out[2 * u]     = bit_cast<float>(v[u] << 16u);
            out[2 * u + 1] = bit_cast<float>(v[u] & 0xffff0000u);
        }
    };

    // K tiles are staged in LDS with the same coalesced row loads as V and
    // read back in the MFMA fragment layout; the row pitch is padded to keep
    // the 16 fragment rows on distinct banks. Every wave runs the same tile
    // count so the workgroup barrier between staging and reading is uniform.
    constexpr index_int krow_pitch = d + 4;
    __shared__ __bf16 lds_k[nwaves][16][krow_pitch];
    const index_int ntiles = (per_wave + 15) / 16;

    for(index_int tt = 0; tt < ntiles; ++tt)
    {
        const index_int t0 = w0 + tt * 16;

        // issue this tile's V row loads first so their latency hides behind
        // the score computation; they are not consumed until after softmax
        kv_vec vraw[4];
        for(index_int i = 0; i < 4; ++i)
        {
            const index_int nn = min(t0 + 4 * part + i, r1 - 1);
            vraw[i]            = load_row(vbase, nn);
        }
        // stage the K tile: same coalesced pattern, two 8-byte LDS writes
        // per row slice (the padded pitch is only 8-byte aligned)
        for(index_int i = 0; i < 4; ++i)
        {
            const index_int nn = min(t0 + 4 * part + i, r1 - 1);
            const kv_vec kraw  = load_row(kbase, nn);
            unsigned int words[dpl / 2];
            __builtin_memcpy(&words, &kraw, sizeof(words));
            auto* dst = &lds_k[wave][4 * part + i][lane * dpl];
            __builtin_memcpy(dst, &words[0], sizeof(words) / 2);
            __builtin_memcpy(dst + dpl / 2, &words[dpl / 4], sizeof(words) / 2);
        }
        // the staging buffer is per-wave, so ordering the LDS accesses is
        // enough and no cross-wave rendezvous is required
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");

        // scores for the 16-row K tile against all head columns in one MFMA
        // chain; out-of-range rows staged a safe address and are masked below
        floatx4 sacc = {0.0f, 0.0f, 0.0f, 0.0f};
        for(index_int kb = 0; kb < kblocks; ++kb)
        {
            bf16x4 ak;
            __builtin_memcpy(&ak, &lds_k[wave][lane][16 * kb + 4 * part], 8);
            sacc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(ak, bq[kb], sacc, 0, 0, 0);
        }

        // this lane's head column: scores for tile rows 4*part+i; masked
        // rows contribute nothing even when the whole tile is masked
        float s[4];
        bool okr[4];
        float tmax = lowest;
        for(index_int i = 0; i < 4; ++i)
        {
            okr[i] = (t0 + 4 * part + i) < w1;
            s[i]   = okr[i] ? sacc[i] * scale : lowest;
            tmax   = max(tmax, s[i]);
        }
        tmax               = max(tmax, __shfl_xor(tmax, 16, wave_size));
        tmax               = max(tmax, __shfl_xor(tmax, 32, wave_size));
        const float mn_own = max(m_own, tmax);
        float p[4];
        float psum = 0.0f;
        for(index_int i = 0; i < 4; ++i)
        {
            p[i] = okr[i] ? __expf(s[i] - mn_own) : 0.0f;
            psum += p[i];
        }
        psum += __shfl_xor(psum, 16, wave_size);
        psum += __shfl_xor(psum, 32, wave_size);
        m_own = mn_own;
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");

        // broadcast each real head's tile statistics and probabilities from
        // its head-column lanes, rescale the accumulators once per tile
        float pj[r][4];
        for(index_int j = 0; j < r; ++j)
        {
            const int src     = static_cast<int>(part * 16 + j);
            const float mn    = __shfl(mn_own, src, wave_size);
            const float lsum  = __shfl(psum, src, wave_size);
            const float alpha = __expf(m[j] - mn);
            l[j]              = l[j] * alpha + lsum;
            m[j]              = mn;
            for(index_int i = 0; i < 4; ++i)
                pj[j][i] = __shfl(p[i], src, wave_size);
            for(index_int ii = 0; ii < dq; ++ii)
                acc[j][ii] *= alpha;
        }

        // P*V with the prefetched coalesced rows: quarter rows, lane dims
        for(index_int i = 0; i < 4; ++i)
        {
            float vf[dq];
            unpack(vraw[i], vf);
            for(index_int j = 0; j < r; ++j)
                for(index_int ii = 0; ii < dq; ++ii)
                    acc[j][ii] += pj[j][i] * vf[ii];
        }
    }

    // the quarters share the same normalization, so merging their partial
    // accumulators is a plain sum across the two lane-group strides
    for(index_int j = 0; j < r; ++j)
    {
        for(index_int i = 0; i < dq; ++i)
        {
            acc[j][i] += __shfl_xor(acc[j][i], 16, wave_size);
            acc[j][i] += __shfl_xor(acc[j][i], 32, wave_size);
        }
    }

    // merge the per-wave partial softmax states
    __shared__ float lds_m[nwaves][r];
    __shared__ float lds_l[nwaves][r];
    __shared__ float lds_acc[nwaves][r][d];
    if(part == 0 and lane == 0)
    {
        for(index_int j = 0; j < r; ++j)
        {
            lds_m[wave][j] = m[j];
            lds_l[wave][j] = l[j];
        }
    }
    if(part == 0)
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
