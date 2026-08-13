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
#ifndef MIGRAPHX_GUARD_KERNELS_SKINNY_GEMM_HPP
#define MIGRAPHX_GUARD_KERNELS_SKINNY_GEMM_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/reduce.hpp>
#include <migraphx/kernels/ops.hpp>

namespace migraphx {

// Split-K GEMM for skinny (M <= 8) bf16 problems where reading the weight
// matrix dominates. The M rows are zero-padded to a 16-row LDS tile and the
// products run on v_mfma_f32_16x16x16bf16_1k so each wave spends its issue
// slots on the weight stream instead of scalar FMAs: per 16-k step a wave
// stages a coalesced 16x64 weight tile in LDS and consumes it as four MFMA
// column tiles. The per-chunk float partials land in a {Splits, M, N} buffer
// that skinny_gemm_reduce collapses.
// When Swiglu is set the skinny input is the packed {m, 2k} gate/up
// projection and the staged value is up * silu(gate), so the activation runs
// once per element at LDS staging time instead of in a separate kernel.
template <index_int Splits, bool Swiglu, class A, class B, class Partials>
__device__ void skinny_gemm_splitk(A a, B b, Partials partials)
{
    auto idx = make_index();

    constexpr auto b_lens      = get_shape_c<B>{}.lens;
    constexpr index_int k      = b_lens[0];
    constexpr index_int n      = b_lens[1];
    constexpr auto p_lens      = get_shape_c<Partials>{}.lens;
    constexpr index_int m      = p_lens[1];
    constexpr index_int kchunk = (k + Splits - 1) / Splits;
    // k-chunk rounded to whole MFMA steps; positions past k stage zeros
    constexpr index_int kc16    = ((kchunk + 15) / 16) * 16;
    constexpr index_int ksteps  = kc16 / 16;
    constexpr index_int a_pitch = kc16 + 4;
    constexpr index_int b_pitch = 64 + 8;
    constexpr index_int nwaves  = 4;

    using bf16x4  = __bf16 __attribute__((ext_vector_type(4)));
    using floatx4 = float __attribute__((ext_vector_type(4)));

    const index_int ntiles = idx.nglobal() / (Splits * idx.nlocal());
    const index_int tile   = idx.group % ntiles;
    const index_int split  = idx.group / ntiles;
    const index_int wave   = idx.local / 64;
    const index_int lane   = idx.local % 64;

    const index_int k0   = split * kchunk;
    const index_int col0 = tile * 256 + wave * 64;

    // Stage the skinny input as a zero-padded 16-row bf16 MFMA tile
    __shared__ __bf16 lds_a[16][a_pitch];
    for(index_int t = idx.local; t < 16 * kc16; t += idx.nlocal())
    {
        const index_int mi = t / kc16;
        const index_int kk = t % kc16;
        float value        = 0.0f;
        if(mi < m and k0 + kk < k)
        {
            if constexpr(Swiglu)
            {
                const float gate = migraphx::convert<float>(a.data()[mi * 2 * k + k0 + kk]);
                const float up   = migraphx::convert<float>(a.data()[mi * 2 * k + k + k0 + kk]);
                value            = gate / (1.0f + __expf(-gate)) * up;
            }
            else
            {
                value = migraphx::convert<float>(a.data()[mi * k + k0 + kk]);
            }
        }
        lds_a[mi][kk] = static_cast<__bf16>(value);
    }
    __syncthreads();

    // Per-wave double-buffered weight staging tiles: 16 k-rows x 64 columns,
    // written with two coalesced 16-byte stores per lane and read back in
    // fragment order. The next k-step's global loads are issued before the
    // fence so the fragment reads and MFMAs of the current step cover their
    // latency.
    __shared__ __bf16 lds_b[nwaves][2][16][b_pitch];
    const index_int brow  = lane / 4;
    const index_int bcol  = (lane % 4) * 16;
    const index_int fcol  = lane % 16; // fragment column within a 16-wide tile
    const index_int fkrow = 4 * (lane / 16);
    const index_int bc    = min(col0 + bcol, n - 16);

    floatx4 dacc[4];
    for(index_int c = 0; c < 4; ++c)
        dacc[c] = {0.0f, 0.0f, 0.0f, 0.0f};

    struct tile_regs
    {
        vec<unsigned int, 4> lo;
        vec<unsigned int, 4> hi;
    };
    const auto* bn = b.data();
    // the weight stream is touched exactly once per token, so nontemporal
    // loads keep it from evicting data that is actually reused
    auto load_tile = [&](index_int ks) {
        const index_int krow = min(k0 + min(ks, ksteps - 1) * 16 + brow, k - 1);
        const auto* src = reinterpret_cast<const vec<unsigned int, 4>*>(bn + krow * n + bc);
        tile_regs out;
        out.lo = __builtin_nontemporal_load(src);
        out.hi = __builtin_nontemporal_load(src + 1);
        return out;
    };

    auto words = load_tile(0);
    for(index_int ks = 0; ks < ksteps; ++ks)
    {
        const index_int buf = ks % 2;
        auto* dst           = &lds_b[wave][buf][brow][bcol];
        __builtin_memcpy(dst, &words.lo, 16);
        __builtin_memcpy(dst + 8, &words.hi, 16);
        words = load_tile(ks + 1);
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");

        // A fragment is shared by the four column tiles of this k-step
        bf16x4 af;
        __builtin_memcpy(&af, &lds_a[fcol][ks * 16 + fkrow], 8);
        for(index_int c = 0; c < 4; ++c)
        {
            bf16x4 bf = {};
            for(index_int i = 0; i < 4; ++i)
                bf[i] = lds_b[wave][buf][fkrow + i][fcol + 16 * c];
            dacc[c] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(af, bf, dacc[c], 0, 0, 0);
        }
    }

    // D fragment: lane holds rows 4*(lane/16)+i of column fcol per tile
    auto* out = partials.data() + split * m * n;
    for(index_int c = 0; c < 4; ++c)
    {
        for(index_int i = 0; i < 4; ++i)
        {
            const index_int mi  = fkrow + i;
            const index_int col = col0 + 16 * c + fcol;
            if(mi < m and col < n)
                out[mi * n + col] = dacc[c][i];
        }
    }
}

// Collapse the {Splits, M, N} float partials, add the residual, and also emit
// the rmsnorm of the result: the raw row feeds the next residual chain while
// the normalized row feeds the next projection, which removes both the
// separate reduce and the rmsnorm kernels. One workgroup per output row so
// the row-wide sum of squares reduces within the block.
template <index_int Block, class Partials, class Residual, class HOut, class NormOut>
__device__ void skinny_gemm_reduce_rmsnorm(Partials partials,
                                           Residual residual,
                                           HOut h_out,
                                           NormOut norm_out,
                                           float eps,
                                           float ss_scale)
{
    auto idx   = make_index();
    using type = typename HOut::type;

    constexpr auto p_lens       = get_shape_c<Partials>{}.lens;
    constexpr index_int nsplits = p_lens[0];
    constexpr index_int m       = p_lens[1];
    constexpr index_int n       = p_lens[2];
    constexpr index_int cpt     = (n + Block - 1) / Block; // columns per thread

    const index_int row = idx.group;

    // sum the split partials once per element (they were just written, so
    // regular cached loads hit the caches), add the residual, and keep the
    // rounded value for the norm so it matches the unfused arithmetic; the
    // split loop is outermost so the column accumulator chains stay
    // independent and their loads pipeline
    float acc[cpt];
    for(index_int c = 0; c < cpt; ++c)
    {
        const index_int col = idx.local + c * Block;
        acc[c] = (col < n) ? migraphx::convert<float>(residual.data()[row * n + col]) : 0.0f;
    }
#pragma unroll 2
    for(index_int s = 0; s < nsplits; ++s)
    {
        for(index_int c = 0; c < cpt; ++c)
        {
            const index_int col = idx.local + c * Block;
            acc[c] += (col < n) ? partials.data()[(s * m + row) * n + col] : 0.0f;
        }
    }
    float hv[cpt];
    float ss = 0.0f;
    for(index_int c = 0; c < cpt; ++c)
    {
        const index_int col = idx.local + c * Block;
        hv[c]               = 0.0f;
        if(col < n)
        {
            const type hb               = migraphx::convert<type>(acc[c]);
            h_out.data()[row * n + col] = hb;
            hv[c]                       = migraphx::convert<float>(hb);
            ss += hv[c] * hv[c] * ss_scale;
        }
    }
    const float total =
        block_reduce(idx, op::sum{}, 0.0f, idx.nlocal(), [&](auto) { return ss; });
    const type rms = migraphx::convert<type>(rsqrt(total + eps));
    for(index_int c = 0; c < cpt; ++c)
    {
        const index_int col = idx.local + c * Block;
        if(col < n)
            norm_out.data()[row * n + col] =
                migraphx::convert<type>(hv[c] * migraphx::convert<float>(rms));
    }
}

// Collapse the {Splits, M, N} float partials into the final output, optionally
// adding a residual. 256 threads per block: 16 output lanes x 16 split
// partitions with four outputs per thread, so every thread keeps enough
// strided partial loads in flight to cover the L2 latency.
template <class Partials, class Output, class... Residual>
__device__ void skinny_gemm_reduce(Partials partials, Output output, Residual... residual)
{
    auto idx   = make_index();
    using type = typename Output::type;

    constexpr auto p_lens       = get_shape_c<Partials>{}.lens;
    constexpr index_int nsplits = p_lens[0];
    constexpr index_int total   = p_lens[1] * p_lens[2];
    constexpr index_int lanes   = 16;
    constexpr index_int parts   = 16;
    constexpr index_int owp     = 4; // outputs per thread

    __shared__ float buffer[parts * owp * lanes];

    const index_int lane = idx.local % lanes;
    const index_int part = idx.local / lanes;
    const index_int i0   = idx.group * lanes * owp + lane;

    float acc[owp];
    for(index_int w = 0; w < owp; ++w)
        acc[w] = 0.0f;
    for(index_int s = part; s < nsplits; s += parts)
    {
        for(index_int w = 0; w < owp; ++w)
        {
            const index_int i = i0 + w * lanes;
            acc[w] += (i < total) ? partials.data()[s * total + i] : 0.0f;
        }
    }
    for(index_int w = 0; w < owp; ++w)
        buffer[(part * owp + w) * lanes + lane] = acc[w];
    __syncthreads();
    for(index_int o = idx.local; o < owp * lanes; o += idx.nlocal())
    {
        const index_int w  = o / lanes;
        const index_int ll = o % lanes;
        const index_int i  = idx.group * lanes * owp + w * lanes + ll;
        if(i >= total)
            continue;
        float sum = 0.0f;
        for(index_int p = 0; p < parts; ++p)
            sum += buffer[(p * owp + w) * lanes + ll];
        ([&] { sum += migraphx::convert<float>(residual.data()[i]); }(), ...);
        output.data()[i] = migraphx::convert<type>(sum);
    }
}

} // namespace migraphx
#endif
