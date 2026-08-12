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

namespace migraphx {

// Reduction-based split-K GEMM for skinny (M <= 8) problems where reading the
// weight matrix dominates. Each thread accumulates Cols adjacent output
// columns over one K chunk so wavefront loads of the row-major {K, N} weight
// are fully coalesced; the per-chunk float partials land in a {Splits, M, N}
// buffer that skinny_gemm_reduce collapses.
// When Swiglu is set the skinny input is the packed {m, 2k} gate/up
// projection and the staged value is up * silu(gate), so the activation runs
// once per element at LDS staging time instead of in a separate kernel.
template <index_int Cols, index_int Splits, bool Swiglu, class A, class B, class Partials>
__device__ void skinny_gemm_splitk(A a, B b, Partials partials)
{
    auto idx = make_index();

    constexpr auto b_lens      = get_shape_c<B>{}.lens;
    constexpr index_int k      = b_lens[0];
    constexpr index_int n      = b_lens[1];
    constexpr auto p_lens      = get_shape_c<Partials>{}.lens;
    constexpr index_int m      = p_lens[1];
    constexpr index_int kchunk = (k + Splits - 1) / Splits;

    const index_int ntiles = idx.nglobal() / (Splits * idx.nlocal());
    const index_int tile   = idx.group % ntiles;
    const index_int split  = idx.group / ntiles;

    const index_int n0 = (tile * idx.nlocal() + idx.local) * Cols;
    const index_int k0 = split * kchunk;

    // Stage this split's slice of the skinny input in LDS so the inner loop
    // issues only the weight stream to global memory
    __shared__ float a_lds[m][kchunk];
    for(index_int t = idx.local; t < m * kchunk; t += idx.nlocal())
    {
        const index_int mi = t / kchunk;
        const index_int kk = t % kchunk;
        float value        = 0.0f;
        if(k0 + kk < k)
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
        a_lds[mi][kk] = value;
    }
    __syncthreads();
    if(n0 >= n)
        return;

    float acc[m][Cols];
    for(index_int mi = 0; mi < m; ++mi)
        for(index_int c = 0; c < Cols; ++c)
            acc[mi][c] = 0.0f;

    // Static trip count so the compiler unrolls and keeps many weight loads
    // in flight; rows past k contribute zero since their staged input is zero
    // and the row index is clamped to stay in bounds.
    const auto* bn = b.data();
    for(index_int kk2 = 0; kk2 < kchunk; ++kk2)
    {
        const index_int kk = migraphx::min(k0 + kk2, k - 1);
        float av[m];
        for(index_int mi = 0; mi < m; ++mi)
            av[mi] = a_lds[mi][kk2];
        const auto* brow = bn + kk * n + n0;
        for(index_int c = 0; c < Cols; ++c)
        {
            float bv = migraphx::convert<float>(brow[c]);
            for(index_int mi = 0; mi < m; ++mi)
                acc[mi][c] += av[mi] * bv;
        }
    }
    auto* out = partials.data() + (split * m) * n + n0;
    for(index_int mi = 0; mi < m; ++mi)
        for(index_int c = 0; c < Cols; ++c)
            out[mi * n + c] = acc[mi][c];
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
