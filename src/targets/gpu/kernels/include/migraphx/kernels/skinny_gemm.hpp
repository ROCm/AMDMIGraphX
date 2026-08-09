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
template <index_int Cols, index_int Splits, class A, class B, class Partials>
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
    if(n0 >= n)
        return;
    const index_int k0 = split * kchunk;
    const index_int k1 = migraphx::min(k, k0 + kchunk);

    float acc[m][Cols];
    for(index_int mi = 0; mi < m; ++mi)
        for(index_int c = 0; c < Cols; ++c)
            acc[mi][c] = 0.0f;

    const auto* an = a.data();
    const auto* bn = b.data();
    for(index_int kk = k0; kk < k1; ++kk)
    {
        float av[m];
        for(index_int mi = 0; mi < m; ++mi)
            av[mi] = migraphx::convert<float>(an[mi * k + kk]);
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
// partitions, combined through LDS so the strided partial reads overlap.
template <class Partials, class Output, class... Residual>
__device__ void skinny_gemm_reduce(Partials partials, Output output, Residual... residual)
{
    auto idx   = make_index();
    using type = typename Output::type;

    constexpr auto p_lens       = get_shape_c<Partials>{}.lens;
    constexpr index_int nsplits = p_lens[0];
    constexpr index_int total   = p_lens[1] * p_lens[2];
    // 16 split partitions per output keep the strided partial reads shallow
    // enough that the L2 latency overlaps instead of serializing.
    constexpr index_int lanes = 16;
    constexpr index_int parts = 16;

    __shared__ float buffer[lanes * parts];

    const index_int lane = idx.local % lanes;
    const index_int part = idx.local / lanes;
    const index_int i    = idx.group * lanes + lane;

    float acc = 0.0f;
    if(i < total)
    {
        for(index_int s = part; s < nsplits; s += parts)
            acc += partials.data()[s * total + i];
    }
    buffer[part * lanes + lane] = acc;
    __syncthreads();
    if(part == 0 and i < total)
    {
        float sum = 0.0f;
        for(index_int p = 0; p < parts; ++p)
            sum += buffer[p * lanes + lane];
        ([&] { sum += migraphx::convert<float>(residual.data()[i]); }(), ...);
        output.data()[i] = migraphx::convert<type>(sum);
    }
}

} // namespace migraphx
#endif
