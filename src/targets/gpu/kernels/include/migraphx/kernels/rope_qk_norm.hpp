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
#ifndef MIGRAPHX_GUARD_KERNELS_ROPE_QK_NORM_HPP
#define MIGRAPHX_GUARD_KERNELS_ROPE_QK_NORM_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/reduce.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/math.hpp>

namespace migraphx {

// Fused q/k head RMS-norm + rotary embedding for a packed qkv projection.
//
// The qkv input is the row-major projection output [batch, 1, (nq+2*nk)*d].
// For each q head (h < nq) and k head (nq <= h < nq+nk) the row of d elements
// is RMS-normalized with the q or k norm weight and then rotated with the
// gathered cos/sin values for the current position. v heads are not touched
// (consumers read them directly from the qkv buffer). The output is
// [batch, nq+nk, 1, d] with q heads first, matching the layout the attention
// kernel and the k-cache append consume.
//
// Arithmetic mirrors the unfused graph: the sum of squares accumulates in
// float, everything after the rsqrt is done in the tensor type per operation.
template <index_int NumQHeads, class QKV, class QW, class KW, class Cos, class SSin, class Output>
__device__ void rope_qk_norm(
    QKV qkv, QW qw, KW kw, Cos cos_in, SSin ssin_in, Output output, float eps, float ss_scale)
{
    auto idx   = make_index();
    using type = typename Output::type;

    constexpr auto out_lens     = get_shape_c<Output>{}.lens;
    constexpr index_int nheads  = out_lens[1];
    constexpr index_int d       = out_lens[3];
    constexpr index_int half    = d / 2;
    constexpr auto qkv_lens     = get_shape_c<QKV>{}.lens;
    constexpr index_int qkv_row = qkv_lens[2];

    const index_int batch = idx.group / nheads;
    const index_int head  = idx.group % nheads;

    // q heads then k heads are contiguous in the qkv row
    const auto* row = qkv.data() + batch * qkv_row + head * d;
    const auto* nw  = (head < NumQHeads) ? qw.data() : kw.data();

    float ss       = block_reduce(idx, op::sum{}, 0.0f, d, [&](auto i) {
        float v = migraphx::convert<float>(row[i]);
        return v * v;
    });
    const auto rms = migraphx::convert<type>(rsqrt(ss * ss_scale + eps));

    const auto* cos_row  = cos_in.data() + batch * d;
    const auto* ssin_row = ssin_in.data() + batch * d;
    auto* out_row        = output.data() + idx.group * d;

    idx.local_stride(half, [&](auto i) {
        const auto j  = i + half;
        const type ni = (row[i] * rms) * nw[i];
        const type nj = (row[j] * rms) * nw[j];
        // rotate_half: partner of i is j and vice versa; the sign is already
        // folded into ssin (negative first half, positive second half)
        out_row[i] = (ni * cos_row[i]) + (nj * ssin_row[i]);
        out_row[j] = (nj * cos_row[j]) + (ni * ssin_row[j]);
    });
}

} // namespace migraphx
#endif
