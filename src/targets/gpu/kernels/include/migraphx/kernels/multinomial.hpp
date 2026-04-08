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
#ifndef MIGRAPHX_GUARD_KERNELS_MULTINOMIAL_HPP
#define MIGRAPHX_GUARD_KERNELS_MULTINOMIAL_HPP

#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>

namespace migraphx {

template <class CDF, class Dist, class Output>
__device__ void multinomial(CDF cdf, Dist dist, Output output)
{
    const index_int class_size = cdf.get_shape().lens[1];

    auto ind = make_index();
    ind.global_stride(output.get_shape().elements(), [&](auto i) {
        auto idx        = output.get_shape().multi(i);
        auto cdf_begin  = cdf.begin() + (idx[0] * class_size);
        auto cdf_end    = cdf_begin + class_size;
        auto last_total = *(cdf_end - 1);
        auto threshold  = dist[i] * last_total;
        auto it         = upper_bound(cdf_begin, cdf_end, threshold, less{});
        output[i]       = it - cdf_begin;
    });
}

} // namespace migraphx

#endif
