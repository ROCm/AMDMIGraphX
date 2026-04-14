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
#ifndef MIGRAPHX_GUARD_KERNELS_NONZERO_HPP
#define MIGRAPHX_GUARD_KERNELS_NONZERO_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/scan.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/float_equal.hpp>

namespace migraphx {

// Use prefix sum to compute index in the output.
// Only 1 block can be used since we have only one prefix sum.
template <class Input, class Output>
__device__ void nonzero(Input input, Output output)
{
    auto idx                = make_index();
    const auto in_shape     = input.get_shape();
    const auto elem_num     = in_shape.elements();
    const auto out_elem_num = output.get_shape().elements();

    // Fill all output to 0 first
    idx.local_stride(out_elem_num, [&](auto j) { output[j] = 0; });

    constexpr auto block_size = decltype(idx.max_nlocal()){};
    static_assert(block_size % MIGRAPHX_WAVEFRONTSIZE == 0,
                  "Block size must be a multiple of wavefront size");
    block_scan(
        idx,
        op::sum{},
        0,
        elem_num,
        [&](auto j) { return float_equal(input[j], 0) ? 0 : 1; },
        [&](auto j, auto value) {
            if(j >= elem_num)
                return;
            const int scanned  = value;
            const auto out_loc = scanned - 1;
            if(float_equal(input[j], 0))
                return;

            const auto multi_idx = in_shape.multi(j);
            for(index_int k = 0; k < multi_idx.size(); ++k)
            {
                output[k * elem_num + out_loc] = multi_idx[k];
            }
        });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_NONZERO_HPP
