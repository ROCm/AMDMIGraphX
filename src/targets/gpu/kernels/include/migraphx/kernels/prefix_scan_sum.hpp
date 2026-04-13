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
#ifndef MIGRAPHX_GUARD_KERNELS_PREFIX_SCAN_SUM_HPP
#define MIGRAPHX_GUARD_KERNELS_PREFIX_SCAN_SUM_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/scan.hpp>
#include <migraphx/kernels/ops.hpp>

namespace migraphx {

template <bool Exclusive, bool Reverse, class Input, class Output>
__device__ void prefix_scan_sum_slice(
    Input input, Output output, index_int offset, index_int n, index_int axis_stride)
{
    auto idx = make_index();

    constexpr index_int block_size = decltype(idx.max_nlocal())::value;
    static_assert(block_size % MIGRAPHX_WAVEFRONTSIZE == 0,
                  "Block size must be a multiple of wavefront size");
    const index_int num_chunks = (n + block_size - 1) / block_size;

    auto axis_index = [&](index_int j) {
        const index_int pos = Reverse ? (n - 1 - j) : j;
        return offset + pos * axis_stride;
    };

    using value_type = remove_reference_t<decltype(input[axis_index(0)])>;

    value_type carry = value_type{0};
    for(index_int chunk = 0; chunk < num_chunks; ++chunk)
    {
        const index_int j = chunk * block_size + idx.local;
        value_type value  = (j < n) ? input[axis_index(j)] : value_type{0};
        carry             = block_scan(idx, value, op::sum{}, carry);
        if(j < n)
        {
            if constexpr(Exclusive)
            {
                if(j == 0)
                    output[axis_index(j)] = value_type{0};
                else
                    output[axis_index(j)] = value - input[axis_index(j)];
            }
            else
            {
                output[axis_index(j)] = value;
            }
        }
    }
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_PREFIX_SCAN_SUM_HPP
