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

template <index_int BlockSize, bool Exclusive, bool Reverse, class Input, class Output>
__device__ void prefix_scan_sum_slice(
    Input input, Output output, index_int offset, index_int n, index_int axis_stride)
{
    auto idx = make_index();

    auto read_input = [&](index_int j) {
        index_int pos = Reverse ? (n - 1 - j) : j;
        return input[offset + pos * axis_stride];
    };

    auto write_output = [&](index_int j, auto x) {
        index_int pos                      = Reverse ? (n - 1 - j) : j;
        output[offset + pos * axis_stride] = x;
    };

    using value_type = decltype(read_input(0));

    if constexpr(Exclusive)
    {
        block_scan<BlockSize>(
            idx, op::sum{}, value_type{0}, n, read_input, [&](index_int j, auto x) {
                if(j == 0)
                    write_output(j, value_type{0});
                else
                    write_output(j, x - read_input(j));
            });
    }
    else
    {
        block_scan<BlockSize>(idx, op::sum{}, value_type{0}, n, read_input, write_output);
    }
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_PREFIX_SCAN_SUM_HPP
