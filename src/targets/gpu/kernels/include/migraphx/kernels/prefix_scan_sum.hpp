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
#include <migraphx/kernels/slice.hpp>

namespace migraphx {

// sliced tensors from slice_schedule cover one axis line
template <bool Exclusive, bool Reverse, class Input, class Output>
__device__ void prefix_scan_sum_slice(Input input, Output output)
{
    auto idx = make_index();

    constexpr auto block_size = decltype(idx.max_nlocal()){};
    static_assert(block_size % MIGRAPHX_WAVEFRONTSIZE == 0,
                  "Block size must be a multiple of wavefront size");
    const index_int n = input.get_shape().elements();

    const auto linear = [&](index_int j) { return Reverse ? (n - 1 - j) : j; };

    using value_type = remove_reference_t<decltype(*input.data())>;

    block_scan(
        idx,
        op::sum{},
        value_type{0},
        n,
        [&](index_int j) { return input[linear(j)]; },
        [&](index_int j, const value_type& value) {
            if(j >= n)
                return;
            const index_int li = linear(j);
            if constexpr(Exclusive)
            {
                if(j == 0)
                    output[li] = value_type{0};
                else
                    output[li] = value - input[li];
            }
            else
            {
                output[li] = value;
            }
        });
}

// prefix sum along Axis
// slice_schedule uses per_block group_stride over slices,
// each slice is scanned along that axis
template <index_int Axis, bool Exclusive, bool Reverse, class Input, class Output>
__device__ void prefix_scan_sum(Input input, Output output)
{
    auto idx = make_index();
    slice_schedule<per_block>(idx, slice_axes<Axis>())(input,
                                                       output)([&](auto in_slice, auto out_slice) {
        prefix_scan_sum_slice<Exclusive, Reverse>(in_slice, out_slice);
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_PREFIX_SCAN_SUM_HPP
