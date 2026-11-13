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
#include <migraphx/gpu/device/dyn_concat.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/reduce_ops.hpp>
#include <migraphx/gpu/device/scan.hpp>
#include <migraphx/gpu/device/visit.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <algorithm>
#include <cassert>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void dyn_concat(hipStream_t stream,
                std::int64_t axis,
                const argument& counts,
                const std::vector<argument>& inputs,
                const argument& output,
                const argument& total)
{
    assert(not inputs.empty());
    const std::int64_t rank = inputs.front().get_shape().ndim();
    if(axis < 0)
        axis += rank;
    assert(axis >= 0 and axis < rank);
    const index_int normalized_axis = axis;
    const index_int ninputs         = inputs.size();
    constexpr index_int block_size = 256;
    const auto n_aligned           = ((ninputs + block_size - 1) / block_size) * block_size;

    hip_pointer_visit_all(counts, total)([&](auto counts_ptr, auto total_ptr) {
        for(std::size_t input_index = 0; input_index < inputs.size(); ++input_index)
        {
            const auto& input = inputs.at(input_index);
            const index_int current_input = input_index;
            const index_int elements      = input.get_shape().elements();
            const auto nblocks = std::max<index_int>(1, (elements + block_size - 1) / block_size);
            hip_visit_all(output, input)([&](auto output_view, auto input_view) {
                launch(stream, nblocks * block_size, block_size)([=](auto idx) __device__ {
                    MIGRAPHX_DEVICE_SHARED int64_t offset;
                    block_scan<block_size>(
                        idx,
                        sum{},
                        int64_t{0},
                        [&](auto f) -> decltype(f(index_int{})) {
                            return idx.local_stride(n_aligned, f);
                        },
                        [&](auto i) { return i < ninputs ? counts_ptr[i] : int64_t{0}; },
                        [&](auto i, auto inclusive_value) {
                            if(i == current_input)
                            {
                                offset = inclusive_value - counts_ptr[i];
                                if(i == ninputs - 1 and idx.group == 0)
                                    total_ptr[0] = inclusive_value;
                            }
                        });
                    __syncthreads();
                    auto count = counts_ptr[current_input];
                    idx.global_stride(elements, [&](auto i) {
                        if(count <= 0)
                            return;
                        auto input_pos = input_view.get_shape().multi(i);
                        if(input_pos[normalized_axis] >= count)
                            return;
                        auto output_pos = input_pos;
                        output_pos[normalized_axis] += offset;
                        output_view[output_pos] = input_view[input_pos];
                    });
                });
            });
        }
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
