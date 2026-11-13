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
#ifndef MIGRAPHX_GUARD_KERNELS_DYN_CONCAT_HPP
#define MIGRAPHX_GUARD_KERNELS_DYN_CONCAT_HPP

#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/scan.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/tuple.hpp>
#include <migraphx/kernels/uninitialized_buffer.hpp>

namespace migraphx {

template <index_int Axis, class Input, class Output>
__device__ void
dyn_concat_copy(index idx, Input input, index_int count, index_int offset, Output output)
{
    constexpr auto input_shape  = get_shape_c<Input>{};
    constexpr auto output_shape = get_shape_c<Output>{};
    static_assert(Axis < input_shape.lens.size());
    static_assert(Axis < output_shape.lens.size());

    idx.local_stride(input_shape.elements(), [&](auto i) {
        auto input_index = input_shape.multi(i);
        if(input_index[Axis] < count)
        {
            auto output_index = input_index;
            output_index[Axis] += offset;
            output[output_index] = input[input_index];
        }
    });
}

//NOTE: this only runs with one block.
//TODO: investigate if we get better performance by doing multiple kernel launches such
//that we concatenate inputs i and i+1 until all the inputs are concatenated.
template <index_int Axis, class Counts, class Output, class Total, class... Inputs>
__device__ void dyn_concat(Counts counts, Output output, Total total, Inputs... inputs)
{
    constexpr index_int ninputs = sizeof...(Inputs);
    static_assert(ninputs > 0);
    static_assert(Axis < get_shape_c<Output>{}.lens.size());
    constexpr array<index_int, ninputs> capacities = {get_shape_c<Inputs>{}.lens[Axis]...};

    auto idx = make_index();
    __shared__ uninitialized_buffer<index_int, ninputs> valid_counts;
    __shared__ uninitialized_buffer<index_int, ninputs> offsets;
    auto get_count = [&](auto i) {
        auto value = counts[i];
        if(value <= 0)
            return index_int{0};
        if(value >= capacities[i])
            return capacities[i];
        return static_cast<index_int>(value);
    };
    const auto total_count = block_scan(
        idx, op::sum{}, index_int{0}, ninputs, get_count, [&](auto i, auto inclusive_value) {
            auto count      = get_count(i);
            valid_counts[i] = count;
            offsets[i]      = inclusive_value - count;
        });
    __syncthreads();

    if(idx.local == 0)
        total[0] = total_count;

    auto input_pack = make_tuple(inputs...);
    sequence_c<ninputs>([&](auto... is) {
        swallow{(dyn_concat_copy<Axis>(idx, input_pack[is], valid_counts[is], offsets[is], output),
                 0)...};
    });
}

} // namespace migraphx

#endif
