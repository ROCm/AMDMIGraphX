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

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>

namespace migraphx {

template <index_int Capacity, class Count>
__device__ index_int dyn_concat_count(Count count)
{
    auto value = count[0];
    if(value <= 0)
        return 0;
    if(value >= Capacity)
        return Capacity;
    return static_cast<index_int>(value);
}

template <index_int Outer,
          index_int InputCapacity,
          index_int OutputCapacity,
          index_int Inner,
          class Input,
          class Output>
__device__ void dyn_concat_copy(index idx,
                                Input input,
                                index_int count,
                                index_int source_capacity,
                                index_int offset,
                                Output output)
{
    if constexpr(Outer > 0 and InputCapacity > 0 and Inner > 0)
    {
        constexpr index_int max_elements = Outer * InputCapacity * Inner;
        idx.local_stride(max_elements, [&](auto i) {
            const index_int inner_index = i % Inner;
            const index_int axis_index  = (i / Inner) % InputCapacity;
            const index_int outer_index = i / (InputCapacity * Inner);
            if(axis_index < count)
            {
                const index_int input_index =
                    (outer_index * source_capacity + axis_index) * Inner + inner_index;
                const index_int output_index =
                    (outer_index * OutputCapacity + offset + axis_index) * Inner + inner_index;
                output[output_index] = input[input_index];
            }
        });
    }
}

} // namespace migraphx

#endif
