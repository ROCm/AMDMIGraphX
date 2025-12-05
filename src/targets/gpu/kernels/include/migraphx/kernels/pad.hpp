/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_KERNELS_PAD_HPP
#define MIGRAPHX_GUARD_KERNELS_PAD_HPP

#include <migraphx/kernels/shape.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/ranges.hpp>
#include <migraphx/kernels/vec.hpp>

namespace migraphx {

struct pad_constant
{
};
struct pad_reflect
{
};
struct pad_edge
{
};

// Unified reflect index using triangle wave formula
// Handles any signed index with proper bouncing pattern
template <class T, class S>
__device__ auto reflect_index(T idx, S size) -> S
{
    if(size == 1)
        return 0;

    auto period = size - 1;

    // Handle negative indices by taking absolute value
    auto shifted = idx < 0 ? static_cast<S>(-idx) : static_cast<S>(idx);

    // Triangle wave: oscillates between 0 and period
    auto mod_val = shifted % (2 * period);
    return (mod_val <= period) ? mod_val : (2 * period - mod_val);
}

// Transform index array for reflect padding - returns new index array
// Uses multi and offsets to compute signed index (avoids unsigned underflow)
template <class MultiType, class OffsetsType, class BoundsType>
__device__ auto transform_reflect(const MultiType& multi,
                                  const OffsetsType& offsets,
                                  const BoundsType& input_bounds)
{
    auto result = multi;
    for(size_t j = 0; j < multi.size(); ++j)
    {
        // Compute signed index to avoid unsigned underflow
        auto idx  = static_cast<int64_t>(multi[j]) - static_cast<int64_t>(offsets[j]);
        result[j] = reflect_index(idx, input_bounds[j]);
    }
    return result;
}

// Transform index array for edge padding - returns clamped index array
// Uses multi and offsets to compute signed index (avoids unsigned underflow)
template <class MultiType, class OffsetsType, class BoundsType>
__device__ auto
transform_edge(const MultiType& multi, const OffsetsType& offsets, const BoundsType& input_bounds)
{
    auto result = multi;
    for(size_t j = 0; j < multi.size(); ++j)
    {
        // Compute signed index to avoid unsigned underflow
        auto idx = static_cast<int64_t>(multi[j]) - static_cast<int64_t>(offsets[j]);
        if(idx < 0)
            result[j] = 0;
        else if(idx >= static_cast<int64_t>(input_bounds[j]))
            result[j] = input_bounds[j] - 1;
        else
            result[j] = idx;
    }
    return result;
}

template <class Offsets, class Input, class Output, class PadVal, class PadMode>
__device__ void pad(const index& idx,
                    const Offsets& offsets,
                    const Input& input,
                    Output& output,
                    const PadVal& pad_val,
                    PadMode)
{
    auto output_shape = output.get_shape();
    auto input_bounds = input.get_shape().lens;

    idx.global_stride(output_shape.elements(), [&](auto i) {
        auto multi = output_shape.multi(i);

        if constexpr(is_same<PadMode, pad_constant>{})
        {
            auto input_idx   = multi - offsets;
            auto range_multi = range(multi.size());

            // Check if we're in the padding region
            bool in_padding = any_of(range_multi.begin(), range_multi.end(), [&](auto j) {
                return multi[j] < offsets[j] or input_idx[j] >= input_bounds[j];
            });

            if(in_padding)
                output[multi] = implicit_conversion(pad_val);
            else
                output[multi] = implicit_conversion(input[input_idx]);
        }
        else if constexpr(is_same<PadMode, pad_reflect>{})
        {
            auto input_idx = transform_reflect(multi, offsets, input_bounds);
            output[multi]  = implicit_conversion(input[input_idx]);
        }
        else if constexpr(is_same<PadMode, pad_edge>{})
        {
            auto input_idx = transform_edge(multi, offsets, input_bounds);
            output[multi]  = implicit_conversion(input[input_idx]);
        }
    });
}

} // namespace migraphx
#endif
