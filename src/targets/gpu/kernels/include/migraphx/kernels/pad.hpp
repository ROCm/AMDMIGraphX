/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

enum class pad_mode_t
{
    constant = 0,
    reflect  = 1,
    edge     = 2
};

// Helper function to calculate reflected index for a single dimension
template <class T>
__device__ auto reflect_index(T idx, T size) -> T
{
    if(size == 1)
        return 0;

    auto period = 2 * (size - 1);

    if(idx < 0)
    {
        // Left padding: -1 -> size-1, -2 -> size-2, etc., with bouncing
        auto dist = -idx;
        auto pos  = (dist - 1) % period;
        return (pos < size) ? (size - 1 - pos) : (pos - (size - 1));
    }

    // Right padding: size -> 0, size+1 -> 1, etc., with bouncing
    auto dist = idx - size + 1;
    auto pos  = (dist - 1) % period;
    return (pos < size - 1) ? pos : (period - pos);
}

// Apply edge padding: clamp to valid range
template <class IndexType, class BoundsType, class Range>
__device__ void apply_edge_padding(IndexType& input_idx, const BoundsType& input_bounds, Range range_multi)
{
    for(auto j : range_multi)
    {
        if(input_idx[j] < 0)
            input_idx[j] = 0;
        else if(input_idx[j] >= input_bounds[j])
            input_idx[j] = input_bounds[j] - 1;
    }
}

// Apply reflect padding: mirror around boundaries
template <class IndexType, class BoundsType, class Range>
__device__ void apply_reflect_padding(IndexType& input_idx, const BoundsType& input_bounds, Range range_multi)
{
    for(auto j : range_multi)
    {
        if(input_idx[j] < 0 or input_idx[j] >= input_bounds[j])
            input_idx[j] = reflect_index(input_idx[j], input_bounds[j]);
    }
}

template <class Offsets, class Input, class Output, class PadVal>
__device__ void pad(const index& idx,
                    const Offsets& offsets,
                    const Input& input,
                    Output& output,
                    const PadVal& pad_val,
                    pad_mode_t mode = pad_mode_t::constant)
{
    auto output_shape = output.get_shape();
    auto input_bounds = input.get_shape().lens;

    idx.global_stride(output_shape.elements(), [&](auto i) {
        auto multi       = output_shape.multi(i);
        auto input_idx   = multi - offsets;
        auto range_multi = range(multi.size());

        // Check if we're in the padding region
        bool in_padding = any_of(range_multi.begin(), range_multi.end(), [&](auto j) {
            return multi[j] < offsets[j] or input_idx[j] >= input_bounds[j];
        });

        if(not in_padding)
        {
            output[multi] = implicit_conversion(input[input_idx]);
            return;
        }

        // Handle padding based on mode
        if(mode == pad_mode_t::constant)
        {
            output[multi] = implicit_conversion(pad_val);
        }
        else if(mode == pad_mode_t::reflect)
        {
            apply_reflect_padding(input_idx, input_bounds, range_multi);
            output[multi] = implicit_conversion(input[input_idx]);
        }
        else // pad_mode_t::edge
        {
            apply_edge_padding(input_idx, input_bounds, range_multi);
            output[multi] = implicit_conversion(input[input_idx]);
        }
    });
}

} // namespace migraphx
#endif
