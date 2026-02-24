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
#include <migraphx/kernels/math.hpp>

namespace migraphx {

struct pad_constant
{
    constexpr diff_int operator()(diff_int idx, diff_int) const { return idx; }
};
struct pad_reflect
{
    constexpr diff_int operator()(diff_int idx, diff_int size) const
    {
        if(size <= 1)
            return 0;

        auto period = size - 1;

        // Triangle wave: oscillates between 0 and period
        // Handle negative indices by taking absolute value
        auto mod_val = abs(idx) % (2 * period);
        return (mod_val <= period) ? mod_val : (2 * period - mod_val);
    }
};
struct pad_edge
{
    constexpr diff_int operator()(diff_int idx, diff_int size) const
    {
        return min(max(idx, 0), size - 1);
    }
};

template <class Offsets, class Input, class Output, class PadVal, class PadMode>
__device__ void pad(const index& idx,
                    const Offsets& offsets,
                    const Input& input,
                    Output& output,
                    const PadVal& pad_val,
                    PadMode pad_mode)
{
    auto output_shape = output.get_shape();
    auto input_bounds = input.get_shape().lens.template to<diff_int>();

    idx.global_stride(output_shape.elements(), [&](auto gid) {
        auto out_idx   = output_shape.multi(gid).template to<diff_int>();
        auto input_idx = array_transform(out_idx - offsets, input_bounds)(pad_mode);
        bool in_bounds = array_transform(input_idx, input_bounds)([&](auto i, auto bound) {
                             return i < bound and i >= 0;
                         }).all();
        if(in_bounds)
            output[out_idx] = implicit_conversion(input[input_idx]);
        else
            output[out_idx] = implicit_conversion(pad_val);
    });
}

} // namespace migraphx
#endif
