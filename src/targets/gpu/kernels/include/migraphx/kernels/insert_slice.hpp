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
#ifndef MIGRAPHX_GUARD_KERNELS_INSERT_SLICE_HPP
#define MIGRAPHX_GUARD_KERNELS_INSERT_SLICE_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/types.hpp>

namespace migraphx {

template <index_int Rank,
          class Offsets,
          class Strides,
          bool DerefDest,
          class Source,
          class Output>
__device__ void insert_slice(const index& idx,
                            const Offsets& offsets,
                            const Strides& strides,
                            const Source& source,
                            Output& output)
{
    auto src_shape = source.get_shape();

    constexpr auto src_elements = decltype(src_shape.elements()){};

    // In-place scatter into `output` at dest_idx = src_idx * strides + offsets
    idx.global_stride(src_elements, [&](auto i) {
        auto src_idx  = src_shape.multi(i);
        auto dest_idx = array_transform(src_idx, offsets, strides)(
            [](auto s, auto o, auto st) { return static_cast<index_int>(s * st + o); });

        if constexpr(DerefDest)
        {
            using value_type = remove_cv_t<remove_reference_t<decltype(source[0])>>;
            auto addr = static_cast<uintptr_t>(output[dest_idx]);
            auto ptr = reinterpret_cast<value_type*>(addr);
            *ptr          = source[src_idx];
        }
        else
        {
            output[dest_idx] = source[src_idx];
        }
    });
}

// Variant with offsets read from a tensor: 1D [rank] per-axis, or 2D [batch, rank] (batch = axis 0).
template <index_int Rank,
          bool BatchedOffsets,
          class Strides,
          bool DerefDest,
          class OffsetsTensor,
          class Source,
          class Output>
__device__ void insert_slice(const index& idx,
                            const OffsetsTensor& offsets_tensor,
                            const Strides& strides,
                            const Source& source,
                            Output& output)
{
    auto src_shape = source.get_shape();

    constexpr auto src_elements = decltype(src_shape.elements()){};

    idx.global_stride(src_elements, [&](auto i) {
        auto src_idx = src_shape.multi(i);
        array<index_int, Rank> dest_idx;
        const index_int b  = BatchedOffsets ? src_idx[0] : 0;
        for(index_int j = 0; j < Rank; j++)
            dest_idx[j] = static_cast<index_int>(src_idx[j] * strides[j] + offsets_tensor[(b * Rank) + j]);

        if constexpr(DerefDest)
        {
            using value_type = remove_cv_t<remove_reference_t<decltype(source[0])>>;
            auto addr = static_cast<uintptr_t>(output[dest_idx]);
            auto ptr = reinterpret_cast<value_type*>(addr);
            *ptr          = source[src_idx];
        }
        else
        {
            output[dest_idx] = source[src_idx];
        }
    });
}

} // namespace migraphx

#endif
