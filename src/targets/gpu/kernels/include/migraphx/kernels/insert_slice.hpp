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
          class Dest,
          class Output>
__device__ void insert_slice(const index& idx,
                            const Offsets& offsets,
                            const Strides& strides,
                            const Source& source,
                            const Dest& dest,
                            Output& output)
{
    auto output_shape = output.get_shape();
    auto src_shape    = source.get_shape();
    auto dest_shape   = dest.get_shape();

    constexpr auto out_elements = decltype(output_shape.elements()){};
    constexpr auto src_elements = decltype(src_shape.elements()){};

    // Phase 1: copy destination to output
    idx.global_stride(out_elements, [&](auto i) { output[i] = dest[i]; });

    // Phase 2: scatter source into output at dest_idx = src_idx * strides + offsets
    idx.global_stride(src_elements, [&](auto i) {
        auto src_idx  = src_shape.multi(i);
        auto dest_idx = array_transform(src_idx, offsets, strides)(
            [](auto s, auto o, auto st) { return static_cast<index_int>(s * st + o); });

        if constexpr(DerefDest)
        {
            using src_type = typename std::remove_cv_t<typename Source::type>;
            auto addr     = static_cast<uintptr_t>(output[dest_idx]);
            auto* ptr     = reinterpret_cast<src_type*>(addr);
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
