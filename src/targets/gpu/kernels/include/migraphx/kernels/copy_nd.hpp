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
#ifndef MIGRAPHX_GUARD_KERNELS_COPY_ND_HPP
#define MIGRAPHX_GUARD_KERNELS_COPY_ND_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/algorithm.hpp>

namespace migraphx {

template <class Src, class Offsets, class Dest>
__device__ void copy_nd(Src src_t, Offsets offsets_t, Dest dest_t, index_int axis)
{
    auto ind       = make_index();
    auto src_shape = src_t.get_shape();
    auto src_lens  = src_shape.lens;

    ind.global_stride(src_t.size(), [&](auto idx) {
        auto src_multi = src_shape.multi(idx);

        // Linear index for the "outer" dimensions (0 .. axis-1)
        index_int outer_linear = 0;
        index_int stride       = 1;
        for(index_int d = axis; d > 0; d--)
        {
            index_int dim = d - 1;
            outer_linear += src_multi[dim] * stride;
            stride *= src_lens[dim];
        }

        index_int off = (offsets_t.size() == 1)
                            ? static_cast<index_int>(offsets_t[0])
                            : static_cast<index_int>(offsets_t[outer_linear]);

        // dest_multi = src_multi with dest_multi[axis] = off + src_multi[axis]
        auto dest_multi = src_multi;
        dest_multi[axis] = off + src_multi[axis];

        dest_t[dest_multi] = src_t[idx];
    });
}

} // namespace migraphx
#endif
