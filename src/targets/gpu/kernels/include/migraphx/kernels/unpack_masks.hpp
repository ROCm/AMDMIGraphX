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
#ifndef MIGRAPHX_GUARD_KERNELS_UNPACK_MASKS_HPP
#define MIGRAPHX_GUARD_KERNELS_UNPACK_MASKS_HPP

#include "migraphx/kernels/array.hpp"
#include <migraphx/kernels/group_query_attention.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>

namespace migraphx {

template <class Output, class BlockRowInd, class BlockColInd>
__device__ void unpack_masks(Output output, BlockRowInd block_row_ind, BlockColInd block_col_ind)
{
    constexpr auto out_shape_lens      = output.get_shape().lens;
    constexpr unsigned int num_layouts = out_shape_lens[0];
    constexpr unsigned int max_blocks  = out_shape_lens[1];
    make_index().global_stride(num_layouts * max_blocks, [&](auto idx) {
        const unsigned int layout_idx = idx / max_blocks;
        const unsigned int row_idx    = idx % max_blocks;
        auto out_idx                  = make_array(layout_idx, row_idx, 0u);

        auto row_ind_idx             = make_array(layout_idx, row_idx);
        const unsigned int col_start = block_row_ind[row_ind_idx];
        ++row_ind_idx[1];
        const unsigned int col_end = block_row_ind[row_ind_idx];

        auto col_ind_idx = make_array(layout_idx, 0);
        for(index_int i = col_start; i < col_end; ++i)
        {
            col_ind_idx[1]  = i;
            out_idx[2]      = block_col_ind[col_ind_idx];
            output[out_idx] = true;
        }
    });
}

} // namespace migraphx
#endif
