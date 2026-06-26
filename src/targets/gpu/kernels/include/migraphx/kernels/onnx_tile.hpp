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
#ifndef MIGRAPHX_GUARD_KERNELS_ONNX_TILE_HPP
#define MIGRAPHX_GUARD_KERNELS_ONNX_TILE_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>

namespace migraphx {

/// ONNX Tile: output[i] = input[i mod input_lens] per dimension.
template <class Input, class Output>
__device__ void onnx_tile(Input input, Output output)
{
    auto ind           = make_index();
    const auto in_lens = input.get_shape().lens;
    ind.global_stride(output.get_shape().elements(), [&](auto i) {
        auto om = output.get_shape().multi(i);
        auto im = om;
        for(index_int d = 0; d < in_lens.size(); ++d)
        {
            im[d] = om[d] % in_lens[d];
        }
        output[i] = input[im];
    });
}

} // namespace migraphx

#endif
