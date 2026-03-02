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
#ifndef MIGRAPHX_GUARD_KERNELS_REVERSE_HPP
#define MIGRAPHX_GUARD_KERNELS_REVERSE_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/array.hpp>

namespace migraphx {

template <class Axes, class Input, class Output>
__device__ void reverse(Axes axes, Input input, Output output)
{
    auto ind  = make_index();
    auto lens = input.get_shape().lens;

    ind.global_stride(output.get_shape().elements(), [&](auto i) {
        auto out_idx = output.get_shape().multi(i);
        auto in_idx  = out_idx;
        for(auto axis : axes)
        {
            in_idx[axis] = lens[axis] - 1 - out_idx[axis];
        }
        output[i] = input[in_idx];
    });
}

} // namespace migraphx
#endif
