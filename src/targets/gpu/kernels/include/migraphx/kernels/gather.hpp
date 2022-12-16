/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_KERNELS_GATHER_HPP
#define MIGRAPHX_GUARD_KERNELS_GATHER_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/shape.hpp>
#include <migraphx/kernels/algorithm.hpp>

namespace migraphx {

template <int Axis, class Input, class Indices>
__device__ constexpr auto gather_shape(Input input, Indices indices)
{
    auto lengths = input.get_shape().lens;

    lengths[Axis] = indices.get_shape().elements();
    return make_shape(lengths, input.get_shape().strides);
}

template <int Axis, class Input, class Indices, class Output>
__device__ void gather(Input input, Indices indices, Output output)
{
    auto ind           = make_index();
    auto axis_dim_size = input.get_shape().lens[Axis];

    const auto out_comp = gather_shape<Axis>(input, indices);

    ind.global_stride(output.get_shape().elements(), [&](auto i) {
        auto idx      = out_comp.multi(i);
        auto in_index = indices[idx[Axis]];

        auto new_in_index = (in_index < 0) ? in_index + axis_dim_size : in_index;

        idx[Axis] = new_in_index;

        output[i] = input[idx];
    });
}

} // namespace migraphx
#endif
