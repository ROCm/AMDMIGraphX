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
#ifndef MIGRAPHX_GUARD_KERNELS_UNPACK_FP4_HPP
#define MIGRAPHX_GUARD_KERNELS_UNPACK_FP4_HPP

#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/fp4_casts.hpp>
#include <migraphx/kernels/float8.hpp>

namespace migraphx {

template <int Axis, class Input, class Output>
__device__ void unpack_fp4(Input input, Output output)
{
    const auto input_shape = input.get_shape();
    make_index().global_stride(input_shape.elements(), [&](auto i) {
        auto in_idx  = input_shape.multi(i);
        auto out_idx = in_idx;
        out_idx[Axis] *= 2;
        // unpacking 2 unsigned parts
        // unpacking 4 least significant bits first
        uint8_t fp4_val = input[in_idx];
        output[out_idx] = cast_from_fp4<fp8::fp8e4m3fn>(fp4_val);
        out_idx[Axis] += 1;
        fp4_val         = fp4_val >> 4u;
        output[out_idx] = cast_from_fp4<fp8::fp8e4m3fn>(fp4_val);
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_UNPACK_FP4_HPP
