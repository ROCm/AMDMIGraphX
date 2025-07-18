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
#ifndef MIGRAPHX_GUARD_KERNELS_PACK_FP4_HPP
#define MIGRAPHX_GUARD_KERNELS_PACK_FP4_HPP

#include "migraphx/kernels/types.hpp"
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>

template<typename T> struct TD;

namespace migraphx {

template <int Axis, class Output, class Input>
__device__ void pack_fp4(Output output, Input input)
{
    const shape output_shape = output.get_shape();
    make_index().global_stride(output_shape.elements(), [&](auto i) {
        auto idx = output_shape.multi(i);
        idx[Axis] *= 2;
        auto inp_val0 = input[idx];
        auto inp_val1 = input[idx+1];
        output[idx] = static_cast<uint8_t>(float_to_fp4(inp_val1) << 4u) | static_cast<uint8_t>(float_to_fp4(inp_val1));
    });
}

}// namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_PACK_FP4_HPP
