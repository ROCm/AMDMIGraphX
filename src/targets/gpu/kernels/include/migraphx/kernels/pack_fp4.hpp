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

#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/fp4_casts.hpp>
// TODO: use hip_fp4 header
// #include <hip/hip_fp4.h>

namespace migraphx {

template <int Axis, class Input, class Output>
__device__ void pack_fp4(Input input, Output output)
{
    const auto output_shape = output.get_shape();
    make_index().global_stride(output_shape.elements(), [&](auto i) {
        auto out_idx = output_shape.multi(i);
        auto in_idx  = out_idx;
        in_idx[Axis] *= 2;
        auto inp_val0 = input[in_idx];
        in_idx[Axis] += 1;
        auto inp_val1    = input[in_idx];
        uint8_t out_val0 = float_to_fp4(inp_val0);
        uint8_t out_val1 = float_to_fp4(inp_val1);
        output[out_idx]  = static_cast<uint8_t>(out_val1 << 4u) | out_val0;
        // TODO: from hip_fp4 header
        // auto fp32x2_val = float2{input[idx], input[idx + 1]};
        // output[idx] = __hip_cvt_float2_to_fp4x2(fp32x2_val, __HIP_E2M1, __HIP_SATFINITE);
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_PACK_FP4_HPP
