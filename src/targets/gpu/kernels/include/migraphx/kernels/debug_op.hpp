/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_KERNELS_GROUP_QUERY_ATTENTION_HPP
#define MIGRAPHX_GUARD_KERNELS_GROUP_QUERY_ATTENTION_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/ck.hpp>
#include <migraphx/kernels/gemm_batcher.hpp>
#include <limits>
#include <migraphx/kernels/type_traits.hpp>

namespace migraphx {

// kernel inputs = query, past_present_key, past_present_value, cos_cache, sin_cache, rotary_qkv, attn_probs, seqlens_k
template <class Output,
          class Input0,
          class Input1,
          class Input2>
__device__ void debug_op(Output, Input0 i0, Input1, Input2)
{
    auto ind = make_index();
    ind.global_stride(i0.get_shape().elements(), [&](auto idx) {
        if(idx == 0)
        {
            for(int i = 0; i < i0.get_shape().elements(); ++i)
            {
                printf("gpu_query%d: %f\n", i, static_cast<double>(i0[i]));
            }
        }
    });
}

} // namespace migraphx
#endif
