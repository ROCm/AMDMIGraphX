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
#ifndef MIGRAPHX_GUARD_KERNELS_CK_GEMM_HPP
#define MIGRAPHX_GUARD_KERNELS_CK_GEMM_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/ck.hpp>
#include <migraphx/kernels/gemm_batcher.hpp>

namespace migraphx {

template <class G, class E1, class A0, class B0, class B1, class... D0s>
__device__ void ck_gemm_gemm_matrix(E1 e1, A0 a0, B0 b0, B1 b1, D0s... d0s)
{
    constexpr auto desc = G::make_descriptor(to_ck_tensor<A0>(),
                                             to_ck_tensor<ck_transposeb<B0>>(),
                                             ck::make_tuple(to_ck_tensor<D0s>()...),
                                             to_ck_tensor<ck_transposeb<B1>>(),
                                             ck::make_tuple(),
                                             to_ck_tensor<E1>());

    MIGRAPHX_STATIC_ASSERT_FOR(desc.IsValid())
    {
        G::Run(desc,
               to_ck_const_pointer(a0.data()),
               to_ck_const_pointer(b0.data()),
               ck::make_tuple(to_ck_const_pointer(d0s.data())...),
               to_ck_const_pointer(b1.data()),
               ck::make_tuple(),
               to_ck_pointer(e1.data()));
    }
}

template <class G, index_int BlocksPerBatch, class... Ts>
__device__ void ck_gemm_gemm(Ts... xs)
{
    gemm_batch_args(make_index(), _c<BlocksPerBatch>, xs...)(
        [](auto... ys) { ck_gemm_gemm_matrix<G>(ys...); });
}

} // namespace migraphx
#endif
