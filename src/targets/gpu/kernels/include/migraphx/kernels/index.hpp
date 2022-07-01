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
#ifndef MIGRAPHX_GUARD_KERNELS_INDEX_HPP
#define MIGRAPHX_GUARD_KERNELS_INDEX_HPP

#include <migraphx/kernels/hip.hpp>
#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/type_traits.hpp>

namespace migraphx {

struct index
{
    index_int global = 0;
    index_int local  = 0;
    index_int group  = 0;

#ifdef MIGRAPHX_NGLOBAL
    constexpr index_constant<MIGRAPHX_NGLOBAL> nglobal() const { return {}; }
#else
    __device__ index_int nglobal() const
    {
        return blockDim.x * gridDim.x; // NOLINT
    }
#endif

#ifdef MIGRAPHX_NLOCAL
    constexpr index_constant<MIGRAPHX_NLOCAL> nlocal() const { return {}; }
#else
    __device__ index_int nlocal() const
    {
        return blockDim.x; // NOLINT
    }
#endif
    template <class N, class Stride>
    static constexpr auto max_stride_iterations(N n, Stride stride)
    {
        return (n - _c<1>) / stride + _c<1>;
    }

    template <class F, class N, class Stride>
    static constexpr void for_stride(index_int start, N n, Stride stride, F f)
    {
        if constexpr(not is_integral<N>{} and not is_integral<Stride>{} and
                     max_stride_iterations(n, stride) == 1)
        {
            if constexpr(stride > n)
            {
                if(start < n)
                    f(start);
            }
            else
            {
                f(start);
            }
        }
        else
        {
            for(index_int i = start; i < n; i += stride)
            {
                f(i);
            }
        }
    }

    template <class F, class N>
    __device__ void global_stride(N n, F f) const
    {
        for_stride(global, n, nglobal(), f);
    }

    template <class F, class N>
    __device__ void local_stride(N n, F f) const
    {
        for_stride(local, n, nlocal(), f);
    }
};

inline __device__ __attribute__((const)) index make_index()
{
    return index{blockIdx.x * blockDim.x + threadIdx.x, threadIdx.x, blockIdx.x}; // NOLINT
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_INDEX_HPP
