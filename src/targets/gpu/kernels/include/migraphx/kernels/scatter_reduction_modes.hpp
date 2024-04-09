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
#ifndef MIGRAPHX_GUARD_KERNELS_SCATTER_REDUCTION_MODES_HPP
#define MIGRAPHX_GUARD_KERNELS_SCATTER_REDUCTION_MODES_HPP

#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/type_traits.hpp>
#include <migraphx/kernels/debug.hpp>

#ifndef MIGRAPHX_ALLOW_ATOMIC_CAS
#define MIGRAPHX_ALLOW_ATOMIC_CAS 0
#endif

#define MIGRAPHX_ATOMIC_CAS_WARNING() \
    MIGRAPHX_ASSERT(MIGRAPHX_ALLOW_ATOMIC_CAS and "Using atomicCAS is slow")

namespace migraphx {

struct assign_none
{
    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR void operator()(T& x, U y) const
    {
        x = y;
    }
};

struct assign_add
{
    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR void operator()(T& x, U y) const
    {
        if constexpr(is_same<T, float>{} or is_same<T, double>{})
        {
            unsafeAtomicAdd(&x, T(y));
        }
        else
        {
            MIGRAPHX_ATOMIC_CAS_WARNING();
            atomicAdd(&x, T(y));
        }
    }
};

struct assign_mul
{
    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR void operator()(T& x, U y) const
    {
        MIGRAPHX_ATOMIC_CAS_WARNING();
        T old = x;
        T assumed;
        do
        {
            assumed = old;
            old     = atomicCAS(&x, assumed, assumed * y);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
        } while(assumed != old);
#pragma clang diagnostic pop
    }
};

struct assign_max
{
    template <typename T, typename U>
    MIGRAPHX_DEVICE_CONSTEXPR void operator()(T& x, U y) const
    {
        atomicMax(&x, T(y));
    }
};

struct assign_min
{
    template <typename T, typename U>
    MIGRAPHX_DEVICE_CONSTEXPR void operator()(T& x, U y) const
    {
        atomicMin(&x, T(y));
    }
};

} // namespace migraphx
#endif
