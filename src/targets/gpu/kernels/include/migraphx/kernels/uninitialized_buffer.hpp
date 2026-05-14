/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 */
#ifndef MIGRAPHX_GUARD_KERNELS_UNINITIALIZED_BUFFER_HPP
#define MIGRAPHX_GUARD_KERNELS_UNINITIALIZED_BUFFER_HPP

#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/debug.hpp>

namespace migraphx {

template <typename T, index_int N>
struct uninitialized_buffer
{
    // Use aligned char storage to avoid initialization
    alignas(T) char storage[sizeof(T) * N];

    __device__ T* data() { return reinterpret_cast<T*>(storage); }

    __device__ const T* data() const { return reinterpret_cast<const T*>(storage); }

    // Array-like interface
    __device__ T& operator[](index_int i)
    {
        MIGRAPHX_ASSERT(i < N);
        return data()[i];
    }

    __device__ const T& operator[](index_int i) const
    {
        MIGRAPHX_ASSERT(i < N);
        return data()[i];
    }

    __device__ constexpr index_int size() const { return N; }

    // Iterator support (if needed)
    __device__ T* begin() { return data(); }
    __device__ T* end() { return data() + N; }
    __device__ const T* begin() const { return data(); }
    __device__ const T* end() const { return data() + N; }
};

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_UNINITIALIZED_BUFFER_HPP
