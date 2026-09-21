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
 *
 */
#ifndef MIGRAPHX_GUARD_KERNELS_NONTEMPORAL_HPP
#define MIGRAPHX_GUARD_KERNELS_NONTEMPORAL_HPP

#include <migraphx/kernels/type_traits.hpp>
#include <migraphx/kernels/vec.hpp>
#include <migraphx/kernels/bit_cast.hpp>
#include <migraphx/kernels/tensor_view.hpp>

// Set to 0 (via MIGRAPHX_GPU_DISABLE_NONTEMPORAL_LOADS) to fall back to cached loads.
#ifndef MIGRAPHX_NONTEMPORAL_LOADS
#define MIGRAPHX_NONTEMPORAL_LOADS 1
#endif

namespace migraphx {

// Load with a nontemporal (streaming) hint: the value is not expected to be reused.
// The builtin only accepts arithmetic and vector types, so struct element types (eg fp8)
// are loaded through a same-sized unsigned integer and bit-cast back.
template <class T>
__device__ T nontemporal_load(const T* ptr)
{
#if MIGRAPHX_NONTEMPORAL_LOADS
    if constexpr(is_integral<T>{} or is_floating_point<T>{} or is_any_vec<T>())
    {
        return __builtin_nontemporal_load(ptr);
    }
    else
    {
        using storage = sized_uint_t<sizeof(T)>;
        static_assert(alignof(T) >= alignof(storage));
        return bit_cast<T>(__builtin_nontemporal_load(reinterpret_cast<const storage*>(ptr)));
    }
#else
    return *ptr;
#endif
}

// Non-broadcasted inputs are streamed with no reuse expected, so use a nontemporal load.
// Broadcasted inputs are re-read by other threads or workgroups, so keep a cached load.
template <class T, class I>
__device__ auto stream_load(const T& x, I i)
{
    if constexpr(get_shape_c<T>{}.broadcasted())
        return x[i];
    else
        return nontemporal_load(&x[i]);
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_NONTEMPORAL_HPP
