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
#ifndef MIGRAPHX_GUARD_KERNELS_POINTWISE_HPP
#define MIGRAPHX_GUARD_KERNELS_POINTWISE_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/preload.hpp>
#include <migraphx/kernels/vectorize.hpp>
#include <migraphx/kernels/args.hpp>
#include <migraphx/kernels/tile.hpp>
#include <migraphx/kernels/tuple.hpp>
#include <migraphx/kernels/bit_cast.hpp>

namespace migraphx {

// Unsigned integer with the same size as T, used to move struct element types (eg fp8)
// through a builtin that only accepts arithmetic and vector types.
template <class T>
using nontemporal_storage =
    conditional_t<sizeof(T) == 1,
                  uint8_t,
                  conditional_t<sizeof(T) == 2,
                                uint16_t,
                                conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>>;

// Load a single value with a nontemporal hint so it bypasses the cache. The builtin only
// accepts arithmetic and vector types, so any other trivially-copyable type is loaded
// through a same-sized integer and bit-cast back.
template <class T>
__device__ T nontemporal_load(const T* ptr)
{
    if constexpr(is_integral<T>{} or is_floating_point<T>{} or is_any_vec<T>())
    {
        return __builtin_nontemporal_load(ptr);
    }
    else
    {
        static_assert(is_trivially_copyable<T>{});
        using storage = nontemporal_storage<T>;
        static_assert(sizeof(storage) == sizeof(T));
        return bit_cast<T>(__builtin_nontemporal_load(reinterpret_cast<const storage*>(ptr)));
    }
}

// Read an element from an input tensor. Inputs that are not broadcasted are read only
// once, so a nontemporal load avoids polluting the cache. Broadcasted inputs reuse the
// same element across threads, so a regular cached load is kept for them.
template <class T, class I>
__device__ auto pointwise_load(const T& x, I i)
{
#if 0
    if constexpr(get_shape_c<T>{}.broadcasted())
        return x[i];
    else
        return nontemporal_load(&x[i]);
#else
    return x[i];
#endif
}

template <class Stride, class F, class Output, class T, class... Ts>
__device__ void pointwise_tensor(Stride stride, F f, Output out, T x, Ts... xs)
{
    stride(x.get_shape().elements(), [&](auto i) {
        auto r = f(pointwise_load(x, i), pointwise_load(xs, i)...);
        out([&](auto... outs) {
            r([&](auto... rs) {
                static_assert(sizeof...(outs) == sizeof...(rs));
                swallow{(outs[i] = implicit_conversion(rs))...};
            });
        });
    });
}

template <index_int N, bool Tiled, class... Transforms>
__device__ auto pointwise(index idx, Transforms... transforms)
{
    return [=](auto f, auto*... ps) {
        auto t = transform_args(make_tensors(), transforms..., rotate_and_pack_last<N>());
        t(ps...)([&](auto... xs) { pointwise_tensor(tile_stride<Tiled>(idx), f, xs...); });
    };
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_POINTWISE_HPP
