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
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/vec.hpp>
#include <migraphx/kernels/bit_cast.hpp>
#include <migraphx/kernels/tensor_view.hpp>

// Set to 0 (via MIGRAPHX_GPU_DISABLE_NONTEMPORAL_LOADS) to fall back to cached loads.
#ifndef MIGRAPHX_NONTEMPORAL_LOADS
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
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

// Non-broadcasted global inputs are streamed with no reuse expected, so use a nontemporal
// load. Broadcasted inputs are re-read by other threads or workgroups, and LDS tiles have
// no cache to bypass, so those keep a plain load.
template <class T, class I>
__device__ auto stream_load(const T& x, I i)
{
    if constexpr(is_same<typename T::memory_tag, lds_memory_tag>{} or
                 get_shape_c<T>{}.broadcasted())
        return x[i];
    else
        return nontemporal_load(&x[i]);
}

// Overload for raw pointers where the shape has been erased; the caller is responsible
// for only streaming data with no reuse.
template <class T, class I>
__device__ T stream_load(const T* x, I i)
{
    return nontemporal_load(x + i);
}

// The position of the element at p within its block of S elements
template <index_int S, class T>
__device__ index_int block_phase(const T* p)
{
    return (bit_cast<uintptr_t>(p) / sizeof(T)) % S;
}

// Read the N elements S apart at p as one vector: the aligned block of N*S
// elements holding them is loaded whole and the lanes at the phase of p
// within the block are selected, so the load stays wide and aligned. The
// block is aligned only when p sits in its first S elements, which the
// views of the fusions guarantee; a view offset past that reads its lanes
// one at a time
template <bool Stream, class T, index_int N, index_int S>
__device__ vec<T, N> load_strided(const strided_vec<T, N, S>* p)
{
    using block_type  = vec<T, N * S>;
    const T* elements = p->data;
    index_int phase   = block_phase<N * S>(elements);
    if(phase >= S)
    {
        return generate_vec(_c<N>, [&](auto i) {
            constexpr index_int lane = decltype(i){} * S;
            if constexpr(Stream)
                return nontemporal_load(elements + lane);
            else
                return elements[lane];
        });
    }
    const auto* block = as_vec<N * S>(elements - phase);
    MIGRAPHX_ASSERT(bit_cast<uintptr_t>(block) % alignof(block_type) == 0);
    block_type v;

    if constexpr(Stream)
        v = nontemporal_load(block);
    else
        v = *block;
    vec<T, N> result = {0};
    repeat_c<S>([&](auto s) {
        if(phase == s)
        {
            result = generate_vec(_c<N>, [&](auto i) {
                constexpr index_int lane = decltype(s){} + decltype(i){} * S;
                return v[lane];
            });
        }
    });
    return result;
}

// Read element i of the input view: a strided vector element is read as the
// vector of its lanes, any other element through stream_load

template <class T, class I>
__device__ auto load_element(const T& x, I i)
{
    using type = remove_cv_t<typename T::type>;
    if constexpr(is_strided_vec<type>{})
    {
        constexpr bool stream = not(is_same<typename T::memory_tag, lds_memory_tag>{} or
                                    get_shape_c<T>{}.broadcasted());
        return load_strided<stream>(&x[i]);
    }
    else
    {
        return stream_load(x, i);
    }
}

// Function objects selecting the load used when copying a tensor
struct cached_load
{
    template <class T, class I>
    constexpr auto operator()(const T& x, I i) const
    {
        return x[i];
    }
};

MIGRAPHX_LIFT_CLASS(streaming_load, stream_load);

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_NONTEMPORAL_HPP
