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
 */
#ifndef MIGRAPHX_GUARD_KERNELS_CK_APPENDKV_HPP
#define MIGRAPHX_GUARD_KERNELS_CK_APPENDKV_HPP

#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/integral_constant.hpp>

namespace migraphx {

namespace detail {

template <class T>
struct appendkv_to_ck_type
{
    using type = T;
};
template <>
struct appendkv_to_ck_type<migraphx::half>
{
    using type = ck_tile::fp16_t;
};
template <class T>
struct appendkv_to_ck_type<const T>
{
    using type = const typename appendkv_to_ck_type<T>::type;
};

} // namespace detail

template <class T>
constexpr auto appendkv_ck_pointer(T* x)
{
    return reinterpret_cast<typename detail::appendkv_to_ck_type<T>::type*>(x);
}

template <class Tensor>
constexpr auto appendkv_ck_dims()
{
    constexpr auto s = get_shape_c<Tensor>{};
    return sequence(s.lens.size(), [&](auto... is) { return ck_tile::make_tuple(s.lens[is]...); });
}

template <class Tensor>
constexpr auto appendkv_ck_strides()
{
    constexpr auto s = get_shape_c<Tensor>{};
    return sequence(s.strides.size() - _c<1>,
                    [&](auto... is) { return ck_tile::make_tuple(s.strides[is]...); });
}

// Argument order: Q, K_cache, Knew, V_cache, Vnew, SeqlenK, (Cos, Sin)
// Without RoPE: 6 tensor_view arguments
// With RoPE:    8 tensor_view arguments
template <class G,
          index_int RotaryDim,
          bool HasMask,
          class Q,
          class K,
          class Knew,
          class V,
          class Vnew,
          class SeqlenK>
__device__ void ck_appendkv(Q q, K k, Knew knew, V v, Vnew vnew, SeqlenK seqlen_k)
{
    constexpr auto desc = G::make_descriptor(appendkv_ck_dims<Q>(),
                                             appendkv_ck_strides<Q>(),
                                             appendkv_ck_dims<K>(),
                                             appendkv_ck_strides<K>(),
                                             appendkv_ck_dims<Knew>(),
                                             appendkv_ck_strides<Knew>(),
                                             appendkv_ck_dims<V>(),
                                             appendkv_ck_strides<V>(),
                                             appendkv_ck_dims<Vnew>(),
                                             appendkv_ck_strides<Vnew>(),
                                             RotaryDim,
                                             HasMask);

    static_assert(desc.IsValid(), "Invalid AppendKV kernel configuration");

    using DataType = typename G::DataType;
    G::Run(desc,
           appendkv_ck_pointer(q.data()),
           appendkv_ck_pointer(k.data()),
           appendkv_ck_pointer(knew.data()),
           appendkv_ck_pointer(v.data()),
           appendkv_ck_pointer(vnew.data()),
           seqlen_k.data(),
           static_cast<const DataType*>(nullptr),
           static_cast<const DataType*>(nullptr),
           nullptr);
}

template <class G,
          index_int RotaryDim,
          bool HasMask,
          class Q,
          class K,
          class Knew,
          class V,
          class Vnew,
          class SeqlenK,
          class Cos,
          class Sin>
__device__ void ck_appendkv(Q q, K k, Knew knew, V v, Vnew vnew, SeqlenK seqlen_k, Cos cos, Sin sin)
{
    constexpr auto desc = G::make_descriptor(appendkv_ck_dims<Q>(),
                                             appendkv_ck_strides<Q>(),
                                             appendkv_ck_dims<K>(),
                                             appendkv_ck_strides<K>(),
                                             appendkv_ck_dims<Knew>(),
                                             appendkv_ck_strides<Knew>(),
                                             appendkv_ck_dims<V>(),
                                             appendkv_ck_strides<V>(),
                                             appendkv_ck_dims<Vnew>(),
                                             appendkv_ck_strides<Vnew>(),
                                             RotaryDim,
                                             HasMask);

    static_assert(desc.IsValid(), "Invalid AppendKV kernel configuration");

    G::Run(desc,
           appendkv_ck_pointer(q.data()),
           appendkv_ck_pointer(k.data()),
           appendkv_ck_pointer(knew.data()),
           appendkv_ck_pointer(v.data()),
           appendkv_ck_pointer(vnew.data()),
           seqlen_k.data(),
           appendkv_ck_pointer(cos.data()),
           appendkv_ck_pointer(sin.data()),
           nullptr);
}

} // namespace migraphx
#endif
