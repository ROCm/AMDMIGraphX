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
#ifndef MIGRAPHX_GUARD_KERNELS_CK_SPLITKV_HPP
#define MIGRAPHX_GUARD_KERNELS_CK_SPLITKV_HPP

#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/integral_constant.hpp>

namespace migraphx {

namespace detail {

template <class T>
struct splitkv_to_ck_type
{
    using type = T;
};
template <>
struct splitkv_to_ck_type<migraphx::half>
{
    using type = ck_tile::fp16_t;
};
template <class T>
struct splitkv_to_ck_type<const T>
{
    using type = const typename splitkv_to_ck_type<T>::type;
};

} // namespace detail

template <class T>
constexpr auto splitkv_ck_pointer(T* x)
{
    return reinterpret_cast<typename detail::splitkv_to_ck_type<T>::type*>(x);
}

template <class Tensor>
constexpr auto splitkv_ck_dims()
{
    constexpr auto s = get_shape_c<Tensor>{};
    return sequence(s.lens.size(), [&](auto... is) { return ck_tile::make_tuple(s.lens[is]...); });
}

template <class Tensor>
constexpr auto splitkv_ck_strides()
{
    constexpr auto s = get_shape_c<Tensor>{};
    return sequence(s.strides.size() - _c<1>,
                    [&](auto... is) { return ck_tile::make_tuple(s.strides[is]...); });
}

// Tensor view argument order after rotate_last<2>(): o_acc, lse_acc, q, k, v
//
// make_descriptor takes 9 tuple args:
//   Q dims, Q strides, K dims, K strides, V dims, V strides,
//   OAcc dims, OAcc strides, LseAcc strides
//
// Run pointer order: q, k, v, lse_acc, o_acc
//
// LseAcc is 4D [batch, nhead, num_splits, M] with 3 strides (M stride is implicit).
// OAcc is 5D [batch, nhead, num_splits, M, O] with 4 strides (O stride is implicit).
// Q/K/V are 4D with 3 strides each (innermost stride is implicit).
template <class G, class OAcc, class LseAcc, class Q, class K, class V>
__device__ void ck_splitkv(OAcc o_acc, LseAcc lse_acc, Q q, K k, V v)
{
    constexpr auto desc = G::make_descriptor(splitkv_ck_dims<Q>(),
                                             splitkv_ck_strides<Q>(),
                                             splitkv_ck_dims<K>(),
                                             splitkv_ck_strides<K>(),
                                             splitkv_ck_dims<V>(),
                                             splitkv_ck_strides<V>(),
                                             splitkv_ck_dims<OAcc>(),
                                             splitkv_ck_strides<OAcc>(),
                                             splitkv_ck_strides<LseAcc>());

    static_assert(desc.IsValid(), "Invalid SplitKV kernel configuration");

    G::Run(desc,
           float{MIGRAPHX_CK_SPLITKV_SCALE},
           splitkv_ck_pointer(q.data()),
           splitkv_ck_pointer(k.data()),
           splitkv_ck_pointer(v.data()),
           splitkv_ck_pointer(lse_acc.data()),
           splitkv_ck_pointer(o_acc.data()));
}

} // namespace migraphx
#endif
