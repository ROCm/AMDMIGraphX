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
#ifndef MIGRAPHX_GUARD_KERNELS_CK_FMHA_FWD_HPP
#define MIGRAPHX_GUARD_KERNELS_CK_FMHA_FWD_HPP

#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/integral_constant.hpp>

namespace migraphx {

namespace detail {

template <class T>
struct to_ck_tile_type_impl
{
    using type = T;
};
template <>
struct to_ck_tile_type_impl<migraphx::half>
{
    using type = ck_tile::fp16_t;
};
template <class T>
struct to_ck_tile_type_impl<const T>
{
    using type = const typename to_ck_tile_type_impl<T>::type;
};

} // namespace detail

template <class T>
using to_ck_tile_type = typename detail::to_ck_tile_type_impl<T>::type;

template <class T>
constexpr auto to_ck_tile_pointer(T* x)
{
    return reinterpret_cast<to_ck_tile_type<T>*>(x);
}

template <class Tensor>
constexpr auto to_ck_tile_dims()
{
    constexpr auto s = get_shape_c<Tensor>{};
    return sequence(s.lens.size(), [&](auto... is) { return ck_tile::make_tuple(s.lens[is]...); });
}

// Extract first N-1 strides from a tensor type as a ck_tile::make_tuple
// (innermost stride = 1 is implicit for the FMHA descriptor)
template <class Tensor>
constexpr auto to_ck_tile_strides()
{
    constexpr auto s = get_shape_c<Tensor>{};
    return sequence(s.strides.size() - _c<1>,
                    [&](auto... is) { return ck_tile::make_tuple(s.strides[is]...); });
}

// No bias overload
template <class G, class O, class Q, class K, class V>
__device__ void ck_fmha_fwd(O o, Q q, K k, V v)
{
    constexpr auto desc = G::make_descriptor(to_ck_tile_dims<Q>(),
                                             to_ck_tile_strides<Q>(),
                                             to_ck_tile_dims<K>(),
                                             to_ck_tile_strides<K>(),
                                             to_ck_tile_dims<V>(),
                                             to_ck_tile_strides<V>(),
                                             to_ck_tile_dims<O>(),
                                             to_ck_tile_strides<O>(),
                                             ck_tile::make_tuple(0, 0, 0, 0),
                                             ck_tile::make_tuple(0, 0, 0));

    static_assert(desc.IsValid(), "Invalid FMHA kernel configuration");

    G::Run(desc,
           float{SCALE},
           to_ck_tile_pointer(q.data()),
           to_ck_tile_pointer(k.data()),
           to_ck_tile_pointer(v.data()),
           nullptr,
           to_ck_tile_pointer(o.data()));
}

template <class G, class O, class Q, class K, class Bias, class V>
__device__ void ck_fmha_fwd(O o, Q q, K k, Bias bias, V v)
{
    constexpr auto desc = G::make_descriptor(to_ck_tile_dims<Q>(),
                                             to_ck_tile_strides<Q>(),
                                             to_ck_tile_dims<K>(),
                                             to_ck_tile_strides<K>(),
                                             to_ck_tile_dims<V>(),
                                             to_ck_tile_strides<V>(),
                                             to_ck_tile_dims<O>(),
                                             to_ck_tile_strides<O>(),
                                             to_ck_tile_dims<Bias>(),
                                             to_ck_tile_strides<Bias>());

    static_assert(desc.IsValid(), "Invalid FMHA kernel configuration");

    G::Run(desc,
           float{SCALE},
           to_ck_tile_pointer(q.data()),
           to_ck_tile_pointer(k.data()),
           to_ck_tile_pointer(v.data()),
           to_ck_tile_pointer(bias.data()),
           to_ck_tile_pointer(o.data()));
}

} // namespace migraphx
#endif
