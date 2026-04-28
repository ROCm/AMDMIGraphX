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
#ifndef MIGRAPHX_GUARD_KERNELS_CK_SPLITKV_COMBINE_HPP
#define MIGRAPHX_GUARD_KERNELS_CK_SPLITKV_COMBINE_HPP

#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/integral_constant.hpp>

namespace migraphx {

namespace detail {

template <class T>
struct combine_to_ck_type
{
    using type = T;
};
template <>
struct combine_to_ck_type<migraphx::half>
{
    using type = ck_tile::fp16_t;
};
template <class T>
struct combine_to_ck_type<const T>
{
    using type = const typename combine_to_ck_type<T>::type;
};

} // namespace detail

template <class T>
constexpr auto combine_ck_pointer(T* x)
{
    return reinterpret_cast<typename detail::combine_to_ck_type<T>::type*>(x);
}

template <class Tensor>
constexpr auto combine_ck_strides()
{
    constexpr auto s = get_shape_c<Tensor>{};
    return sequence(s.strides.size() - _c<1>,
                    [&](auto... is) { return ck_tile::make_tuple(s.strides[is]...); });
}

// Tensor view argument order after rotate_last<1>(): o, o_acc, lse_acc
//
// make_descriptor takes:
//   batch, nhead, seqlen_q, hdim_v, num_splits,
//   lse_acc_strides (3 elements: batch, nhead, split),
//   o_acc_strides   (4 elements: batch, nhead, split, m),
//   o_strides       (3 elements: batch, nhead, m)
//
// Run pointer order: lse_acc, o_acc, o
//
// O is 4D [batch, nhead, M, O] with 3 strides (O stride is implicit).
// OAcc is 5D [batch, nhead, num_splits, M, O] with 4 strides (O stride is implicit).
// LseAcc is 4D [batch, nhead, num_splits, M] with 3 strides (M stride is implicit).
template <class G, class O, class OAcc, class LseAcc>
__device__ void ck_splitkv_combine(O o, OAcc o_acc, LseAcc lse_acc)
{
    constexpr auto o_shape     = get_shape_c<O>{};
    constexpr auto o_acc_shape = get_shape_c<OAcc>{};

    constexpr auto desc = G::make_descriptor(
        o_shape.lens[_c<0>],                  // batch
        o_shape.lens[_c<1>],                  // nhead
        o_shape.lens[_c<2>],                  // seqlen_q (M)
        o_shape.lens[_c<3>],                  // hdim_v (O)
        o_acc_shape.lens[_c<2>],              // num_splits
        combine_ck_strides<LseAcc>(),         // lse_acc strides
        combine_ck_strides<OAcc>(),           // o_acc strides
        combine_ck_strides<O>());             // o strides

    static_assert(desc.IsValid(), "Invalid SplitKV Combine kernel configuration");

    G::Run(desc,
           combine_ck_pointer(lse_acc.data()),
           combine_ck_pointer(o_acc.data()),
           combine_ck_pointer(o.data()));
}

} // namespace migraphx
#endif
