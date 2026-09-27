/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_KERNELS_TOPK_HPP
#define MIGRAPHX_GUARD_KERNELS_TOPK_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/bit.hpp>
#include <migraphx/kernels/ranges.hpp>
#include <migraphx/kernels/slice.hpp>
#include <migraphx/kernels/sort.hpp>
#include <migraphx/kernels/operators.hpp>
#include <migraphx/kernels/float_equal.hpp>

namespace migraphx {

constexpr auto select_key()
{
    return [](const auto& p) { return p.key; };
}

template <class T, class Type = typename T::type>
constexpr auto
    get_index_type(T) -> conditional_t<(sizeof(Type) < sizeof(index_int)), Type, index_int>;

constexpr auto get_index_type() -> uint16_t;

template <class... X>
constexpr auto make_get_index(X... x_idxs)
{
    if constexpr(sizeof...(x_idxs) == 1)
    {
        auto x_idx = arg_c<0>()(x_idxs...);
        return [=](auto i) { return i < x_idx.get_shape().elements() ? x_idx[i] : -1; };
    }
    else
    {
        return [](auto i) { return i; };
    }
}

template <index_int Axis, class Compare, class T, class Y, class YIndex, class X, class... XIndices>
__device__ void
topk_impl(index idx, Compare compare, T init, Y y, YIndex y_idx, X x, XIndices... x_idxs)
{
    using type       = typename X::type;
    constexpr auto n = _c<get_shape_c<X>{}.get_shape().lens[Axis]>;
    constexpr auto k = _c<get_shape_c<Y>{}.get_shape().lens[Axis]>;
    using pair =
        topk_pair<type, conditional_t<(n > 32768), index_int, decltype(get_index_type(x_idxs...))>>;
    auto get_index = make_get_index(x_idxs...);
    select_topk<k>(
        idx,
        compare,
        init,
        n,
        [&](auto j, auto) { return make_topk_pair<pair>(x[j], get_index(j)); },
        [&](auto i, auto, const pair& p) {
            y[i]     = p.key;
            y_idx[i] = p.val;
        });
}

template <index_int Axis, class Compare, class T>
__device__ auto topk(Compare compare, T init)
{
    return [=](auto output, auto out_indices, auto input, auto... in_indices) {
        auto idx = make_index();
        slice_schedule<per_block>(idx,
                                  slice_axes<Axis>())(output, out_indices, input, in_indices...)(
            [&](auto y, auto y_idx, auto x, auto... x_idxs) {
                topk_impl<Axis>(idx, compare, init, y, y_idx, x, x_idxs...);
            });
    };
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_TOPK_HPP
