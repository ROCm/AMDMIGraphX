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

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/tensor_view.hpp>

#ifndef MIGRAPHX_GUARD_KERNELS_CONCAT_HPP
#define MIGRAPHX_GUARD_KERNELS_CONCAT_HPP

namespace migraphx {

#define MIGRAPHX_CONCAT_GROUP_CASE(x) \
    case x: { \
        if constexpr(x < Max) \
            return f(_c<x>);\
        break; \
    }
template<index_int Max, class I, class F>
constexpr auto visit_concat_group(I i, F f)
{
#if 1
    if constexpr(Max > 0)
    {
        if(i == (Max - 1)) 
            return f(_c<Max - 1>);
        if constexpr(Max > 1)
            return visit_concat_group<Max - 1>(i, f);
    }
#else
    static_assert(Max <= 8);
    MIGRAPHX_ASSUME(i < Max);
    switch(i)
    {
        MIGRAPHX_CONCAT_GROUP_CASE(0)
        MIGRAPHX_CONCAT_GROUP_CASE(1)
        MIGRAPHX_CONCAT_GROUP_CASE(2)
        MIGRAPHX_CONCAT_GROUP_CASE(3)
        MIGRAPHX_CONCAT_GROUP_CASE(4)
        MIGRAPHX_CONCAT_GROUP_CASE(5)
        MIGRAPHX_CONCAT_GROUP_CASE(6)
        MIGRAPHX_CONCAT_GROUP_CASE(7)
    }
#endif
    MIGRAPHX_UNREACHABLE();
}

template<class I, class... InputPacks>
constexpr auto visit_concat_pack(I i, InputPacks... input_packs)
{
    return [=](auto f) {
        return visit_concat_group<sizeof...(InputPacks)>(i, [&](auto x) {
            return f(arg(x)(input_packs...));
        });
    };
}

template <index_int Axis, class Output, class Start>
constexpr auto concat_offset(Output, Start start)
{
    if constexpr(is_integral<Start>{})
        return start * get_shape_c<Output>{}.strides[Axis];
    else
        return return_c([] { return Start{} * get_shape_c<Output>{}.strides[Axis]; });
}

template <index_int Axis, class Output, class Input, class Start>
constexpr auto concat_slice(Output out, Input, Start start)
{
    constexpr auto lens    = get_shape_c<Input>{}.lens;
    constexpr auto strides = get_shape_c<Output>{}.strides;
    auto offset = concat_offset<Axis>(out, start);
    constexpr auto s       = make_shape(lens, strides);
    MIGRAPHX_ASSUME(offset < out.get_shape().element_space());
    MIGRAPHX_ASSUME((s.element_space() + offset) <= out.get_shape().element_space());
    return make_tensor_view(out.data() + offset, s);
}

template <index_int Axis, class Input, class Start, class... Ts>
constexpr auto concat_slices(Input input, Start start, Ts... xs)
{
    return [=](auto f) { return f(concat_slice<Axis>(xs, input, start)...); };
}

template <index_int Axis, class Input>
constexpr auto concat_ends(Input)
{
    constexpr auto lens = get_shape_c<Input>{}.lens;
    return _c<lens[Axis]>;
}

template <index_int Axis, class N, class InputPack, class... InputPacks>
constexpr auto concat_starts(N n, InputPack input_pack, InputPacks... input_packs)
{
    static_assert(n <= sizeof...(InputPacks));
    if constexpr(n == 0) {
        return _c<0>;
    }
    else {
        return input_pack([&](auto, auto x, auto...) {
            return concat_starts<Axis>(n - _c<1>, input_packs...) +  concat_ends<Axis>(x);
        });        
    }
}

// template <index_int Axis>
// constexpr index_int concat_starts(index_int)
// {
//     MIGRAPHX_UNREACHABLE();
//     return 0;
// }

// template <index_int Axis, class InputPack, class... InputPacks>
// constexpr index_int concat_starts(index_int n, InputPack input_pack, InputPacks... input_packs)
// {
//     MIGRAPHX_ASSUME(n <= sizeof...(InputPacks));
//     if (n == 0)
//         return 0;
//     return input_pack([&](auto, auto x, auto...) {
//         return concat_starts<Axis>(n - 1, input_packs...) +  concat_ends<Axis>(x);
//     });        
// }

template <index_int Axis, class... InputPacks>
constexpr auto concat_elements(index_int n, InputPacks... input_packs)
{
    return visit_concat_group<sizeof...(InputPacks)>(n, [&](auto i) {
        return arg(i)(input_packs...)([&](auto, auto x, auto...) -> index_int {
            return x.get_shape().elements();
        });
    });
            
}

template<class... InputPacks>
constexpr auto concat_pack_x(index_int n, InputPacks... input_packs)
{
    return [=](auto i) {
        return visit_concat_group<sizeof...(InputPacks)>(n, [&](auto j) {
            return arg(j)(input_packs...)([&](auto g, auto... xs) {
                return g(xs[i]...);
            });
        });
    };
}

template<class... InputPacks>
constexpr auto concat_visit_x(index_int n, InputPacks... input_packs)
{
    return [=](auto f) {
        return visit_concat_group<sizeof...(InputPacks)>(n, [&](auto j) {
            return arg(j)(input_packs...)([&](auto g, auto... xs) {
                return f(g, xs...);
            });
        });
    };
}

template<index_int Axis, class... InputPacks>
constexpr auto concat_slice_y(index_int n, InputPacks... input_packs)
{
    return [=](auto y) {
        return visit_concat_group<sizeof...(InputPacks)>(n, [&](auto j) {
            auto start = concat_starts<Axis>(j, input_packs...);
            auto input_pack = arg(j)(input_packs...);
            return input_pack([&](auto, auto x, auto...) {
                return concat_slice<Axis>(y, x, start);
            });
        });
    };
}

template <index_int Axis, class ForStride, class Start, class InputPack, class F, class... Ts>
__device__ auto concat_each(ForStride for_stride, Start start, InputPack input_pack, F f, Ts... ts)
{
    return input_pack([&](auto g, auto x, auto... xs) {
        return concat_slices<Axis>(x, start, ts...)([&](auto z, auto... ys) {
            for_stride(x.get_shape().elements(),
                              [&](auto i) { z[i] = f(g(x[i], xs[i]...), ys[i]...); });

            return start + concat_ends<Axis>(x);
        });
    });
}

template <index_int Axis, class... InputPacks>
__device__ auto concat(InputPacks... input_packs)
{
    return [=](auto f, auto... ts) {
        auto idx = make_index();
        fold([&](auto start, auto input_pack) {
            auto fs = [&](auto n, auto g) { idx.global_stride(n, g); };
            return concat_each<Axis>(fs, start, input_pack, f, ts...);
        })(_c<0>, input_packs...);
    };
}

template <index_int Axis, class... InputPacks>
__device__ auto par_concat(InputPacks... input_packs)
{
#if 0
    return [=](auto f, auto z, auto... ys) {
        auto idx = make_index();
        auto ninputs = _c<sizeof...(input_packs)>;
        auto nteams = idx.ngroup() / ninputs;
        auto team = idx.team<nteams>();
        // auto start = concat_starts<Axis>(team, input_packs...);
        // auto start = visit_concat_group<ninputs>(team, [&](auto i) -> index_int {
        //     return concat_starts<Axis>(i, input_packs...);
        // });
        auto x = concat_pack_x(team, input_packs...);
        auto slice = concat_slice_y<Axis>(team, input_packs...);

        idx.local_team_stride<nteams>(concat_elements<Axis>(team, input_packs...), [&](auto i) {
            // slice(z)[i] = concat_visit_x(team, input_packs...)([&](auto g, auto... xs) {
            //     return f(g(xs[i]...), slice(ys)[i]...);
            // });
            slice(z)[i] = f(x(i), slice(ys)[i]...);
        });
    };
#else
    return [=](auto f, auto... ts) {
        auto idx = make_index();
        auto ninputs = _c<sizeof...(input_packs)>;
        auto nteams = idx.ngroup() / ninputs;
        auto team = idx.team<nteams>();
        visit_concat_group<ninputs>(team, [&](auto gidx) {
            auto start = concat_starts<Axis>(gidx, input_packs...);
            auto fs = [&](auto n, auto g) { idx.local_team_stride<nteams>(n, g); };
            concat_each<Axis>(fs, start, arg(gidx)(input_packs...), f, ts...);
        });
    };
#endif
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_CONCAT_HPP
