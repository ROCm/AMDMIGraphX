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

#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/slice.hpp>
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/uninitialized_buffer.hpp>

#ifndef MIGRAPHX_GUARD_KERNELS_CONCAT_HPP
#define MIGRAPHX_GUARD_KERNELS_CONCAT_HPP

namespace migraphx {

namespace concat {
template <index_int Axis, class Output, class Input, class Start>
constexpr auto concat_slice(Output out, Input, Start)
{
    constexpr auto lens    = get_shape_c<Input>{}.lens;
    constexpr auto strides = get_shape_c<Output>{}.strides;
    constexpr auto offset  = return_c([] {
        constexpr auto output_shape = get_shape_c<Output>{};
        return Start{} * output_shape.strides[Axis];
    });
    constexpr auto s       = make_shape(lens, strides);
    MIGRAPHX_ASSERT(offset < out.get_shape().element_space());
    MIGRAPHX_ASSERT((s.element_space() + offset) <= out.get_shape().element_space());
    return make_tensor_view(out.data() + offset, s);
}

template <index_int Axis, class Input, class Start, class... Ts>
constexpr auto concat_slices(Input input, Start start, Ts... xs)
{
    return [=](auto f) { return f(concat_slice<Axis>(xs, input, start)...); };
}

// Extracts the input from a pack of (op, input, extra inputs...)
struct get_input
{
    template <class G, class X, class... Xs>
    constexpr auto operator()(G, X x, Xs...) const
    {
        return x;
    }
};

// Extent along Axis of the input carried by a pack, from its type alone
template <index_int Axis, class InputPack>
constexpr index_int pack_end()
{
    using input = decltype(declval<InputPack>()(get_input{}));
    return get_shape_c<input>{}.lens[Axis];
}

// Start offset of each input along the concat axis: an exclusive prefix sum of the extents
template <index_int N>
constexpr array<index_int, N> concat_offsets(const array<index_int, N>& ends)
{
    array<index_int, N> result{};
    partial_sum(ends.begin(), ends.end() - 1, result.begin() + 1);
    return result;
}

// Per-input extents, start offsets and the largest extent along Axis, computed once as static
// constants so each input is a flat lookup. A recursive fold over the packs instead made compile
// time quadratic in the number of inputs and hit the template depth limit near 1024 of them.
template <index_int Axis, class... InputPacks>
struct layout
{
    static constexpr auto ends =
        array<index_int, sizeof...(InputPacks)>{pack_end<Axis, InputPacks>()...};
    static constexpr auto offsets  = concat_offsets(ends);
    static constexpr auto max_size = accumulate(
        ends.begin(), ends.end(), index_int{0}, [](index_int a, index_int b) { return max(a, b); });
};

template <class Axis, class NArgs, class MaxSize>
struct info
{
    Axis axis;
    NArgs nargs;
    MaxSize max_size;
};
MIGRAPHX_AUTO_DEDUCE(info);

template <class R>
struct basic_algo
{
    R r;

    template <class... Ts>
    constexpr auto run(Ts... xs)
    {
        return r(xs...);
    }

    template <class... Ts>
    constexpr void finish(Ts...) const
    {
    }
};
MIGRAPHX_AUTO_DEDUCE(basic_algo);

struct simple
{
    template <class Info, class Output>
    static __device__ auto make(index idx, Info, Output)
    {
        return basic_algo{[=](auto, auto g, auto x, auto... xs) {
            return [=](auto z, auto f, auto... ys) {
                idx.global_stride(x.get_shape().elements(),
                                  [&](auto i) { z[i] = f(g(x[i], xs[i]...), ys[i]...); });
            };
        }};
    }
};

template <index_int NGroups>
struct block_tile
{
    template <class T, index_int N, index_int MaxSize>
    struct algo
    {
        constexpr auto slice() const
        {
            return slice_schedule<per_block>(idx, slice_axes<-1>(), slice_group<NGroups>());
        }

        static __device__ auto output_data()
        {
            constexpr auto s = make_shape(index_ints<NGroups, N, MaxSize>{});
            __shared__ uninitialized_buffer<T, s.element_space()> storage;
            return make_tensor_view(storage.data(), s);
        }

        index idx;

        template <class Depth, class G, class... Xs>
        __device__ void run(Depth depth, G g, Xs... xs)
        {
            auto output = output_data();
            slice()(xs...)([&](auto w, auto... ws) {
                MIGRAPHX_ASSERT(w.get_shape().lens.back() == MaxSize);
                // slices are enumerated group-major, so view the slice as {group, k}
                constexpr auto gshape = make_shape(index_ints<NGroups, MaxSize>{});
                MIGRAPHX_ASSERT(w.get_shape().elements() == gshape.elements());
                idx.local_stride(gshape.elements(), [&](auto i) {
                    auto m                      = gshape.multi(i);
                    output[{m[0], depth, m[1]}] = g(w[i], ws[i]...);
                });
            });
        }

        template <class F, class... Outputs>
        __device__ void finish(F f, Outputs... outputs) const
        {
            __syncthreads();
            auto output = output_data();
            slice()(outputs...)([&](auto z, auto... ys) {
                MIGRAPHX_ASSERT(z.get_shape().lens.back() == N * MaxSize);
                MIGRAPHX_ASSERT(z.get_shape().elements() == output.get_shape().elements());
                block_stride<per_block, 8>(idx, z.get_shape().elements())(
                    [&](auto i) { z[i] = f(output[i], ys[i]...); });
            });
        }
    };

    template <class Info, class Output>
    static __device__ auto make(index idx, Info info, Output)
    {
        MIGRAPHX_ASSERT(info.axis == get_shape_c<Output>{}.lens.size() - 1);
        return algo<typename Output::type, info.nargs, info.max_size>{idx};
    }
};

// Writes one input's slice of the output; depth is the input's index among the packs
template <index_int Axis,
          class Algo,
          class F,
          class Depth,
          class Start,
          class InputPack,
          class T,
          class... Ts>
__device__ void
run_pack(Algo& algo, F f, Depth depth, Start start, InputPack input_pack, T t, Ts... ts)
{
    input_pack([&](auto g, auto x, auto... xs) {
        concat_slices<Axis>(x, start, t, ts...)([&](auto z, auto... ys) {
            if constexpr(is_void<decltype(algo.run(depth, g, x, xs...))>{})
                algo.run(depth, g, x, xs...);
            else
                algo.run(depth, g, x, xs...)(z, f, ys...);
        });
    });
}

template <class Algo, index_int Axis, class... InputPacks>
__device__ auto run(InputPacks... input_packs)
{
    return [=](auto f, auto t, auto... ts) {
        using layout_t = layout<Axis, InputPacks...>;
        auto idx       = make_index();
        auto algo      = Algo::make(idx,
                                    info{.axis     = _c<Axis>,
                                         .nargs    = _c<sizeof...(InputPacks)>,
                                         .max_size = _c<layout_t::max_size>},
                                    t);
        sequence_c<sizeof...(InputPacks)>([&](auto... is) {
            (run_pack<Axis>(algo, f, is, _c<layout_t::offsets[is]>, input_packs, t, ts...), ...);
        });
        algo.finish(f, t, ts...);
    };
}
} // namespace concat

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_CONCAT_HPP
