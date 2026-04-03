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
#include <migraphx/kernels/slice.hpp>
#include <migraphx/kernels/dpp.hpp>
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/ops.hpp>

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
 
template <index_int Axis, class Input>
constexpr auto concat_ends(Input)
{
    constexpr auto lens = get_shape_c<Input>{}.lens;
    return _c<lens[Axis]>;
}

template <index_int Axis, class... InputPacks>
constexpr auto concat_max(InputPacks... input_packs)
{
    return fold([&](auto start, auto input_pack) {
        return input_pack([&](auto, auto x, auto...) { return max(start, concat_ends<Axis>(x)); });
    })(_c<0>, input_packs...);
}

template<class InputPack, class...>
struct get_base_type
{
    static constexpr auto apply(InputPack input_pack)
    {
        return input_pack([&](auto g, auto... xs) {
            return g(xs[0]...);
        });
    }
    using type = decltype(declval<InputPack>());
};

template<class T, class U>
struct concat_pair
{
    T offset;
    U depth;
};
MIGRAPHX_AUTO_DEDUCE(concat_pair);

template <class Axis, class NArgs, class MaxSize>
struct info
{
    Axis axis;
    NArgs nargs;
    MaxSize max_size;
};
MIGRAPHX_AUTO_DEDUCE(info);

template <index_int Axis, class Start, class InputPack, class F, class... Ts>
__device__ auto concat_each(index idx, Start start, InputPack input_pack, F f, Ts... ts)
{
    return input_pack([&](auto g, auto x, auto... xs) {
        return concat_slices<Axis>(x, start, ts...)([&](auto z, auto... ys) {
            idx.global_stride(x.get_shape().elements(),
                              [&](auto i) { z[i] = f(g(x[i], xs[i]...), ys[i]...); });

            return start + concat_ends<Axis>(x);
        });
    });
}

template<class R>
struct basic_algo
{
    R r;

    template<class... Ts>
    constexpr auto run(Ts... xs)
    {
        return r(xs...);
    }

    template<class... Ts>
    constexpr void finish(Ts...) const
    {}

};
MIGRAPHX_AUTO_DEDUCE(basic_algo);

struct simple
{
    template<class T, class Info, class Output>
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
    template <class T, class Output, index_int N, index_int MaxSize>
    struct algo
    {
        constexpr auto slice() const
        {
            return slice_schedule<single_group<per_block>>(
                idx, slice_axes<-1>(), slice_group<NGroups>());
        }

        static __device__ auto output_data()
        {
            constexpr auto s = make_shape(index_ints<NGroups, N, MaxSize>{});
            // constexpr auto stride = ceil_div(MaxSize, MIGRAPHX_WAVEFRONTSIZE) *
            // MIGRAPHX_WAVEFRONTSIZE; constexpr auto s = make_shape(
            //     index_ints<NGroups, N, MaxSize>{},
            //     index_ints<N * stride,
            //                 stride,
            //                1>{});
            __shared__ T storage[s.element_space()];
            return make_tensor_view(storage, s);
        }

        template <class Array>
        static constexpr index_int compute_group(Array a)
        {
            return accumulate(a.begin(), a.end() - 1, index_int{1}, op::product{});
        }

        index idx;

        template <class Depth, class G, class... Xs>
        __device__ void run(Depth depth, G g, Xs... xs)
        {
            auto output = output_data();
            slice()(xs...)([&](auto w, auto... ws) {
                MIGRAPHX_ASSERT(w.get_shape().lens.back() == MaxSize);
                idx.local_stride(w.get_shape().elements(), [&](auto i) {
                    auto multi_idx     = w.get_shape().multi(i);
                    auto k             = multi_idx.back();
                    auto group                = compute_group(multi_idx);
                    output[{group, depth, k}] = g(w[i], ws[i]...);
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
                MIGRAPHX_ASSERT(compute_group(z.get_shape().lens) == NGroups);
                block_stride<per_block, 8>(idx, z.get_shape().elements())(
                    [&](auto i) { z[i] = f(output[i], ys[i]...); });
            });
        }
    };

    template <class T, class Info, class Output>
    static __device__ auto make(index idx, Info info, Output)
    {
        MIGRAPHX_ASSERT(info.axis == get_shape_c<Output>{}.lens.size() - 1);
        return algo<typename Output::type, Output, info.nargs, info.max_size>{idx};
    }
};

template<class...>
class static_print;

template<int64_t...>
class static_print1;

struct transpose2d
{
    template<class T, index_int N, index_int Pad>
    struct padded_array
    {
        static constexpr auto size()
        {
            return _c<(N + Pad)>;
        }

        template<class Index>
        constexpr auto get(Index i) const
        {
            if constexpr(i < N)
                return data[i];
            else
                return T{0};
        }

        template<class Index>
        constexpr void set(Index i, T x)
        {
            if constexpr(i < N)
                data[i] = x;
        }

        array<T, N> data{};
    };


    template<index_int Width, class T>
    static constexpr auto blocked_to_striped_shuffle(index idx, T input)
    {
        T output;
        constexpr auto n = T::size();
        const auto id = idx.local_subwave<Width>();
        static_assert(Width % n == 0, "Size must be a divisor of Width");

        repeat(n, [&](auto dst) {
            repeat(n, [&](auto src) {
                auto target = id / n + dst * (Width / n);
                auto x = input.get(src);
                auto y = readlane<Width>(x, target);
                if(src == id % n)
                    output.set(dst, y);
            });
        });

        return output;
    }

    template<index_int Width, class T>
    static constexpr auto striped_to_block_shuffle(index idx, T input)
    {
        T output;
        constexpr auto n = T::size();
        const auto id = idx.local_subwave<Width>();
        static_assert(Width % n == 0, "Size must be a divisor of Width");

        repeat(n, [&](auto dst) {
            repeat(n, [&](auto src) {
                auto target = (n * id + dst) % Width;
                auto x = input.get(src);
                auto y = readlane<Width>(x, target);
                if((id / (Width / n)) == src)
                    output.set(dst, y);
            });
        });

        return output;
    }

    template<index_int Pad, class T, index_int N>
    static constexpr auto make_pad_array(array<T, N> x)
    {
        return padded_array<T, N, Pad>{x};
    }

    template<index_int K, class T, index_int N, class F>
    static constexpr auto transform_split(array<T, N> input, F f)
    {
        if constexpr(K < 2)
        {
            return input;
        }
        else
        {
            array<T, N> output;
            repeat_c<K>([&](auto k) {
                constexpr auto width = _c<N/K>;
                array<T, width> x;;
                repeat_c<width>([&](auto i) {
                    x[i] = input[i + k*width];
                });
                auto y = f(x);
                repeat_c<width>([&](auto i) {
                    output[i + k*width] = y[i];
                });
            });
            return output;
        }
    }

    template<class A, class B>
    static constexpr auto next_divisor(A a, B b)
    {
        static_assert(b <= a);
        if constexpr(a % b == 0)
            return b;
        else
            return next_divisor(a, b + _c<1>);
    }

    template<index_int Batch, index_int Stride, class T, index_int PerLane>
    static __device__ array<T, PerLane> wave_shuffle(index idx, array<T, PerLane> x)
    {
        constexpr auto width = MIGRAPHX_WAVEFRONTSIZE / Batch;
        constexpr auto total = (PerLane * MIGRAPHX_WAVEFRONTSIZE) / Batch;
        constexpr auto rows = total / Stride;
        auto padx = make_pad_array<next_divisor(_c<width>, _c<PerLane>) - PerLane>(x);
        // static_print1<PerLane>{};
        // static_print1<Stride>{};
        // static_print1<rows>{};
        // static_print1<width>{};
        if constexpr(PerLane == Stride)
        {
            return blocked_to_striped_shuffle<width>(idx, padx).data;
        }
        else if constexpr(PerLane == rows)
        {
            return striped_to_block_shuffle<width>(idx, padx).data;
        }
        else
        {
            // TODO: Use shared memory to transpose
            static_assert(false, "Unsupported wave transpose");
        }

    }

    template<index_int Stride, class T, index_int PerLane>
    static __device__ array<T, PerLane> wave_shuffle(index idx, const array<T, PerLane>& x)
    {
        return wave_shuffle<1, Stride>(idx, x);
    }
};

template <index_int NGroups>
struct wave_interleave
{
    template<class T, index_int N>
    struct algo
    {
        constexpr auto slice() const
        {
            return slice_schedule<per_wave>(idx, slice_axes<-1>(), slice_group<NGroups>());
        }

        static constexpr auto per_lane() { return N * NGroups; }

        using data_shape = decltype(make_shape(index_ints<N, NGroups>{}));

        template <class Depth, class G>
        static constexpr auto data_index(Depth, G)
        {
            return return_c([] { return data_shape{}.index({Depth{}, G{}}); });
        }

        index idx;
        array<T, per_lane()> data{};

        template<class Depth, class G, class... Xs>
        __device__ void run(Depth depth, G g, Xs... xs)
        {
            slice()(xs...)([&](auto w, auto... ws) {
                repeat_c<NGroups>([&](auto group) {
                    auto i = group + idx.local_wave() * NGroups;
                    if(i < w.get_shape().elements())
                        data[data_shape{}.index({depth, group})] = g(w[i], ws[i]...);
                });
                // idx.local_wave_stride(w.get_shape().elements(), [&](auto i, auto k) {
                //     data[k + depth * Group] = g(w[i], ws[i]...);
                // });
            });
        }

        template<class F, class... Outputs>
        __device__ void finish(F f, Outputs... outputs) const
        {
            // auto out = transpose2d::wave_shuffle<Group*N>(idx, data);
            slice()(outputs...)([&](auto z, auto... ys) {
                constexpr auto nlocal_wave  = decltype(idx.nlocal_wave()){};
                constexpr auto output_shape = make_shape(index_ints<N, nlocal_wave, NGroups>{});
                repeat_c<N>([&](auto depth) {
                    repeat_c<NGroups>([&](auto group) {
                        auto k = data_shape{}.index({depth, group});
                        auto i = output_shape.index({depth, idx.local_wave(), group});
                        if(i < z.get_shape().elements())
                            z[i] = f(data[k], ys[i]...);
                    });
                });
                // idx.local_wave_stride(z.get_shape().elements(), [&](auto i, auto k) {
                //     z[i] = f(data[k], ys[i]...);
                // });
            });
        }
    };

    template<class T, class Info, class Output>
    static __device__ auto make(index idx, Info info, Output)
    {
        return algo<typename Output::type, info.nargs>{idx};
    }
};

template <class Algo, index_int Axis, class... InputPacks>
__device__ auto run(InputPacks... input_packs)
{
    return [=](auto f, auto t, auto... ts) {
        auto idx = make_index();
        auto algo = Algo::template make<typename get_base_type<InputPacks...>::type>(
            idx,
            info{.axis     = _c<Axis>,
                 .nargs    = _c<sizeof...(InputPacks)>,
                 .max_size = concat_max<Axis>(input_packs...)},
            t);
        fold([&](auto p, auto input_pack) {
            return input_pack([&](auto g, auto x, auto... xs) {
                return concat_slices<Axis>(x, p.offset, t, ts...)([&](auto z, auto... ys) {
                    if constexpr(is_void<decltype(algo.run(p.depth, g, x, xs...))>{})
                        algo.run(p.depth, g, x, xs...);
                    else
                        algo.run(p.depth, g, x, xs...)(z, f, ys...);
                    return concat_pair{p.offset + concat_ends<Axis>(x), p.depth + _c<1>};
                });
            });
        })(concat_pair{_c<0>, _c<0>}, input_packs...);
        algo.finish(f, t, ts...);
    };
}
} // concat

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_CONCAT_HPP
