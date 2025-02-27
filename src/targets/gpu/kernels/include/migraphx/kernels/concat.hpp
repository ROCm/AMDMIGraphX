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
#include <migraphx/kernels/dpp.hpp>

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

template<class Axis, class Size>
struct info
{
    Axis axis;
    Size size;
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

template<class... Ss>
constexpr auto per_wave_slice(index idx, Ss... ss)
{
    return [=](auto... xs) {
        return [=](auto f) {
            // TODO: Assert nslices is the same for all xs
            constexpr auto n = nslices(get_shape_c<decltype(arg_c<0>()(xs...))>{}, ss...);
            idx.wave_stride(n, [&](auto i) {
                f(tensor_slice(xs, i, ss...)...);
            });
        };
    };
}

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

template<index_int Group>
struct wave_interleave
{
    template<class T, index_int N>
    struct algo
    {
        static constexpr auto per_lane()
        {
            return N * Group;
        }

        index idx;
        array<T, per_lane()> data{};

        constexpr auto slice() const
        {
            return per_wave_slice(idx, slice_axes<-1>(), slice_group<Group>());
        }

        template<class Depth, class G, class... Xs>
        __device__ void run(Depth depth, G g, Xs... xs)
        {
            slice()(xs...)([&](auto w, auto... ws) {
                repeat_c<Group>([&](auto group) {
                    auto i = group + idx.local_wave() * Group;
                    if(i < w.get_shape().elements())
                        data[depth + group * N] = g(w[i], ws[i]...);
                });
                // idx.local_wave_stride(w.get_shape().elements(), [&](auto i, auto k) {
                //     data[depth + k * N] = g(w[i], ws[i]...);
                // });
            });
        }

        template<class F, class... Outputs>
        __device__ void finish(F f, Outputs... outputs) const
        {
            auto out = transpose2d::wave_shuffle<Group*N>(idx, data);
            slice()(outputs...)([&](auto z, auto... ys) {
                repeat_c<per_lane()>([&](auto k) {
                    auto i = k + idx.local_wave() * per_lane();
                    if(i < z.get_shape().elements())
                        z[i] = f(out[k], ys[i]...);
                });
                // repeat_c<N>([&](auto depth) {
                //     repeat_c<Group>([&](auto group) {

                //     });
                // });
                // idx.local_wave_stride(z.get_shape().elements(), [&](auto i, auto k) {
                //     z[i] = f(data[k], ys[i]...);
                // });
            });
        }
    };
    template<class T, class Info, class Output>
    static __device__ auto make(index idx, Info info, Output)
    {
        return algo<typename Output::type, info.size>{idx};
    }
};

template <class Algo, index_int Axis, class... InputPacks>
__device__ auto run(InputPacks... input_packs)
{
    return [=](auto f, auto t, auto... ts) {
        auto idx = make_index();
        auto algo = Algo::template make<typename get_base_type<InputPacks...>::type>(idx, info{.axis = _c<Axis>, .size = _c<sizeof...(InputPacks)>}, t);
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
