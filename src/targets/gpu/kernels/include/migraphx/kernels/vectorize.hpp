#ifndef MIGRAPHX_GUARD_KERNELS_VECTORIZE_HPP
#define MIGRAPHX_GUARD_KERNELS_VECTORIZE_HPP

#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/vec.hpp>

namespace migraphx {

template <class T>
constexpr auto tensor_vec_size(T)
{
    return vec_size<typename T::type>();
}

template <index_int N, class Shape>
constexpr auto as_vec_shape(Shape s)
{
    auto lens    = transform(s.lens, s.strides, [](auto len, auto stride) {
        if(stride == 1)
            return len / N;
        else
            return len;
    });
    auto strides = transform(s.strides, [](auto stride) {
        if(stride == 1)
            return stride;
        return stride / N;
    });
    MIGRAPHX_ASSERT(make_shape(lens, strides).element_space() * N == s.element_space());
    return make_shape(lens, strides);
}

template <index_int N, class T>
__device__ __host__ auto as_vec(T x)
{
    if constexpr(N == 0)
        return x;
    else
        return make_tensor_view(as_vec<N>(x.data()), as_vec_shape<N>(x.get_shape()));
}

template <index_int N, class T, class Axis>
constexpr auto tensor_step(T x, Axis)
{
    if constexpr(N == 0)
    {
        return x;
    }
    else
    {
        constexpr auto s = decltype(x.get_shape()){};
        MIGRAPHX_ASSERT(s.strides[Axis{}] == 0);
        return sequence(x.get_shape().lens.size(), [&](auto... is) {
            auto lens = transform(s.lens, index_ints<is...>{}, [&](auto i, auto j) {
                constexpr auto axis = Axis{};
                if(j == axis)
                    return i / N;
                else
                    return i;
            });
            return make_tensor_view(x.data(), make_shape(lens, s.strides));
        });
    }
}

template <class IntegralConstant, class T>
__device__ __host__ auto as_vec(IntegralConstant ic, T&& x)
{
    return as_vec<ic>(x);
}

template <class... Shapes>
constexpr index_int find_vector_axis(Shapes... ss)
{
    index_int axis = 0;
    bool b         = false;
    by([&](auto s) {
        if(s.broadcasted() or b)
            return;
        auto it = find(s.strides.begin(), s.strides.end(), 1);
        if(it == s.strides.end())
            return;
        axis = it - s.strides.begin();
        b    = true;
    })(ss...);
    return axis;
}

template <index_int N, class Axis, class... Shapes>
constexpr auto is_vectorizable(Axis axis, Shapes... ss)
{
    return (((ss.lens[axis] % N) == 0 and (ss.strides[axis] == 1 or ss.strides[axis] == 0)) and
            ...);
}

template <index_int N, class... Shapes>
constexpr bool is_vectorizable(Shapes... ss)
{
    return (is_vectorizable<N>(ss, find_vector_axis(ss)) and ...);
}

template <class P>
constexpr auto find_vectorize_size(P pred)
{
    if constexpr(pred(_c<4>))
        return _c<4>;
    else if constexpr(pred(_c<2>))
        return _c<2>;
    else
        return _c<0>;
}

template <class T>
__host__ __device__ auto vectorize(T x)
{
    if constexpr(vec_size<T>() == 0)
    {
        constexpr auto n =
            find_vectorize_size([&](auto i) { return _c<is_vectorizable<i>(x.get_shape())>; });
        return as_vec<n>(x);
    }
    else
    {
        return x;
    }
}

inline __device__ __host__ auto auto_vectorize()
{
    return [](auto... xs) {
        return [=](auto f) {
            // TODO: Just check there a single axis of 1
            constexpr bool packed_or_broadcasted =
                ((xs.get_shape().packed() or xs.get_shape().broadcasted()) and ...);
            if constexpr(packed_or_broadcasted)
            {
                constexpr auto axis = find_vector_axis(xs.get_shape()...);
                constexpr auto n    = find_vectorize_size(
                    [&](auto i) { return _c<is_vectorizable<i>(axis, xs.get_shape()...)>; });
                by(
                    [&](auto x) {
                        constexpr auto s = x.get_shape();
                        if constexpr(s.strides[axis] == 0)
                            return tensor_step<n>(x, axis);
                        else
                            return as_vec<n>(x);
                    },
                    f)(xs...);
            }
            else
            {
                f(xs...);
            }
        };
    };
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_VECTORIZE_HPP
