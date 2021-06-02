#ifndef MIGRAPHX_GUARD_KERNELS_VECTORIZE_HPP
#define MIGRAPHX_GUARD_KERNELS_VECTORIZE_HPP

#include <migraphx/kernels/tensor_view.hpp>

namespace migraphx {

template <class T, index_int N>
constexpr auto vec_size(vec<T, N>)
{
    return index_constant<N>{};
}

template <class T>
constexpr auto vec_size(T, ...)
{
    return index_constant<0>{};
}

template <class T>
constexpr auto vec_size()
{
    return decltype(vec_size(T{})){};
}

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
__device__ __host__ auto as_vec(T* x)
{
    if constexpr(N == 0)
        return x;
    else
        return reinterpret_cast<vec<T, N>*>(x);
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

template <class... Shapes>
constexpr auto find_vector_axis()
{
    return index_constant<find_vector_axis(Shapes{}...)>{};
}

template <index_int N, class Shape, class Axis>
constexpr auto is_vectorizable(Shape s, Axis axis)
{
    return (s.lens[axis] % N) == 0 and (s.strides[axis] == 1 or s.strides[axis] == 0);
}

template <index_int N, class Shape>
constexpr bool is_vectorizable(Shape s)
{
    return is_vectorizable<N>(s, find_vector_axis<Shape>());
}

template <index_int N, index_int Axis, class... Shapes>
constexpr auto is_vectorizable()
{
    return bool_constant<(is_vectorizable<N>(Shapes{}, index_constant<Axis>{}) and ...)>{};
}

template <index_int N, class... Shapes>
constexpr auto is_vectorizable()
{
    return bool_constant<(is_vectorizable<N>(Shapes{}) and ...)>{};
}

template <class P>
constexpr auto find_vectorize_size(P pred)
{
    if constexpr(pred(index_constant<4>{}))
        return index_constant<4>{};
    else
        return index_constant<0>{};
}

template <class T>
__device__ __host__ auto vectorize(T x)
{
    if constexpr(is_vectorizable<4, decltype(x.get_shape())>())
        return as_vec<4>(x);
    else
        return x;
}

template <class... Ts>
__device__ __host__ auto auto_vectorize(Ts... xs)
{
    return [=](auto f) {
        constexpr bool packed = (decltype(xs.get_shape()){}.packed() or ...);
        if constexpr(packed)
        {
            constexpr auto axis = find_vector_axis<decltype(xs.get_shape())...>();
            constexpr auto n    = find_vectorize_size(
                [&](auto i) { return is_vectorizable<i, axis, decltype(xs.get_shape())...>(); });
            by(
                [&](auto x) {
                    constexpr auto s = decltype(x.get_shape()){};
                    if constexpr(s.broadcasted())
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
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_VECTORIZE_HPP
