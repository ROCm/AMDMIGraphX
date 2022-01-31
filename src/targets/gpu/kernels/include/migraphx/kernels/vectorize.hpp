#ifndef MIGRAPHX_GUARD_KERNELS_VECTORIZE_HPP
#define MIGRAPHX_GUARD_KERNELS_VECTORIZE_HPP

#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/vec.hpp>

namespace migraphx {

template <class T>
constexpr auto tensor_vec_size()
{
    return vec_size<typename T::type>();
}

template <class T>
constexpr auto tensor_vec_size(T)
{
    return tensor_vec_size<T>();
}

template <index_int N, class Shape, class Axis>
constexpr auto shape_step(Shape s, Axis)
{
    static_assert(N > 0, "Vector size must be non-zero");
    return sequence(s.lens.size(), [&](auto... is) {
        auto lens    = transform(s.lens, index_ints<is...>{}, [&](auto i, auto j) {
            constexpr auto axis = Axis::to();
            MIGRAPHX_ASSERT(i != 0);
            MIGRAPHX_ASSERT(j != axis or i % N == 0);
            if(j == axis)
                return i / N;
            else
                return i;
        });
        auto strides = transform(s.strides, index_ints<is...>{}, [&](auto i, auto j) {
            constexpr auto axis = Axis::to();
            // If stride of the axis is zero then we dont need to adjust the other strides
            if(Shape{}.strides[axis] == 0)
                return i;
            MIGRAPHX_ASSERT(j == axis or i % N == 0);
            if(j == axis)
                return i;
            else
                return i / N;
        });
        MIGRAPHX_ASSERT(make_shape(lens, strides).elements() * N == s.elements());
        MIGRAPHX_ASSERT(strides[Axis{}] == 0 or
                        make_shape(lens, strides).element_space() * N == s.element_space());
        return make_shape(lens, strides);
    });
}

// Bools can not be used as a vector type so convert it to int8
template <class T>
__device__ __host__ T* remove_bool(T* x)
{
    return x;
}

inline __device__ __host__ int8_t* remove_bool(bool* x) { return reinterpret_cast<int8_t*>(x); }

template <index_int N, class T, class Axis>
__device__ __host__ auto as_vec(T x, Axis axis)
{
    if constexpr(N == 0)
        return x;
    else
        return make_tensor_view(as_vec<N>(remove_bool(x.data())),
                                shape_step<N>(x.get_shape(), axis));
}

template <index_int N, class T, class Axis>
constexpr auto tensor_step(T x, Axis axis)
{
    if constexpr(N == 0)
    {
        return x;
    }
    else
    {
        constexpr auto s = decltype(x.get_shape()){};
        MIGRAPHX_ASSERT(s.strides[axis] == 0);
        return make_tensor_view(x.data(), shape_step<N>(s, axis));
    }
}

template <class IntegralConstant, class T>
__device__ __host__ auto as_vec(IntegralConstant ic, T&& x)
{
    return as_vec<ic>(x);
}

template <class Shape>
constexpr index_int find_vector_axis_c(Shape s)
{
    // Find the fastest axis that is not broadcasted
    index_int axis = 0;
    for(index_int i = 1; i < s.lens.size(); i++)
    {
        if(s.strides[i] == 0)
            continue;
        if(s.strides[axis] == 0 or
           pack_compare(less{}, pack(s.strides[i], s.lens[i]), pack(s.strides[axis], s.lens[axis])))
            axis = i;
    }
    return axis;
}

template <class... Shapes>
constexpr index_int find_vector_axis_c(Shapes... ss)
{
    const bool all_broadcasted = (ss.broadcasted() and ...);
    index_int axis             = 0;
    bool b                     = false;
    by([&](auto s) {
        if(b)
            return;
        // Skip broadcasted shapes if there are shapes not broadcasted
        if(not all_broadcasted and s.broadcasted())
            return;
        axis = find_vector_axis_c(s);
        if(s.strides[axis] == 1)
            b = true;
    })(ss...);
    if(not b)
        return -1;
    return axis;
}

template <class... Shapes>
constexpr auto find_vector_axis(Shapes...)
{
    return _c<find_vector_axis_c(Shapes{}...)>;
}

template <index_int N, class Axis, class... Shapes>
constexpr auto is_vectorizable_c(Axis axis, Shapes... ss)
{
    return ((axis < ss.lens.size() and ss.lens[axis] % N == 0 and
             // Only vectorize broadcasted types with stride 0, since this causes issues in the
             // preloader
             ((not ss.broadcasted() and ss.strides[axis] == 1) or ss.strides[axis] == 0)) and
            ...);
}

template <index_int N, class Axis, class... Shapes>
constexpr auto is_vectorizable(Axis, Shapes...)
{
    return _c<is_vectorizable_c<N>(Axis::to(), Shapes{}...)>;
}

template <class P>
constexpr auto find_vectorize_size(P pred)
{
    if constexpr(decltype(pred(_c<4>)){})
        return _c<4>;
    else if constexpr(decltype(pred(_c<2>)){})
        return _c<2>;
    else
        return _c<0>;
}

template <class T>
__host__ __device__ auto vectorize(T x)
{
    if constexpr(tensor_vec_size<T>() == 0)
    {
        constexpr auto axis = find_vector_axis(x.get_shape());
        constexpr auto n =
            find_vectorize_size([&](auto i) { return is_vectorizable<i>(axis, x.get_shape()); });
        return as_vec<n>(x, axis);
    }
    else
    {
        return x;
    }
}

template <class F, class... Ts>
inline __device__ __host__ auto auto_vectorize_impl(F f, Ts... xs)
{
    // TODO: Just check there a single axis of 1
    constexpr bool packed_or_broadcasted =
        ((xs.get_shape().packed() or xs.get_shape().broadcasted()) and ...);
    if constexpr(packed_or_broadcasted)
    {
        constexpr auto axis = decltype(find_vector_axis(xs.get_shape()...)){};
        constexpr auto n    = find_vectorize_size(
            [&](auto i) { return is_vectorizable<i>(axis, xs.get_shape()...); });
        by(
            [&](auto x) {
                constexpr auto s = decltype(x.get_shape()){};
                if constexpr(axis < s.strides.size())
                {
                    MIGRAPHX_ASSERT(s.strides[axis] == 0 or s.strides[axis] == 1);
                    MIGRAPHX_ASSERT(s.lens[axis] > 0);
                    MIGRAPHX_ASSERT(n == 0 or s.lens[axis] % n == 0);
                    if constexpr(s.strides[axis] == 0)
                        return tensor_step<n>(x, axis);
                    else
                        return as_vec<n>(x, axis);
                }
                else
                {
                    return x;
                }
            },
            f)(xs...);
    }
    else
    {
        f(xs...);
    }
}

inline __device__ __host__ auto auto_vectorize()
{
    return [](auto... xs) { return [=](auto f) { auto_vectorize_impl(f, xs...); }; };
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_VECTORIZE_HPP
