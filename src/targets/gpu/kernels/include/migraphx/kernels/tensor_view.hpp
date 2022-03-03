#ifndef MIGRAPHX_GUARD_KERNELS_TENSOR_VIEW_HPP
#define MIGRAPHX_GUARD_KERNELS_TENSOR_VIEW_HPP

#include <migraphx/kernels/shape.hpp>
#include <migraphx/kernels/debug.hpp>

namespace migraphx {

template <class T, class Shape>
struct tensor_view
{
    using type       = T;
    using shape_type = Shape;

    constexpr Shape get_shape() const { return Shape{}; }
    constexpr index_int size() const { return get_shape().elements(); }

    template <class U>
    constexpr T& operator[](U i) const
    {
        MIGRAPHX_ASSERT(get_shape().index(i) < get_shape().element_space());
        return x[get_shape().index(i)];
    }

    constexpr T* data() const { return x; }

    constexpr T* begin() const { return data(); }
    constexpr T* end() const { return data() + size(); }

    template <class U>
    constexpr tensor_view<U, Shape> with(U* y) const
    {
        static_assert(sizeof(T) == sizeof(U), "Not the same size");
        return {y};
    }

    T* x;
};

template <class T, class Shape>
constexpr tensor_view<T, Shape> make_tensor_view(T* x, Shape)
{
    return {x};
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_TENSOR_VIEW_HPP
