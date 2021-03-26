#ifndef MIGRAPHX_GUARD_KERNELS_TENSOR_VIEW_HPP
#define MIGRAPHX_GUARD_KERNELS_TENSOR_VIEW_HPP

#include <migraphx/kernels/shape.hpp>

namespace migraphx {

template <class T, class Shape>
struct tensor_view
{
    constexpr Shape get_shape() const { return Shape{}; }
    constexpr index_int size() const { return get_shape().elements(); }

    template <class U>
    constexpr T& operator[](U i) const
    {
        return x[get_shape().index(i)];
    }

    constexpr T* data() const { return x; }

    constexpr T* begin() const { return data(); }
    constexpr T* end() const { return data() + size(); }

    T* x;
};

template <class T, class Shape>
constexpr tensor_view<T, Shape> make_tensor_view(T* x, Shape)
{
    return {x};
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_TENSOR_VIEW_HPP
