#ifndef MIGRAPHX_GUARD_KERNELS_TENSOR_VIEW_HPP
#define MIGRAPHX_GUARD_KERNELS_TENSOR_VIEW_HPP

#include <migraphx/kernels/shape.hpp>
#include <migraphx/kernels/debug.hpp>
#include <migraphx/kernels/iota_iterator.hpp>

namespace migraphx {

template <class T>
struct tensor_view_iterator_read
{
    T* view;
    constexpr auto& operator()(std::size_t n) const
    {
        MIGRAPHX_ASSERT(view != nullptr);
        return (*view)[n];
    }
};

template <class T, class Shape>
struct tensor_view
{
    using type       = T;
    using shape_type = Shape;
    using iterator   = basic_iota_iterator<tensor_view_iterator_read<const tensor_view>, index_int>;

    constexpr Shape get_shape() const { return Shape{}; }
    constexpr auto size() const { return get_shape().elements(); }

    template <class U>
    constexpr T& operator[](U i) const
    {
        MIGRAPHX_ASSERT(get_shape().index(i) < get_shape().element_space());
        return x[get_shape().index(i)];
    }

    constexpr T* data() const { return x; }

    constexpr auto begin() const { return iterator{0, {this}}; }
    constexpr auto end() const { return iterator{this->size(), {this}}; }

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
