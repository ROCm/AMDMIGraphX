#ifndef MIGRAPHX_GUARD_KERNELS_IOTA_ITERATOR_HPP
#define MIGRAPHX_GUARD_KERNELS_IOTA_ITERATOR_HPP

#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/type_traits.hpp>

namespace migraphx {

template <class F, class Iterator = diff_int>
struct basic_iota_iterator
{
    Iterator index;
    F f;

    using difference_type = diff_int;
    using reference       = decltype(f(declval<Iterator>()));
    using value_type      = remove_reference_t<reference>;
    using pointer         = add_pointer_t<value_type>;

    constexpr basic_iota_iterator& operator+=(diff_int n)
    {
        index += n;
        return *this;
    }

    constexpr basic_iota_iterator& operator-=(diff_int n)
    {
        index -= n;
        return *this;
    }

    constexpr basic_iota_iterator& operator++()
    {
        index++;
        return *this;
    }

    constexpr basic_iota_iterator& operator--()
    {
        index--;
        return *this;
    }

    constexpr basic_iota_iterator operator++(int) // NOLINT
    {
        basic_iota_iterator it = *this;
        index++;
        return it;
    }

    constexpr basic_iota_iterator operator--(int) // NOLINT
    {
        basic_iota_iterator it = *this;
        index--;
        return it;
    }
    // TODO: operator->
    constexpr reference operator*() const { return f(index); }

    template <class T>
    constexpr reference operator[](T x) const
    {
        return f(index + x);
    }
};

template <class T, class F>
constexpr basic_iota_iterator<F, T> make_basic_iota_iterator(T x, F f)
{
    return basic_iota_iterator<F, T>{x, f};
}

template <class F, class Iterator>
constexpr basic_iota_iterator<F, Iterator> operator+(basic_iota_iterator<F, Iterator> x, diff_int y)
{
    return x += y;
}

template <class F, class Iterator>
constexpr basic_iota_iterator<F, Iterator> operator+(diff_int x, basic_iota_iterator<F, Iterator> y)
{
    return y + x;
}

template <class F, class Iterator>
constexpr diff_int operator-(basic_iota_iterator<F, Iterator> x, basic_iota_iterator<F, Iterator> y)
{
    return x.index - y.index;
}

template <class F, class Iterator>
constexpr basic_iota_iterator<F, Iterator> operator-(basic_iota_iterator<F, Iterator> x, diff_int y)
{
    return x -= y;
}

template <class F, class Iterator>
constexpr bool operator==(basic_iota_iterator<F, Iterator> x, basic_iota_iterator<F, Iterator> y)
{
    return x.index == y.index;
}

template <class F, class Iterator>
constexpr bool operator!=(basic_iota_iterator<F, Iterator> x, basic_iota_iterator<F, Iterator> y)
{
    return x.index != y.index;
}

template <class F, class Iterator>
constexpr bool operator<(basic_iota_iterator<F, Iterator> x, basic_iota_iterator<F, Iterator> y)
{
    return x.index < y.index;
}

template <class F, class Iterator>
constexpr bool operator>(basic_iota_iterator<F, Iterator> x, basic_iota_iterator<F, Iterator> y)
{
    return x.index > y.index;
}

template <class F, class Iterator>
constexpr bool operator>=(basic_iota_iterator<F, Iterator> x, basic_iota_iterator<F, Iterator> y)
{
    return x.index >= y.index;
}

template <class F, class Iterator>
constexpr bool operator<=(basic_iota_iterator<F, Iterator> x, basic_iota_iterator<F, Iterator> y)
{
    return x.index <= y.index;
}

struct defaul_iota_iterator
{
    template <class T>
    constexpr auto operator()(T x) const
    {
        return x;
    }
};

using iota_iterator = basic_iota_iterator<defaul_iota_iterator>;

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_IOTA_ITERATOR_HPP
