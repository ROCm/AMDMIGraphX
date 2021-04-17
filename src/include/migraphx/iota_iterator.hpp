#ifndef MIGRAPHX_GUARD_RTGLIB_IOTA_ITERATOR_HPP
#define MIGRAPHX_GUARD_RTGLIB_IOTA_ITERATOR_HPP

#include <migraphx/config.hpp>
#include <migraphx/functional.hpp>
#include <iterator>
#include <type_traits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class F, class Iterator = std::ptrdiff_t>
struct basic_iota_iterator
{
    Iterator index;
    F f;

    using difference_type   = std::ptrdiff_t;
    using reference         = decltype(f(std::declval<Iterator>()));
    using value_type        = typename std::remove_reference<reference>::type;
    using pointer           = typename std::add_pointer<value_type>::type;
    using iterator_category = std::random_access_iterator_tag;

    basic_iota_iterator& operator+=(int n)
    {
        index += n;
        return *this;
    }

    basic_iota_iterator& operator-=(int n)
    {
        index -= n;
        return *this;
    }

    basic_iota_iterator& operator++()
    {
        index++;
        return *this;
    }

    basic_iota_iterator& operator--()
    {
        index--;
        return *this;
    }

    basic_iota_iterator operator++(int) // NOLINT
    {
        basic_iota_iterator it = *this;
        index++;
        return it;
    }

    basic_iota_iterator operator--(int) // NOLINT
    {
        basic_iota_iterator it = *this;
        index--;
        return it;
    }
    // TODO: operator->
    reference operator*() const { return f(index); }
};

template <class T, class F>
inline basic_iota_iterator<F, T> make_basic_iota_iterator(T x, F f)
{
    return basic_iota_iterator<F, T>{x, f};
}

template <class F, class Iterator>
inline basic_iota_iterator<F, Iterator> operator+(basic_iota_iterator<F, Iterator> x,
                                                  std::ptrdiff_t y)
{
    return x += y;
}

template <class F, class Iterator>
inline basic_iota_iterator<F, Iterator> operator+(std::ptrdiff_t x,
                                                  basic_iota_iterator<F, Iterator> y)
{
    return y + x;
}

template <class F, class Iterator>
inline std::ptrdiff_t operator-(basic_iota_iterator<F, Iterator> x,
                                basic_iota_iterator<F, Iterator> y)
{
    return x.index - y.index;
}

template <class F, class Iterator>
inline bool operator==(basic_iota_iterator<F, Iterator> x, basic_iota_iterator<F, Iterator> y)
{
    return x.index == y.index;
}

template <class F, class Iterator>
inline bool operator!=(basic_iota_iterator<F, Iterator> x, basic_iota_iterator<F, Iterator> y)
{
    return x.index != y.index;
}

template <class F, class Iterator>
inline bool operator<(basic_iota_iterator<F, Iterator> x, basic_iota_iterator<F, Iterator> y)
{
    return x.index < y.index;
}

template <class F, class Iterator>
inline bool operator>(basic_iota_iterator<F, Iterator> x, basic_iota_iterator<F, Iterator> y)
{
    return x.index > y.index;
}

template <class F, class Iterator>
inline bool operator>=(basic_iota_iterator<F, Iterator> x, basic_iota_iterator<F, Iterator> y)
{
    return x.index >= y.index;
}

template <class F, class Iterator>
inline bool operator<=(basic_iota_iterator<F, Iterator> x, basic_iota_iterator<F, Iterator> y)
{
    return x.index <= y.index;
}

using iota_iterator = basic_iota_iterator<id>;

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
