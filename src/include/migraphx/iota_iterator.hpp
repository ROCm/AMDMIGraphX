#ifndef MIGRAPHX_GUARD_RTGLIB_IOTA_ITERATOR_HPP
#define MIGRAPHX_GUARD_RTGLIB_IOTA_ITERATOR_HPP

#include <migraphx/config.hpp>
#include <iterator>
#include <type_traits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class F, class Iterator = std::size_t>
struct iota_iterator
{
    Iterator index;
    F f;

    using difference_type   = std::ptrdiff_t;
    using reference         = decltype(f(std::declval<Iterator>()));
    using value_type        = typename std::remove_reference<reference>::type;
    using pointer           = typename std::add_pointer<value_type>::type;
    using iterator_category = std::random_access_iterator_tag;

    iota_iterator& operator+=(int n)
    {
        index += n;
        return *this;
    }

    iota_iterator& operator-=(int n)
    {
        index -= n;
        return *this;
    }

    iota_iterator& operator++()
    {
        index++;
        return *this;
    }

    iota_iterator& operator--()
    {
        index--;
        return *this;
    }

    iota_iterator operator++(int) // NOLINT
    {
        iota_iterator it = *this;
        index++;
        return it;
    }

    iota_iterator operator--(int) // NOLINT
    {
        iota_iterator it = *this;
        index--;
        return it;
    }
    // TODO: operator->
    reference operator*() const { return f(index); }
};

template <class F, class Iterator>
inline iota_iterator<F, Iterator> operator+(iota_iterator<F, Iterator> x,
                                            iota_iterator<F, Iterator> y)
{
    return iota_iterator<F, Iterator>(x.index + y.index, x.f);
}

template <class F, class Iterator>
inline std::ptrdiff_t operator-(iota_iterator<F, Iterator> x, iota_iterator<F, Iterator> y)
{
    return x.index - y.index;
}

template <class F, class Iterator>
inline bool operator==(iota_iterator<F, Iterator> x, iota_iterator<F, Iterator> y)
{
    return x.index == y.index;
}

template <class F, class Iterator>
inline bool operator!=(iota_iterator<F, Iterator> x, iota_iterator<F, Iterator> y)
{
    return x.index != y.index;
}

template <class F, class Iterator>
inline bool operator<(iota_iterator<F, Iterator> x, iota_iterator<F, Iterator> y)
{
    return x.index < y.index;
}

template <class F, class Iterator>
inline bool operator>(iota_iterator<F, Iterator> x, iota_iterator<F, Iterator> y)
{
    return x.index > y.index;
}

template <class F, class Iterator>
inline bool operator>=(iota_iterator<F, Iterator> x, iota_iterator<F, Iterator> y)
{
    return x.index >= y.index;
}

template <class F, class Iterator>
inline bool operator<=(iota_iterator<F, Iterator> x, iota_iterator<F, Iterator> y)
{
    return x.index <= y.index;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
