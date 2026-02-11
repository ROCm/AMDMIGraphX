#ifndef ROCM_GUARD_ITERATOR_REVERSE_ITERATOR_HPP
#define ROCM_GUARD_ITERATOR_REVERSE_ITERATOR_HPP

#include <rocm/iterator/iterator_traits.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator>
struct reverse_iterator
{
    using iterator_type     = Iterator;
    using difference_type   = typename iterator_traits<Iterator>::difference_type;
    using value_type        = typename iterator_traits<Iterator>::value_type;
    using pointer           = typename iterator_traits<Iterator>::pointer;
    using reference         = typename iterator_traits<Iterator>::reference;
    using iterator_category = typename iterator_traits<Iterator>::iterator_category;

    iterator_type current;

    constexpr reverse_iterator() : current() {}

    constexpr explicit reverse_iterator(iterator_type it) : current(it) {}

    template <class U>
    constexpr reverse_iterator(const reverse_iterator<U>& other) : current(other.base())
    {
    }

    template <class U>
    constexpr reverse_iterator& operator=(const reverse_iterator<U>& other)
    {
        current = other.base();
        return *this;
    }

    constexpr iterator_type base() const { return current; }

    constexpr reference operator*() const
    {
        iterator_type tmp = current;
        --tmp;
        return *tmp;
    }

    constexpr pointer operator->() const
    {
        iterator_type tmp = current;
        --tmp;
        return tmp;
    }

    constexpr reference operator[](difference_type n) const { return *(*this + n); }

    constexpr reverse_iterator& operator++()
    {
        --current;
        return *this;
    }

    constexpr reverse_iterator& operator--()
    {
        ++current;
        return *this;
    }

    constexpr reverse_iterator operator++(int) // NOLINT
    {
        reverse_iterator tmp = *this;
        --current;
        return tmp;
    }

    constexpr reverse_iterator operator--(int) // NOLINT
    {
        reverse_iterator tmp = *this;
        ++current;
        return tmp;
    }

    constexpr reverse_iterator& operator+=(difference_type n)
    {
        current -= n;
        return *this;
    }

    constexpr reverse_iterator& operator-=(difference_type n)
    {
        current += n;
        return *this;
    }

    friend constexpr reverse_iterator operator+(reverse_iterator it, difference_type n)
    {
        return it += n;
    }

    friend constexpr reverse_iterator operator+(difference_type n, reverse_iterator it)
    {
        return it += n;
    }

    friend constexpr reverse_iterator operator-(reverse_iterator it, difference_type n)
    {
        return it -= n;
    }

    template <class Iterator2>
    friend constexpr auto operator-(const reverse_iterator& lhs,
                                    const reverse_iterator<Iterator2>& rhs)
        -> decltype(rhs.base() - lhs.base())
    {
        return rhs.base() - lhs.base();
    }

    template <class Iterator2>
    friend constexpr bool operator==(const reverse_iterator& lhs,
                                     const reverse_iterator<Iterator2>& rhs)
    {
        return lhs.base() == rhs.base();
    }

    template <class Iterator2>
    friend constexpr bool operator!=(const reverse_iterator& lhs,
                                     const reverse_iterator<Iterator2>& rhs)
    {
        return lhs.base() != rhs.base();
    }

    template <class Iterator2>
    friend constexpr bool operator<(const reverse_iterator& lhs,
                                    const reverse_iterator<Iterator2>& rhs)
    {
        return lhs.base() > rhs.base();
    }

    template <class Iterator2>
    friend constexpr bool operator>(const reverse_iterator& lhs,
                                    const reverse_iterator<Iterator2>& rhs)
    {
        return lhs.base() < rhs.base();
    }

    template <class Iterator2>
    friend constexpr bool operator<=(const reverse_iterator& lhs,
                                     const reverse_iterator<Iterator2>& rhs)
    {
        return lhs.base() >= rhs.base();
    }

    template <class Iterator2>
    friend constexpr bool operator>=(const reverse_iterator& lhs,
                                     const reverse_iterator<Iterator2>& rhs)
    {
        return lhs.base() <= rhs.base();
    }
};

template <class Iterator>
constexpr reverse_iterator<Iterator> make_reverse_iterator(Iterator it)
{
    return reverse_iterator<Iterator>(it);
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ITERATOR_REVERSE_ITERATOR_HPP
