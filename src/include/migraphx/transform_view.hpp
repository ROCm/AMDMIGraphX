#ifndef MIGRAPHX_GUARD_MIGRAPHX_TRANSFORM_VIEW_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_TRANSFORM_VIEW_HPP

#include <migraphx/config.hpp>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace views {

template <class Range, class F>
struct transform_view
{

    constexpr transform_view(Range& prng, F pf) : rng(&prng), f(std::move(pf)) {}

    struct iterator
    {
        using underlying_iterator = decltype(std::begin(std::declval<Range&>()));
        using reference           = decltype(std::declval<const F>()(
            std::declval<typename std::iterator_traits<underlying_iterator>::reference>()));
        using value_type          = std::decay_t<reference>;

        using iterator_category =
            typename std::iterator_traits<underlying_iterator>::iterator_category;
        using difference_type = typename std::iterator_traits<underlying_iterator>::difference_type;
        using pointer         = std::add_pointer_t<std::remove_reference_t<reference>>;

        struct arrow_proxy
        {
            reference value;
            constexpr std::add_pointer_t<std::remove_reference_t<reference>> operator->() &&
            {
                return std::addressof(value);
            }
        };

        constexpr iterator() = default;

        constexpr iterator(const transform_view* pparent, underlying_iterator it)
            : parent(pparent), current(it)
        {
        }

        constexpr reference operator*() const { return parent->f(*current); }

        constexpr arrow_proxy operator->() const { return arrow_proxy{parent->f(*current)}; }

        constexpr iterator& operator++()
        {
            ++current;
            return *this;
        }
        constexpr iterator operator++(int)
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        constexpr iterator& operator--()
        {
            --current;
            return *this;
        }

        constexpr iterator operator--(int)
        {
            iterator temp = *this;
            --(*this);
            return temp;
        }

        constexpr iterator operator+(difference_type n) const
        {
            return iterator(parent, current + n);
        }

        constexpr iterator operator-(difference_type n) const
        {
            return iterator(parent, current - n);
        }

        constexpr difference_type operator-(const iterator& other) const
        {
            return current - other.current;
        }

        constexpr iterator& operator+=(difference_type n)
        {
            current += n;
            return *this;
        }

        constexpr iterator& operator-=(difference_type n)
        {
            current -= n;
            return *this;
        }

        constexpr reference operator[](difference_type n) const
        {
            return parent->f(*(current + n));
        }

        constexpr bool operator==(const iterator& other) const { return current == other.current; }

        constexpr bool operator!=(const iterator& other) const { return not(*this == other); }

        constexpr bool operator<(const iterator& other) const { return current < other.current; }

        constexpr bool operator<=(const iterator& other) const { return current <= other.current; }

        constexpr bool operator>(const iterator& other) const { return current > other.current; }

        constexpr bool operator>=(const iterator& other) const { return current >= other.current; }

        private:
        const transform_view* parent = nullptr;
        underlying_iterator current{};
    };

    constexpr iterator begin() const { return {this, std::begin(*rng)}; }
    constexpr iterator end() const { return {this, std::end(*rng)}; }

    friend constexpr bool operator==(const transform_view& a, const transform_view& b)
    {
        return std::equal(a.begin(), a.end(), b.begin(), b.end());
    }

    friend constexpr bool operator!=(const transform_view& a, const transform_view& b)
    {
        return not(a == b);
    }

    friend constexpr bool operator<(const transform_view& a, const transform_view& b)
    {
        return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
    }

    friend constexpr bool operator>(const transform_view& a, const transform_view& b)
    {
        return b < a;
    }

    friend constexpr bool operator<=(const transform_view& a, const transform_view& b)
    {
        return not(b < a);
    }

    friend constexpr bool operator>=(const transform_view& a, const transform_view& b)
    {
        return not(a < b);
    }

    private:
    Range* rng = nullptr;
    F f;
};

// helper for type deduction
template <class Range, class F>
auto transform(Range& rng, F f)
{
    return transform_view<Range, F>(rng, std::move(f));
}

} // namespace views
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_TRANSFORM_VIEW_HPP
