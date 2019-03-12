#ifndef MIGRAPHX_GUARD_RTGLIB_ITERATOR_FOR_HPP
#define MIGRAPHX_GUARD_RTGLIB_ITERATOR_FOR_HPP

#include <cassert>
#include <type_traits>
#include <iterator>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct iterator_for_select
{
    template <class T>
    static T deref(T x)
    {
        return x;
    }

    template <class T>
    static auto begin(T* x)
    {
        return x->begin();
    }

    template <class T>
    static auto end(T* x)
    {
        return x->end();
    }
};

struct iterator_for_select_reverse
{
    template <class T>
    static auto deref(T x)
    {
        return std::prev(x.base());
    }

    template <class T>
    static auto begin(T* x)
    {
        return std::make_reverse_iterator(x->end());
    }

    template <class T>
    static auto end(T* x)
    {
        return std::make_reverse_iterator(x->begin());
    }
};

template <class T, class Selector = iterator_for_select>
struct iterator_for_range
{
    T* base;
    using base_iterator = std::remove_reference_t<decltype(Selector::begin(base))>;

    struct iterator
    {
        base_iterator i;
        auto operator*() const { return Selector::deref(i); }
        base_iterator operator++() { return ++i; }
        bool operator!=(const iterator& rhs) const { return i != rhs.i; }
    };

    iterator begin()
    {
        assert(base != nullptr);
        return {Selector::begin(base)};
    }
    iterator end()
    {
        assert(base != nullptr);
        return {Selector::end(base)};
    }
};
template <class T>
iterator_for_range<T> iterator_for(T& x)
{
    return {&x};
}

template <class T>
iterator_for_range<T, iterator_for_select_reverse> reverse_iterator_for(T& x)
{
    return {&x};
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
