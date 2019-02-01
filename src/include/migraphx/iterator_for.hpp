#ifndef MIGRAPHX_GUARD_RTGLIB_ITERATOR_FOR_HPP
#define MIGRAPHX_GUARD_RTGLIB_ITERATOR_FOR_HPP

#include <cassert>
#include <type_traits>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class T>
struct iterator_for_range
{
    T* base;
    using base_iterator = std::remove_reference_t<decltype(base->begin())>;

    struct iterator
    {
        base_iterator i;
        base_iterator operator*() const { return i; }
        base_iterator operator++() { return ++i; }
        bool operator!=(const iterator& rhs) const { return i != rhs.i; }
    };

    iterator begin()
    {
        assert(base != nullptr);
        return {base->begin()};
    }
    iterator end()
    {
        assert(base != nullptr);
        return {base->end()};
    }
};
template <class T>
iterator_for_range<T> iterator_for(T& x)
{
    return {&x};
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
