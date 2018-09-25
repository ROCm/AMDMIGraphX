#ifndef MIGRAPH_GUARD_MIGRAPHLIB_RANGES_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_RANGES_HPP

#include <algorithm>
#include <initializer_list>
#include <migraph/rank.hpp>

namespace migraph {

namespace detail {

template <class String, class T>
auto generic_find_impl(rank<2>, String&& s, const T& x) -> decltype(s.begin() + s.find(x), s.npos)
{
    auto index = s.find(x);
    if(index == s.npos)
        return s.end();
    else
        return s.begin() + index;
}

template <class C, class T>
auto generic_find_impl(rank<1>, C&& c, const T& x) -> decltype(c.find(x))
{
    return c.find(x);
}

template <class C, class T>
auto generic_find_impl(rank<0>, C&& c, const T& x)
{
    return std::find(c.begin(), c.end(), x);
}

} // namespace detail

template <class C, class T>
auto generic_find(C&& c, const T& x)
{
    return detail::generic_find_impl(rank<2>{}, c, x);
}

template <class C, class T>
bool contains(const C& c, const T& x)
{
    return generic_find(c, x) != c.end();
}

template <class T, class U>
bool contains(const std::initializer_list<T>& c, const U& x)
{
    return generic_find(c, x) != c.end();
}

template <class C, class Predicate>
bool all_of(const C& c, const Predicate& p)
{
    return std::all_of(c.begin(), c.end(), p);
}

template <class T, class Predicate>
bool all_of(const std::initializer_list<T>& c, const Predicate& p)
{
    return std::all_of(c.begin(), c.end(), p);
}

template <class Range, class Iterator>
void copy(Range&& r, Iterator it)
{
    std::copy(r.begin(), r.end(), it);
}

template <class Range, class T>
void replace(Range&& r, const T& old, const T& new_x)
{
    std::replace(r.begin(), r.end(), old, new_x);
}

template <class Iterator>
struct iterator_range
{
    Iterator start;
    Iterator last;

    Iterator begin() const { return start; }

    Iterator end() const { return last; }
};

template <class Iterator>
iterator_range<Iterator> range(Iterator start, Iterator last)
{
    return {start, last};
}

} // namespace migraph

#endif
