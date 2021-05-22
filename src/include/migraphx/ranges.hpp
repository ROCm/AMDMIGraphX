#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_RANGES_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_RANGES_HPP

#include <algorithm>
#include <vector>
#include <initializer_list>
#include <migraphx/rank.hpp>
#include <migraphx/iota_iterator.hpp>
#include <migraphx/type_name.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/requires.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace detail {

template <class String, class T>
auto generic_find_impl(rank<2>, String&& s, const T& x) -> decltype(s.npos, s.begin() + s.find(x))
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

template <class C, class T>
auto generic_find_at_impl(rank<1>, C&& c, const T& x) -> decltype(c.find(x))
{
    return c.find(x);
}

template <class C, class T>
auto generic_find_at_impl(rank<0>, C&& c, const T& x)
{
    auto n = std::distance(c.begin(), c.end());
    if(x >= n)
        return c.end();
    return std::next(c.begin(), x);
}

template <class C, class T, class = typename C::mapped_type>
decltype(auto) generic_at_impl(rank<1>, const C&, T&& it)
{
    return it->second;
}

template <class C, class T>
decltype(auto) generic_at_impl(rank<0>, const C&, T&& it)
{
    return *it;
}

struct empty
{
};

} // namespace detail

template <class C, class T>
auto generic_find(C&& c, const T& x)
{
    return detail::generic_find_impl(rank<2>{}, c, x);
}

template <class C, class T>
decltype(auto) at(C&& c, const T& x, const std::string& msg = "")
{
    auto it = detail::generic_find_at_impl(rank<2>{}, c, x);
    if(it == c.end())
    {
        if(msg.empty())
            MIGRAPHX_THROW("At operator out of range for " + get_type_name(c));
        else
            MIGRAPHX_THROW(msg);
    }
    return detail::generic_at_impl(rank<2>{}, c, it);
}

template <class C, class T>
bool contains(const C& c, const T& x)
{
    return generic_find(c, x) != c.end();
}

template <class T>
bool contains(const std::initializer_list<T>& c, const T& x)
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

template <class Predicate>
bool all_of(detail::empty, const Predicate&)
{
    return true;
}

template <class C, class Predicate>
bool any_of(const C& c, const Predicate& p)
{
    return std::any_of(c.begin(), c.end(), p);
}

template <class T, class Predicate>
bool any_of(const std::initializer_list<T>& c, const Predicate& p)
{
    return std::any_of(c.begin(), c.end(), p);
}

template <class Predicate>
bool any_of(detail::empty, const Predicate&)
{
    return false;
}

template <class C, class Predicate>
bool none_of(const C& c, const Predicate& p)
{
    return std::none_of(c.begin(), c.end(), p);
}

template <class T, class Predicate>
bool none_of(const std::initializer_list<T>& c, const Predicate& p)
{
    return std::none_of(c.begin(), c.end(), p);
}

template <class Predicate>
bool none_of(detail::empty, const Predicate&)
{
    return true;
}

template <class Range, class Iterator>
void copy(Range&& r, Iterator it)
{
    std::copy(r.begin(), r.end(), it);
}

template <class Range>
auto reverse(Range& r)
{
    return range(std::make_reverse_iterator(r.end()), std::make_reverse_iterator(r.begin()));
}

template <class Range, class T>
void replace(Range&& r, const T& old, const T& new_x)
{
    std::replace(r.begin(), r.end(), old, new_x);
}

template <class R1, class R2>
bool equal(R1&& r1, R2&& r2)
{
    return std::equal(r1.begin(), r1.end(), r2.begin(), r2.end());
}

template <class R>
using range_value = std::decay_t<decltype(*std::declval<R>().begin())>;

template <class Range, class Predicate>
std::vector<range_value<Range>> find_all(Range&& r, Predicate p)
{
    std::vector<range_value<Range>> result;
    std::copy_if(r.begin(), r.end(), std::back_inserter(result), p);
    return result;
}

template <class Iterator>
struct iterator_range
{
    Iterator start;
    Iterator last;

    Iterator begin() const { return start; }

    Iterator end() const { return last; }
};

template <class Iterator, MIGRAPHX_REQUIRES(not std::is_integral<Iterator>{})>
iterator_range<Iterator> range(Iterator start, Iterator last)
{
    return {start, last};
}

inline iterator_range<iota_iterator> range(std::ptrdiff_t start, std::ptrdiff_t last)
{
    return {{start, {}}, {last, {}}};
}
inline iterator_range<iota_iterator> range(std::ptrdiff_t last) { return range(0, last); }

template <class Iterator>
iterator_range<Iterator> range(std::pair<Iterator, Iterator> p)
{
    return {p.first, p.second};
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
