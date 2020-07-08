#ifndef MIGRAPHX_GUARD_RTGLIB_SERIALIZE_HPP
#define MIGRAPHX_GUARD_RTGLIB_SERIALIZE_HPP

#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/requires.hpp>
#include <migraphx/rank.hpp>
#include <type_traits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class T>
value to_value(const T& x);

template <class T>
void from_value(const value& v, T& x);

template <class T>
T from_value(const value& v)
{
    T x;
    from_value(v, x);
    return x;
}

namespace detail {

template <class T, class U>
value to_value_impl(rank<0>, const std::pair<T, U>& x)
{

    return {x.first, x.second};
}

template <class T>
auto to_value_impl(rank<1>, const T& x) -> decltype(x.begin(), x.end(), value{})
{
    value result;
    for(auto&& y : x)
    {
        result.insert(to_value(y));
    }
    return result;
}

template <class T, MIGRAPHX_REQUIRES(is_reflectable<T>{})>
value to_value_impl(rank<2>, const T& x)
{
    value result;
    reflect_each(x, [&](auto&& y, std::string name) { result.emplace(name, to_value(y)); });
    return result;
}

template <class T, MIGRAPHX_REQUIRES(std::is_signed<T>{})>
value to_value_impl(rank<3>, const T& x)
{
    return {std::int64_t{x}};
}

template <class T, MIGRAPHX_REQUIRES(std::is_unsigned<T>{})>
value to_value_impl(rank<4>, const T& x)
{
    return {std::uint64_t{x}};
}

template <class T, MIGRAPHX_REQUIRES(std::is_floating_point<T>{})>
value to_value_impl(rank<5>, const T& x)
{
    return {double{x}};
}

inline value to_value_impl(rank<6>, const std::string& x) { return {x}; }

template <class T>
auto to_value_impl(rank<7>, const T& x) -> decltype(migraphx_to_value(x))
{
    return migraphx_to_value(x);
}

template <class T>
auto from_value_impl(rank<0>, const value& v, T& x)
    -> decltype(x.insert(x.end(), *x.begin()), void())
{
    x.clear();
    for(auto&& e : v)
        x.insert(x.end(), from_value<typename T::value_type>(e));
}

template <class T>
auto from_value_impl(rank<0>, const value& v, T& x) -> decltype(x.insert(*x.begin()), void())
{
    x.clear();
    for(auto&& e : v)
        x.emplace(e.get_key(), from_value<typename T::mapped_type>(e));
}

template <class T, MIGRAPHX_REQUIRES(is_reflectable<T>{})>
void from_value_impl(rank<0>, const value& v, T& x)
{
    reflect_each(x, [&](auto&& y, std::string name) {
        using type = std::decay_t<decltype(y)>;
        y          = from_value<type>(v.at(name));
    });
}

template <class T, MIGRAPHX_REQUIRES(std::is_signed<T>{})>
void from_value_impl(rank<0>, const value& v, T& x)
{
    x = v.get_int64();
}

template <class T, MIGRAPHX_REQUIRES(std::is_unsigned<T>{})>
void from_value_impl(rank<0>, const value& v, T& x)
{
    x = v.get_uint64();
}

template <class T, MIGRAPHX_REQUIRES(std::is_floating_point<T>{})>
void from_value_impl(rank<0>, const value& v, T& x)
{
    x = v.get_float();
}

} // namespace detail

template <class T>
value to_value(const T& x)
{
    return detail::to_value_impl(rank<8>{}, x);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
