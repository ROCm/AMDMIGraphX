#ifndef MIGRAPHX_GUARD_MIGRAPHX_UTILITY_OPERATORS_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_UTILITY_OPERATORS_HPP

#include <migraphx/config.hpp>
#include <migraphx/requires.hpp>
#include <migraphx/returns.hpp>
#include <type_traits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class X>
struct equality_comparable
{
    struct private_ops
    {
        template<class U, MIGRAPHX_REQUIRES(std::is_same<U, X>{})>
        static constexpr auto equal1(const U& x, const X& y) MIGRAPHX_RETURNS(x == y);

        template<class T, class U, MIGRAPHX_REQUIRES(std::is_same<T, X>{} and not std::is_convertible<U, X>{})>
        static constexpr auto equal2(const T& x, const U& y) MIGRAPHX_RETURNS(x.operator==(y));
    };

    template <class T, class U>
    friend constexpr auto operator !=(const T& x, const U& y) MIGRAPHX_RETURNS(not (private_ops::equal1(x, y)));

    template <class U, class T>
    friend constexpr auto operator ==(const U& x, const T& y) MIGRAPHX_RETURNS(private_ops::equal2(y, x));
    template <class U, class T>
    friend constexpr auto operator !=(const U& x, const T& y) MIGRAPHX_RETURNS(not (private_ops::equal2(y, x)));
};

template <class X>
struct less_than_comparable
{

    struct private_ops
    {
        // template<class U, MIGRAPHX_REQUIRES(std::is_same<U, X>{})>
        // static constexpr auto less1(const U& x, const X& y) MIGRAPHX_RETURNS(x.operator<(y));

        // template<class T, class U, MIGRAPHX_REQUIRES(std::is_same<T, X>{} and not std::is_convertible<U, X>{})>
        // static constexpr auto less2(const T& x, const U& y) MIGRAPHX_RETURNS(x.operator<(y));
        // template<class T, class U, MIGRAPHX_REQUIRES(std::is_same<T, X>{} and not std::is_same<U, X>{})>
        // static constexpr auto greater2(const T& x, const U& y) MIGRAPHX_RETURNS(x.operator>(y));

        template<class U, MIGRAPHX_REQUIRES(std::is_same<U, X>{})>
        static constexpr auto less1(const U& x, const X& y) MIGRAPHX_RETURNS(x < y);
        template<class U, MIGRAPHX_REQUIRES(not std::is_convertible<U, X>{})>
        static constexpr auto less2(const X& x, const U& y) MIGRAPHX_RETURNS(x.operator<(y));
        template<class U, MIGRAPHX_REQUIRES(not std::is_same<U, X>{})>
        static constexpr auto greater2(const X& x, const U& y) MIGRAPHX_RETURNS(x.operator>(y));

    };

    template <class T, class U>
    friend constexpr auto operator>(const T& x, const U& y) MIGRAPHX_RETURNS(private_ops::less1(y, x));
    template <class T, class U>
    friend constexpr auto operator <=(const T& x, const U& y) MIGRAPHX_RETURNS(not (private_ops::less1(y, x)));
    template <class T, class U>
    friend constexpr auto operator >=(const T& x, const U& y) MIGRAPHX_RETURNS(not (private_ops::less1(x, y)));

    template <class T, class U>
    friend constexpr auto operator <=(const T& x, const U& y) MIGRAPHX_RETURNS(not (private_ops::greater2(x, y)));
    template <class T, class U>
    friend constexpr auto operator >=(const T& x, const U& y) MIGRAPHX_RETURNS(not (private_ops::less2(x, y)));

    template <class U, class T>
    friend constexpr auto operator<(const U& x, const T& y) MIGRAPHX_RETURNS(private_ops::greater2(y, x));
    template <class U, class T>
    friend constexpr auto operator>(const U& x, const T& y) MIGRAPHX_RETURNS(private_ops::less2(y, x));
    template <class U, class T>
    friend constexpr auto operator <=(const U& x, const T& y) MIGRAPHX_RETURNS(not (private_ops::less2(y, x)));
    template <class U, class T>
    friend constexpr auto operator >=(const U& x, const T& y) MIGRAPHX_RETURNS(not (private_ops::greater2(y, x)));
};

template<class X>
struct totally_ordered
: equality_comparable<X>, less_than_comparable<X>
{};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_UTILITY_OPERATORS_HPP
