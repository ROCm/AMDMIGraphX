#ifndef MIGRAPHX_GUARD_RTGLIB_REFLECT_HPP
#define MIGRAPHX_GUARD_RTGLIB_REFLECT_HPP

#include <migraphx/functional.hpp>
#include <migraphx/rank.hpp>
#include <migraphx/config.hpp>
#include <functional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace detail {

struct reflect_placeholder
{
    template <class... Ts>
    int operator()(Ts&&...) const
    {
        return 0;
    }
};

template <class T, class Selector>
auto reflect_impl(rank<1>, T& x, Selector f) -> decltype(T::reflect(x, f))
{
    return T::reflect(x, std::move(f));
}

template <class T, class Selector>
auto reflect_impl(rank<0>, T&, Selector)
{
    return pack();
}

template <class T>
auto reflectable_impl(rank<1>, T&& x)
    -> decltype(T::reflect(x, reflect_placeholder{}), std::true_type{});

template <class T>
auto reflectable_impl(rank<0>, T &&) -> decltype(std::false_type{});

template <class T>
struct remove_rvalue_reference
{
    using type = T;
};

template <class T>
struct remove_rvalue_reference<T&&>
{
    using type = T;
};

template <class T>
struct wrapper
{
    using type = typename remove_rvalue_reference<T>::type;
    type data;
    type get() const { return data; }
};

template <class T>
wrapper<T> wrap(std::remove_reference_t<T>& x)
{
    return wrapper<T>{std::forward<T>(x)};
}

template <class... Ts>
std::tuple<typename remove_rvalue_reference<Ts>::type...> auto_tuple(Ts&&... xs)
{
    return {std::forward<Ts>(xs)...};
}

} // namespace detail

template <class T>
using is_reflectable = decltype(detail::reflectable_impl(rank<1>{}, std::declval<T>()));

template <class T, class Selector>
auto reflect(T& x, Selector f)
{
    return detail::reflect_impl(rank<1>{}, x, std::move(f));
}

template <class T>
auto reflect_tie(T& x)
{
    return reflect(x, [](auto&& y, auto&&...) { return detail::wrap<decltype(y)>(y); })(
        [](auto&&... xs) { return detail::auto_tuple(xs.get()...); });
}

template <class T, class F>
void reflect_each(T& x, F f)
{
    return reflect(x, [](auto&& y, auto... ys) {
        return pack(detail::wrap<decltype(y)>(y), ys...);
    })([&](auto&&... xs) {
        each_args([&](auto p) { p([&](auto&& y, auto... ys) { f(y.get(), ys...); }); }, xs...);
    });
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
