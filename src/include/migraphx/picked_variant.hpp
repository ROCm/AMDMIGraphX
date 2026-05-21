/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef MIGRAPHX_GUARD_MIGRAPHX_PICKED_VARIANT_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_PICKED_VARIANT_HPP

#include <migraphx/config.hpp>
#include <migraphx/requires.hpp>
#include <migraphx/returns.hpp>
#include <type_traits>
#include <utility>
#include <variant>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class Picker, class... Ts>
struct picked_variant;

template <class T>
struct is_picked_variant : std::false_type
{
};
template <class Picker, class... Ts>
struct is_picked_variant<picked_variant<Picker, Ts...>> : std::true_type
{
};

// Namespace-scope as_variant overloads for plain std::variant. Declared
// before picked_variant so its hidden-friend visit body can find them via
// ordinary lookup. For picked_variant arguments the class's hidden-friend
// as_variant wins (exact-type match beats the derived-to-base conversion
// these overloads would otherwise need).
template <class... Ts>
constexpr std::variant<Ts...>& as_variant(std::variant<Ts...>& v)
{
    return v;
}
template <class... Ts>
constexpr const std::variant<Ts...>& as_variant(const std::variant<Ts...>& v)
{
    return v;
}
template <class... Ts>
constexpr std::variant<Ts...>&& as_variant(std::variant<Ts...>&& v)
{
    return std::move(v);
}

template <class Picker, class... Ts>
struct picked_variant : std::variant<Ts...>
{
    using base_t = std::variant<Ts...>;
    using base_t::base_t; // inherit default, in_place_type, in_place_index ctors

    template <class T, MIGRAPHX_REQUIRES(not std::is_base_of<base_t, std::decay_t<T>>{}), class = decltype(Picker::apply(std::declval<T>()))>
    constexpr picked_variant(T&& x) : base_t(Picker::apply(std::forward<T>(x)))
    {
    }

    friend constexpr base_t& as_variant(picked_variant& x) { return x; }
    friend constexpr const base_t& as_variant(const picked_variant& x) { return x; }
    friend constexpr base_t&& as_variant(picked_variant&& x) { return std::move(x); }

    // Hidden friends.
    //
    // One overload per "position" of the picked_variant argument. Each takes a
    // forwarding reference constrained via SFINAE on the decayed type so a
    // single overload covers `&`, `const&`, and `&&`.
    //
    // Return type is `decltype(auto)` rather than a trailing `decltype(...)`
    // so the body is only substituted when the overload is actually selected.
    // std::visit is not required by the standard to be SFINAE-friendly and on
    // older libstdc++ (e.g. GCC 7 on SLES) it isn't, so a trailing decltype
    // around a std::visit call would produce hard errors during overload
    // resolution rather than cleanly removing the overload.
    //
    // Every variant argument is routed through `as_variant`. The namespace-
    // scope overloads above cover plain std::variant; together they ensure
    // std::visit only ever sees std::variant arguments and never the raw
    // picked_variant (which doesn't have std::variant_size specialized for
    // it).
    //
    // The position-2 overload handles calls where picked_variant is the
    // second variant (e.g. visit(f, std_v, pv)). Its SFINAE excludes the case
    // where the first variant is also a picked_variant -- those are handled
    // by the position-1 overload.

    template <class Visitor,
              class V,
              class... Variants,
              MIGRAPHX_REQUIRES(std::is_same<std::decay_t<V>, picked_variant>{})>
    friend constexpr decltype(auto) visit(Visitor&& vis, V&& pv, Variants&&... vars)
    {
        return std::visit(std::forward<Visitor>(vis),
                          as_variant(std::forward<V>(pv)),
                          as_variant(std::forward<Variants>(vars))...);
    }

    template <class Visitor,
              class V0,
              class V1,
              class... Variants,
              MIGRAPHX_REQUIRES(not is_picked_variant<std::decay_t<V0>>{} and
                                std::is_same<std::decay_t<V1>, picked_variant>{})>
    friend constexpr decltype(auto) visit(Visitor&& vis, V0&& v0, V1&& v1, Variants&&... vars)
    {
        return std::visit(std::forward<Visitor>(vis),
                          as_variant(std::forward<V0>(v0)),
                          as_variant(std::forward<V1>(v1)),
                          as_variant(std::forward<Variants>(vars))...);
    }
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_MIGRAPHX_PICKED_VARIANT_HPP
