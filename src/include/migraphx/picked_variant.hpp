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

    template <class T, MIGRAPHX_REQUIRES(not std::is_base_of<base_t, std::decay_t<T>>{})>
    constexpr picked_variant(T&& x) : base_t(Picker::apply(std::forward<T>(x)))
    {
    }

    friend constexpr base_t& as_variant(picked_variant& x) { return x; }
    friend constexpr const base_t& as_variant(const picked_variant& x) { return x; }
    friend constexpr base_t&& as_variant(picked_variant&& x) { return std::move(x); }

    // Hidden friends.
    //
    // Each takes picked_variant as a concrete (non-deduced) first variant
    // parameter, which is strictly more specialized than std::visit's deduced
    // `Variants&&...` per partial ordering on every compiler we target
    // (including GCC 7) -- unlike a forwarding-reference + SFINAE constraint,
    // whose effect on partial ordering older compilers don't always honor.
    //
    // The return type is deduced from the body (plain `auto`, not a trailing
    // `decltype`). This is deliberate: overload resolution only needs the
    // parameter types, so the body is only substituted into the overload that
    // is actually selected. If we used a trailing `decltype(std::visit(...))`,
    // all three overloads would have to substitute their return type during
    // viability checking -- and the const-lvalue overload would then invoke
    // std::visit on a `const base_t&`, which would instantiate the visitor
    // lambda's body with const arguments. That body isn't immediate context,
    // so a `x += 4`-style mutation would become a hard error during overload
    // resolution for a perfectly legal non-const call.
    //
    // Every variant argument is routed through `as_variant`, including the
    // trailing pack. The namespace-scope `as_variant` overloads below cover
    // plain std::variant; together they ensure std::visit only ever sees
    // std::variant arguments and never the raw picked_variant (which doesn't
    // have std::variant_size specialized for it).

    template <class Visitor, class... Variants>
    friend constexpr auto visit(Visitor&& vis, picked_variant& pv, Variants&&... vars)
    {
        return std::visit(std::forward<Visitor>(vis),
                          as_variant(pv),
                          as_variant(std::forward<Variants>(vars))...);
    }

    template <class Visitor, class... Variants>
    friend constexpr auto visit(Visitor&& vis, const picked_variant& pv, Variants&&... vars)
    {
        return std::visit(std::forward<Visitor>(vis),
                          as_variant(pv),
                          as_variant(std::forward<Variants>(vars))...);
    }

    template <class Visitor, class... Variants>
    friend constexpr auto visit(Visitor&& vis, picked_variant&& pv, Variants&&... vars)
    {
        return std::visit(std::forward<Visitor>(vis),
                          as_variant(std::move(pv)),
                          as_variant(std::forward<Variants>(vars))...);
    }

    // Position-2 overloads handle calls where picked_variant is the second
    // variant argument (e.g. visit(f, some_std_variant, pv)). The SFINAE on V0
    // excludes the case where the first variant is also a picked_variant --
    // those are handled by the position-1 overloads above, which would
    // otherwise be ambiguous with these.
    template <class Visitor,
              class V0,
              class... Variants,
              MIGRAPHX_REQUIRES(not is_picked_variant<std::decay_t<V0>>{})>
    friend constexpr auto visit(Visitor&& vis, V0&& v0, picked_variant& pv, Variants&&... vars)
    {
        return std::visit(std::forward<Visitor>(vis),
                          as_variant(std::forward<V0>(v0)),
                          as_variant(pv),
                          as_variant(std::forward<Variants>(vars))...);
    }

    template <class Visitor,
              class V0,
              class... Variants,
              MIGRAPHX_REQUIRES(not is_picked_variant<std::decay_t<V0>>{})>
    friend constexpr auto
    visit(Visitor&& vis, V0&& v0, const picked_variant& pv, Variants&&... vars)
    {
        return std::visit(std::forward<Visitor>(vis),
                          as_variant(std::forward<V0>(v0)),
                          as_variant(pv),
                          as_variant(std::forward<Variants>(vars))...);
    }

    template <class Visitor,
              class V0,
              class... Variants,
              MIGRAPHX_REQUIRES(not is_picked_variant<std::decay_t<V0>>{})>
    friend constexpr auto visit(Visitor&& vis, V0&& v0, picked_variant&& pv, Variants&&... vars)
    {
        return std::visit(std::forward<Visitor>(vis),
                          as_variant(std::forward<V0>(v0)),
                          as_variant(std::move(pv)),
                          as_variant(std::forward<Variants>(vars))...);
    }
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_MIGRAPHX_PICKED_VARIANT_HPP
