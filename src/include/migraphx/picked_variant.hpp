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

    // Hidden friend. The first variant parameter is constrained (via SFINAE on
    // its decayed type) to be this picked_variant specialization, which makes
    // this overload strictly more specialized than std::visit's fully-generic
    // `Variants&&...` for partial ordering. ADL through any picked_variant
    // argument routes here; subsequent variant arguments may be other variants
    // (std::variant or different picked_variant specializations) and are
    // forwarded to std::visit, which slices any picked_variant to its base.
    template <class Visitor,
              class V,
              class... Variants,
              MIGRAPHX_REQUIRES(std::is_same<std::decay_t<V>, picked_variant>{})>
    friend constexpr auto visit(Visitor&& vis, V&& pv, Variants&&... vars)
        MIGRAPHX_RETURNS(std::visit(std::forward<Visitor>(vis),
                                    as_variant(std::forward<V>(pv)),
                                    std::forward<Variants>(vars)...));
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_MIGRAPHX_PICKED_VARIANT_HPP
