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
};

// template<class Visitor, class... Variants>
// constexpr auto visit(Visitor&& vis, Variants&&... vars)
// MIGRAPHX_RETURNS(std::visit(std::forward<Visitor>(vis),
// as_variant(std::forward<Variants>(vars))...));

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_MIGRAPHX_PICKED_VARIANT_HPP
