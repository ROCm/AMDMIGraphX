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
#include <type_traits>
#include <utility>
#include <variant>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class Picker, class... Ts>
struct picked_variant : std::variant<Ts...>
{
    using base = std::variant<Ts...>;
    using base::base; // inherit default, in_place_type, in_place_index ctors

    template <class T, MIGRAPHX_REQUIRES(not std::is_base_of<base, std::decay_t<T>>{})>
    picked_variant(T&& x) : base(Picker::apply(std::forward<T>(x)))
    {
    }
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_MIGRAPHX_PICKED_VARIANT_HPP
