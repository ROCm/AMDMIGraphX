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
#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_DIM_LIKE_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_DIM_LIKE_HPP

#include <cstdint>
#include <ostream>
#include <type_traits>
#include <utility>
#include <variant>

#include <migraphx/config.hpp>
#include <migraphx/requires.hpp>
#include <migraphx/shape.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct value;

// A dim attribute entry that may be either a plain int64_t or a
// dynamic_dimension. Used by ops whose dim-valued attributes need to carry
// either static integers or dynamic/symbolic dimensions.
struct MIGRAPHX_EXPORT dim_like
{
    std::variant<int64_t, shape::dynamic_dimension> value = int64_t{0};

    constexpr dim_like() = default;

    template <class T, MIGRAPHX_REQUIRES(std::is_integral<T>{})>
    constexpr dim_like(T v) : value{static_cast<int64_t>(v)} // NOLINT(google-explicit-constructor)
    {
    }

    dim_like(shape::dynamic_dimension d) // NOLINT(google-explicit-constructor)
        : value{std::move(d)}
    {
    }

    friend bool operator==(const dim_like& a, const dim_like& b) { return a.value == b.value; }
    friend bool operator!=(const dim_like& a, const dim_like& b) { return not(a == b); }

    MIGRAPHX_EXPORT friend std::ostream& operator<<(std::ostream& os, const dim_like& d);

    migraphx::value to_value() const;
    void from_value(const migraphx::value& v);
};

template <class T>
bool holds_alternative(const dim_like& d)
{
    return std::holds_alternative<T>(d.value);
}

template <class T>
const T& get(const dim_like& d)
{
    return std::get<T>(d.value);
}

template <class T>
T& get(dim_like& d)
{
    return std::get<T>(d.value);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
