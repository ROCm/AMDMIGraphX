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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_BIT_FLAG_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_BIT_FLAG_HPP

#include <migraphx/config.hpp>
#include <migraphx/requires.hpp>
#include <type_traits>
#include <cstdint>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// Trait to enable type-safe bitwise operators for an enum class. Specialize
/// to std::true_type for any enum whose values are power-of-two bitmasks:
///
///     enum class my_flags : std::uint32_t
///     {
///         none  = 0,
///         alpha = 1 << 0,
///         beta  = 1 << 1,
///         gamma = 1 << 2,
///     };
///     template <>
///     struct bit_flag<my_flags> : std::true_type {};
///
/// This enables |, &, ^, ~, |=, &=, ^= on the enum, plus has_flag():
///
///     auto f = my_flags::alpha | my_flags::gamma;
///     if(has_flag(f, my_flags::alpha)) { /* ... */ }
///
template <class E>
struct bit_flag : std::false_type
{
};

template <class E, MIGRAPHX_REQUIRES(bit_flag<E>{})>
constexpr E operator|(E lhs, E rhs)
{
    using U = std::underlying_type_t<E>;
    return static_cast<E>(static_cast<U>(lhs) | static_cast<U>(rhs));
}

template <class E, MIGRAPHX_REQUIRES(bit_flag<E>{})>
constexpr E operator&(E lhs, E rhs)
{
    using U = std::underlying_type_t<E>;
    return static_cast<E>(static_cast<U>(lhs) & static_cast<U>(rhs));
}

template <class E, MIGRAPHX_REQUIRES(bit_flag<E>{})>
constexpr E operator^(E lhs, E rhs)
{
    using U = std::underlying_type_t<E>;
    return static_cast<E>(static_cast<U>(lhs) ^ static_cast<U>(rhs));
}

template <class E, MIGRAPHX_REQUIRES(bit_flag<E>{})>
constexpr E operator~(E val)
{
    using U = std::underlying_type_t<E>;
    return static_cast<E>(~static_cast<U>(val));
}

template <class E, MIGRAPHX_REQUIRES(bit_flag<E>{})>
constexpr E& operator|=(E & lhs, E rhs)
{ return lhs = lhs | rhs; }

template <class E, MIGRAPHX_REQUIRES(bit_flag<E>{})>
constexpr E& operator&=(E & lhs, E rhs)
{ return lhs = lhs & rhs; }

template <class E, MIGRAPHX_REQUIRES(bit_flag<E>{})>
constexpr E& operator^=(E & lhs, E rhs)
{ return lhs = lhs ^ rhs; }

template <class E, MIGRAPHX_REQUIRES(bit_flag<E>{})>
constexpr bool has_flag(E val, E flag)
{
    using U = std::underlying_type_t<E>;
    return (static_cast<U>(val) & static_cast<U>(flag)) == static_cast<U>(flag);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
