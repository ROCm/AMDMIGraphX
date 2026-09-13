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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_CHECKED_OPS_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_CHECKED_OPS_HPP

#include <migraphx/errors.hpp>
#include <migraphx/requires.hpp>
#include <type_traits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class T, MIGRAPHX_REQUIRES(std::is_integral<T>{})>
constexpr T checked_mul(T a, T b)
{
    T c{};
    if(__builtin_mul_overflow(a, b, &c))
        MIGRAPHX_THROW("Integer overflow in multiplication");
    return c;
}

template <class T, MIGRAPHX_REQUIRES(std::is_integral<T>{})>
constexpr T checked_add(T a, T b)
{
    T c{};
    if(__builtin_add_overflow(a, b, &c))
        MIGRAPHX_THROW("Integer overflow in addition");
    return c;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
