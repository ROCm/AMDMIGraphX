/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef MIGRAPHX_GUARD_RTGLIB_HALF_HPP
#define MIGRAPHX_GUARD_RTGLIB_HALF_HPP

#if !defined(_WIN32)
#include <half/half.hpp>
#else
#include <half.hpp>
#endif
#include <migraphx/config.hpp>
#include <migraphx/float8.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

using half = half_float::half;

namespace detail {
template <class T>
struct deduce
{
    using type = T;
};

#ifdef HAS_HALF_V1
template <>
struct deduce<half_float::detail::expr>
{
    using type = half;
};
#endif
} // namespace detail

template <class T>
using deduce = typename detail::deduce<T>::type;

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

namespace std {

template <class T>
struct common_type<migraphx::half, T> : std::common_type<float, T> // NOLINT
{
};

template <class T>
struct common_type<T, migraphx::half> : std::common_type<float, T> // NOLINT
{
};

template <>
struct common_type<migraphx::fp8::fp8e4m3fnuz, migraphx::half>
{
    using type = float;
};

template <>
struct common_type<migraphx::half, migraphx::fp8::fp8e4m3fnuz>
{
    using type = float;
};

template <>
struct common_type<migraphx::half, migraphx::half>
{
    using type = migraphx::half;
};

} // namespace std

#endif
