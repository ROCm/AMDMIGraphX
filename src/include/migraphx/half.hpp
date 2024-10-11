/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <half/half.hpp>
#include <migraphx/config.hpp>
#include <migraphx/float8.hpp>
#include <migraphx/generic_float.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

using half = migraphx::generic_float<10,5>;

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

template <class T>
struct is_floating_point : std::false_type {};

template <unsigned int E, unsigned int M, unsigned int F>
struct is_floating_point<generic_float<E, M, F>> : std::true_type {};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

namespace std {

template<unsigned int E, unsigned int M, unsigned int F>
class numeric_limits<migraphx::generic_float<E, M, F>>
{
    public:
    static constexpr bool has_infinity = false;
    static constexpr migraphx::generic_float<E, M, F> epsilon() { return migraphx::generic_float<E, M, F>::epsilon(); }

    static constexpr migraphx::generic_float<E, M, F> quiet_NaN() { return migraphx::generic_float<E, M, F>::quiet_NaN(); }

    static constexpr migraphx::generic_float<E, M, F> max() { return migraphx::generic_float<E, M, F>::max(); }

    static constexpr migraphx::generic_float<E, M, F> min() { return migraphx::generic_float<E, M, F>::min(); }

    static constexpr migraphx::generic_float<E, M, F> lowest() { return migraphx::generic_float<E, M, F>::lowest(); }

};

template<unsigned int E, unsigned int M, unsigned int F, class T>
struct common_type<migraphx::generic_float<E, M, F>, T> : std::common_type<float, T> // NOLINT
{
};

template<unsigned int E, unsigned int M, unsigned int F, class T>
struct common_type<T, migraphx::generic_float<E, M, F>> : std::common_type<float, T> // NOLINT
{
};

template<unsigned int E, unsigned int M, unsigned int F, migraphx::fp8::f8_type T, bool FNUZ>
struct common_type<migraphx::generic_float<E, M, F>, migraphx::fp8::float8<T, FNUZ>> : std::common_type<float, float>
{};

template<unsigned int E, unsigned int M, unsigned int F, migraphx::fp8::f8_type T, bool FNUZ>
struct common_type<migraphx::fp8::float8<T, FNUZ>, migraphx::generic_float<E, M, F>> : std::common_type<float, float>
{};

template<unsigned int E, unsigned int M, unsigned int F>
struct common_type<migraphx::generic_float<E, M, F>,  migraphx::generic_float<E, M, F>>
{
    using type = migraphx::generic_float<E, M, F>;
};

} // namespace std

#endif
