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
#ifndef MIGRAPHX_GUARD_KERNELS_MATH_HPP
#define MIGRAPHX_GUARD_KERNELS_MATH_HPP

#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/vec.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/type_traits.hpp>
#include <hip/hip_fp16.h>
#include <hip/math_functions.h>

namespace migraphx {

namespace math {
constexpr float as_float(migraphx::half x) { return x; }
template <class T>
constexpr T as_float(T x)
{
    return x;
}
} // namespace math

// NOLINTNEXTLINE
#define MIGRAPHX_DEVICE_MATH(name, fname)                              \
    template <class... Ts, MIGRAPHX_REQUIRES(not is_any_vec<Ts...>())> \
    auto __device__ name(Ts... xs) MIGRAPHX_RETURNS(fname(xs...))

// NOLINTNEXTLINE
#define MIGRAPHX_DEVICE_MATH_VEC(name)                                       \
    template <class... Ts, MIGRAPHX_REQUIRES(is_any_vec<Ts...>())>           \
    auto __device__ name(Ts... xs)                                           \
    {                                                                        \
        return vec_transform(xs...)([](auto... ys) { return name(ys...); }); \
    }

// NOLINTNEXTLINE
#define MIGRAPHX_DEVICE_MATH_FOR(type, name, fname)                    \
    template <class... Ts, MIGRAPHX_REQUIRES(not is_any_vec<Ts...>())> \
    auto __device__ name(type x, Ts... xs)->type                       \
    {                                                                  \
        return fname(x, xs...);                                        \
    }

// NOLINTNEXTLINE
#define MIGRAPHX_DEVICE_MATH_BINARY_FOR(type, name, fname) \
    inline auto __device__ name(type x, type y)->type { return fname(x, y); }

// NOLINTNEXTLINE
#define MIGRAPHX_DEVICE_MATH_HALF(name, fname)                         \
    template <class... Ts, MIGRAPHX_REQUIRES(not is_any_vec<Ts...>())> \
    auto __device__ name(migraphx::half x, Ts... xs)                   \
        MIGRAPHX_RETURNS(fname(math::as_float(x), math::as_float(xs)...))

// Template with two overloads for math functions, one for half2 type and one for more generic
// <half, N> vectorization where N is 4 or another even number.

// NOLINTNEXTLINE
#define MIGRAPHX_DEVICE_MATH_HALF2(name, fname)                                           \
    template <class... Ts>                                                                \
    auto __device__ name(migraphx::vec<migraphx::half, 2> x, Ts... xs)                    \
        MIGRAPHX_RETURNS(migraphx::vec<migraphx::half, 2>{fname(x, xs...)});              \
    template <class... Ts, index_int N, MIGRAPHX_REQUIRES(N % 2 == 0 && (N > 2))>         \
    auto __device__ name(migraphx::vec<migraphx::half, N> x, Ts... xs)                    \
    {                                                                                     \
        return vec_packed_transform<2>(x, xs...)(                                         \
            [](auto... ys) -> migraphx::vec<migraphx::half, 2> { return fname(ys...); }); \
    }

MIGRAPHX_DEVICE_MATH(abs, ::abs)
MIGRAPHX_DEVICE_MATH(acos, ::acos)
MIGRAPHX_DEVICE_MATH(acosh, ::acosh)
MIGRAPHX_DEVICE_MATH(asin, ::asin)
MIGRAPHX_DEVICE_MATH(asinh, ::asinh)
MIGRAPHX_DEVICE_MATH(atan, ::atan)
MIGRAPHX_DEVICE_MATH(atanh, ::atanh)
MIGRAPHX_DEVICE_MATH(ceil, ::ceil)
MIGRAPHX_DEVICE_MATH(cos, ::cos)
MIGRAPHX_DEVICE_MATH(cosh, ::cosh)
MIGRAPHX_DEVICE_MATH(erf, ::erf)
MIGRAPHX_DEVICE_MATH(exp, ::exp)
MIGRAPHX_DEVICE_MATH(floor, ::floor)
MIGRAPHX_DEVICE_MATH(isnan, ::isnan)
MIGRAPHX_DEVICE_MATH(log, ::log)
MIGRAPHX_DEVICE_MATH(pow, ::pow)
MIGRAPHX_DEVICE_MATH(round, ::round)
MIGRAPHX_DEVICE_MATH(rsqrt, ::rsqrt)
MIGRAPHX_DEVICE_MATH(sin, ::sin)
MIGRAPHX_DEVICE_MATH(sinh, ::sinh)
MIGRAPHX_DEVICE_MATH(sqrt, ::sqrt)
MIGRAPHX_DEVICE_MATH(tan, ::tan)
MIGRAPHX_DEVICE_MATH(tanh, ::tanh)

// Float overloads
MIGRAPHX_DEVICE_MATH_FOR(float, acos, ::acosf)
MIGRAPHX_DEVICE_MATH_FOR(float, acosh, ::acoshf)
MIGRAPHX_DEVICE_MATH_FOR(float, asin, ::asinf)
MIGRAPHX_DEVICE_MATH_FOR(float, asinh, ::asinhf)
MIGRAPHX_DEVICE_MATH_FOR(float, atan, ::atanf)
MIGRAPHX_DEVICE_MATH_FOR(float, atanh, ::atanhf)
MIGRAPHX_DEVICE_MATH_FOR(float, cos, ::cosf)
MIGRAPHX_DEVICE_MATH_FOR(float, cosh, ::coshf)
MIGRAPHX_DEVICE_MATH_FOR(float, rsqrt, ::rsqrtf)
MIGRAPHX_DEVICE_MATH_FOR(float, sin, ::sinf)
MIGRAPHX_DEVICE_MATH_FOR(float, sinh, ::sinhf)
MIGRAPHX_DEVICE_MATH_FOR(float, tan, ::tanf)
MIGRAPHX_DEVICE_MATH_FOR(float, tanh, ::tanhf)

// Builtin half functions
MIGRAPHX_DEVICE_MATH_FOR(migraphx::half, abs, ::__habs)
MIGRAPHX_DEVICE_MATH_FOR(migraphx::half, exp, ::hexp)
MIGRAPHX_DEVICE_MATH_FOR(migraphx::half, log, ::hlog)
MIGRAPHX_DEVICE_MATH_FOR(migraphx::half, rsqrt, ::hrsqrt)
MIGRAPHX_DEVICE_MATH_FOR(migraphx::half, sqrt, ::hsqrt)

// Use float to compute half overload
MIGRAPHX_DEVICE_MATH_HALF(acos, ::acos)
MIGRAPHX_DEVICE_MATH_HALF(acosh, ::acosh)
MIGRAPHX_DEVICE_MATH_HALF(asin, ::asin)
MIGRAPHX_DEVICE_MATH_HALF(asinh, ::asinh)
MIGRAPHX_DEVICE_MATH_HALF(atan, ::atan)
MIGRAPHX_DEVICE_MATH_HALF(atanh, ::atanh)
MIGRAPHX_DEVICE_MATH_HALF(ceil, ::ceil)
MIGRAPHX_DEVICE_MATH_HALF(cos, ::cos)
MIGRAPHX_DEVICE_MATH_HALF(cosh, ::cosh)
MIGRAPHX_DEVICE_MATH_HALF(erf, ::erf)
MIGRAPHX_DEVICE_MATH_HALF(floor, ::floor)
MIGRAPHX_DEVICE_MATH_HALF(isnan, ::isnan)
MIGRAPHX_DEVICE_MATH_HALF(pow, ::pow)
MIGRAPHX_DEVICE_MATH_HALF(round, ::round)
MIGRAPHX_DEVICE_MATH_HALF(sin, ::sin)
MIGRAPHX_DEVICE_MATH_HALF(sinh, ::sinh)
MIGRAPHX_DEVICE_MATH_HALF(tan, ::tan)
MIGRAPHX_DEVICE_MATH_HALF(tanh, ::tanh)

// Map math functions to hip half2 functions
// The half2 type is defined in include/hip/amd_detail/hip_fp16_gcc.h and is 2 16-bit floats
// packed into a 32-bit number.  See include/hip/amd_detail/hip_fp16_math_fwd.h for the HIP names
// Most but not all of these math ops have operators of the same names.  Ones not yet implemented
// at this time are: exp2, exp10, log2, log10, isinf
MIGRAPHX_DEVICE_MATH_HALF2(abs, ::__habs2)
MIGRAPHX_DEVICE_MATH_HALF2(ceil, ::h2ceil)
MIGRAPHX_DEVICE_MATH_HALF2(floor, ::h2floor)
MIGRAPHX_DEVICE_MATH_HALF2(sin, ::h2sin)
MIGRAPHX_DEVICE_MATH_HALF2(cos, ::h2cos)
MIGRAPHX_DEVICE_MATH_HALF2(exp, ::h2exp)
MIGRAPHX_DEVICE_MATH_HALF2(exp2, ::h2exp2)
MIGRAPHX_DEVICE_MATH_HALF2(exp10, ::h2exp10)
MIGRAPHX_DEVICE_MATH_HALF2(log2, ::h2log2)
MIGRAPHX_DEVICE_MATH_HALF2(log, ::h2log)
MIGRAPHX_DEVICE_MATH_HALF2(log10, ::h2log10)
MIGRAPHX_DEVICE_MATH_HALF2(rsqrt, ::h2rsqrt)
MIGRAPHX_DEVICE_MATH_HALF2(sqrt, ::h2sqrt)
MIGRAPHX_DEVICE_MATH_HALF2(isinf, ::__hisinf2)
MIGRAPHX_DEVICE_MATH_HALF2(isnan, ::__hisnan2)

template <class T, class U>
constexpr auto where(bool cond, const T& a, const U& b)
{
    return cond ? a : b;
}

MIGRAPHX_DEVICE_MATH_BINARY_FOR(float, max, ::max)
MIGRAPHX_DEVICE_MATH_BINARY_FOR(float, min, ::min)
MIGRAPHX_DEVICE_MATH_BINARY_FOR(double, max, ::max)
MIGRAPHX_DEVICE_MATH_BINARY_FOR(double, min, ::min)
// Add overloads for half that calls the float version
MIGRAPHX_DEVICE_MATH_BINARY_FOR(migraphx::half, max, ::fmaxf)
MIGRAPHX_DEVICE_MATH_BINARY_FOR(migraphx::half, min, ::fminf)

template <class T, MIGRAPHX_REQUIRES(not is_any_vec<T>())>
constexpr auto max(const T& a, const T& b)
{
    return where(a < b, b, a);
}

template <class T, MIGRAPHX_REQUIRES(not is_any_vec<T>())>
constexpr auto min(const T& a, const T& b)
{
    return where(a < b, a, b);
}

template <class T, class U, MIGRAPHX_REQUIRES(not is_same<T, U>{} and not is_any_vec<T, U>())>
constexpr auto max(const T& a, const U& b)
{
    return max<common_type_t<T, U>>(a, b);
}

template <class T, class U, MIGRAPHX_REQUIRES(not is_same<T, U>{} and not is_any_vec<T, U>())>
constexpr auto min(const T& a, const U& b)
{
    return min<common_type_t<T, U>>(a, b);
}

MIGRAPHX_DEVICE_MATH_VEC(abs)
MIGRAPHX_DEVICE_MATH_VEC(acos)
MIGRAPHX_DEVICE_MATH_VEC(acosh)
MIGRAPHX_DEVICE_MATH_VEC(asin)
MIGRAPHX_DEVICE_MATH_VEC(asinh)
MIGRAPHX_DEVICE_MATH_VEC(atan)
MIGRAPHX_DEVICE_MATH_VEC(atanh)
MIGRAPHX_DEVICE_MATH_VEC(ceil)
MIGRAPHX_DEVICE_MATH_VEC(cos)
MIGRAPHX_DEVICE_MATH_VEC(cosh)
MIGRAPHX_DEVICE_MATH_VEC(erf)
MIGRAPHX_DEVICE_MATH_VEC(exp)
MIGRAPHX_DEVICE_MATH_VEC(floor)
MIGRAPHX_DEVICE_MATH_VEC(isnan)
MIGRAPHX_DEVICE_MATH_VEC(log)
MIGRAPHX_DEVICE_MATH_VEC(max)
MIGRAPHX_DEVICE_MATH_VEC(min)
MIGRAPHX_DEVICE_MATH_VEC(pow)
MIGRAPHX_DEVICE_MATH_VEC(round)
MIGRAPHX_DEVICE_MATH_VEC(rsqrt)
MIGRAPHX_DEVICE_MATH_VEC(sin)
MIGRAPHX_DEVICE_MATH_VEC(sinh)
MIGRAPHX_DEVICE_MATH_VEC(sqrt)
MIGRAPHX_DEVICE_MATH_VEC(tan)
MIGRAPHX_DEVICE_MATH_VEC(tanh)
MIGRAPHX_DEVICE_MATH_VEC(where)

template <class T, class U>
constexpr auto convert(U v)
{
    return vec_transform(v)([](auto x) -> T { return x; });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_MATH_HPP
