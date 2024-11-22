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
#ifndef MIGRAPHX_GUARD_KERNELS_MATH_HPP
#define MIGRAPHX_GUARD_KERNELS_MATH_HPP

#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/vec.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/type_traits.hpp>
#include <migraphx/kernels/hip.hpp>
#include <migraphx/kernels/float8.hpp>
#include <migraphx/kernels/pp.hpp>

namespace migraphx {

namespace math {
// constexpr float as_float(migraphx::half x) { return x; }

// constexpr float as_float(migraphx::fp8::fp8e4m3fnuz x) { return x; }
// constexpr float as_float(migraphx::fp8::fp8e4m3fn x) { return x; }
// constexpr float as_float(migraphx::fp8::fp8e5m2 x) { return x; }

template <class T>
constexpr auto as_float(T x)
{
    if constexpr(is_integral<T>{})
        return x;
    else
        return float(x);
}

template<class Sig, class F>
struct function_wrapper;

template<class Result, class... Ts, class F>
struct function_wrapper<Result(*)(Ts...) noexcept, F>
{
    F f;

    constexpr Result operator()(Ts... xs) const noexcept
    {
        return f(xs...);
    }
};

template<class Result, class... Ts, class F>
struct function_wrapper<Result(*)(Ts...), F>
{
    F f;

    constexpr Result operator()(Ts... xs) const
    {
        return f(xs...);
    }
};

// template<class F>
// struct function_wrapper;

// template<class Result, class... Ts>
// struct function_wrapper<Result(*)(Ts...) noexcept>
// {
//     using type = Result(*)(Ts...) noexcept;
//     type f;

//     constexpr Result operator()(Ts... xs) const noexcept
//     {
//         return f(xs...);
//     }
// };

// template<class Result, class... Ts>
// struct function_wrapper<Result(*)(Ts...)>
// {
//     using type = Result(*)(Ts...);
//     type f;

//     constexpr Result operator()(Ts... xs) const
//     {
//         return f(xs...);
//     }
// };

template<class Sig, class F>
constexpr function_wrapper<Sig, F> make_function_wrapper(Sig, F f)
{
    return {f};
}

template<class Key, class T, class = void>
struct disable_wrap : false_type {};

template<class F, class T, class... Ts, MIGRAPHX_REQUIRES(not is_any_vec<T, Ts...>())>
__device__ auto wrap(F f, T x, Ts... xs)
{
    if constexpr(is_integral<T>{})
    {
        return wrap(f, double(x), double(xs)...);
    }
    else if constexpr(is_callable<F, T, Ts...>{})
    {
        return f(x, xs...);
    }
    else
    {
        T result = f(as_float(x), as_float(xs)...);
        return result;
    }
}

} // namespace math

#define MIGRAPHX_DEVICE_MATH_LIFT(...) make_function_wrapper(&__VA_ARGS__, MIGRAPHX_LIFT(__VA_ARGS__))

// NOLINTNEXTLINE
#define MIGRAPHX_DEVICE_MATH_WRAP(name, ...)                              \
    namespace math { inline static constexpr auto wrap_##name = overload(MIGRAPHX_PP_TRANSFORM_ARGS(MIGRAPHX_DEVICE_MATH_LIFT, __VA_ARGS__)); } \
    template <class... Ts> \
    auto __device__ name(Ts... xs) MIGRAPHX_RETURNS(math::wrap(math::wrap_##name, xs...))


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
    auto __device__ name(type x, Ts... xs) -> type                     \
    {                                                                  \
        return fname(x, xs...);                                        \
    }

// NOLINTNEXTLINE
#define MIGRAPHX_DEVICE_MATH_BINARY_FOR(type, name, fname) \
    inline auto __device__ name(type x, type y) -> type { return fname(x, y); }

// NOLINTNEXTLINE
#define MIGRAPHX_DEVICE_MATH_HALF(name, fname)                         \
    template <class... Ts, MIGRAPHX_REQUIRES(not is_any_vec<Ts...>())> \
    auto __device__ name(migraphx::half x, Ts... xs)                   \
        MIGRAPHX_RETURNS(fname(math::as_float(x), math::as_float(xs)...))

// NOLINTNEXTLINE
#define MIGRAPHX_DEVICE_MATH_FP8(name, fname)                                          \
    template <class... Ts, MIGRAPHX_REQUIRES(not is_any_vec<Ts...>())>                 \
    auto __device__ name(migraphx::fp8::fp8e4m3fnuz x, Ts... xs) MIGRAPHX_RETURNS(     \
        migraphx::fp8::fp8e4m3fnuz(fname(math::as_float(x), math::as_float(xs)...)))   \
                                                                                       \
        template <class... Ts, MIGRAPHX_REQUIRES(not is_any_vec<Ts...>())>             \
        auto __device__ name(migraphx::fp8::fp8e4m3fn x, Ts... xs) MIGRAPHX_RETURNS(   \
            migraphx::fp8::fp8e4m3fn(fname(math::as_float(x), math::as_float(xs)...))) \
                                                                                       \
            template <class... Ts, MIGRAPHX_REQUIRES(not is_any_vec<Ts...>())>         \
            auto __device__ name(migraphx::fp8::fp8e5m2 x, Ts... xs) MIGRAPHX_RETURNS( \
                migraphx::fp8::fp8e5m2(fname(math::as_float(x), math::as_float(xs)...)))

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

MIGRAPHX_DEVICE_MATH_WRAP(abs, ::abs, ::__habs)
MIGRAPHX_DEVICE_MATH_WRAP(acos, ::acos, ::acosf)
MIGRAPHX_DEVICE_MATH_WRAP(acosh, ::acosh, ::acoshf)
MIGRAPHX_DEVICE_MATH_WRAP(asin, ::asin, ::asinf)
MIGRAPHX_DEVICE_MATH_WRAP(asinh, ::asinh, ::asinh)
MIGRAPHX_DEVICE_MATH_WRAP(atan, ::atan, ::atan)
MIGRAPHX_DEVICE_MATH_WRAP(atanh, ::atanh, ::atanh)
MIGRAPHX_DEVICE_MATH_WRAP(ceil, ::ceil, ::hceil)
MIGRAPHX_DEVICE_MATH_WRAP(cos, ::cos, ::cosf, ::hcos)
MIGRAPHX_DEVICE_MATH_WRAP(cosh, ::cosh, ::coshf)
MIGRAPHX_DEVICE_MATH_WRAP(erf, ::erf)
MIGRAPHX_DEVICE_MATH_WRAP(exp, ::exp, ::hexp)
MIGRAPHX_DEVICE_MATH_WRAP(floor, ::floor, ::hfloor)
MIGRAPHX_DEVICE_MATH_WRAP(isnan, ::isnan, ::__hisinf)
MIGRAPHX_DEVICE_MATH_WRAP(isinf, ::isinf, ::__hisnan)
MIGRAPHX_DEVICE_MATH_WRAP(log, ::log, ::hlog)
MIGRAPHX_DEVICE_MATH_WRAP(log2, ::log2, ::hlog2)
MIGRAPHX_DEVICE_MATH_WRAP(nearbyint, ::nearbyint)
MIGRAPHX_DEVICE_MATH_WRAP(pow, ::pow)
MIGRAPHX_DEVICE_MATH_WRAP(remainder, ::remainder)
MIGRAPHX_DEVICE_MATH_WRAP(round, ::round)
MIGRAPHX_DEVICE_MATH_WRAP(rsqrt, ::rsqrt, ::rsqrtf, ::hrsqrt)
MIGRAPHX_DEVICE_MATH_WRAP(sin, ::sin, ::sinf, ::hsin)
MIGRAPHX_DEVICE_MATH_WRAP(sinh, ::sinh, ::sinhf)
MIGRAPHX_DEVICE_MATH_WRAP(sqrt, ::sqrt, ::hsqrt)
MIGRAPHX_DEVICE_MATH_WRAP(tan, ::tan, ::tanf)
MIGRAPHX_DEVICE_MATH_WRAP(tanh, ::tanh, ::tanhf)
MIGRAPHX_DEVICE_MATH_WRAP(fmod, ::fmod, ::fmodf)

// // Float overloads
// MIGRAPHX_DEVICE_MATH_FOR(float, acos, ::acosf)
// MIGRAPHX_DEVICE_MATH_FOR(float, acosh, ::acoshf)
// MIGRAPHX_DEVICE_MATH_FOR(float, asin, ::asinf)
// MIGRAPHX_DEVICE_MATH_FOR(float, asinh, ::asinhf)
// MIGRAPHX_DEVICE_MATH_FOR(float, atan, ::atanf)
// MIGRAPHX_DEVICE_MATH_FOR(float, atanh, ::atanhf)
// MIGRAPHX_DEVICE_MATH_FOR(float, cos, ::cosf)
// MIGRAPHX_DEVICE_MATH_FOR(float, cosh, ::coshf)
// MIGRAPHX_DEVICE_MATH_FOR(float, rsqrt, ::rsqrtf)
// MIGRAPHX_DEVICE_MATH_FOR(float, sin, ::sinf)
// MIGRAPHX_DEVICE_MATH_FOR(float, sinh, ::sinhf)
// MIGRAPHX_DEVICE_MATH_FOR(float, tan, ::tanf)
// MIGRAPHX_DEVICE_MATH_FOR(float, tanh, ::tanhf)
// MIGRAPHX_DEVICE_MATH_FOR(float, fmod, ::fmodf)

// // Builtin half functions
// MIGRAPHX_DEVICE_MATH_FOR(migraphx::half, abs, ::__habs)
// MIGRAPHX_DEVICE_MATH_FOR(migraphx::half, ceil, ::hceil)
// MIGRAPHX_DEVICE_MATH_FOR(migraphx::half, cos, ::hcos)
// MIGRAPHX_DEVICE_MATH_FOR(migraphx::half, exp, ::hexp)
// MIGRAPHX_DEVICE_MATH_FOR(migraphx::half, floor, ::hfloor)
// MIGRAPHX_DEVICE_MATH_FOR(migraphx::half, isinf, ::__hisinf)
// MIGRAPHX_DEVICE_MATH_FOR(migraphx::half, isnan, ::__hisnan)
// MIGRAPHX_DEVICE_MATH_FOR(migraphx::half, log, ::hlog)
// MIGRAPHX_DEVICE_MATH_FOR(migraphx::half, log2, ::hlog2)
// MIGRAPHX_DEVICE_MATH_FOR(migraphx::half, rsqrt, ::hrsqrt)
// MIGRAPHX_DEVICE_MATH_FOR(migraphx::half, sin, ::hsin)
// MIGRAPHX_DEVICE_MATH_FOR(migraphx::half, sqrt, ::hsqrt)

// Use float to compute half overload
// MIGRAPHX_DEVICE_MATH_HALF(acos, ::acos)
// MIGRAPHX_DEVICE_MATH_HALF(acosh, ::acosh)
// MIGRAPHX_DEVICE_MATH_HALF(asin, ::asin)
// MIGRAPHX_DEVICE_MATH_HALF(asinh, ::asinh)
// MIGRAPHX_DEVICE_MATH_HALF(atan, ::atan)
// MIGRAPHX_DEVICE_MATH_HALF(atanh, ::atanh)
// MIGRAPHX_DEVICE_MATH_HALF(cosh, ::cosh)
// MIGRAPHX_DEVICE_MATH_HALF(erf, ::erf)
// MIGRAPHX_DEVICE_MATH_HALF(nearbyint, ::nearbyint)
// MIGRAPHX_DEVICE_MATH_HALF(pow, ::pow)
// MIGRAPHX_DEVICE_MATH_HALF(remainder, ::remainder)
// MIGRAPHX_DEVICE_MATH_HALF(round, ::round)
// MIGRAPHX_DEVICE_MATH_HALF(sinh, ::sinh)
// MIGRAPHX_DEVICE_MATH_HALF(tan, ::tan)
// MIGRAPHX_DEVICE_MATH_HALF(tanh, ::tanh)
// MIGRAPHX_DEVICE_MATH_HALF(fmod, ::fmod)

// // use float to compute fp8 overload
// MIGRAPHX_DEVICE_MATH_FP8(abs, ::abs)
// MIGRAPHX_DEVICE_MATH_FP8(acos, ::acos)
// MIGRAPHX_DEVICE_MATH_FP8(acosh, ::acosh)
// MIGRAPHX_DEVICE_MATH_FP8(asin, ::asin)
// MIGRAPHX_DEVICE_MATH_FP8(asinh, ::asinh)
// MIGRAPHX_DEVICE_MATH_FP8(atan, ::atan)
// MIGRAPHX_DEVICE_MATH_FP8(atanh, ::atanh)
// MIGRAPHX_DEVICE_MATH_FP8(ceil, ::ceil)
// MIGRAPHX_DEVICE_MATH_FP8(cos, ::cos)
// MIGRAPHX_DEVICE_MATH_FP8(cosh, ::cosh)
// MIGRAPHX_DEVICE_MATH_FP8(erf, ::erf)
// MIGRAPHX_DEVICE_MATH_FP8(exp, ::exp)
// MIGRAPHX_DEVICE_MATH_FP8(floor, ::floor)
// MIGRAPHX_DEVICE_MATH_FP8(isnan, ::isnan)
// MIGRAPHX_DEVICE_MATH_FP8(log, ::log)
// MIGRAPHX_DEVICE_MATH_FP8(log2, ::log2)
// MIGRAPHX_DEVICE_MATH_FP8(pow, ::pow)
// MIGRAPHX_DEVICE_MATH_FP8(remainder, ::remainder)
// MIGRAPHX_DEVICE_MATH_FP8(round, ::round)
// MIGRAPHX_DEVICE_MATH_FP8(rsqrt, ::rsqrt)
// MIGRAPHX_DEVICE_MATH_FP8(sin, ::sin)
// MIGRAPHX_DEVICE_MATH_FP8(sinh, ::sinh)
// MIGRAPHX_DEVICE_MATH_FP8(sqrt, ::sqrt)
// MIGRAPHX_DEVICE_MATH_FP8(tan, ::tan)
// MIGRAPHX_DEVICE_MATH_FP8(tanh, ::tanh)
// MIGRAPHX_DEVICE_MATH_FP8(fmod, ::fmod)

// Map math functions to hip half2 functions
// The half2 type is defined in include/hip/amd_detail/hip_fp16_gcc.h and is 2 16-bit floats
// packed into a 32-bit number.  See include/hip/amd_detail/hip_fp16_math_fwd.h for the HIP names
// Most but not all of these math ops have operators of the same names.
MIGRAPHX_DEVICE_MATH_HALF2(abs, ::__habs2)
MIGRAPHX_DEVICE_MATH_HALF2(ceil, ::h2ceil)
MIGRAPHX_DEVICE_MATH_HALF2(cos, ::h2cos)
MIGRAPHX_DEVICE_MATH_HALF2(exp, ::h2exp)
MIGRAPHX_DEVICE_MATH_HALF2(exp10, ::h2exp10)
MIGRAPHX_DEVICE_MATH_HALF2(exp2, ::h2exp2)
MIGRAPHX_DEVICE_MATH_HALF2(floor, ::h2floor)
MIGRAPHX_DEVICE_MATH_HALF2(isinf, ::__hisinf2)
MIGRAPHX_DEVICE_MATH_HALF2(isnan, ::__hisnan2)
MIGRAPHX_DEVICE_MATH_HALF2(log, ::h2log)
MIGRAPHX_DEVICE_MATH_HALF2(log10, ::h2log10)
MIGRAPHX_DEVICE_MATH_HALF2(log2, ::h2log2)
MIGRAPHX_DEVICE_MATH_HALF2(rsqrt, ::h2rsqrt)
MIGRAPHX_DEVICE_MATH_HALF2(sin, ::h2sin)
MIGRAPHX_DEVICE_MATH_HALF2(sqrt, ::h2sqrt)

template <class T, class U>
constexpr auto where(bool cond, const T& a, const U& b)
{
    return cond ? a : b;
}

MIGRAPHX_DEVICE_MATH_BINARY_FOR(float, max, ::max)
MIGRAPHX_DEVICE_MATH_BINARY_FOR(float, min, ::min)
MIGRAPHX_DEVICE_MATH_BINARY_FOR(double, max, ::max)
MIGRAPHX_DEVICE_MATH_BINARY_FOR(double, min, ::min)
MIGRAPHX_DEVICE_MATH_BINARY_FOR(migraphx::half, max, ::__hmax)
MIGRAPHX_DEVICE_MATH_BINARY_FOR(migraphx::half, min, ::__hmin)

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

template <class T, MIGRAPHX_REQUIRES(not is_any_vec<T>())>
constexpr T mod(const T& a, const T& b)
{
    if constexpr(is_integral<T>{})
        // onnx mod operator requires numpy style modulus
        return ((a % b) + b) % b;
    return static_cast<T>(fmod(remainder(a, b) + b, b));
}

template <class T, class U, MIGRAPHX_REQUIRES(not is_same<T, U>{} and not is_any_vec<T, U>())>
constexpr auto mod(const T& a, const U& b)
{
    return mod<common_type_t<T, U>>(a, b);
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
MIGRAPHX_DEVICE_MATH_VEC(fmod)
MIGRAPHX_DEVICE_MATH_VEC(isinf)
MIGRAPHX_DEVICE_MATH_VEC(isnan)
MIGRAPHX_DEVICE_MATH_VEC(log)
MIGRAPHX_DEVICE_MATH_VEC(log2)
MIGRAPHX_DEVICE_MATH_VEC(max)
MIGRAPHX_DEVICE_MATH_VEC(min)
MIGRAPHX_DEVICE_MATH_VEC(mod)
MIGRAPHX_DEVICE_MATH_VEC(nearbyint)
MIGRAPHX_DEVICE_MATH_VEC(pow)
MIGRAPHX_DEVICE_MATH_VEC(remainder)
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
    return vec_transform(v)([](auto x) -> T { return static_cast<T>(x); });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_MATH_HPP
