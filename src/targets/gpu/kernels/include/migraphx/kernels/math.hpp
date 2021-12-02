#ifndef MIGRAPHX_GUARD_KERNELS_MATH_HPP
#define MIGRAPHX_GUARD_KERNELS_MATH_HPP

#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/vec.hpp>
#include <migraphx/kernels/functional.hpp>
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
#define MIGRAPHX_DEVICE_MATH(name, fname) \
    template <class... Ts>                \
    auto __device__ name(Ts... xs) MIGRAPHX_RETURNS(fname(xs...))

// NOLINTNEXTLINE
#define MIGRAPHX_DEVICE_MATH_FOR(type, name, fname) \
    template <class... Ts>                          \
    auto __device__ name(type x, Ts... xs) MIGRAPHX_RETURNS(fname(x, xs...))

// NOLINTNEXTLINE
#define MIGRAPHX_DEVICE_MATH_HALF(name, fname)       \
    template <class... Ts>                           \
    auto __device__ name(migraphx::half x, Ts... xs) \
        MIGRAPHX_RETURNS(fname(math::as_float(x), math::as_float(xs)...))

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
MIGRAPHX_DEVICE_MATH_HALF(pow, ::pow)
MIGRAPHX_DEVICE_MATH_HALF(round, ::round)
MIGRAPHX_DEVICE_MATH_HALF(sin, ::sin)
MIGRAPHX_DEVICE_MATH_HALF(sinh, ::sinh)
MIGRAPHX_DEVICE_MATH_HALF(tan, ::tan)
MIGRAPHX_DEVICE_MATH_HALF(tanh, ::tanh)

template <class T, class U>
constexpr auto& max(const T& a, const U& b)
{
    return (a < b) ? b : a;
}

template <class T, class U>
constexpr auto& min(const T& a, const U& b)
{
    return (a > b) ? b : a;
}

template <class T, class U>
constexpr T convert(U x)
{
    return x;
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_MATH_HPP
