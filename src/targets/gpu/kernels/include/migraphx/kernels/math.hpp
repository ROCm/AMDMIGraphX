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
#define MIGRAPHX_DEVICE_MATH_HALF(name, fname)                         \
    template <class... Ts, MIGRAPHX_REQUIRES(not is_any_vec<Ts...>())> \
    auto __device__ name(migraphx::half x, Ts... xs)                   \
        MIGRAPHX_RETURNS(fname(math::as_float(x), math::as_float(xs)...))

// Template with two overloads for math functions, one for half2 type and one for more generic
// <half, N> vectorization where N is 4 or another even number. NOLINTNEXTLINE
#define MIGRAPHX_DEVICE_MATH_HALF_2(name, fname)                                                 \
    template <class... Ts>                                                                       \
    auto __device__ name(migraphx::vec<migraphx::half, 2> x, Ts... xs) MIGRAPHX_RETURNS(fname(   \
        x, xs...)) template <class... Ts, index_int N, MIGRAPHX_REQUIRES(N % 2 == 0 && (N > 2))> \
    auto __device__ name(migraphx::vec<migraphx::half, N> x, Ts... xs)                           \
    {                                                                                            \
        return vec_packed_transform<2>(x, xs...)(                                                \
            [](auto... ys) -> migraphx::vec<migraphx::half, 2> { return fname(ys...); });        \
    }

// NOLINTNEXTLINE
#define MIGRAPHX_DEVICE_MATH_HALF_VEC(name)                                  \
    template <class... Ts, MIGRAPHX_REQUIRES(is_any_vec<Ts...>())>           \
    auto __device__ name(Ts... xs)                                           \
    {                                                                        \
        return vec_transform(xs...)([](auto... ys) { return name(ys...); }); \
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
MIGRAPHX_DEVICE_MATH_HALF_2(sqrt, ::h2sqrt)
// __half __low2half(__half2 x)
// __half __high2half(__half2 x)
// __half2 __low2half2(__half2 x)
// __half2 __high2half2(__half2 x)
// __half2 __lowhigh2highlow(__half2 x)
// float __low2float(__half2 x)
// float __high2float(__half2 x)
// float2 __half22float2(__half2 x)
// __half2 __habs2(__half2 x)
// __half2 h2trunc(__half2 x)
MIGRAPHX_DEVICE_MATH_HALF_2(ceil, ::h2ceil)
MIGRAPHX_DEVICE_MATH_HALF_2(floor, ::h2floor)
// __half2 h2rint(__half2 x)
MIGRAPHX_DEVICE_MATH_HALF_2(sin, ::h2sin)
MIGRAPHX_DEVICE_MATH_HALF_2(cos, ::h2cos)
MIGRAPHX_DEVICE_MATH_HALF_2(exp, ::h2exp)
MIGRAPHX_DEVICE_MATH_HALF_2(exp2, ::h2exp2)
// __half2 h2exp10(__half2 x)
MIGRAPHX_DEVICE_MATH_HALF_2(log2, ::h2log2)
MIGRAPHX_DEVICE_MATH_HALF_2(log, ::h2log)
MIGRAPHX_DEVICE_MATH_HALF_2(log10, ::h2log10)
// __half2 h2rcp(__half2 x) { return __llvm_amdgcn_rcp_2f16(x); }
// __half2 h2rsqrt(__half2 x) { return __ocml_rsqrt_2f16(x); }
MIGRAPHX_DEVICE_MATH_HALF_2(isinf, ::__hisinf2)
MIGRAPHX_DEVICE_MATH_HALF_2(isnan, ::__hisnan2)
// __half2 __hneg2(__half2 x)

template <class T, class U>
constexpr auto where(bool cond, const T& a, const U& b)
{
    return cond ? a : b;
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
constexpr auto max(const T& a, const U& b)
{
    return where(a < b, b, a);
}

template <class T, class U>
constexpr auto min(const T& a, const U& b)
{
    return where(a > b, b, a);
}

template <class T, class U>
constexpr auto convert(U v)
{
    return vec_transform(v)([](auto x) -> T { return x; });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_MATH_HPP
