#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_MATH_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_MATH_HPP

#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

// NOLINTNEXTLINE
#define MIGRAPHX_DEVICE_MATH(name, fname) \
template<class... Ts> \
auto __device__ name(Ts... xs) -> decltype(fname(xs...)) \
{ \
    return fname(xs...); \
} \
template <class... Ts, index_int N> \
auto __device__ name(vec<Ts, N>... xs) -> vec<decltype(name(xs[0]...)), N> \
{ \
    vec<decltype(name(xs[0]...)), N> y; \
    for(index_int k = 0; k < N; k++) \
        y[k] = name(xs[k]...); \
    return y; \
}

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
MIGRAPHX_DEVICE_MATH(max, ::max)
MIGRAPHX_DEVICE_MATH(min, ::min)
MIGRAPHX_DEVICE_MATH(pow, ::pow)
MIGRAPHX_DEVICE_MATH(round, ::round)
MIGRAPHX_DEVICE_MATH(rsqrt, ::rsqrt)
MIGRAPHX_DEVICE_MATH(sin, ::sin)
MIGRAPHX_DEVICE_MATH(sinh, ::sinh)
MIGRAPHX_DEVICE_MATH(sqrt, ::sqrt)
MIGRAPHX_DEVICE_MATH(tan, ::tan)
MIGRAPHX_DEVICE_MATH(tanh, ::tanh)

vec<half, 2> __device__ cos(vec<half, 2> x)
{ 
    return h2cos(x); 
}


} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
