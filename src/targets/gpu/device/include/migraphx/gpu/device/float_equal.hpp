#ifndef MIGRAPHX_GUARD_RTGLIB_GPU_DEVICE_FLOAT_EQUAL_HPP
#define MIGRAPHX_GUARD_RTGLIB_GPU_DEVICE_FLOAT_EQUAL_HPP

#include <migraphx/requires.hpp>
#include <migraphx/config.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class... Ts>
using common_type = typename std::common_type<Ts...>::type;

template <class T, MIGRAPHX_REQUIRES(is_floating_point<T>{})>
__device__ bool float_equal_device(T x, T y)
{
    return std::isfinite(x) and std::isfinite(y) and
           std::nextafter(x, std::numeric_limits<T>::lowest()) <= y and
           std::nextafter(x, std::numeric_limits<T>::max()) >= y;
}

template <class T, MIGRAPHX_REQUIRES(not is_floating_point<T>{})>
__device__ bool float_equal_device(T x, T y)
{
    return x == y;
}

template <class T, class U>
__device__ bool float_equal(T x, U y)
{
    return float_equal_device<common_type<T, U>>(x, y);
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
