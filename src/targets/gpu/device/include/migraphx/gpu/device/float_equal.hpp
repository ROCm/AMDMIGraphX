#ifndef MIGRAPHX_GUARD_RTGLIB_GPU_DEVICE_FLOAT_EQUAL_HPP
#define MIGRAPHX_GUARD_RTGLIB_GPU_DEVICE_FLOAT_EQUAL_HPP

#include <hip/hip_runtime.h>
#include <migraphx/half.hpp>
#include <migraphx/config.hpp>
#include <migraphx/tensor_view.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class... Ts>
using common_type = typename std::common_type<Ts...>::type;

struct float_equal_fn
{
    template <class T, MIGRAPHX_REQUIRES(is_floating_point<T>{})>
    __device__ __host__ static bool apply(T x, T y)
    {
        return std::isfinite(x) and std::isfinite(y) and
               std::nextafter(x, std::numeric_limits<T>::lowest()) <= y and
               std::nextafter(x, std::numeric_limits<T>::max()) >= y;
    }

    template <class T, MIGRAPHX_REQUIRES(not is_floating_point<T>{})>
    __device__ __host__ static bool apply(T x, T y)
    {
        return x == y;
    }

    template <class T, class U>
    __device__ __host__ bool operator()(T x, U y) const
    {
        return float_equal_fn::apply<common_type<T, U>>(x, y);
    }
};

__device__ static constexpr float_equal_fn float_equal{};

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
