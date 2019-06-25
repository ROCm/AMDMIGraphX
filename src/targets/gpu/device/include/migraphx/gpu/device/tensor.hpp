#ifndef MIGRAPHX_GUARD_RTGLIB_DEAVICE_TENSOR_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEAVICE_TENSOR_HPP

#include <migraphx/gpu/device/visit.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <std::size_t NDim>
using hip_tensor_index = hip_array<std::size_t, NDim>;

template <std::size_t NDim>
struct hip_tensor_descriptor
{
    __device__ __host__ hip_tensor_descriptor() = default;

    hip_tensor_descriptor(const shape& s)
    {
        std::copy(s.lens().begin(), s.lens().end(), lens);
        std::copy(s.strides().begin(), s.strides().end(), strides);
    }

    __device__ __host__ hip_tensor_index<NDim> multi(std::size_t idx) const
    {
        hip_tensor_index<NDim> result{};
        std::size_t tidx = idx;
        for(std::size_t is = 0; is < NDim; is++)
        {
            result[is] = tidx / strides[is];
            tidx       = tidx % strides[is];
        }
        return result;
    }
    __device__ __host__ std::size_t linear(hip_tensor_index<NDim> s) const
    {
        std::size_t idx = 0;
        for(std::size_t i = 0; i < NDim; i++)
            idx += s[i] * strides[i];
        return idx;
    }
    std::size_t lens[NDim]    = {};
    std::size_t strides[NDim] = {};
};

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
