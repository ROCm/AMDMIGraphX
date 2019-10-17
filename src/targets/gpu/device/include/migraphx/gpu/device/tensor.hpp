#ifndef MIGRAPHX_GUARD_RTGLIB_DEAVICE_TENSOR_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEAVICE_TENSOR_HPP

#include <migraphx/gpu/device/visit.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <index_int NDim>
using hip_tensor_index = hip_array<index_int, NDim>;

template <index_int NDim>
struct hip_tensor_descriptor
{
    __device__ __host__ hip_tensor_descriptor() = default;

    hip_tensor_descriptor(const shape& s)
    {
        std::copy(s.lens().begin(), s.lens().end(), lens);
        std::copy(s.strides().begin(), s.strides().end(), strides);
    }

    __device__ __host__ hip_tensor_index<NDim> multi(index_int idx) const
    {
        hip_tensor_index<NDim> result{};
        index_int tidx = idx;
        for(index_int is = 0; is < NDim; is++)
        {
            result[is] = tidx / strides[is];
            tidx       = tidx % strides[is];
        }

        return result;
    }
    __device__ __host__ index_int linear(hip_tensor_index<NDim> s) const
    {
        index_int idx = 0;
        for(index_int i = 0; i < NDim; i++)
            idx += s[i] * strides[i];
        return idx;
    }
    index_int lens[NDim]    = {};
    index_int strides[NDim] = {};
};

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
