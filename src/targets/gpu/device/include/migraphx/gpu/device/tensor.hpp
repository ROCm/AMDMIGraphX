#ifndef MIGRAPHX_GUARD_RTGLIB_DEAVICE_TENSOR_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEAVICE_TENSOR_HPP

#include <hip/hip_runtime.h>
#include <migraphx/functional.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class F>
void visit_tensor_size(std::size_t n, F f)
{
    switch(n)
    {
    case 1:
    {
        f(std::integral_constant<std::size_t, 1>{});
        break;
    }
    case 2:
    {
        f(std::integral_constant<std::size_t, 2>{});
        break;
    }
    case 3:
    {
        f(std::integral_constant<std::size_t, 3>{});
        break;
    }
    case 4:
    {
        f(std::integral_constant<std::size_t, 4>{});
        break;
    }
    case 5:
    {
        f(std::integral_constant<std::size_t, 5>{});
        break;
    }
    default: throw std::runtime_error("Unknown tensor size");
    }
}

template <size_t NDim>
struct hip_index
{
    size_t d[NDim];
    __device__ __host__ size_t& operator[](size_t i) { return d[i]; }
    __device__ __host__ size_t operator[](size_t i) const { return d[i]; }
};

template <size_t NDim>
struct hip_tensor_descriptor
{
    __device__ __host__ hip_tensor_descriptor() = default;

    hip_tensor_descriptor(const shape& s)
    {
        std::copy(s.lens().begin(), s.lens().end(), lens);
        std::copy(s.strides().begin(), s.strides().end(), strides);
        std::vector<std::size_t> vec_idx(s.lens().size());
        std::iota(vec_idx.begin(), vec_idx.end(), 0);
        std::sort(vec_idx.begin(), vec_idx.end(), [&](size_t i, size_t j) {
            return strides[i] > strides[j];
        });
        std::copy(vec_idx.begin(), vec_idx.end(), indices);
    }

    __device__ __host__ hip_index<NDim> multi(size_t idx) const
    {
        hip_index<NDim> result{};
        size_t tidx = idx;
        for(size_t is = 0; is < NDim; is++)
        {
            result[indices[is]] = tidx / strides[indices[is]];
            tidx                = tidx % strides[indices[is]];
        }
        return result;
    }

    __device__ __host__ size_t linear(hip_index<NDim> s) const
    {
        size_t idx = 0;
        for(size_t i = 0; i < NDim; i++)
            idx += s[i] * strides[i];
        return idx;
    }
    size_t lens[NDim]    = {};
    size_t strides[NDim] = {};
    size_t indices[NDim] = {};
};

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
