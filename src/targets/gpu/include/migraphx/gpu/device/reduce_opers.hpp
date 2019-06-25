#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_REDUCE_OPERS_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_REDUCE_OPERS_HPP

#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class T>
inline __device__ void reduce_max(T* data_ptr, size_t block_size, size_t thr_idx, size_t item_num)
{
    while(true)
    {
        auto stride = (item_num + 1) / 2;
        auto size = item_num / 2;
        for (size_t i = thr_idx; i < size; i += block_size)
        {
            data_ptr[i] =
                ::max(to_hip_type(data_ptr[i]), to_hip_type(data_ptr[i + stride]));
        }
        __syncthreads();
        item_num = stride;

        if(item_num == 1)
            break;
    }

    if(thr_idx == 0)
    {
        data_ptr[block_size] =
            (data_ptr[0] < data_ptr[block_size]) ? data_ptr[block_size] : data_ptr[0];
    }

    __syncthreads();
}

template <class T>
inline __device__ void reduce_sum(T* data_ptr, size_t block_size, size_t thr_idx, size_t item_num)
{
    while(true)
    {
        auto stride = (item_num + 1) / 2;
        auto size = item_num / 2;
        for (size_t i = thr_idx; i < size; i += block_size)
        {
            data_ptr[i] += data_ptr[i + stride];
        }
        __syncthreads();
        item_num = stride;

        if(item_num == 1)
            break;
    }

    if(thr_idx == 0)
    {
        data_ptr[block_size] += data_ptr[0];
    }

    __syncthreads();
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
