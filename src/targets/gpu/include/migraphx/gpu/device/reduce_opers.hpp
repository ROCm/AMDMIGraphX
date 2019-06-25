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
inline __device__ void
reduce_max(T* data_ptr, size_t block_size, size_t thr_idx, size_t item_num, size_t max_index)
{
    while(true)
    {
        auto stride = (item_num + 1) / 2;
        auto size   = item_num / 2;
        for(size_t i = thr_idx; i < size; i += block_size)
        {
            data_ptr[i] = ::max(to_hip_type(data_ptr[i]), to_hip_type(data_ptr[i + stride]));
        }
        __syncthreads();
        item_num = stride;

        if(item_num == 1)
            break;
    }

    if(thr_idx == 0)
    {
        data_ptr[max_index] =
            (data_ptr[0] < data_ptr[max_index]) ? data_ptr[max_index] : data_ptr[0];
    }

    __syncthreads();
}

template <class T>
inline __device__ void
reduce_min(T* data_ptr, size_t block_size, size_t thr_idx, size_t item_num, size_t min_index)
{
    while(true)
    {
        auto stride = (item_num + 1) / 2;
        auto size   = item_num / 2;
        for(size_t i = thr_idx; i < size; i += block_size)
        {
            data_ptr[i] = ::min(to_hip_type(data_ptr[i]), to_hip_type(data_ptr[i + stride]));
        }
        __syncthreads();
        item_num = stride;

        if(item_num == 1)
            break;
    }

    if(thr_idx == 0)
    {
        data_ptr[min_index] =
            (data_ptr[0] > data_ptr[min_index]) ? data_ptr[min_index] : data_ptr[0];
    }

    __syncthreads();
}

template <class T>
inline __device__ void reduce_argmax(T* data_ptr,
                                     int64_t* index_ptr,
                                     size_t block_size,
                                     size_t thr_idx,
                                     size_t item_num,
                                     size_t max_index)
{
    while(true)
    {
        auto stride = (item_num + 1) / 2;
        auto size   = item_num / 2;
        for(size_t i = thr_idx; i < size; i += block_size)
        {
            if(data_ptr[i] < data_ptr[i + stride])
            {
                data_ptr[i]  = data_ptr[i + stride];
                index_ptr[i] = index_ptr[i + stride];
            }
        }
        __syncthreads();
        item_num = stride;

        if(item_num == 1)
            break;
    }

    if(thr_idx == 0)
    {
        if(data_ptr[max_index] < data_ptr[0])
        {
            data_ptr[max_index]  = data_ptr[0];
            index_ptr[max_index] = index_ptr[0];
        }
    }

    __syncthreads();
}

template <class T>
inline __device__ void reduce_argmin(T* data_ptr,
                                     int64_t* index_ptr,
                                     size_t block_size,
                                     size_t thr_idx,
                                     size_t item_num,
                                     size_t min_index)
{
    while(true)
    {
        auto stride = (item_num + 1) / 2;
        auto size   = item_num / 2;
        for(size_t i = thr_idx; i < size; i += block_size)
        {
            if(data_ptr[i] > data_ptr[i + stride])
            {
                data_ptr[i]  = data_ptr[i + stride];
                index_ptr[i] = index_ptr[i + stride];
            }
        }
        __syncthreads();
        item_num = stride;

        if(item_num == 1)
            break;
    }

    if(thr_idx == 0)
    {
        if(data_ptr[min_index] > data_ptr[0])
        {
            data_ptr[min_index]  = data_ptr[0];
            index_ptr[min_index] = index_ptr[0];
        }
    }

    __syncthreads();
}

template <class T>
inline __device__ void
reduce_sum(T* data_ptr, size_t block_size, size_t thr_idx, size_t item_num, size_t sum_index)
{
    while(true)
    {
        auto stride = (item_num + 1) / 2;
        auto size   = item_num / 2;
        for(size_t i = thr_idx; i < size; i += block_size)
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
        data_ptr[sum_index] += data_ptr[0];
    }

    __syncthreads();
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
