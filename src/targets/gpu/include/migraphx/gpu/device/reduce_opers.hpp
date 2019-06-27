#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_REDUCE_OPERS_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_REDUCE_OPERS_HPP

#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/device/reduce_opers.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class T>
struct max_op
{
    T operator()(T x, T y) const { return (x > y) ? x : y; }
};

template <class T>
struct min_op
{
    T operator()(T x, T y) const { return (x < y) ? x : y; }
};

template <class T>
struct sum_op
{
    T operator()(T x, T y) const { return x + y; }
};

template <class T, class Op>
inline __device__ void block_reduce(T* data_ptr,
                                    Op op,
                                    std::size_t block_size,
                                    std::size_t thr_idx,
                                    std::size_t item_num,
                                    std::size_t output_index)
{
    while(true)
    {
        auto stride = (item_num + 1) / 2;
        auto size   = item_num / 2;
        for(std::size_t i = thr_idx; i < size; i += block_size)
        {
            data_ptr[i] = op(data_ptr[i], data_ptr[i + stride]);
        }
        __syncthreads();
        item_num = stride;

        if(item_num == 1)
            break;
    }

    if(thr_idx == 0)
    {
        data_ptr[output_index] = op(data_ptr[output_index], data_ptr[0]);
    }

    __syncthreads();
}

template <class T, class F>
struct pair_max_op
{
    using type = std::pair<T, F>;
    type operator()(type x, type y) const { return (x.first > y.first) ? x : y; }
};

template <class T, class F>
struct pair_min_op
{
    using type = std::pair<T, F>;
    type operator()(type x, type y) const { return (x.first < y.first) ? x : y; }
};

template <class T, class Op>
inline __device__ void block_reduce_pair(T* data_ptr,
                                         int64_t* index_ptr,
                                         Op op,
                                         std::size_t block_size,
                                         std::size_t thr_idx,
                                         std::size_t item_num,
                                         std::size_t output_index)
{
    while(true)
    {
        auto stride = (item_num + 1) / 2;
        auto size   = item_num / 2;
        for(std::size_t i = thr_idx; i < size; i += block_size)
        {
            auto output =
                op({data_ptr[i], index_ptr[i]}, {data_ptr[i + stride], index_ptr[i + stride]});
            data_ptr[i]  = output.first;
            index_ptr[i] = output.second;
        }
        __syncthreads();
        item_num = stride;

        if(item_num == 1)
            break;
    }

    if(thr_idx == 0)
    {
        auto output =
            op({data_ptr[output_index], index_ptr[output_index]}, {data_ptr[0], index_ptr[0]});
        data_ptr[output_index]  = output.first;
        index_ptr[output_index] = output.second;
    }

    __syncthreads();
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
