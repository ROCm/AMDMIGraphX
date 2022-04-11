#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/gpu/device/softmax.hpp>
#include <migraphx/gpu/device/reduce.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

struct half2_sum
{
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(__half2 x, __half2 y) const { return __hadd2(x, y); }
};

inline __device__ __half2 hmax2(__half2 x, __half2 y)
{
    auto fx2 = __half22float2(x);
    auto fy2 = __half22float2(y);
    auto fx  = fx2.x > fy2.x ? fx2.x : fy2.x;
    auto fy  = fx2.y > fy2.y ? fx2.y : fy2.y;
    return __floats2half2_rn(fx, fy);
}

struct half2_max
{
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(__half2 x, __half2 y) const { return hmax2(x, y); }
};

// in_data is in shared memory
template <class Op>
__device__ __half2 block_reduce_half2(
    __half2* buffer, index_int batch_item_num, index_int tid, index_int block_size, Op op)
{
    __syncthreads();
    for(index_int s = block_size; s > 0; s >>= 1)
    {
        if(tid < s and tid + s < batch_item_num)
        {
            buffer[tid] = op(buffer[tid], buffer[tid + s]);
        }
        __syncthreads();
    }

    auto lows2  = __low2half2(buffer[0]);
    auto highs2 = __high2half2(buffer[0]);

    return op(lows2, highs2);
}

__global__ void
softmax_kernel_half2(void* data_in, index_int batch_item_num, index_int block_size, void* data_out)
{
    __half2* input  = reinterpret_cast<__half2*>(data_in);
    __half2* output = reinterpret_cast<__half2*>(data_out);
    batch_item_num /= 2;
    extern MIGRAPHX_DEVICE_SHARED __half2 buffer2[];

    __half2* in_data_reduce = buffer2;
    __half2* in_data        = buffer2 + batch_item_num;
    int start               = blockIdx.x * batch_item_num;
    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        auto d            = input[i + start];
        in_data[i]        = d;
        in_data_reduce[i] = d;
    }

    auto batch_max =
        block_reduce_half2(in_data_reduce, batch_item_num, threadIdx.x, block_size, half2_max{});

    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        in_data[i]        = h2exp(__hsub2(in_data[i], batch_max));
        in_data_reduce[i] = in_data[i];
    }

    auto batch_sum =
        block_reduce_half2(in_data_reduce, batch_item_num, threadIdx.x, block_size, half2_sum{});

    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        output[i + start] = __h2div(in_data[i], batch_sum);
    }
}

// in_data is in shared memory
template <class Op>
__device__ __half block_reduce_half(
    __half* data, index_int batch_item_num, index_int tid, index_int block_size, Op op)
{
    __syncthreads();
    for(index_int s = block_size / 2; s > 0; s >>= 1)
    {
        if(tid < s and tid + s < batch_item_num)
        {
            data[tid] = op(data[tid], data[tid + s]);
        }
        __syncthreads();
    }

    return data[0];
}

__global__ void
softmax_kernel_half(void* data_in, index_int batch_item_num, index_int block_size, void* data_out)
{
    __half* input  = reinterpret_cast<__half*>(data_in);
    __half* output = reinterpret_cast<__half*>(data_out);
    extern MIGRAPHX_DEVICE_SHARED __half buffer[];

    __half* in_data_reduce = buffer;
    __half* in_data        = buffer + batch_item_num;
    int start              = blockIdx.x * batch_item_num;
    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        auto d            = input[i + start];
        in_data[i]        = d;
        in_data_reduce[i] = d;
    }

    auto batch_max =
        block_reduce_half(in_data_reduce, batch_item_num, threadIdx.x, block_size, max{});
    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        in_data[i]        = __float2half(::exp(__half2float(in_data[i]) - __half2float(batch_max)));
        in_data_reduce[i] = in_data[i];
    }

    auto batch_sum =
        block_reduce_half(in_data_reduce, batch_item_num, threadIdx.x, block_size, sum{});
    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        output[i + start] = __float2half(__half2float(in_data[i]) / __half2float(batch_sum));
    }
}

void softmax(hipStream_t stream, const argument& result, const argument& arg, int64_t axis)
{
    auto batch_lens          = result.get_shape().lens();
    index_int batch_item_num = batch_lens[axis];
    batch_lens[axis]         = 1;
    migraphx::shape batch_shape{result.get_shape().type(), batch_lens};

    hip_visit_all(result, arg, batch_shape)([&](auto output, auto input, auto batch) {
        const index_int max_block_size = 128;
        const index_int block_size     = compute_block_size(batch_item_num, max_block_size);
        using type = device_type<std::remove_cv_t<typename decltype(input)::value_type>>;
        type init  = lowest();

        if(axis == batch_lens.size() - 1)
        {
            auto in_type = result.get_shape().type();
            if(in_type == shape::half_type and batch_item_num <= 1024)
            {
                auto half2_block_size = compute_block_size(batch_item_num, 1024);
                int block_num         = batch_shape.elements();
                int shared_size       = batch_item_num * 2 * result.get_shape().type_size();
                half2_block_size      = half2_block_size / 4;
                softmax_kernel_half2<<<block_num, half2_block_size, shared_size, stream>>>(
                    arg.data(), batch_item_num, half2_block_size, result.data());
            }
            else
            {
                gs_launch(stream, batch_shape.elements() * block_size, block_size)(
                    [=](auto i, auto idx) __device__ {
                        auto start_loc = i / block_size * batch_item_num;
                        auto batch_max = block_reduce<max_block_size>(
                            idx, max{}, init, batch_item_num, [&](auto j) __device__ {
                                return input[start_loc + j];
                            });

                        auto batch_sum = block_reduce<max_block_size>(
                            idx, sum{}, 0, batch_item_num, [&](auto j) __device__ {
                                auto val = input[start_loc + j] - batch_max;
                                return ::exp(to_hip_type(val));
                            });

                        idx.local_stride(batch_item_num, [&](auto j) __device__ {
                            auto val              = input[start_loc + j] - batch_max;
                            output[start_loc + j] = ::exp(to_hip_type(val)) / batch_sum;
                        });
                    });
            }
        }
        else
        {
            gs_launch(stream, batch_shape.elements() * block_size, block_size)(
                [=](auto i, auto idx) __device__ {
                    auto data_idx  = batch.multi(i / block_size);
                    auto batch_max = block_reduce<max_block_size>(
                        idx, max{}, init, batch_item_num, [&](auto j) __device__ {
                            data_idx[axis] = j;
                            return input[data_idx];
                        });

                    auto batch_sum = block_reduce<max_block_size>(
                        idx, sum{}, 0, batch_item_num, [&](auto j) __device__ {
                            data_idx[axis] = j;
                            auto val       = input[data_idx] - batch_max;
                            return ::exp(to_hip_type(val));
                        });

                    idx.local_stride(batch_item_num, [&](auto j) __device__ {
                        data_idx[axis]   = j;
                        auto val         = input[data_idx] - batch_max;
                        output[data_idx] = ::exp(to_hip_type(val)) / batch_sum;
                    });
                });
        }
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
