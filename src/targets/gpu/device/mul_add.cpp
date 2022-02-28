#include "migraphx/gpu/device/launch.hpp"
#include <hip/amd_detail/amd_device_functions.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <migraphx/gpu/device/mul_add.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

//__global__ void mul_add_kernel(void* a, void* x, void* b, void* r, int n)
//{
//    int id = blockDim.x * blockIdx.x + threadIdx.x;
//    __half* ha = reinterpret_cast<__half*>(a);
//    __half* hb = reinterpret_cast<__half*>(b);
//    __half* hx = reinterpret_cast<__half*>(x);
//    __half* hr = reinterpret_cast<__half*>(r);
//    if (id < n)
//    {
//        hr[id] = __float2half(__half2float(ha[id]) * __half2float(hx[id]) + __half2float(hb[id]));
//    }
//}

// __global__ void mul_add_kernel(void* a, int an, void* x, int xn, void* b, int bn, void* r, int n)
// {
//     int id      = blockDim.x * blockIdx.x + threadIdx.x;
//     __half2* ha = reinterpret_cast<__half2*>(a);
//     __half2* hb = reinterpret_cast<__half2*>(b);
//     __half2* hx = reinterpret_cast<__half2*>(x);
//     __half2* hr = reinterpret_cast<__half2*>(r);
//     if(id < n)
//     {
//         hr[id] = __hadd2(__hmul2(ha[id % an], hx[id % xn]), hb[id % bn]);
//     }
// }

__global__ void mul_add_kernel(void* a, void* x, void* b, void* r, int* strides, int elem_num)
{
    __shared__ int shared_strides[18];
    int tid = threadIdx.x * (blockDim.y * blockDim.z) + threadIdx.y * blockDim.z + threadIdx.z;
    if(tid < 18)
    {
        shared_strides[tid] = strides[tid];
    }
    __syncthreads();

    __half2* ha = reinterpret_cast<__half2*>(a);
    __half2* hb = reinterpret_cast<__half2*>(b);
    __half2* hx = reinterpret_cast<__half2*>(x);
    __half2* hr = reinterpret_cast<__half2*>(r);

    tid = tid + (blockIdx.x * (gridDim.y * gridDim.z) + blockIdx.y * gridDim.z + blockIdx.z) *
                    blockDim.x * blockDim.y * blockDim.z;
    if(tid < elem_num)
    {
        int tida = shared_strides[1] * blockIdx.x + shared_strides[2] * blockIdx.y +
                   shared_strides[3] * blockIdx.z + shared_strides[4] * threadIdx.x +
                   shared_strides[5] * threadIdx.y + threadIdx.z;
        int tidx = shared_strides[7] * blockIdx.x + shared_strides[8] * blockIdx.y +
                   shared_strides[9] * blockIdx.z + shared_strides[10] * threadIdx.x +
                   shared_strides[11] * threadIdx.y + threadIdx.z;
        int tidb = shared_strides[13] * blockIdx.x + shared_strides[14] * blockIdx.y +
                   shared_strides[15] * blockIdx.z + shared_strides[16] * threadIdx.x +
                   shared_strides[17] * threadIdx.y + threadIdx.z;
        hr[tid] = __hadd2(__hmul2(ha[tida], hx[tidx]), hb[tidb]);
    }
}

// void mul_add(hipStream_t stream,
//              const argument& result,
//              const argument& arg1,
//              const argument& arg2,
//              const argument& arg3)
// {
//     auto type = result.get_shape().type();
//     if(type == shape::half_type)
//     {
//         std::cout << "case1" << std::endl;
//         mul_add_kernel<<<block_num, block_size>>>(
//             arg1.data(), s1e, arg2.data(), s2e, arg3.data(), s3e, result.data(), elem_num);
//     }
//     else
//     {
//         std::cout << "mul_add" << std::endl;
//         nary(stream, result, arg1, arg2, arg3)([](auto x, auto a, auto b)
//                                                    __device__ { return a * x + b; });
//     }
// }

void mul_add(hipStream_t stream,
             const argument& result,
             const argument& arg1,
             const argument& arg2,
             const argument& arg3)
{
    auto sr   = result.get_shape();
    auto s1   = arg1.get_shape();
    auto s2   = arg2.get_shape();
    auto s3   = arg3.get_shape();
    auto type = sr.type();

    if(type == sr.type())
    {
        hip_visit_all(result, arg1, arg2, arg3, sr, s1, s2, s3)(
            [&](auto r, auto i1, auto i2, auto i3, auto dsr, auto ds1, auto ds2, auto ds3) {
                __half2* rp  = reinterpret_cast<__half2*>(r.data());
                __half2* i1p = reinterpret_cast<__half2*>(i1.data());
                __half2* i2p = reinterpret_cast<__half2*>(i2.data());
                __half2* i3p = reinterpret_cast<__half2*>(i3.data());
                gs_launch(stream, sr.elements() / 2)([=](auto i) __device__ {
                    auto idx  = dsr.multi(i);
                    auto idx1 = ds1.index(idx);
                    auto idx2 = ds2.index(idx);
                    auto idx3 = ds3.index(idx);
                    rp[i]     = __hadd2(__hmul2(i1p[idx1], i2p[idx2]), i3p[idx3]);
                });
            });
    }
    else
    {
        nary(stream, result, arg1, arg2, arg3)([](auto x, auto a, auto b)
                                                   __device__ { return a * x + b; });
    }
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
