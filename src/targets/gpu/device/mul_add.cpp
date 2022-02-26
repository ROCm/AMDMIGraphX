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

__global__ void mul_add_kernel(void* a, void* x, void* b, void* r, int n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    __half2* ha = reinterpret_cast<__half2*>(a);
    __half2* hb = reinterpret_cast<__half2*>(b);
    __half2* hx = reinterpret_cast<__half2*>(x);
    __half2* hr = reinterpret_cast<__half2*>(r);
    if (id < n)
    {
        hr[id] = __hadd2(__hmul2(ha[id], hx[id]), hb[id]);
    }
}


void mul_add(hipStream_t stream,
             const argument& result,
             const argument& arg1,
             const argument& arg2,
             const argument& arg3)
{
    auto type = result.get_shape().type();
    if (type == shape::half_type)
    {
        auto elem_num = result.get_shape().elements() / 2;
        int block_size = 1024;
        int block_num = (elem_num + block_size - 1) / block_size;
        mul_add_kernel<<<block_num, block_size>>>(arg1.data(), arg2.data(), arg3.data(), result.data(), elem_num);
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
