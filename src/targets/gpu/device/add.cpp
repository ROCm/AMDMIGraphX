#include <migraphx/gpu/device/add.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

__global__ void add_kernel(__half* a, __half* b, __half* r, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        r[tid] = a[tid] + b[tid%768];
    }
}

void add(hipStream_t stream, const argument& result, const argument& arg1, const argument& arg2)
{
    auto s2 = arg2.get_shape();
    if (s2.element_space() == 768 and s2.type() == shape::half_type)
    {
        auto elem_num = s2.elements();
        int block_size = 1024;
        int block_num = (elem_num + block_size - 1) / block_size;
        add_kernel<<<block_num, block_size>>>(reinterpret_cast<__half*>(arg1.data()),
                                              reinterpret_cast<__half*>(arg2.data()), 
                                              reinterpret_cast<__half*>(result.data()), elem_num);
    }
    else
    {
        nary(stream, result, arg1, arg2)([](auto x, auto y) __device__ { return x + y; });        
    }
}

void add(hipStream_t stream,
         const argument& result,
         const argument& arg1,
         const argument& arg2,
         const argument& arg3)
{
    nary(stream, result, arg1, arg2, arg3)([](auto x, auto y, auto z)
                                               __device__ { return x + y + z; });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
