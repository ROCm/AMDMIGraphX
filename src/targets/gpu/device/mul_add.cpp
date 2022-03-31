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

__global__ void mul_add_kernel_dim3(void* a, void* x, void* b, int dim3, void* r, int n)
{
    int id      = blockDim.x * blockIdx.x + threadIdx.x;
    __half2* ha = reinterpret_cast<__half2*>(a);
    __half2* hb = reinterpret_cast<__half2*>(b);
    __half2* hx = reinterpret_cast<__half2*>(x);
    __half2* hr = reinterpret_cast<__half2*>(r);
    if(id < n)
    {
        auto id1 = id % dim3;
        hr[id]   = __hfma2(ha[id], hx[id1], hb[id1]);
    }
}

__global__ void mul_add_kernel_dim4(void* a, void* x, void* b, int factor, int dim4, void* r, int n)
{
    int id      = blockDim.x * blockIdx.x + threadIdx.x;
    __half2* ha = reinterpret_cast<__half2*>(a);
    __half2* hb = reinterpret_cast<__half2*>(b);
    __half2* hx = reinterpret_cast<__half2*>(x);
    __half2* hr = reinterpret_cast<__half2*>(r);
    if(id < n)
    {
        int idb = id / (factor * dim4) * dim4 + id % dim4;
        hr[id]  = __hfma2(ha[id], hx[id], hb[idb]);
    }
}

static bool is_bert(const std::vector<shape>& ss)
{
    auto last_dim = ss.front().lens().back();
    if(last_dim % 2 != 0)
    {
        return false;
    }

    auto n_dim = ss.front().lens().size();
    if(n_dim == 3)
    {
        auto stride = ss.at(2).strides();
        return (stride[1] == 0);
    }
    else if(n_dim == 2)
    {
        auto stride1 = ss.at(1).strides();
        auto stride2 = ss.at(2).strides();
        return (stride1 == stride2 and stride1[0] == 0);
    }

    return false;
}

void mul_add(hipStream_t stream,
             const argument& result,
             const argument& arg1,
             const argument& arg2,
             const argument& arg3)
{
    auto sr   = result.get_shape();
    auto type = sr.type();

    std::vector<shape> ss;
    ss.push_back(arg1.get_shape());
    ss.push_back(arg2.get_shape());
    ss.push_back(arg3.get_shape());
    auto lens    = sr.lens();
    int last_dim = lens.back() / 2;
    auto n_dim   = lens.size();
    if(type == shape::half_type and is_bert(ss))
    {
        auto elem_num  = sr.elements() / 2;
        int block_size = 1024;
        int block_num  = (elem_num + block_size - 1) / block_size;
        if(n_dim == 2)
        {
            mul_add_kernel_dim3<<<block_num, block_size, 0, stream>>>(
                arg1.data(), arg2.data(), arg3.data(), last_dim, result.data(), elem_num);
        }
        else
        {
            int factor = lens[1];
            mul_add_kernel_dim4<<<block_num, block_size, 0, stream>>>(
                arg1.data(), arg2.data(), arg3.data(), factor, last_dim, result.data(), elem_num);
        }
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
