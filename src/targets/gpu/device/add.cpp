#include <migraphx/gpu/device/add.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

static bool is_bert(const std::vector<shape>& ss)
{
    auto n_dim = ss.front().lens().size();
    if(n_dim == 2)
    {
        auto stride = ss.at(1).strides();
        return (stride[0] == 0);
    }

    return false;
}

__global__ void add_kernel(void* a, void* b, int n_dim, void* r, int n)
{
    __half2* ha = reinterpret_cast<__half2*>(a);
    __half2* hb = reinterpret_cast<__half2*>(b);
    __half2* hr = reinterpret_cast<__half2*>(r);
    int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n)
    {
        int idb = tid % n_dim;
        hr[tid] = __hadd2(ha[tid], hb[idb]);
    }
}

void add(hipStream_t stream, const argument& result, const argument& arg1, const argument& arg2)
{
    auto sr = result.get_shape();
    std::vector<shape> ss;
    ss.push_back(arg1.get_shape());
    ss.push_back(arg2.get_shape());
    if(sr.type() == shape::half_type and is_bert(ss))
    {
        auto elem_num  = sr.elements() / 2;
        auto last_dim  = sr.lens().back() / 2;
        int block_size = 1024;
        int block_num  = (elem_num + block_size - 1) / block_size;
        add_kernel<<<block_num, block_size>>>(
            arg1.data(), arg2.data(), last_dim, result.data(), elem_num);
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
