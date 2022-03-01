#include <migraphx/gpu/device/gelu.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

// x * 0.5 * (1.0 + erf(x / sqrt(2.0)))
template <class T>
auto gelu_fn(T x) __device__
{
    return x * 0.5 * (1 + ::erf(x * M_SQRT1_2));
}

// 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * pow(x, 3))))
template <class T>
auto gelu_fn_new(T x) __device__
{
    return 0.5 * x * (1 + tanh(sqrt(M_2_PI) * (x + 0.044715 * x * x * x)));
}

void gelu(hipStream_t stream, const argument& result, const argument& arg)
{
    nary(stream, result, arg)([](auto x) __device__ { return gelu_fn(to_hip_type(x)); });
}

void gelu_new(hipStream_t stream, const argument& result, const argument& arg)
{
    nary(stream, result, arg)([](auto x) __device__ { return gelu_fn_new(to_hip_type(x)); });
}

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

__global__ void add_gelu_kernel(void* a, void* b, int n_dim, void* r, int n)
{
    __half2* ha = reinterpret_cast<__half2*>(a);
    __half2* hb = reinterpret_cast<__half2*>(b);
    __half2* hr = reinterpret_cast<__half2*>(r);
    int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n)
    {
        int idb       = tid % n_dim;
        auto sum      = __hadd2(ha[tid], hb[idb]);
        __half2 sqrt2 = __float2half2_rn(M_SQRT1_2);
        sum           = __hmul2(sum, sqrt2);
        auto f2       = __half22float2(sum);
        f2 += 1.0f;
        auto h2 = __floats2half2_rn(f2.x, f2.y);

        __half2 point5 = __float2half2_rn(0.5f);
        hr[tid]        = __hmul2(sum, __hmul2(point5, h2));
    }
}

void add_gelu(hipStream_t stream,
              const argument& result,
              const argument& arg1,
              const argument& arg2)
{
    auto sr   = result.get_shape();
    auto type = sr.type();
    std::vector<shape> ss;
    ss.push_back(arg1.get_shape());
    ss.push_back(arg2.get_shape());
    if(type == shape::half_type and is_bert(ss))
    {
        auto elem_num  = sr.elements() / 2;
        auto last_dim  = sr.lens().back() / 2;
        int block_size = 1024;
        int block_num  = (elem_num + block_size - 1) / block_size;
        add_gelu_kernel<<<block_num, block_size>>>(
            arg1.data(), arg2.data(), last_dim, result.data(), elem_num);
    }
    else
    {
        nary(stream, result, arg1, arg2)([](auto x, auto y) __device__ {
            auto sum = to_hip_type(x + y);
            return gelu_fn(sum);
        });
    }
}

void add_gelu_new(hipStream_t stream,
                  const argument& result,
                  const argument& arg1,
                  const argument& arg2)
{
    nary(stream, result, arg1, arg2)([](auto x, auto y) __device__ {
        auto sum = to_hip_type(x + y);
        return gelu_fn(sum);
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
