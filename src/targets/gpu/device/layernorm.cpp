#include <migraphx/gpu/device/layernorm.hpp>
#include <migraphx/gpu/device/reduce.hpp>
#include <migraphx/gpu/device/pow.hpp>
#include <migraphx/gpu/device/fast_div.hpp>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

#ifndef MIGRAPHX_WORKAROUND_NAVI_DPP_SYNC
#if __AMDGCN_WAVEFRONT_SIZE == 32
#define MIGRAPHX_WORKAROUND_NAVI_DPP_SYNC 1
#else
#define MIGRAPHX_WORKAROUND_NAVI_DPP_SYNC 0
#endif
#endif

template <class T>
struct vector_type
{
};

template <class T, index_int N>
struct vector_type<vec<T, N>>
{
    using type = T;
};

template <class T>
using vector_type_t = typename vector_type<T>::type;

template <class T>
struct vector_size : std::integral_constant<index_int, 1>
{
};

template <class T, index_int N>
struct vector_size<vec<T, N>> : std::integral_constant<index_int, N>
{
};

template <class T, class F>
__device__ auto vec_transform(T x, F f)
{
    return f(x);
}

template <class T, index_int N, class F>
__device__ auto vec_transform(vec<T, N> x, F f)
{
    vec<T, N> y = x;
    // cppcheck-suppress useStlAlgorithm
    for(index_int k = 0; k < N; k++)
        y[k] = f(x[k]);
    return y;
}

template <class T, class U, class Op>
__device__ auto vec_reduce(T x, U, Op)
{
    return x;
}

template <class T, index_int N, class U, class Op>
__device__ auto vec_reduce(vec<T, N> x, U init, Op op)
{
    T r = init;
    for(index_int k = 0; k < N; k++)
        r = op(r, x[k]);
    return r;
}

template <index_int N, class Op, class T, class F>
__device__ auto auto_block_reduce(index idx, Op op, T init, index_int n, F f)
{
    auto r = block_reduce<N>(idx, op, init, n, f);
    return vec_reduce(r, 0, op);
}

template <index_int MaxBlockSize, class Input, class Output>
__device__ void layernorm(index idx,
                          index_int relements,
                          Input input,
                          Output output)
{
    using value_type       = decltype(input(idx.local));
    const auto relements_v = relements / vector_size<value_type>{};
    const auto out_idx     = blockIdx.x;
    const auto base_idx    = out_idx * relements_v;
    const auto input_idx   = base_idx + idx.local;
    const bool in_range    = idx.local < relements_v;

    auto mean = [&](auto z) {
        auto m = auto_block_reduce<MaxBlockSize>(idx, sum{}, value_type(0), relements_v, [=](auto) {
            return z / value_type(relements);
        });
#if MIGRAPHX_WORKAROUND_NAVI_DPP_SYNC
        __builtin_amdgcn_s_barrier();
#endif
        return m;
    };

    // m = x - mean(x)
    value_type x = in_range ? input(input_idx) : 0;
    value_type m = x - mean(x);

    // mean(m ^ 2) + 1e-12
    value_type r = mean(m * m) + value_type(1e-12);

    // m * rsqrt(mean(m ^ 2) + 1e-12)
    if(in_range)
        output(input_idx, m * vec_transform(r, &rsqrt));
}

// m = x - mean(x)
// m / sqrt(mean(m ^ 2) + 1e-12)

template <index_int N, class Input, class Output, class... Arguments>
void layernorm_vec_impl(hipStream_t stream,
                        index_int nelements,
                        index_int relements,
                        Input in,
                        Output out,
                        const argument& result,
                        const Arguments&... args)
{
    hip_vec_visit_all<N>(result, args...)([&](auto output, auto... inputs) {
        const auto relements_v           = relements / N;
        const std::size_t max_block_size = 256;
        const std::size_t block_size     = compute_block_size(relements_v, max_block_size);
        assert(relements_v <= block_size);

        gs_launch(stream, nelements * block_size, block_size)([=](auto, auto idx) __device__ {
            layernorm<max_block_size>(
                idx,
                relements,
                [&](auto input_idx) { return in(inputs.data()[input_idx]...); },
                [&](auto input_idx, auto x) {
                    out(x, output.data()[input_idx], inputs.data()[input_idx]...);
                });
        });
    });
}

template <class Input, class Output, class... Arguments>
void layernorm_impl(hipStream_t stream,
                    index_int nelements,
                    index_int relements,
                    Input in,
                    Output out,
                    const argument& result,
                    const Arguments&... args)
{
    hip_visit_all(result, args...)([&](auto output, auto... inputs) {
        const std::size_t max_block_size = 256;
        const std::size_t block_size     = compute_block_size(relements, max_block_size);
        assert(relements <= block_size);

        gs_launch(stream, nelements * block_size, block_size)([=](auto, auto idx) __device__ {
            layernorm<max_block_size>(
                idx,
                relements,
                [&](auto input_idx) { return in(inputs.data()[input_idx]...); },
                [&](auto input_idx, auto x) {
                    out(x, output.data()[input_idx], inputs.data()[input_idx]...);
                });
        });
    });
}

template <class... Arguments>
auto layernorm_fusion(hipStream_t stream,
                      const argument& result,
                      const argument& arg1,
                      const Arguments&... args)
{
    return [=](auto input, auto output) {
        auto relements = arg1.get_shape().lens().back();
        auto nelements = result.get_shape().elements() / relements;
        if((relements % 4) == 0)
            layernorm_vec_impl<4>(
                stream, nelements, relements, input, output, result, arg1, args...);
        else if(relements < 256)
            layernorm_impl(stream, nelements, relements, input, output, result, arg1, args...);
        else
            MIGRAPHX_THROW("No kernel for layernorm");
    };
}

struct half2_sum
{
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(__half2 x, __half2 y) const { return __hadd2(x, y); }
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

// m = x - mean(x)
// m / sqrt(mean(m ^ 2) + 1e-12)
__global__ void triadd_layernorm_kernel_half2(
    void* in1, void* in2, void* in3, void* data_out, index_int batch_item_num, index_int block_size)
{
    __half2* input1 = reinterpret_cast<__half2*>(in1);
    __half2* input2 = reinterpret_cast<__half2*>(in2);
    __half2* input3 = reinterpret_cast<__half2*>(in3);
    __half2* output = reinterpret_cast<__half2*>(data_out);
    auto rnum       = __float2half2_rn(1.0f / batch_item_num);
    batch_item_num /= 2;
    extern MIGRAPHX_DEVICE_SHARED __half2 buffer2[];
    __half2* in_data_reduce = buffer2;
    __half2* in_data        = buffer2 + batch_item_num;

    int start = blockIdx.x * batch_item_num;
    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        int idx           = i + start;
        in_data[i]        = __hadd2(__hadd2(input1[idx], input2[idx]), input3[idx]);
        in_data_reduce[i] = in_data[i];
        // in_data_reduce[i] = __hmul2(in_data[i], rnum);
    }

    auto m =
        block_reduce_half2(in_data_reduce, batch_item_num, threadIdx.x, block_size, half2_sum{});
    m = __hmul2(m, rnum);

    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        in_data[i] = __hsub2(in_data[i], m);
        // in_data_reduce[i] = __hmul2(__hmul2(in_data[i], in_data[i]), rnum);
        in_data_reduce[i] = __hmul2(in_data[i], in_data[i]);
    }

    m = block_reduce_half2(in_data_reduce, batch_item_num, threadIdx.x, block_size, half2_sum{});
    m = __hmul2(m, rnum);

    auto eps = __float2half2_rn(1.0e-12f);
    auto r   = __hadd2(m, eps);
    r        = h2rsqrt(r);

    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        int idx     = i + start;
        output[idx] = __hmul2(in_data[i], r);
    }
}

template <class T>
__device__ T
block_reduce_half(T* buffer, index_int batch_item_num, index_int tid, index_int block_size)
{
    __syncthreads();
    for(index_int s = block_size; s > 0; s >>= 1)
    {
        if(tid < s and tid + s < batch_item_num)
        {
            buffer[tid] = __float2half(__half2float(buffer[tid]) + __half2float(buffer[tid + s]));
        }
        __syncthreads();
    }

    return buffer[0];
}

// m = x - mean(x)
// m / sqrt(mean(m ^ 2) + 1e-12)
__global__ void triadd_layernorm_kernel_half(
    void* in1, void* in2, void* in3, void* data_out, index_int batch_item_num, index_int block_size)
{
    __half* input1 = reinterpret_cast<__half*>(in1);
    __half* input2 = reinterpret_cast<__half*>(in2);
    __half* input3 = reinterpret_cast<__half*>(in3);
    __half* output = reinterpret_cast<__half*>(data_out);
    extern MIGRAPHX_DEVICE_SHARED __half bufferh[];
    __half* in_data_reduce = bufferh;
    __half* in_data        = bufferh + batch_item_num;

    int start = blockIdx.x * batch_item_num;
    auto rnum = 1.0f / batch_item_num;
    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        int idx           = i + start;
        in_data[i]        = __float2half(__half2float(input1[idx]) + __half2float(input2[idx]) +
                                  __half2float(input3[idx]));
        in_data_reduce[i] = __float2half(__half2float(in_data[i]) * __half2float(rnum));
    }

    auto m = block_reduce_half(in_data_reduce, batch_item_num, threadIdx.x, block_size);
    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        in_data[i] = __float2half(__half2float(in_data[i]) - __half2float(m));
        in_data_reduce[i] =
            __float2half(__half2float(in_data[i]) * __half2float(in_data[i]) * __half2float(rnum));
    }

    m = __float2half(
        __half2float(block_reduce_half(in_data_reduce, batch_item_num, threadIdx.x, block_size)) +
        1.0e-12f);
    auto r = __float2half(rsqrt(__half2float(m)));

    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        int idx     = i + start;
        output[idx] = __float2half(__half2float(in_data[i]) * __half2float(r));
    }
}

template <class T>
__device__ T block_reduce(T* buffer, index_int batch_item_num, index_int tid, index_int block_size)
{
    __syncthreads();
    for(index_int s = block_size; s > 0; s >>= 1)
    {
        if(tid < s and tid + s < batch_item_num)
        {
            buffer[tid] = buffer[tid] + buffer[tid + s];
        }
        __syncthreads();
    }

    return buffer[0];
}

// m = x - mean(x)
// m / sqrt(mean(m ^ 2) + 1e-12)
template <class T>
__global__ void triadd_layernorm_kernel(
    void* in1, void* in2, void* in3, void* data_out, index_int batch_item_num, index_int block_size)
{
    T* input1 = reinterpret_cast<T*>(in1);
    T* input2 = reinterpret_cast<T*>(in2);
    T* input3 = reinterpret_cast<T*>(in3);
    T* output = reinterpret_cast<T*>(data_out);
    extern MIGRAPHX_DEVICE_SHARED T buffer[];
    T* in_data_reduce = buffer;
    T* in_data        = buffer + batch_item_num;

    int start = blockIdx.x * batch_item_num;
    auto rnum = 1.0f / batch_item_num;
    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        int idx           = i + start;
        in_data[i]        = input1[idx] + input2[idx] + input3[idx];
        in_data_reduce[i] = in_data[i];
        // in_data_reduce[i] = __half2float(in_data[i]) * rnum;
    }

    auto m = block_reduce(in_data_reduce, batch_item_num, threadIdx.x, block_size);
    m      = m * rnum;
    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        in_data[i]        = in_data[i] - m;
        in_data_reduce[i] = in_data[i] * in_data[i];
        // in_data_reduce[i] = __half2float(in_data[i] * in_data[i]) * rnum;
    }
    m      = block_reduce(in_data_reduce, batch_item_num, threadIdx.x, block_size);
    m      = m * rnum + 1.0e-12f;
    auto r = rsqrt(m);

    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        int idx = i + start;
        // output[idx] = __half2float(in_data[i]) * r;
        output[idx] = in_data[i] * r;
    }
}

void triadd_layernorm(hipStream_t stream,
                      const argument& result,
                      const argument& arg1,
                      const argument& arg2,
                      const argument& arg3)
{
    auto in_s           = arg1.get_shape();
    auto type           = in_s.type();
    auto batch_item_num = in_s.lens().back();
    if(type == shape::half_type and (batch_item_num % 2) == 0)
    {
        auto half2_block_size = compute_block_size(batch_item_num, 1024);
        int block_num         = in_s.elements() / batch_item_num;
        int shared_size       = batch_item_num * 2 * in_s.type_size();
        half2_block_size      = half2_block_size / 4;
        triadd_layernorm_kernel_half2<<<block_num, half2_block_size, shared_size, stream>>>(
            arg1.data(), arg2.data(), arg3.data(), result.data(), batch_item_num, half2_block_size);
    }
    else
    {
        layernorm_fusion(stream, result, arg1, arg2, arg3)(
            [](auto x, auto y, auto z) { return x + y + z; },
            [](auto x, auto& y, auto...) { y = x; });
    }
}

__global__ void
layernorm_kernel_half2(void* in1, void* data_out, index_int batch_item_num, index_int block_size)
{
    __half2* input1 = reinterpret_cast<__half2*>(in1);
    __half2* output = reinterpret_cast<__half2*>(data_out);
    auto rnum       = __float2half2_rn(1.0f / batch_item_num);
    batch_item_num /= 2;
    extern MIGRAPHX_DEVICE_SHARED __half2 buffer2[];
    __half2* in_data_reduce = buffer2;
    __half2* in_data        = buffer2 + batch_item_num;

    int start = blockIdx.x * batch_item_num;
    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        int idx           = i + start;
        in_data[i]        = input1[idx];
        in_data_reduce[i] = in_data[i];
    }

    auto m =
        block_reduce_half2(in_data_reduce, batch_item_num, threadIdx.x, block_size, half2_sum{});
    m = __hmul2(m, rnum);

    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        in_data[i]        = __hsub2(in_data[i], m);
        in_data_reduce[i] = __hmul2(in_data[i], in_data[i]);
    }

    m = block_reduce_half2(in_data_reduce, batch_item_num, threadIdx.x, block_size, half2_sum{});
    m = __hmul2(m, rnum);

    auto eps = __float2half2_rn(1.0e-12f);
    auto r   = __hadd2(m, eps);
    r        = h2rsqrt(r);

    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        int idx     = i + start;
        output[idx] = __hmul2(in_data[i], r);
    }
}

void layernorm(hipStream_t stream, const argument& result, const argument& arg1)
{
    auto in_s           = arg1.get_shape();
    auto type           = in_s.type();
    auto batch_item_num = in_s.lens().back();
    if(type == shape::half_type and (batch_item_num % 2) == 0)
    {
        auto half2_block_size = compute_block_size(batch_item_num, 1024);
        int block_num         = in_s.elements() / batch_item_num;
        int shared_size       = batch_item_num * 2 * in_s.type_size();
        half2_block_size      = half2_block_size / 4;
        layernorm_kernel_half2<<<block_num, half2_block_size, shared_size, stream>>>(
            arg1.data(), result.data(), batch_item_num, half2_block_size);
    }
    else
    {
        layernorm_fusion(stream, result, arg1)([](auto x) { return x; },
                                               [](auto x, auto& y, auto) { y = x; });
    }
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
