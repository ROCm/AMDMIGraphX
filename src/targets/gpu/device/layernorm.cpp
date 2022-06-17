/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/gpu/device/layernorm.hpp>
#include <migraphx/gpu/device/reduce.hpp>
#include <migraphx/gpu/device/pow.hpp>
#include <migraphx/gpu/device/fast_div.hpp>

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
__device__ void layernorm(index_int i,
                          index idx,
                          std::size_t block_size_div,
                          index_int relements,
                          Input input,
                          Output output)
{
    using value_type       = decltype(input(idx.local));
    const auto relements_v = relements / vector_size<value_type>{};
    const auto out_idx     = fast_div(i, block_size_div);
    const auto base_idx    = out_idx * relements_v;
    const auto input_idx   = base_idx + idx.local;
    const bool in_range    = idx.local < relements_v;

    auto mean = [&](auto z) {
        auto m = auto_block_reduce<MaxBlockSize>(
                     idx, sum{}, value_type(0), relements_v, [=](auto) { return z; }) /
                 value_type(relements);
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
        const std::size_t block_size_div = encode_divisor(block_size);
        assert(relements_v <= block_size);

        gs_launch(stream, nelements * block_size, block_size)([=](auto i, auto idx) __device__ {
            layernorm<max_block_size>(
                i,
                idx,
                block_size_div,
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
        const std::size_t block_size_div = encode_divisor(block_size);
        assert(relements <= block_size);

        gs_launch(stream, nelements * block_size, block_size)([=](auto i, auto idx) __device__ {
            layernorm<max_block_size>(
                i,
                idx,
                block_size_div,
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
        auto relements    = arg1.get_shape().lens().back();
        auto nelements    = result.get_shape().elements() / relements;
        auto output_shape = result.get_shape();
        auto reduce_output_lens(output_shape.lens());
        reduce_output_lens.back() = 1;

        if((relements % 4) == 0)
            layernorm_vec_impl<4>(
                stream, nelements, relements, input, output, result, arg1, args...);
        else if(relements < 256)
            layernorm_impl(stream, nelements, relements, input, output, result, arg1, args...);
        else
            MIGRAPHX_THROW("No kernel for layernorm");
    };
}

void triadd_layernorm(hipStream_t stream,
                      const argument& result,
                      const argument& arg1,
                      const argument& arg2,
                      const argument& arg3)
{
    layernorm_fusion(stream, result, arg1, arg2, arg3)(
        [](auto x, auto y, auto z) { return x + y + z; }, [](auto x, auto& y, auto...) { y = x; });
}

void layernorm(hipStream_t stream, const argument& result, const argument& arg1)
{
    layernorm_fusion(stream, result, arg1)([](auto x) { return x; },
                                           [](auto x, auto& y, auto) { y = x; });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
