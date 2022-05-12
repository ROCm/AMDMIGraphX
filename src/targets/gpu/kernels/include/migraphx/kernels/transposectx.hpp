#ifndef MIGRAPHX_GUARD_KERNELS_TRANSPOSECTX_HPP
#define MIGRAPHX_GUARD_KERNELS_TRANSPOSECTX_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>

namespace migraphx {

template <class T, class U>
struct transposectx_settings
{
    T head_size{};
    U reversed_bs{};
};

template <class... Ts>
constexpr transposectx_settings<Ts...> make_transposectx_settings(Ts... xs)
{
    return {xs...};
}

template <class T, class U, class Settings>
__device__ void transposectx(const T& input_t, const U& output_t, Settings st)
{
    // Input:  BxNxSxH
    // Output: BxSxNxH

    auto head_size = st.head_size;
    auto reversed_bs = st.reversed_bs;
    
    int n = threadIdx.y;
    int s = blockIdx.x;
    int b = blockIdx.y;

    int num_heads = blockDim.y;
    int sequence_length = gridDim.x;

    const int NH = num_heads * head_size;
    const int NHS = NH * sequence_length;
    const int in_offset = s * head_size + n * sequence_length * head_size + b * NHS;

    int out_offset = 0;
    if (reversed_bs) {
        const int batch_size = gridDim.y;
        const int BNH = NH * batch_size;
        out_offset = n * head_size + b * NH + s * BNH;
    } else {
        out_offset = n * head_size + s * NH + b * NHS;

    }

    const int i = threadIdx.x;
    if (i < head_size) {
        output_t[out_offset + i] = input_t[in_offset + i];
    }
}

} // namespace migraphx
#endif
