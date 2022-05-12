#ifndef MIGRAPHX_GUARD_KERNELS_TRANSPOSEQKV_HPP
#define MIGRAPHX_GUARD_KERNELS_TRANSPOSEQKV_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>

namespace migraphx {

template <class T, class U>
struct transposeqkv_settings
{
    T head_size{};
    U reversed_bs{};
};

template <class... Ts>
constexpr transposeqkv_settings<Ts...> make_transposeqkv_settings(Ts... xs)
{
    return {xs...};
}

template <class T, class U, class Settings>
__device__ void transposeqkv(const T& input_t, const U& output_t, Settings st)
{
    // Input:  BxSxKxNxH or SxBxKxNxH
    // Output: KxBxNxSxH
    // K is the number of identical matrix

    auto H = st.head_size;
    auto reversed_bs = st.reversed_bs;

    int n = threadIdx.y;
    int s = blockIdx.x;
    int b = blockIdx.y;
    int m = blockIdx.z;  // matrix id

    const int num_heads = blockDim.y;

    const int sequence_length = gridDim.x;
    const int batch_size = gridDim.y;
    const int chunk_num = gridDim.z;
    const int NH = num_heads * H;
    const int NHS = NH * sequence_length;

    int in_offset = 0;
    if (reversed_bs) {
        const int BNH = NH * batch_size;
        in_offset = n * H + (m + b * chunk_num) * NH + s * BNH * chunk_num;
    } else {
        in_offset = n * H + (m + s * chunk_num) * NH + b * NHS * chunk_num;
    }
    const int out_offset = s * H + n * sequence_length * H + b * NHS + m * NHS * batch_size;

    const int i = threadIdx.x;
    if (i < H) {
        output_t[out_offset + i] = input_t[in_offset + i];
    }
}

} // namespace migraphx
#endif
