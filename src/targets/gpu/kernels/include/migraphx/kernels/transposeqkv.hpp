#ifndef MIGRAPHX_GUARD_KERNELS_TRANSPOSEQKV_HPP
#define MIGRAPHX_GUARD_KERNELS_TRANSPOSEQKV_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>

namespace migraphx {

template <class T, class U>
__device__ void transposeqkv(const T& input_t, const U& output_t)
{
    // Input:  BxSxKxNxH or SxBxKxNxH
    // Output: KxBxNxSxH
    // K is the number of identical matrix

    auto index       = make_index();
    auto input_shape = input_t.get_shape();
    auto lens        = input_shape.lens;

    auto idx = input_shape.multi(index.global);

    const int b = idx[0];
    const int s = idx[1];
    const int m = idx[2];
    const int n = idx[3];
    // const int j = idx[4];

    const int num_heads       = lens[3];
    const int sequence_length = lens[1];
    const int batch_size      = lens[0];
    const int H               = lens[4];
    const int NH              = num_heads * H;
    const int NHS             = NH * sequence_length;

    const int out_offset = s * H + n * sequence_length * H + b * NHS + m * NHS * batch_size;

    if(index.global < input_shape.elements())
    {
        output_t[out_offset + idx[4]] = input_t[index.global];
    }
}

} // namespace migraphx
#endif
