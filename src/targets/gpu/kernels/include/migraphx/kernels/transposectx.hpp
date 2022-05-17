#ifndef MIGRAPHX_GUARD_KERNELS_TRANSPOSECTX_HPP
#define MIGRAPHX_GUARD_KERNELS_TRANSPOSECTX_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>

namespace migraphx {

template <class T, class U>
__device__ void transposectx(const T& input_t, const U& output_t)
{
    // Input:  BxNxSxH
    // Output: BxSxNxH
    auto index                = make_index();
    auto input_shape          = input_t.get_shape();
    auto lens                 = input_shape.lens;
    const int num_heads       = lens[1];
    const int sequence_length = lens[2];
    int head_size             = lens[3];

    auto idx = input_shape.multi(index.global);

    int n = idx[1];
    int s = idx[2];
    int b = idx[0];

    const int NH         = num_heads * head_size;
    const int NHS        = NH * sequence_length;
    const int out_offset = n * head_size + s * NH + b * NHS;

    if(index.local < 1024)
        output_t[out_offset + idx[3]] = input_t[index.global];
}

} // namespace migraphx
#endif
