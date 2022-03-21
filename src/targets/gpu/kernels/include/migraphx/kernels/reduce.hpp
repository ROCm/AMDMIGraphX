#ifndef MIGRAPHX_GUARD_KERNELS_REDUCE_HPP
#define MIGRAPHX_GUARD_KERNELS_REDUCE_HPP

#include <migraphx/kernels/dpp.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/print.hpp>

namespace migraphx {

// #if MIGRAPHX_HAS_DPP
#if 0

#else
template <class Op, class T, class F>
__device__ auto block_reduce(index idx, Op op, T init, index_int n, F f)
{

    using type = decltype(f(0));
    __shared__ type buffer[idx.nlocal()];
    type x = init;
    idx.local_stride(n, [&](auto i) { x = op(x, f(i)); });
    buffer[idx.local] = x;
    __syncthreads();

    for(index_int s = 1; s < idx.nlocal(); s *= 2)
    {
        const index_int index = 2 * s * idx.local;
        if(index + s < idx.nlocal())
        {
            buffer[index] = op(buffer[index], buffer[index + s]);
        }
        __syncthreads();
    }
    return buffer[0];
}
#endif

template <class Input, class T, class Output>
constexpr auto reduce_slice(Input input, T i, Output output)
{
    auto lens = transform(
        input.get_shape().lens, output.get_shape().lens, [](index_int x, index_int y) -> index_int {
            if(x == y)
                return 1;
            return y;
        });
    ;
    auto s = make_shape(lens, input.get_shape().strides);
    return make_tensor_view(&input[i], s);
}

template <class Op, class T, class Input, class Output, class ReadInput, class WriteOuput>
__device__ void
simple_reduce(Op op, T init, Input input, Output output, ReadInput read, WriteOuput write)
{
    auto idx = make_index();
    idx.global_stride(output.get_shape().elements(), [&](auto i) {
        auto rs = reduce_slice(input, i, output);
        auto r  = block_reduce(
            idx, op, init, rs.get_shape().elements(), [&](auto j) { return read(rs[j]); });
        if(idx.local == 0)
            output[i] = write(r);
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_REDUCE_HPP
