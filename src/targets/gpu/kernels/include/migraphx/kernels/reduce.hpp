#ifndef MIGRAPHX_GUARD_KERNELS_REDUCE_HPP
#define MIGRAPHX_GUARD_KERNELS_REDUCE_HPP

#include <migraphx/kernels/dpp.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>

namespace migraphx {

// #if MIGRAPHX_HAS_DPP
#if 0

#else
template <class Op,
          class T,
          class F>
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

template<class T, class U>
constexpr auto get_reduce_lens(T input_lens, U ouput_lens)
{
    transform(input_lens, ouput_lens, [](auto x, auto y) {
        if (x == y)
            return 1;
        return y;
    });
}

template<class Op, class T, class Input, class Output, class ReadInput, class WriteOuput>
__device__ void simple_reduce(Op op, T init, Input input, Output output, ReadInput read, WriteOuput write)
{
    auto idx = make_index();
    constexpr auto reduce_elements = get_shape_c<Input>{}.elements() - get_shape_c<Output>{}.elements();
    idx.global_stride(output.get_shape().elements(), [&](auto i) {
        auto out_idx = output.get_shape().multi(i);
        auto it = input.begin_at(out_idx);
        auto r =
            block_reduce(idx, op, init, reduce_elements, [&](auto j) {
                return read(it[j]);
            });
        if(idx.local == 0)
            output[out_idx] = write(r);
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_REDUCE_HPP
