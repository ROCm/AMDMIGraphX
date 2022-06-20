#ifndef MIGRAPHX_GUARD_DEVICE_SCAN_HPP
#define MIGRAPHX_GUARD_DEVICE_SCAN_HPP

#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/visit.hpp>
#include <migraphx/gpu/device/multi_index.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <index_int N,
          class Op,
          class T,
          class ForStride,
          class Input,
          class Output,
          MIGRAPHX_REQUIRES(not std::is_integral<ForStride>{})>
__device__ void block_scan(index idx, Op op, T init, ForStride fs, Input input, Output output)
{
    using type = decltype(input(deduce_for_stride(fs)));
    MIGRAPHX_DEVICE_SHARED type buffer[N];
    type x = init;
    fs([&](auto i) {
        if(idx.local == 0)
            buffer[idx.local] = op(input(i), x);
        else
            buffer[idx.local] = input(i);
        __syncthreads();
        for(index_int s = 1; s < idx.nlocal(); s *= 2)
        {
            if(idx.local + s < idx.nlocal())
            {
                buffer[idx.local + s] = op(buffer[idx.local], buffer[idx.local + s]);
            }
            __syncthreads();
        }
        x = buffer[idx.nlocal() - 1];
        output(i, buffer[idx.local]);
    });
}

template <index_int N, class Op, class T, class Input, class Output>
__device__ void block_scan(index idx, Op op, T init, index_int n, Input input, Output output)
{
    block_scan<N>(
        idx,
        op,
        init,
        [&](auto f) -> decltype(f(index_int{})) { return idx.local_stride(n, f); },
        input,
        output);
}

template <class F>
constexpr auto reverse_scan(index_int n, F f)
{
    return [=](auto i, auto&&... xs) { return f(n - i - 1, xs...); };
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_DEVICE_SCAN_HPP
