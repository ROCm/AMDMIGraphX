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
          class Output>
__device__ void block_scan(index idx, Op op, T init, ForStride fs, Input input, Output output)
{
    using type = decltype(input(deduce_for_stride(fs)));
    MIGRAPHX_DEVICE_SHARED type buffer[N];
    type x = init;
    fs([&](auto i) {
        buffer[idx.local] = op(input(i), x);
        __syncthreads();

        for(index_int s = 1; s < idx.nlocal(); s *= 2)
        {
            const index_int index = 2 * s * idx.local;
            if(index + s < idx.nlocal())
            {
                buffer[index + s] = op(buffer[index], buffer[index + s]);
            }
            __syncthreads();
        }
        x = buffer[N-1];
        output(i, buffer[idx.local]);
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_DEVICE_SCAN_HPP
