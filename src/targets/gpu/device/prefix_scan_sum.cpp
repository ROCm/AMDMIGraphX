#include <migraphx/gpu/device/prefix_scan_sum.hpp>
#include <migraphx/gpu/device/scan.hpp>
#include <migraphx/gpu/device/reduce_ops.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void prefix_scan_sum(hipStream_t stream, const argument& result, const argument& arg, int32_t axis)
{
    const index_int block_size = 256;
    const index_int n          = arg.get_shape().lens()[axis];
    auto rlens                 = result.get_shape().lens();
    rlens[axis]                = 1;
    hip_visit_all(result, arg, result.get_shape().with_lens(rlens))(
        [=](auto output, auto input, auto rshape) {
            gs_launch(stream, rshape.elements() * block_size, block_size)(
                [=](auto i, auto idx) __device__ {
                    const auto ridx  = rshape.multi(i / block_size);
                    auto compute_idx = [&](auto j) {
                        auto k  = ridx;
                        k[axis] = j;
                        return k;
                    };
                    block_scan<block_size>(idx,
                                           sum{},
                                           0,
                                           n,
                                           [&](auto j) { return input[compute_idx(j)]; },
                                           [&](auto j, auto x) { output[compute_idx(j)] = x; });
                });
        });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
