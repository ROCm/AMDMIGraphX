#include <migraphx/gpu/device/nonzero.hpp>
#include <migraphx/gpu/device/float_equal.hpp>
#include <migraphx/gpu/device/scan.hpp>
#include <migraphx/gpu/device/fill.hpp>
#include <migraphx/gpu/device/reduce_ops.hpp>
#include <hip/hip_runtime.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument nonzero(hipStream_t stream, const argument& result, const argument& arg_data)
{
    auto elem_num              = arg_data.get_shape().elements();
    const index_int block_size = 256;
    auto out_elem_num          = result.get_shape().elements();
    hip_visit_all(arg_data, arg_data.get_shape())([&](auto input, auto si) {
        auto* out_ptr    = result.cast<int64_t>();
        const auto* data = device_cast(input.data());
        gs_launch(stream, block_size, block_size)([=](auto, auto idx) __device__ {
            idx.local_stride(out_elem_num, [&](auto j) { out_ptr[j] = 0; });
            block_scan<block_size>(idx,
                                   sum{},
                                   0,
                                   elem_num,
                                   [&](auto j) { return float_equal(data[j], 0) ? 0 : 1; },
                                   [&](auto, auto x) {
                                       auto out_idx = x - 1;
                                       auto index   = si.multi(out_idx);
                                       for(std::size_t k = 0; k < index.size(); ++k)
                                       {
                                           out_ptr[k * elem_num + out_idx] = index[k];
                                       }
                                   });
        });
    });

    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
