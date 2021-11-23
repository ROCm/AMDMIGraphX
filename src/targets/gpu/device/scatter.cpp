#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/scatter.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument scatter(
    hipStream_t stream, argument result, argument arg0, argument arg1, argument arg2, int64_t axis)
{
    auto ds            = arg0.get_shape();
    auto inds          = arg1.get_shape();
    auto upds          = arg2.get_shape();
    auto axis_dim_size = ds.lens()[axis];
    hip_visit_all(result, arg0, inds, upds)([&](auto output, auto data, auto inds1, auto upds1) {
        auto* output_ptr     = device_cast(output.data());
        const auto* data_ptr = device_cast(data.data());
        gs_launch(stream, ds.elements())([=](auto i) __device__ { output_ptr[i] = data_ptr[i]; });
        hip_visit_all(arg1, arg2)([&](auto indices, auto update) {
            const auto* upd_ptr     = device_cast(update.data());
            const auto* indices_ptr = device_cast(indices.data());
            gs_launch(stream, inds.elements())([=](auto i) __device__ {
                auto out_idx    = inds1.multi(i);
                auto upd_idx = upds1.index(out_idx);
                auto idx      = indices_ptr[i];
                idx           = idx < 0 ? idx + axis_dim_size : idx;
                out_idx[axis] = idx;
                output[out_idx] = upd_ptr[upd_idx];
            });
        });
    });

    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
