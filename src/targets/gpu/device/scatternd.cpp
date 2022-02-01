#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/scatternd.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument scatternd(hipStream_t stream, argument result, argument arg0, argument arg1, argument arg2, const std::string reduction)
{
    auto k = arg1.get_shape().lens().back();
    bool add = reduction == "add";
    bool mul = reduction == "mul";
    hip_visit_all(result, arg0, arg2, result.get_shape(), arg2.get_shape())([&](auto output, auto data, auto updates, auto output_shape, auto updates_shape) {
        auto* output_ptr     = device_cast(output.data());
        const auto* data_ptr = device_cast(data.data());
        gs_launch(stream, output_shape.elements())([=](auto i) __device__ { output_ptr[i] = data_ptr[i]; });
        arg1.visit([&](auto indices_view) {
            hip_visit_views(indices_view)([&](auto indices) {
                const auto* updates_ptr = device_cast(updates.data());
                const auto* indices_ptr = device_cast(indices.data());
                gs_launch(stream, updates_shape.elements())([=](auto i) __device__ {
                    auto offset  = updates_shape.multi(i).front();
                    auto* index  = indices_ptr + (offset * k);
                    auto out_idx = output_shape.multi(i);
                    for (std::size_t j = 0; j < k; ++j)
                        out_idx[j] = index[j];
                    if(add)
                        output[out_idx] += updates_ptr[i];
                    else if(mul)
                        output[out_idx] *= updates_ptr[i];
                    else
                        output[out_idx] = updates_ptr[i];
                });
            });
        });
    });

    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
