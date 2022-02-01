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

argument scatternd(hipStream_t stream, argument result, argument arg0, argument arg1, argument arg2)
{
    auto updates_shape = arg2.get_shape();
    auto output_shape  = result.get_shape();
    // k = index length, r = rank(data)
    // k<r => update slices, k=r => update elements
    auto k = arg1.get_shape().lens().back();
    hip_visit_all(result, arg0, arg2)([&](auto output, auto data, auto updates) {
        auto* output_ptr     = device_cast(output.data());
        const auto* data_ptr = device_cast(data.data());
        gs_launch(stream, ds.elements())([=](auto i) __device__ { output_ptr[i] = data_ptr[i]; });
        //hip_visit_all(arg1)([&](auto indices) {
        arg1.visit([&](auto indices_view){
            hip_visit_views(indices_view)([&](auto indices){
                const auto* updates_ptr = device_cast(updates.data());
                const auto* indices_ptr = device_cast(indices.data());
                gs_launch(stream, updates_shape.elements())([=](auto i) __device__ {
                    printf("i: %i", i);
                    auto offset       = updates_shape.multi(i).front();
                    auto* index_start = indices_ptr + (offset * k);
                    auto* index_end   = index_start + k;
                    auto out_idx      = output_shape.multi(i);
                    std::copy(index_start, index_end, out_idx.begin());
                    if(op.reduction == "add")
                        output[output_shape.index(out_idx)] += updates[i];
                    else if(op.reduction == "mul")
                        output[output_shape.index(out_idx)] *= updates[i];
                    else
                        output[output_shape.index(out_idx)] = updates[i];
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