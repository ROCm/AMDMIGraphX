#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/reverse.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument reverse(hipStream_t stream, argument result, argument arg1, int64_t axis)
{
    const auto& input_shape = arg1.get_shape(); 
    auto lens               = input_shape.lens();

    auto axis_dim_size      = lens[axis];
    shape out_comp_shape{result.get_shape().type(), lens};
    std::size_t nelements = result.get_shape().elements();

    visit_all(result, arg1)([&](auto output, auto input_val) {
        hip_visit_views(input_val, out_comp_shape)([&](auto input, auto out_comp) {
            auto* output_ptr        = device_cast(output.data());
            gs_launch(stream, nelements, 256)([=](auto i) __device__ {
                auto idx      = out_comp.multi(i);
                auto in_index = output_ptr[idx[axis]];
                in_index      = (in_index < 0) ? axis_dim_size - in_index : in_index;
                idx[axis]     = in_index;
                output_ptr[i] = input[idx];
            });
        });
    });
    

    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
