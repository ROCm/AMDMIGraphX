#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/gather.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument gather(hipStream_t stream, argument result, argument arg1, argument arg2, int axis)
{
    auto axis_index    = (axis < 0) ? (axis + arg1.get_shape().lens().size()) : axis;
    auto& input_shape  = arg1.get_shape();
    auto lens          = input_shape.lens();
    auto axis_dim_size = lens[axis_index];
    lens[axis_index]   = arg2.get_shape().elements();
    shape out_comp_shape{result.get_shape().type(), lens};
    std::size_t nelements = result.get_shape().elements();

    visit_all(result, arg1)([&](auto output, auto input_v) {
        hip_visit_views(input_v, out_comp_shape)([&](auto input, auto out_comp) {
            arg2.visit([&](auto indices) {
                const auto* indices_ptr = device_cast(indices.data());
                auto* output_ptr        = device_cast(output.data());
                gs_launch(stream, nelements, 256)([=](auto i) {
                    auto idx        = out_comp.multi(i);
                    auto in_index   = indices_ptr[idx[axis_index]];
                    in_index        = (in_index < 0) ? in_index + axis_dim_size : in_index;
                    idx[axis_index] = in_index;
                    output_ptr[i]   = input[idx];
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
