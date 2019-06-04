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

argument gather(hipStream_t stream,
                const migraphx::shape& output_shape,
                std::vector<migraphx::argument> args,
                int axis)
{
    auto axis_index = (axis < 0) ? (axis + args[0].get_shape().lens().size()) : axis;
    visit_all(args.back(), args[0])([&](auto output, auto input) {
        std::size_t nelements = output_shape.elements();
        args[1].visit([&](auto indices) {
            const auto* indices_ptr = device_cast(indices.data());
            auto* out_ptr           = device_cast(output.data());
            const auto* in_ptr      = device_cast(input.data());
            auto& input_shape       = args[0].get_shape();
            auto lens               = input_shape.lens();
            lens[axis_index]        = args[1].get_shape().elements();
            migraphx::shape out_comp_shape{output_shape.type(), lens};
            visit_tensor_size(out_comp_shape.lens().size(), [&](auto n_out_dim) {
                hip_tensor_descriptor<n_out_dim> desc_input(input_shape);
                hip_tensor_descriptor<n_out_dim> desc_output(out_comp_shape);
                gs_launch(stream, nelements)([=](auto ii) {
                    auto in_idx        = desc_output.multi(ii);
                    in_idx[axis_index] = indices_ptr[in_idx[axis_index]];
                    out_ptr[ii]        = in_ptr[desc_input.linear(in_idx)];
                });
            });
        });
    });

    return args.back();
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
