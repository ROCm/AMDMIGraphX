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
    int axis_index = (axis < 0) ? (axis + output_shape.lens().size()) : axis;
    visit_all(args.back(), args[0])([&](auto output, auto input) {
        std::size_t nelements = output_shape.elements();
        args[1].visit([&](auto indices) {
            visit_tensor_size(output_shape.lens().size(), [&](auto ndim) {
                const auto* indices_ptr = device_cast(indices.data());
                auto* outptr            = device_cast(output.data());
                const auto* inptr       = device_cast(input.data());
                hip_tensor_descriptor<ndim> desc_input(input.get_shape());
                hip_tensor_descriptor<ndim> desc_output(output.get_shape());
                gs_launch(stream, nelements)([=](auto i) {
                    auto lens        = desc_output.multi(i);
                    lens[axis_index] = indices_ptr[lens[axis_index]];
                    outptr[i]        = inptr[desc_input.linear(lens)];
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
