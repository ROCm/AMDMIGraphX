#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/concat.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace gpu {
namespace device {

argument concat(hipStream_t stream,
                const migraphx::shape& output_shape,
                std::vector<migraphx::argument> args,
                std::vector<std::size_t> offsets)
{
    for(std::size_t l = 0; l < args.size() - 1; l++)
    {
        auto argl             = args[l];
        std::size_t nelements = argl.get_shape().elements();
        visit_all(args.back(), argl)([&](auto output, auto input) {
            visit_tensor_size(output_shape.lens().size(), [&](auto ndim) {
                auto* outptr      = output.data() + offsets[l];
                const auto* inptr = input.data();
                hip_tensor_descriptor<ndim> desc_input(input.get_shape());
                hip_tensor_descriptor<ndim> desc_output(output.get_shape());
                gs_launch(stream, nelements)(
                    [=](auto i) { outptr[desc_output.linear(desc_input.multi(i))] = inptr[i]; });
            });
        });
    }
    return args.back();
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx
