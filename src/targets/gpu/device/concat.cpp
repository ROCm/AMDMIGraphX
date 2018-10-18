#include <migraph/shape.hpp>
#include <migraph/argument.hpp>
#include <migraph/gpu/device/concat.hpp>
#include <migraph/gpu/device/tensor.hpp>
#include <migraph/gpu/device/launch.hpp>

namespace migraph {
namespace gpu {
namespace device {

argument concat(const migraph::shape& output_shape, std::vector<migraph::argument> args, std::vector<std::size_t> offsets) 
{
    //migraph::argument& result = args.back();
    for(std::size_t l = 0; l < args.size()-1; l++)
    {
        auto argl = args[l];
        std::size_t nelements = argl.get_shape().elements();
        visit_all(args.back(), argl)([&](auto output, auto input) {
                visit_tensor_size(output_shape.lens().size(), [&](auto ndim) {
                auto* outptr      = output.data() + offsets[l];
                const auto* inptr = input.data();
                hip_tensor_descriptor<ndim> desc_input(input.get_shape());
                hip_tensor_descriptor<ndim> desc_output(output.get_shape());
                gs_launch(nelements)([=](auto i) {
                    outptr[desc_output.linear(desc_input.multi(i))] = inptr[i];
                });
            });
        });
    }
    //return result;
    return args.back();
}

} // namespace device
} // namespace gpu
} // namespace migraph

