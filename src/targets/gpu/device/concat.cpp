#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/concat.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument concat(hipStream_t stream,
                const migraphx::shape&,
                std::vector<migraphx::argument> args,
                std::vector<std::size_t> offsets)
{
    auto ninputs = args.size() - 1;
    for(std::size_t j = 0; j < ninputs; j++)
    {
        auto&& arg            = args[j];
        std::size_t nelements = arg.get_shape().elements();
        auto offset           = offsets[j];
        hip_visit_all(args.back(), arg)([&](auto output, auto input) {
            gs_launch(stream, nelements)([=](auto i) {
                auto idx                    = output.get_shape().index(input.get_shape().multi(i));
                output.data()[idx + offset] = input.data()[i];
            });
        });
    }
    return args.back();
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
