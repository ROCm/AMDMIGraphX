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
        shape arg_shape{arg.get_shape().type(), arg.get_shape().lens()};
        hip_visit_all(args.back(), arg, arg_shape)([&](auto output, auto input, auto input_shape) {
            gs_launch(stream, nelements)([=] __device__ (auto i) {
                auto input_idx              = input_shape.multi(i);
                auto idx                    = output.get_shape().index(input_idx);
                output.data()[idx + offset] = input[input_idx];
            });
        });
    }
    return args.back();
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
