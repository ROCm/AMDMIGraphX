#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/concat.hpp>
#include <migraphx/gpu/device/contiguous.hpp>

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
        auto&& arg        = args[j];
        auto offset       = offsets[j];
        auto byte_offset  = offset * arg.get_shape().type_size();
        auto output_shape = shape{
            arg.get_shape().type(), arg.get_shape().lens(), args.back().get_shape().strides()};
        auto output = argument{output_shape, args.back().data() + byte_offset};
        contiguous(stream, std::move(output), arg);
    }
    return args.back();
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
