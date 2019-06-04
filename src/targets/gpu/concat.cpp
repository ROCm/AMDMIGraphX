#include <migraphx/gpu/concat.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/concat.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_concat::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.compute_shape(inputs);
}

argument hip_concat::compute(context& ctx,
                             const shape& output_shape,
                             const std::vector<argument>& args) const
{
    std::vector<std::size_t> offsets = op.compute_offsets(output_shape, args);
    return device::concat(ctx.get_stream().get(), output_shape, args, offsets);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
