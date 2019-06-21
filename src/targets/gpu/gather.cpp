#include <migraphx/gpu/gather.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/gather.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_gather::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.compute_shape(inputs);
}

argument hip_gather::compute(context& ctx,
                             const shape&,
                             const std::vector<argument>& args) const
{
    return device::gather(ctx.get_stream().get(), args.back(), args[0], args[1], op.axis);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
