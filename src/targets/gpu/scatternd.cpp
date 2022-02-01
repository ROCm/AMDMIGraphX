#include <migraphx/gpu/scatternd.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/scatternd.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_scatternd::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.compute_shape(inputs);
}

argument hip_scatternd::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    return device::scatternd(ctx.get_stream().get(), args.back(), args[0], args[1], args[2], op.reduction);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
