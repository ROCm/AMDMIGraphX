#include <migraphx/gpu/gathernd.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/gathernd.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_gathernd::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.compute_shape(inputs);
}

argument hip_gathernd::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    return device::gathernd(ctx.get_stream().get(), args.back(), args[0], args[1], op.batch_dims);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx