#include <migraphx/gpu/scatter.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/scatter.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_scatter::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.normalize_compute_shape(inputs);
}

argument hip_scatter::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    return device::scatter(ctx.get_stream().get(), args.back(), args[0], args[1], args[2], op.axis);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
