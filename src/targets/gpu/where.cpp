#include <migraphx/gpu/where.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/where.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_where::compute_shape(std::vector<shape> inputs) const
{
    return op.compute_shape(inputs);
}

argument hip_where::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    return device::where(ctx.get_stream().get(), args[0], args[1], args[2]);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
