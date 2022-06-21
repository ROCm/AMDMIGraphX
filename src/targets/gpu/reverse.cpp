#include <migraphx/gpu/reverse.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/reverse.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_reverse::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.normalize_compute_shape(inputs);
}

argument hip_reverse::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    return device::reverse(ctx.get_stream().get(), args.back(), args[0], op.axes);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
