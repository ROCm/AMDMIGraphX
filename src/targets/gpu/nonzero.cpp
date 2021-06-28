#include <migraphx/gpu/nonzero.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/nonzero.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_nonzero::compute_shape(std::vector<shape> inputs) const
{
    inputs.erase(inputs.begin() + 1, inputs.end());
    return op.compute_shape(inputs);
}

argument hip_nonzero::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    return device::nonzero(ctx.get_stream().get(), args.back(), args.at(1), args.front());
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
