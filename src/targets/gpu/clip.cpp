#include <migraphx/gpu/clip.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/clip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_clip::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.compute_shape(inputs);
}

argument hip_clip::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    device::clip(ctx.get_stream().get(), args.back(), args.front(), args.at(1), args.at(2));
    return args.back();
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
