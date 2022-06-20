#include <migraphx/gpu/pad.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/pad.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_pad::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    check_shapes{inputs, *this}.has(1).standard();
    return op.compute_shape(inputs);
}

argument hip_pad::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    return device::pad(ctx.get_stream().get(), args.back(), args.front(), op.value, op.pads);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
