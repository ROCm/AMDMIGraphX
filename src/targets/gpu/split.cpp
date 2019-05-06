#include <migraphx/gpu/split.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/split.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_split::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    check_shapes{inputs, *this}.has(2);
    op.compute_shape(inputs);
    return op.compute_shape(inputs);
}

argument
hip_split::compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
{
    unsigned offset = 0;
    if(op.slice_selector.first >= 0)
        offset = op.compute_offset(args.at(0).get_shape());
    return device::split(ctx.get_stream().get(), output_shape, args, offset);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
