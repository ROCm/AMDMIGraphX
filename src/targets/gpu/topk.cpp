#include <migraphx/gpu/topk.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/topk.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_topk::compute_shape(std::vector<shape> inputs) const
{
    return op.normalize_compute_shape({inputs.front()});
}

argument hip_topk::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    auto outputs = args.back().get_sub_objects();
    return op.largest ? device::topk_largest(ctx.get_stream().get(),
                                             outputs.front(),
                                             outputs.back(),
                                             args[0],
                                             op.k,
                                             op.axis)
                      : device::topk_smallest(ctx.get_stream().get(),
                                              outputs.front(),
                                              outputs.back(),
                                              args[0],
                                              op.k,
                                              op.axis);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
