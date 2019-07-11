#include <migraphx/gpu/reduce_mean.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/reduce_mean.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_reduce_mean::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.compute_shape(inputs);
}

argument
hip_reduce_mean::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    device::reduce_mean(ctx.get_stream().get(), args.back(), args.front());
    return args.back();
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
