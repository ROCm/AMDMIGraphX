#include <migraphx/gpu/gather.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <migraphx/gpu/device/concat.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_gather::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.compute_shape(inputs);
}

argument hip_gather::compute(context& ctx,
                             const shape& output_shape,
                             const std::vector<argument>& args) const
{
    return device::gather(ctx.get_stream().get(), output_shape, args, op.axis);
}

shape hip_gather_torch::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.compute_shape(inputs);
}

argument hip_gather_torch::compute(context& ctx,
                                   const shape& output_shape,
                                   const std::vector<argument>& args) const
{
    return device::gather_torch(ctx.get_stream().get(), output_shape, args, op.axis);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
