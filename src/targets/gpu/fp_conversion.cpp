#include <migraphx/gpu/fp_conversion.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/gather.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_fp_conversion::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.compute_shape(inputs);
}

argument hip_fp_conversion::compute(context& ctx,
                                    const shape& output_shape,
                                    const std::vector<argument>& args) const
{
    return device::fp_conversion(ctx.get_stream().get(), output_shape, args);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
