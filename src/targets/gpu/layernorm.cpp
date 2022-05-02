#include <migraphx/gpu/layernorm.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/layernorm.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_layernorm::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.normalize_compute_shape(inputs);
}

argument hip_layernorm::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    device::layernorm(ctx.get_stream().get(), args.back(), args[0]);
    
    return args.back();
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
