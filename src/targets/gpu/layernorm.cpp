#include <migraphx/gpu/layernorm.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/layernorm.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_layernorm::compute_shape(std::vector<shape> inputs) const
{
    std::cout << "compute shape" << std::endl;
    inputs.pop_back();
    return op.normalize_compute_shape(inputs);
}

argument hip_layernorm::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    /* if (args.size() == 3)
    {
        auto n_dim      = args.front().get_shape().lens().size();
        auto tuned_axis = tune_axis(n_dim, op.axis, op.name());
        device::layernorm(ctx.get_stream().get(), args.back(), args[0], args[1], args[2], tuned_axis);
    }
    else */
    std::cout << "calling device::ln" << std::endl;
    {
        
        device::layernorm(ctx.get_stream().get(), args.back(), args[0]);
        std::cout << "called device::ln" << std::endl;
    }
    return args.back();
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
