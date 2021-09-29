#include <migraphx/gpu/roialign.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/roialign.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_roialign::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.compute_shape(inputs);
}

argument hip_roialign::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    auto result = args.back();
    std::vector<argument> in_args(args);
    in_args.pop_back();

    return device::roialign(ctx.get_stream().get(),
                            args.back(),
                            in_args,
                            op.coord_trans_mode,
                            op.mode,
                            op.output_height,
                            op.output_width,
                            op.sampling_ratio,
                            op.spatial_scale);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
