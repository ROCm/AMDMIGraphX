#include <migraphx/gpu/concat.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <migraphx/gpu/device/concat.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace gpu {

shape hip_concat::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.compute_shape(inputs);
}

argument hip_concat::compute(context& ctx,
                             const shape& output_shape,
                             const std::vector<argument>& args) const
{
    std::vector<std::size_t> offsets = op.compute_offsets(output_shape, args);
    return device::concat(ctx.get_stream().get(), output_shape, args, offsets);
}

} // namespace gpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx
