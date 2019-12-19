#include <migraphx/gpu/logsoftmax.hpp>
#include <migraphx/gpu/device/logsoftmax.hpp>
#include <migraphx/op/logsoftmax.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_logsoftmax::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(2).standard();
    return op.compute_shape({inputs.at(0)});
}

argument
hip_logsoftmax::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    auto n_dim = args.front().get_shape().lens().size();
    auto tuned_axis = (op.axis < 0) ? op.axis + n_dim : op.axis;
    device::logsoftmax(ctx.get_stream().get(), args.back(), args.front(), tuned_axis);
    return args.back();
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
