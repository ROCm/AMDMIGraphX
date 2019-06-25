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
    return device::logsoftmax(ctx.get_stream().get(), args[1], args[0], op.axis);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
