#include <migraphx/gpu/softmax.hpp>
#include <migraphx/gpu/device/softmax.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_softmax::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(2).standard();
    return op.compute_shape({inputs.at(0)});
}

argument hip_softmax::compute(context& ctx,
                              const shape& output_shape,
                              const std::vector<argument>& args) const
{
    return device::softmax(ctx.get_stream().get(), output_shape, args, 1);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
