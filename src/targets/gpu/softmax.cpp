#include <migraphx/gpu/softmax.hpp>
#include <migraphx/gpu/device/softmax.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape miopen_softmax::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(2).standard();
    return op.compute_shape({inputs.at(0)});
}

argument miopen_softmax::compute(context& ctx,
                                 const shape& output_shape,
                                 const std::vector<argument>& args) const
{
    float alpha = 1;
    float beta  = 0;
    auto x_desc = make_tensor(args[0].get_shape());
    auto y_desc = make_tensor(output_shape);
    miopenSoftmaxForward(ctx.get_stream().get_miopen(),
                         &alpha,
                         x_desc.get(),
                         args[0].implicit(),
                         &beta,
                         y_desc.get(),
                         args[1].implicit());

    return args[1];
}

shape hip_softmax::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(2).standard();
    return op.compute_shape({inputs.at(0)});
}

argument hip_softmax::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    return device::softmax(ctx.get_stream().get(), args[1], args[0], op.axis);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
