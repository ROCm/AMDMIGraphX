#include <migraphx/gpu/logsoftmax.hpp>
#include <migraphx/gpu/device/log.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape miopen_logsoftmax::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(2).standard();
    return op.compute_shape({inputs.at(0)});
}

argument miopen_logsoftmax::compute(context& ctx,
                                 const shape& output_shape,
                                 const std::vector<argument>& args) const
{
    float alpha = 1;
    float beta  = 0;
    // temporarily reshape the input to a(0)...a(axis-1)
    // and a(axis)....a(n)
    auto lens = output_shape.lens();
    std::size_t batch_size = std::accumulate(
                            lens.begin(), lens.begin() + op.axis, 
                            std::size_t{1}, std::multiplies<std::size_t>());
    std::size_t n_dims = std::accumulate(lens.begin() + op.axis,
                            lens.end(), std::size_t{1}, std::multiplies<std::size_t>());
    migraphx::shape comp_shape{output_shape.type(), {batch_size, n_dims, 1, 1}};
    auto x_desc = make_tensor(args[0].get_shape());
    auto y_desc = make_tensor(output_shape);

    miopenSoftmaxForward(ctx.get_stream().get_miopen(),
                         &alpha,
                         x_desc.get(),
                         args[0].implicit(),
                         &beta,
                         y_desc.get(),
                         args[1].implicit());

    // call the device::log function to perform the log operation
    device::log(ctx.get_stream().get(), args[1], args[0]);

    return args[1];
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
