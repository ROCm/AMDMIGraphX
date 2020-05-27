#include <migraphx/gpu/pooling.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape miopen_pooling::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(2).standard().only_dims(4);
    return op.compute_shape({inputs.at(0)});
}
argument miopen_pooling::compute(context& ctx,
                                 const shape& output_shape,
                                 const std::vector<argument>& args) const
{
    auto x_desc = make_tensor(args[0].get_shape());
    auto y_desc = make_tensor(output_shape);

    float alpha = 1;
    float beta  = 0;

    miopenPoolingForward(ctx.get_stream().get_miopen(),
                         pd.get(),
                         &alpha,
                         x_desc.get(),
                         args[0].implicit(),
                         &beta,
                         y_desc.get(),
                         args[1].implicit(),
                         false,
                         nullptr,
                         0);

    return args[1];
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
