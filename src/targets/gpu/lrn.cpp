#include <migraphx/gpu/lrn.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape miopen_lrn::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(2).not_broadcasted();
    return inputs.at(1);
}

argument miopen_lrn::compute(context& ctx,
                             const shape& output_shape,
                             const std::vector<argument>& args) const
{
    float alpha = 1;
    float beta  = 0;
    auto x_desc = make_tensor(args[0].get_shape());
    auto y_desc = make_tensor(output_shape);
    miopenLRNForward(ctx.get_stream().get_miopen(),
                     ldesc.get(),
                     &alpha,
                     x_desc.get(),
                     args[0].implicit(),
                     &beta,
                     y_desc.get(),
                     args[1].implicit(),
                     false,
                     nullptr);

    return args[1];
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
